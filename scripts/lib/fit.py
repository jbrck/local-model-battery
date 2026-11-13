"""Fit math: can this model, at this window and this KV type, run on this host.

The answer is a tier, because the interesting cases are the ones that run only by
spilling to system RAM or by leaving the placement to the engine. A model whose weights
fit in VRAM stays tier one even when the standard's window does not: the tier says where
the weights live, `window_is_limited` says whether the standard is reachable.

The cache has two parts. The scaling part grows with the window; the bounded part does not,
because windowed layers stop caching at their window and state-space layers hold a fixed
state. Models in the reference collection use both: Gemma-4-12B spends 40 of 48 layers on a
1024-token window, and Nemotron-3.5-Bolt-74B keeps a growing cache on only 8 of 88 layers.
Treating either as uniform is wrong by 6x and 11x.
"""
from __future__ import annotations

from dataclasses import dataclass

from scripts.lib.env_probe import Gpu

BYTES_PER_GB = 1024 ** 3
# KV cache bits per element. q8_0 stores 8-bit values with 16-bit scales, which rounds to
# 8.5 in practice; the constant stays at the nominal width and the VRAM headroom absorbs it.
KV_BITS = {"f32": 32, "f16": 16, "bf16": 16, "q8_0": 8, "q8": 8, "q4_0": 4, "q4": 4}
VRAM_HEADROOM_GB = 1.0
RAM_HEADROOM_GB = 20.0
# Share of VRAM an engine will spend on weights when it has to offload the rest. An estimate,
# and the planner says so: layer placement is the engine's call, not ours.
OFFLOAD_WEIGHT_FRACTION = 0.9

TIER_VRAM = "T1"
TIER_OFFLOAD = "T2"
TIER_MANAGED = "T3"
TIER_NO_FIT = "T4"
VERDICTS = {
    TIER_VRAM: "fits_in_vram",
    TIER_OFFLOAD: "fits_with_offload",
    TIER_MANAGED: "engine_managed",
    TIER_NO_FIT: "will_not_fit",
}


@dataclass(frozen=True)
class Fit:
    """The tier this model lands in, the largest window it allows, and why."""

    tier: str
    vram_total_gb: float
    vram_free_gb: float
    weights_gb: float
    vram_needed_gb: float
    headroom_gb: float
    max_ctx: int
    preferred_ctx: int
    kv_bytes_per_token: int
    reason: str
    estimate: bool = False
    kv_fixed_bytes: int = 0

    @property
    def verdict(self) -> str:
        return VERDICTS[self.tier]

    @property
    def fits(self) -> bool:
        return self.tier != TIER_NO_FIT

    @property
    def window_is_limited(self) -> bool:
        """True when the host cannot serve the standard's window."""
        return self.max_ctx < self.preferred_ctx

    def to_dict(self) -> dict:
        return {
            "tier": self.tier, "verdict": self.verdict, "estimate": self.estimate,
            "vram_total_gb": round(self.vram_total_gb, 1),
            "vram_free_gb": round(self.vram_free_gb, 1),
            "weights_gb": round(self.weights_gb, 1),
            "vram_needed_gb": round(self.vram_needed_gb, 1),
            "headroom_gb": self.headroom_gb,
            "max_ctx": self.max_ctx, "preferred_ctx": self.preferred_ctx,
            "window_is_limited": self.window_is_limited,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "kv_fixed_bytes": self.kv_fixed_bytes,
            "reason": self.reason,
        }


def parse_kv_bits(name: str) -> int:
    """Bits per KV element for a llama.cpp cache type name."""
    key = (name or "").strip().lower()
    if key not in KV_BITS:
        raise KeyError(f"unknown KV cache type {name!r}; known: {sorted(KV_BITS)}")
    return KV_BITS[key]


def kv_bytes_per_token(n_layers: int, n_kv_heads: int, head_dim: int, kv_bits: int) -> int:
    """KV bytes for one token: two tensors (K and V) per layer, at `kv_bits` each.

    Only for a model whose layers all hold a full-context cache. A probed model carries its
    own element counts instead, because recurrent and windowed layers cost something else.
    """
    return 2 * n_layers * n_kv_heads * head_dim * kv_bits // 8


def bytes_at_ctx(kv_per_token: int, ctx: int, kv_fixed_bytes: int = 0) -> int:
    """Total cache bytes for a window: the scaling part plus the bounded part."""
    return max(0, kv_per_token * ctx) + max(0, kv_fixed_bytes)


def max_ctx_for(available_gb: float, kv_per_token: int, ceiling: int,
                kv_fixed_bytes: int = 0) -> int:
    """Largest window that fits in `available_gb`, capped at `ceiling`.

    The bounded part is spent first, since it does not get smaller as the window does.
    """
    if kv_per_token <= 0 or available_gb <= 0:
        return 0
    budget = available_gb * BYTES_PER_GB - max(0, kv_fixed_bytes)
    if budget <= 0:
        return 0
    return int(min(ceiling, budget / kv_per_token))


def compute_fit(weights_bytes: int, kv_per_token: int, gpus: tuple[Gpu, ...],
                ram_available_gb: float | None, preferred_ctx: int,
                foreign_vram_used_gb: float = 0.0, kv_fixed_bytes: int = 0) -> Fit:
    """Decide the tier and the largest window this host can serve.

    `foreign_vram_used_gb` is VRAM held by other processes. Anything resident takes the
    room before this model gets it, which is why the probe lists compute apps.
    `kv_fixed_bytes` is the cache that does not grow with the window; it is spent before any
    token of window is.
    """
    weights_gb = weights_bytes / BYTES_PER_GB
    vram_total_gb = sum(gpu.vram_total_gb for gpu in gpus)
    vram_free_gb = max(0.0, vram_total_gb - foreign_vram_used_gb)
    kv_at_preferred_gb = bytes_at_ctx(kv_per_token, preferred_ctx, kv_fixed_bytes) / BYTES_PER_GB
    needed_gb = weights_gb + kv_at_preferred_gb + VRAM_HEADROOM_GB

    if not gpus:
        return Fit(tier=TIER_MANAGED, vram_total_gb=0.0, vram_free_gb=0.0, weights_gb=weights_gb,
                   vram_needed_gb=needed_gb, headroom_gb=VRAM_HEADROOM_GB, max_ctx=preferred_ctx,
                   preferred_ctx=preferred_ctx, kv_bytes_per_token=kv_per_token,
                   kv_fixed_bytes=kv_fixed_bytes,
                   reason="no GPU facts available; the engine decides placement",
                   estimate=True)

    if weights_gb + VRAM_HEADROOM_GB <= vram_free_gb:
        spare_gb = vram_free_gb - weights_gb - VRAM_HEADROOM_GB
        max_ctx = max_ctx_for(spare_gb, kv_per_token, preferred_ctx, kv_fixed_bytes)
        if needed_gb <= vram_free_gb:
            reason = (f"weights and the standard window fit in {vram_free_gb:.1f} GB "
                      f"of VRAM")
        else:
            reason = (f"weights fit in {vram_free_gb:.1f} GB of VRAM but the window stops "
                      f"at {max_ctx} tokens; the standard asks for {preferred_ctx}")
        return Fit(tier=TIER_VRAM, vram_total_gb=vram_total_gb, vram_free_gb=vram_free_gb,
                   weights_gb=weights_gb, vram_needed_gb=needed_gb,
                   headroom_gb=VRAM_HEADROOM_GB, max_ctx=max_ctx,
                   preferred_ctx=preferred_ctx, kv_bytes_per_token=kv_per_token,
                   kv_fixed_bytes=kv_fixed_bytes, reason=reason)

    ram_budget_gb = (ram_available_gb or 0.0) - RAM_HEADROOM_GB
    if weights_gb <= vram_free_gb + ram_budget_gb:
        weight_resident_gb = vram_free_gb * OFFLOAD_WEIGHT_FRACTION
        spare_gb = max(0.0, vram_free_gb - weight_resident_gb - VRAM_HEADROOM_GB)
        return Fit(tier=TIER_OFFLOAD, vram_total_gb=vram_total_gb, vram_free_gb=vram_free_gb,
                   weights_gb=weights_gb, vram_needed_gb=needed_gb,
                   headroom_gb=VRAM_HEADROOM_GB,
                   max_ctx=max_ctx_for(spare_gb, kv_per_token, preferred_ctx, kv_fixed_bytes),
                   preferred_ctx=preferred_ctx, kv_bytes_per_token=kv_per_token,
                   kv_fixed_bytes=kv_fixed_bytes,
                   reason=(f"{needed_gb:.1f} GB needed against {vram_free_gb:.1f} GB free, "
                           f"weights spill to RAM; the window is an estimate because the "
                           f"engine picks the split"),
                   estimate=True)

    settled_gb = vram_free_gb + max(0.0, ram_budget_gb)
    return Fit(tier=TIER_NO_FIT, vram_total_gb=vram_total_gb, vram_free_gb=vram_free_gb,
               weights_gb=weights_gb, vram_needed_gb=needed_gb, headroom_gb=VRAM_HEADROOM_GB,
               max_ctx=0, preferred_ctx=preferred_ctx, kv_bytes_per_token=kv_per_token,
               kv_fixed_bytes=kv_fixed_bytes,
               reason=(f"weights are {weights_gb:.1f} GB and the host has "
                       f"{settled_gb:.1f} GB settled memory"), estimate=True)
