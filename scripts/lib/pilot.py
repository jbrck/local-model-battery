"""Pilot speed: what this host delivers for this model before the budget is set.

One timed generation decides the cap, the per-case timeout and whether the full
battery is worth starting at all.
"""
from __future__ import annotations

from dataclasses import dataclass

from scripts.lib.model import ModelConfig, post_chat

PILOT_ITEMS = 3
PILOT_CAP = 256
CASE_TIMEOUT_FLOOR_S = 600
CASE_TIMEOUT_MARGIN = 1.25
SHORT_PROMPT = "Write the numbers from 1 to 20, one per line, and nothing else."
LONG_PROMPT = ("List 30 short bullet points describing how a bicycle wheel is built and "
               "trued, one point per line, no preamble.")


@dataclass(frozen=True)
class PilotResult:
    """Measured speed and the budget numbers derived from it."""

    tps: float
    latency_s: float
    generated_tokens: int
    requests: int
    failed: int
    stalled: bool
    errors: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return self.tps > 0 and not self.stalled

    def case_timeout_s(self, cap: int) -> int:
        """Per-case wall clock that will not clip a slow model at this cap.

        A timeout below the cap's own wall time is a hidden cap, and it lands differently on
        every host. This is the number the hermesagent20 leg needs.
        """
        if self.tps <= 0:
            return CASE_TIMEOUT_FLOOR_S
        return max(CASE_TIMEOUT_FLOOR_S, int(CASE_TIMEOUT_MARGIN * cap / self.tps))

    def hours_for(self, items: int, cap: int, overhead_s: float = 2.0) -> float:
        """Rough wall time for `items` items at this cap, for planning the session."""
        if self.tps <= 0:
            return float("inf")
        return items * (cap / self.tps + overhead_s) / 3600

    def to_dict(self) -> dict:
        return {
            "tps": round(self.tps, 2), "latency_s": round(self.latency_s, 2),
            "generated_tokens": self.generated_tokens, "requests": self.requests,
            "failed": self.failed, "stalled": self.stalled, "errors": list(self.errors),
            "usable": self.usable,
        }


def run_pilot(cfg: ModelConfig, items: int = PILOT_ITEMS, cap: int = PILOT_CAP) -> PilotResult:
    """Time a few generations and report tokens per second. Never raises.

    `items - 1` short prompts and one long prompt, so the number reflects a long generation
    rather than a one-line answer. A model that never emits tokens is reported as stalled
    instead of as a suspiciously fast one.
    """
    prompts = [SHORT_PROMPT] * max(0, items - 1) + [LONG_PROMPT]
    generated = 0
    elapsed = 0.0
    failed = 0
    errors: list[str] = []
    for prompt in prompts:
        response = post_chat(cfg, [{"role": "user", "content": prompt}], max_tokens=cap,
                             temperature=0.0)
        if not response.ok or not response.answer():
            failed += 1
            errors.append(response.error or "no usable answer")
            continue
        generated += response.completion_tokens
        elapsed += response.elapsed_s
    tps = generated / elapsed if elapsed > 0 else 0.0
    return PilotResult(
        tps=tps,
        latency_s=elapsed / max(1, len(prompts) - failed),
        generated_tokens=generated,
        requests=len(prompts),
        failed=failed,
        stalled=generated == 0,
        errors=tuple(errors[:3]),
    )
