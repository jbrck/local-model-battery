"""Fit math: tiers, windows, and the cases where the answer is 'only by offloading'."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.lib.env_probe import Gpu  # noqa: E402
from scripts.lib.fit import (  # noqa: E402
    BYTES_PER_GB,
    Fit,
    TIER_MANAGED,
    TIER_NO_FIT,
    TIER_OFFLOAD,
    TIER_VRAM,
    compute_fit,
    kv_bytes_per_token,
    max_ctx_for,
    parse_kv_bits,
)

GPU_24 = Gpu(index=0, name="RTX 3090", vram_total_mb=24576, vram_used_mb=0, power_limit_w=350.0)
GPU_20S = (Gpu(index=0, name="RTX 3080", vram_total_mb=20480, vram_used_mb=0),
           Gpu(index=1, name="RTX 3080", vram_total_mb=20480, vram_used_mb=0))
# 27B dense shape: 64 layers, 8 KV heads, 128-wide heads.
LAYERS, KV_HEADS, HEAD_DIM = 64, 8, 128
KV_Q8 = kv_bytes_per_token(LAYERS, KV_HEADS, HEAD_DIM, parse_kv_bits("q8_0"))


class KvMathTest(unittest.TestCase):
    def test_kv_bytes_per_token_counts_both_tensors(self):
        self.assertEqual(kv_bytes_per_token(2, 4, 8, 8), 2 * 2 * 4 * 8 * 8 // 8)
        self.assertEqual(KV_Q8, 131072)

    def test_f16_is_twice_q8(self):
        self.assertEqual(kv_bytes_per_token(LAYERS, KV_HEADS, HEAD_DIM, 16), KV_Q8 * 2)

    def test_unknown_kv_type_raises_and_names_the_known_ones(self):
        with self.assertRaises(KeyError) as caught:
            parse_kv_bits("q3_imaginary")
        self.assertIn("q8_0", str(caught.exception))

    def test_max_ctx_obeys_the_ceiling(self):
        self.assertEqual(max_ctx_for(10.0, KV_Q8, 8192), 8192)

    def test_max_ctx_is_zero_when_nothing_is_left(self):
        self.assertEqual(max_ctx_for(0.0, KV_Q8, 8192), 0)
        self.assertEqual(max_ctx_for(1.0, 0, 8192), 0)


class TierTest(unittest.TestCase):
    def test_small_model_fits_entirely_in_vram(self):
        fit = compute_fit(weights_bytes=6 * BYTES_PER_GB, kv_per_token=KV_Q8,
                          gpus=(GPU_24,), ram_available_gb=64.0, preferred_ctx=32768)
        self.assertEqual(fit.tier, TIER_VRAM)
        self.assertEqual(fit.verdict, "fits_in_vram")
        self.assertFalse(fit.window_is_limited)

    def test_the_standard_window_is_limited_by_a_full_24gb_card(self):
        # 16 GB of weights plus q8_0 KV: the weights fit, the 131072 window does not, and the
        # fit says so instead of starting a run that dies on the first long prompt.
        fit = compute_fit(weights_bytes=16 * BYTES_PER_GB, kv_per_token=KV_Q8,
                          gpus=(GPU_24,), ram_available_gb=64.0, preferred_ctx=131072)
        self.assertEqual(fit.tier, TIER_VRAM)
        self.assertTrue(fit.window_is_limited)
        self.assertLess(fit.max_ctx, 131072)
        self.assertGreater(fit.max_ctx, 32768)
        self.assertIn("window stops at", fit.reason)

    def test_q8_kv_buys_window_over_f16(self):
        def fitted(kv: int) -> Fit:
            return compute_fit(weights_bytes=16 * BYTES_PER_GB, kv_per_token=kv,
                               gpus=(GPU_24,), ram_available_gb=64.0, preferred_ctx=131072)
        self.assertGreater(fitted(KV_Q8).max_ctx, fitted(KV_Q8 * 2).max_ctx)

    def test_moe_bigger_than_vram_fits_by_offloading_to_ram(self):
        fit = compute_fit(weights_bytes=int(75.8 * BYTES_PER_GB), kv_per_token=KV_Q8,
                          gpus=GPU_20S, ram_available_gb=200.0, preferred_ctx=32768)
        self.assertEqual(fit.tier, TIER_OFFLOAD)
        self.assertEqual(fit.verdict, "fits_with_offload")
        self.assertTrue(fit.estimate)
        self.assertIn("spill", fit.reason)

    def test_model_too_big_for_vram_and_ram_does_not_fit(self):
        fit = compute_fit(weights_bytes=300 * BYTES_PER_GB, kv_per_token=KV_Q8,
                          gpus=(GPU_24,), ram_available_gb=16.0, preferred_ctx=8192)
        self.assertEqual(fit.tier, TIER_NO_FIT)
        self.assertEqual(fit.verdict, "will_not_fit")
        self.assertFalse(fit.fits)
        self.assertEqual(fit.max_ctx, 0)

    def test_no_gpu_facts_means_the_engine_decides(self):
        fit = compute_fit(weights_bytes=8 * BYTES_PER_GB, kv_per_token=KV_Q8, gpus=(),
                          ram_available_gb=64.0, preferred_ctx=32768)
        self.assertEqual(fit.tier, TIER_MANAGED)
        self.assertTrue(fit.estimate)
        self.assertIn("engine", fit.reason)

    def test_missing_ram_facts_do_not_make_a_big_model_look_fine(self):
        fit = compute_fit(weights_bytes=75 * BYTES_PER_GB, kv_per_token=KV_Q8,
                          gpus=(GPU_24,), ram_available_gb=None, preferred_ctx=8192)
        self.assertEqual(fit.tier, TIER_NO_FIT)
        self.assertEqual(fit.reason.count("None"), 0)

    def test_foreign_vram_takes_room_from_the_model(self):
        def fitted(foreign: float) -> Fit:
            return compute_fit(weights_bytes=19 * BYTES_PER_GB, kv_per_token=KV_Q8,
                               gpus=(GPU_24,), ram_available_gb=64.0, preferred_ctx=32768,
                               foreign_vram_used_gb=foreign)
        free = fitted(0.0)
        occupied = fitted(11.0)
        self.assertEqual(free.tier, TIER_VRAM)
        self.assertEqual(occupied.tier, TIER_OFFLOAD)
        self.assertLess(occupied.vram_free_gb, free.vram_free_gb)

    def test_fit_serialises_without_losing_the_verdict(self):
        fit = compute_fit(weights_bytes=6 * BYTES_PER_GB, kv_per_token=KV_Q8, gpus=(GPU_24,),
                          ram_available_gb=64.0, preferred_ctx=8192)
        payload = fit.to_dict()
        self.assertEqual(payload["verdict"], "fits_in_vram")
        self.assertEqual(payload["tier"], TIER_VRAM)
        self.assertIn("reason", payload)


if __name__ == "__main__":
    unittest.main()
