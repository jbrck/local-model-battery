"""Host probe behaviour. Values are host-dependent, so the assertions are about shape."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.lib.env_probe import (  # noqa: E402
    DEFAULT_TOOL_SPECS,
    Gpu,
    HostFacts,
    ToolFact,
    ToolSpec,
    _optional_float,
    _probe_tool,
    _read_meminfo,
    probe_host,
)


class GpuTest(unittest.TestCase):
    def test_free_vram_is_total_minus_used(self):
        gpu = Gpu(index=0, name="RTX 3090", vram_total_mb=24576, vram_used_mb=11264)
        self.assertEqual(gpu.vram_free_mb, 13312)
        self.assertAlmostEqual(gpu.vram_total_gb, 24.0)

    def test_used_above_total_never_reports_negative_free(self):
        gpu = Gpu(index=0, name="odd", vram_total_mb=1024, vram_used_mb=2048)
        self.assertEqual(gpu.vram_free_mb, 0)

    def test_to_dict_carries_the_free_figure_the_fit_math_uses(self):
        payload = Gpu(index=1, name="RTX 3080", vram_total_mb=20480, vram_used_mb=0).to_dict()
        self.assertEqual(payload["vram_free_mb"], 20480)
        self.assertEqual(payload["index"], 1)


class ToolTest(unittest.TestCase):
    def test_available_follows_the_path(self):
        self.assertTrue(ToolFact("docker", "/usr/bin/docker").available)
        self.assertFalse(ToolFact("docker").available)

    def test_missing_binary_is_reported_as_unavailable(self):
        fact = _probe_tool(ToolSpec("definitely-not-installed-xyz"))
        self.assertFalse(fact.available)
        self.assertIsNone(fact.version)

    def test_present_binary_reports_a_version_line(self):
        fact = _probe_tool(ToolSpec("python3", ("--version",)))
        if not fact.available:
            self.skipTest("python3 is not on PATH")
        self.assertTrue(fact.version)
        self.assertLessEqual(len(fact.version or ""), 80)

    def test_environment_variable_overrides_the_search_path(self):
        spec = ToolSpec("definitely-not-installed-xyz", ("--version",), "PROBE_TEST_BIN")
        os.environ["PROBE_TEST_BIN"] = "/bin/echo"
        try:
            fact = _probe_tool(spec)
        finally:
            del os.environ["PROBE_TEST_BIN"]
        self.assertEqual(fact.path, "/bin/echo")


class HostFactsTest(unittest.TestCase):
    def test_vram_totals_sum_across_cards(self):
        facts = HostFacts(gpus=(Gpu(0, "a", 20480, 0), Gpu(1, "b", 20480, 0)))
        self.assertAlmostEqual(facts.vram_total_gb, 40.0)
        self.assertTrue(facts.has_gpu)

    def test_tool_lookup_by_name(self):
        facts = HostFacts(tools=(ToolFact("docker", "/usr/bin/docker"),))
        self.assertTrue(facts.tool("docker").available)  # type: ignore[union-attr]
        self.assertIsNone(facts.tool("nope"))

    def test_to_dict_is_json_ready(self):
        facts = HostFacts(gpus=(Gpu(0, "a", 20480, 0),), ram_available_gb=64.0,
                          tools=(ToolFact("docker", "/usr/bin/docker", "Docker version 27"),))
        payload = facts.to_dict()
        self.assertEqual(payload["vram_total_gb"], 20.0)
        self.assertEqual(payload["tools"][0]["available"], True)
        self.assertIn("compute_apps", payload)


class ProcessProbeTest(unittest.TestCase):
    def test_optional_float_accepts_numbers_and_rejects_junk(self):
        self.assertEqual(_optional_float("350.00"), 350.0)
        self.assertIsNone(_optional_float("[N/A]"))
        self.assertIsNone(_optional_float(""))

    def test_meminfo_yields_three_numbers_or_three_nones(self):
        total, available, swap = _read_meminfo()
        if total is None:
            self.assertIsNone(available)
            self.assertIsNone(swap)
        else:
            self.assertGreater(total, 0)
            self.assertGreater(available or 0, 0)


class LiveProbeTest(unittest.TestCase):
    def test_probe_host_returns_a_complete_fact_sheet(self):
        facts = probe_host()
        self.assertGreater(facts.ram_total_gb or 0, 0)
        self.assertTrue(facts.cpu_model)
        self.assertGreater(facts.cpu_threads or 0, 0)
        self.assertEqual([tool.name for tool in facts.tools],
                         [spec.name for spec in DEFAULT_TOOL_SPECS])
        if facts.gpus:
            self.assertIsNone(facts.gpu_error)
        else:
            self.assertTrue(facts.gpu_error)

    def test_probe_host_never_raises_on_a_missing_tool_set(self):
        facts = probe_host(tool_specs=(ToolSpec("definitely-not-installed-xyz"),))
        self.assertEqual(len(facts.tools), 1)
        self.assertFalse(facts.tools[0].available)


if __name__ == "__main__":
    unittest.main()
