"""Host probe behaviour. Values are host-dependent, so the assertions are about shape."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.lib import env_probe  # noqa: E402
from scripts.lib.env_probe import (  # noqa: E402
    DEFAULT_TOOL_SPECS,
    Gpu,
    HostFacts,
    ToolFact,
    ToolSpec,
    _optional_float,
    _probe_tool,
    _query_gpus,
    _read_meminfo,
    _run,
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


class RunTest(unittest.TestCase):
    """The three ways a probe command can fail: missing, hanging, or unrunnable."""

    def test_a_missing_binary_returns_none(self):
        self.assertIsNone(_run(["definitely-not-installed-xyz"]))

    def test_a_command_that_hangs_is_cut_off_at_the_timeout(self):
        started = time.monotonic()
        self.assertIsNone(_run(["sleep", "60"], timeout_s=1))
        self.assertLess(time.monotonic() - started, 30)

    def test_an_unrunnable_binary_returns_none_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "broken-bin")
            with open(path, "wb") as handle:
                handle.write(b"\x7f-not-an-elf\x00\x01")
            os.chmod(path, 0o755)
            self.assertTrue(shutil.which(path))
            self.assertIsNone(_run([path]))


class FakeNvidiaSmiTest(unittest.TestCase):
    """_query_gpus against fake nvidia-smi scripts, so every branch is driven for real."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        # Only the fake directory is on PATH, so absence is also testable.
        self._old_path = os.environ["PATH"]
        os.environ["PATH"] = self.directory

    def tearDown(self):
        os.environ["PATH"] = self._old_path
        shutil.rmtree(self.directory)

    def install_smi(self, script: str) -> None:
        path = os.path.join(self.directory, "nvidia-smi")
        with open(path, "w") as handle:
            handle.write(script)
        os.chmod(path, 0o755)

    def test_a_missing_nvidia_smi_says_why_there_are_no_gpus(self):
        gpus, error = _query_gpus()
        self.assertEqual(gpus, ())
        self.assertIn("not found", error or "")

    def test_a_failing_nvidia_smi_reports_its_stderr(self):
        self.install_smi('#!/bin/sh\necho "driver mismatch" >&2\nexit 1\n')
        gpus, error = _query_gpus()
        self.assertEqual(gpus, ())
        self.assertEqual(error, "driver mismatch")

    def test_a_failing_silent_nvidia_smi_still_names_the_tool(self):
        self.install_smi("#!/bin/sh\nexit 1\n")
        gpus, error = _query_gpus()
        self.assertEqual(gpus, ())
        self.assertEqual(error, "nvidia-smi failed")

    def test_a_clean_exit_with_no_rows_reports_no_devices(self):
        self.install_smi("#!/bin/sh\nexit 0\n")
        gpus, error = _query_gpus()
        self.assertEqual(gpus, ())
        self.assertEqual(error, "nvidia-smi reported no devices")

    def test_unusable_rows_are_skipped_and_good_rows_parsed(self):
        self.install_smi(
            "#!/bin/sh\n"
            "echo 'N/A, junk, 1, 2, 3'\n"
            "echo '0, NVIDIA GeForce RTX 3090, 24576, 19727, 300.00'\n"
            "echo '1, Small Card, 8192, 0, [N/A]'\n"
        )
        gpus, error = _query_gpus()
        self.assertIsNone(error)
        self.assertEqual([gpu.index for gpu in gpus], [0, 1])
        self.assertEqual(gpus[0].vram_free_mb, 24576 - 19727)
        self.assertEqual(gpus[0].power_limit_w, 300.0)
        self.assertIsNone(gpus[1].power_limit_w)


class ToolVersionTest(unittest.TestCase):
    """_probe_tool against real little scripts, so version reporting is measured."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self._old_path = os.environ["PATH"]
        os.environ["PATH"] = f"{self.directory}{os.pathsep}{self._old_path}"

    def tearDown(self):
        os.environ["PATH"] = self._old_path
        shutil.rmtree(self.directory)

    def install(self, name: str, body: str) -> None:
        path = os.path.join(self.directory, name)
        with open(path, "w") as handle:
            handle.write(body)
        os.chmod(path, 0o755)

    def test_version_comes_from_stdout(self):
        self.install("fake-tool", '#!/bin/sh\necho "fake-tool 1.2.3"\n')
        fact = _probe_tool(ToolSpec("fake-tool"))
        self.assertTrue(fact.available)
        self.assertEqual(fact.version, "fake-tool 1.2.3")

    def test_version_comes_from_stderr_when_stdout_is_silent(self):
        self.install("quiet-tool", '#!/bin/sh\necho "quiet-tool 0.1" >&2\n')
        fact = _probe_tool(ToolSpec("quiet-tool"))
        self.assertEqual(fact.version, "quiet-tool 0.1")

    def test_a_failing_tool_reports_a_path_but_no_version(self):
        self.install("grumpy-tool", '#!/bin/sh\necho "usage: no" >&2\nexit 1\n')
        fact = _probe_tool(ToolSpec("grumpy-tool"))
        self.assertTrue(fact.available)
        self.assertIsNone(fact.version)

    def test_a_silent_tool_reports_no_version(self):
        self.install("mute-tool", "#!/bin/sh\nexit 0\n")
        fact = _probe_tool(ToolSpec("mute-tool"))
        self.assertIsNone(fact.version)

    def test_a_broken_binary_keeps_its_path_without_a_version(self):
        path = os.path.join(self.directory, "broken-tool")
        with open(path, "wb") as handle:
            handle.write(b"\x7f-not-an-elf\x00\x01")
        os.chmod(path, 0o755)
        fact = _probe_tool(ToolSpec("broken-tool"))
        self.assertEqual(fact.path, path)
        self.assertIsNone(fact.version)


class MeminfoFixtureTest(unittest.TestCase):
    """_read_meminfo against fixture files shaped like /proc/meminfo."""

    def setUp(self):
        self._old_path = env_probe.MEMINFO_PATH
        self._fixture = None

    def tearDown(self):
        env_probe.MEMINFO_PATH = self._old_path
        if self._fixture:
            os.unlink(self._fixture)

    def point_at(self, text: str) -> None:
        handle = tempfile.NamedTemporaryFile("w", suffix=".meminfo", delete=False)
        handle.write(text)
        handle.close()
        self._fixture = handle.name
        env_probe.MEMINFO_PATH = self._fixture

    def test_totals_swap_and_junk_rows(self):
        self.point_at(
            "MemTotal:       66055992 kB\n"
            "MemFree:         8118904 kB\n"
            "MemAvailable:   51234567 kB\n"
            "SwapTotal:       2097148 kB\n"
            "SwapFree:        1048574 kB\n"
            "HugePages_Total:       0\n"
            "VmallocChunk:   (null)\n"
        )
        total, available, swap = _read_meminfo()
        self.assertEqual(total, round(66055992 / (1024 * 1024), 1))
        self.assertEqual(available, round(51234567 / (1024 * 1024), 1))
        self.assertEqual(swap, round((2097148 - 1048574) / (1024 * 1024), 1))

    def test_a_machine_without_swap_reports_no_swap_used(self):
        self.point_at("MemTotal:       66055992 kB\nMemAvailable:   51234567 kB\n")
        total, available, swap = _read_meminfo()
        self.assertGreater(total or 0, 0)
        self.assertGreater(available or 0, 0)
        self.assertIsNone(swap)

    def test_a_missing_proc_file_reports_unknowns(self):
        env_probe.MEMINFO_PATH = "/nonexistent/meminfo"
        self.assertEqual(_read_meminfo(), (None, None, None))


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
