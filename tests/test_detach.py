"""Detached runs: the child outlives its terminal, and --status reads the truth back.

The child is spawned in a new session with start_new_session=True (the setsid
equivalent), output appends to run.log, and the pid lands in run.pid. A pid whose
process is gone is finished, never running. The tests spawn real short-lived children
and reap everything they start.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import run_battery  # noqa: E402
from scripts.lib import detach  # noqa: E402

SLEEPER = "import time; time.sleep(0.5); print('detached child ran')\n"


class TempRun(unittest.TestCase):
    """A throwaway run directory, and the discipline to reap every child."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "run"

    def reap_child(self) -> int:
        """A pid that is certainly dead: spawn, wait, reap."""
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        return proc.pid

    def wait_gone(self, pid: int, deadline_s: float = 10.0) -> bool:
        """Bounded poll until the pid is dead, reaping as we go; never asserts on timing."""
        limit = time.monotonic() + deadline_s
        while time.monotonic() < limit:
            if not detach.pid_alive(pid):
                return True
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                return True  # already reaped, or never ours
            time.sleep(0.05)
        return False

    def kill_if_alive(self, pid: int) -> None:
        if detach.pid_alive(pid):
            os.kill(pid, signal.SIGKILL)
            self.wait_gone(pid)


class ChildArgv(TempRun):
    """The child argv is the same command minus --detach, under the interpreter."""

    def test_strips_detach_and_prefixes_interpreter(self):
        child = detach.child_argv(
            ["scripts/run_battery.py", "--model", "m", "--detach", "--out", "x"],
            script_path="/repo/scripts/run_battery.py")
        self.assertEqual(child, [sys.executable, "/repo/scripts/run_battery.py",
                                 "--model", "m", "--out", "x"])

    def test_defaults_the_script_to_argv0(self):
        child = detach.child_argv(["somewhere/run_battery.py", "--detach"])
        self.assertEqual(child, [sys.executable, "somewhere/run_battery.py"])


class ResumeLine(TempRun):
    """The resume line is the child command with --resume, exactly once."""

    def test_appends_resume_once(self):
        child = detach.child_argv(["run_battery.py", "--out", "x", "--resume"])
        self.assertEqual(detach.resume_argv(child),
                         [sys.executable, "run_battery.py", "--out", "x", "--resume"])

    def test_adds_resume_when_missing(self):
        child = detach.child_argv(["run_battery.py", "--out", "x"])
        self.assertEqual(detach.resume_argv(child)[-1], "--resume")


class PidFiles(TempRun):
    """The pid file is the run's handle on its process."""

    def test_roundtrip(self):
        detach.write_pid_file(self.out, 4242)
        self.assertEqual(detach.read_pid(self.out), 4242)

    def test_missing_pid_file_reads_none(self):
        self.assertIsNone(detach.read_pid(self.out))


class PidAlive(TempRun):
    """Liveness is asked with os.kill(pid, 0), never with a signal."""

    def test_live_pid_is_alive(self):
        self.assertTrue(detach.pid_alive(os.getpid()))

    def test_dead_pid_is_not_alive(self):
        dead = self.reap_child()
        self.assertFalse(detach.pid_alive(dead))


class RunState(TempRun):
    """run_state answers from the pid file and says plainly what it asserts."""

    def test_stale_pid_reports_finished(self):
        dead = self.reap_child()
        detach.write_pid_file(self.out, dead)
        state = detach.run_state(self.out)
        self.assertEqual(state["state"], "finished")
        self.assertEqual(state["pid"], dead)

    def test_live_pid_reports_running(self):
        detach.write_pid_file(self.out, os.getpid())
        state = detach.run_state(self.out)
        self.assertEqual(state["state"], "running")
        self.assertIn(str(os.getpid()), state["detail"])

    def test_missing_pid_file_is_unknown(self):
        state = detach.run_state(self.out)
        self.assertEqual(state["state"], "unknown")
        self.assertIsNone(state["pid"])

    def test_pid_reuse_is_flagged_not_hidden(self):
        """A live pid whose command line is not the orchestrator gets a plain warning."""
        detach.write_pid_file(self.out, os.getpid())
        state = detach.run_state(self.out, needle="definitely-not-this-process")
        self.assertEqual(state["state"], "running")
        self.assertIn("pid reuse", state["detail"])


class SpawnDetached(TempRun):
    """The detached child is in its own session, and the log exists before we return."""

    def test_spawn_leaves_a_new_session_a_log_and_a_pid_file(self):
        pid, log_path = detach.spawn_detached([sys.executable, "-c", SLEEPER], self.out)
        self.addCleanup(self.kill_if_alive, pid)
        self.assertTrue(log_path.is_file(),
                        "the log file must exist before the parent returns")
        self.assertEqual(detach.read_pid(self.out), pid)
        self.assertNotEqual(os.getsid(pid), os.getsid(0),
                            "the child must be in its own session")
        self.assertTrue(self.wait_gone(pid))
        self.assertEqual(detach.run_state(self.out)["state"], "finished")


class LaunchResume(TempRun):
    """The parent's own mkdir must not read back as a previous run to the child."""

    def test_directory_the_parent_creates_starts_resuming(self):
        self.assertFalse(self.out.exists())
        pid, _, child = detach.launch(
            [sys.executable, "-c", SLEEPER, "--detach", "--out", str(self.out)], self.out)
        self.addCleanup(self.kill_if_alive, pid)
        self.assertNotIn("--detach", child)
        self.assertIn("--resume", child,
                      "a fresh run directory is not a previous run: the child resumes")
        self.assertTrue(self.out.is_dir())
        self.assertTrue(self.wait_gone(pid))

    def test_directory_that_was_already_there_is_not_resumed_for_you(self):
        self.out.mkdir(parents=True)
        (self.out / "run-plan.json").write_text(json.dumps({"legs": [], "skips": []}))
        pid, _, child = detach.launch(
            [sys.executable, "-c", SLEEPER, "--detach", "--out", str(self.out)], self.out)
        self.addCleanup(self.kill_if_alive, pid)
        self.assertNotIn("--resume", child,
                         "an existing run directory still has to be resumed on purpose")
        self.assertNotIn("--detach", child)
        self.assertTrue(self.wait_gone(pid))


class LegStates(TempRun):
    """Leg state comes from run-plan.json for the ids and legs/ for the outcomes."""

    def test_reads_plan_and_results(self):
        self.out.mkdir(parents=True)
        (self.out / "run-plan.json").write_text(json.dumps(
            {"legs": [{"id": "math500"}, {"id": "ifbench"}],
             "skips": [{"leg": "writing", "reason": "no judge"}]}))
        legs_dir = self.out / "legs"
        legs_dir.mkdir()
        (legs_dir / "math500.result.json").write_text(json.dumps({"status": "pass"}))
        states = detach.leg_states(self.out)
        self.assertEqual(states["math500"]["status"], "pass")
        self.assertEqual(states["ifbench"]["status"], "not run")
        self.assertEqual(states["writing"]["status"], "not run")
        self.assertEqual(states["writing"]["reason"], "no judge")

    def test_no_plan_means_no_states(self):
        self.assertEqual(detach.leg_states(self.out), {})


class MainDetach(TempRun):
    """--detach starts the child, prints where to watch, and returns 0."""

    def setUp(self):
        super().setUp()
        self.stub = Path(self.tmp.name) / "stub_orchestrator.py"
        self.stub.write_text(SLEEPER)

    def test_detach_prints_pid_log_and_resume_and_returns_zero(self):
        buffer = io.StringIO()
        with mock_argv([str(self.stub), "--detach", "--out", str(self.out)]):
            with contextlib.redirect_stdout(buffer):
                code = run_battery.main(["--detach", "--out", str(self.out)])
        printed = buffer.getvalue()
        self.assertEqual(code, 0)
        pid = detach.read_pid(self.out)
        self.assertIsNotNone(pid)
        assert pid is not None  # for the type checker; the assert above already proved it
        self.addCleanup(self.kill_if_alive, pid)
        self.assertIn(f"run directory: {self.out}", printed)
        self.assertIn(f"pid: {pid}", printed)
        self.assertIn(f"tail -f {self.out / 'run.log'}", printed)
        self.assertIn("resume:", printed)
        self.assertIn("--resume", printed)
        self.assertNotIn("--detach", printed.split("resume:")[-1].splitlines()[0])
        self.assertTrue((self.out / "run.log").is_file())
        self.assertNotEqual(os.getsid(pid), os.getsid(0))
        self.assertTrue(self.wait_gone(pid))

    def test_detach_without_out_is_refused(self):
        with mock_argv(["stub", "--detach"]):
            self.assertEqual(run_battery.main(["--detach"]), 2)


class MainStatus(TempRun):
    """--status reads a run directory with no --detach and exits 0."""

    def test_status_reports_finished_run_and_leg_states(self):
        dead = self.reap_child()
        self.out.mkdir(parents=True)
        detach.write_pid_file(self.out, dead)
        (self.out / "run-plan.json").write_text(json.dumps(
            {"legs": [{"id": "math500"}], "skips": []}))
        legs_dir = self.out / "legs"
        legs_dir.mkdir()
        (legs_dir / "math500.result.json").write_text(json.dumps({"status": "pass"}))
        buffer = io.StringIO()
        with mock_argv(["stub", "--status", "--out", str(self.out)]):
            with contextlib.redirect_stdout(buffer):
                code = run_battery.main(["--status", "--out", str(self.out)])
        printed = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("finished", printed)
        self.assertIn("math500", printed)
        self.assertIn("pass", printed)

    def test_status_without_out_is_refused(self):
        self.assertEqual(run_battery.main(["--status"]), 2)


class UnreadableState(TempRun):
    """Every way the run directory can fail to answer: 'unknown', never a guess at running."""

    def test_pid_file_that_is_not_a_number_reads_as_none(self):
        self.out.mkdir(parents=True)
        (self.out / "run.pid").write_text("not-a-pid\n")
        self.assertIsNone(detach.read_pid(self.out))

    def test_pid_file_that_is_missing_reads_as_none(self):
        self.assertIsNone(detach.read_pid(self.out))

    def test_pid_owned_by_another_user_counts_as_alive(self):
        # os.kill(pid, 0) on a process we may not signal raises PermissionError, which
        # means the process exists. pid 1 does not belong to this user.
        self.assertTrue(detach.pid_alive(1))

    def test_cmdline_of_a_dead_pid_is_unknown(self):
        self.assertEqual(detach._cmdline_mentions(self.reap_child(), "run_battery"), "unknown")

    def test_cmdline_of_this_process_matches_and_mismatches(self):
        self.assertEqual(detach._cmdline_mentions(os.getpid(), "python"), "match")
        self.assertEqual(detach._cmdline_mentions(os.getpid(), "no-such-needle"), "mismatch")

    def test_a_result_block_that_is_not_json_reads_as_none(self):
        legs_dir = self.out / "legs"
        legs_dir.mkdir(parents=True)
        (legs_dir / "math500.result.json").write_text("{not json")
        self.assertIsNone(detach.read_block(self.out, "math500"))

    def test_a_result_block_that_is_missing_reads_as_none(self):
        self.assertIsNone(detach.read_block(self.out, "math500"))

    def test_run_state_is_unknown_without_a_pid_file(self):
        state = detach.run_state(self.out)
        self.assertEqual(state["state"], "unknown")
        self.assertIsNone(state["pid"])
        self.assertIn("run.pid", state["detail"])


class mock_argv:
    """Point sys.argv at a stub for the duration of a with block."""

    def __init__(self, argv):
        self.argv = argv

    def __enter__(self):
        self.saved = sys.argv
        sys.argv = list(self.argv)
        return self

    def __exit__(self, *exc):
        sys.argv = self.saved
        return False


if __name__ == "__main__":
    unittest.main()
