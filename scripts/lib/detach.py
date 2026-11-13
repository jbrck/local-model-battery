"""Detached runs: start the orchestrator in its own session, and read a run back later.

`--detach` re-executes the orchestrator in a new session — start_new_session=True is the
setsid equivalent, so closing the terminal does not kill the run — with stdout and stderr
appended to run.log and the pid written to run.pid. The parent prints where to watch and
the exact resume line, then exits without waiting or polling.

`--status` answers from the run directory alone: the pid file, run-plan.json for the
planned legs, and legs/*.result.json for what actually happened. A pid whose process is
gone is finished, never running. Liveness is asked with os.kill(pid, 0), which touches
nothing. A pid that is alive but does not look like the orchestrator is a pid-reuse
hazard, and the status says so plainly instead of asserting.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.lib.harness import RESULT_SUFFIX

RUN_LOG = "run.log"
RUN_PID = "run.pid"
PLAN_FILE = "run-plan.json"
LEGS_DIR = "legs"
NOT_RUN = "not run"


def child_argv(argv: list[str], script_path: str | None = None) -> list[str]:
    """The child's command line: the same argv minus --detach, under the interpreter."""
    script = script_path or (argv[0] if argv else "run_battery.py")
    rest = [item for item in argv[1:] if item != "--detach"]
    return [sys.executable, script, *rest]


def resume_argv(child: list[str]) -> list[str]:
    """The exact line to continue this run: the child's command plus --resume, once."""
    without = [item for item in child if item != "--resume"]
    return [*without, "--resume"]


def write_pid_file(out_dir: Path, pid: int) -> Path:
    """Record the detached child's pid in the run directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RUN_PID
    path.write_text(f"{pid}\n")
    return path


def read_pid(out_dir: Path) -> int | None:
    """The pid in run.pid, or None when the file is missing or unreadable."""
    path = out_dir / RUN_PID
    if not path.is_file():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def pid_alive(pid: int) -> bool:
    """Is this process alive? Asked with os.kill(pid, 0), which signals nothing."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_state(out_dir: Path, needle: str = "run_battery") -> dict:
    """What the pid file says about the run: finished, running, or unknown.

    `needle` is matched against /proc/<pid>/cmdline when it can be read, to catch a
    reused pid. On systems without /proc the check reports unknown and the detail says
    the assertion rests on the pid file alone.
    """
    pid = read_pid(out_dir)
    if pid is None:
        return {"pid": None, "state": "unknown",
                "detail": f"no readable {RUN_PID} in {out_dir}: the run state is unknown"}
    if not pid_alive(pid):
        return {"pid": pid, "state": "finished",
                "detail": f"pid {pid} is gone: the run is finished, or was stopped"}
    match = _cmdline_mentions(pid, needle)
    if match == "match":
        detail = (f"pid {pid} is alive and its command line mentions the orchestrator: "
                  f"the run is going")
    elif match == "mismatch":
        detail = (f"pid {pid} is alive but its command line does not mention the "
                  f"orchestrator: possible pid reuse — verify that pid {pid} belongs to "
                  f"this run in {out_dir} before trusting 'running'")
    else:
        detail = (f"pid {pid} is alive; this status asserts the pid file is this run's "
                  f"process (pid reuse cannot be ruled out here)")
    return {"pid": pid, "state": "running", "detail": detail}


def _cmdline_mentions(pid: int, needle: str) -> str:
    """'match', 'mismatch', or 'unknown', from /proc/<pid>/cmdline. Best effort."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            cmdline = handle.read().replace(b"\x00", b" ").decode(errors="replace")
    except OSError:
        return "unknown"
    return "match" if needle in cmdline else "mismatch"


def spawn_detached(argv: list[str], out_dir: Path) -> tuple[int, Path]:
    """Start argv in its own session, output appending to run.log, pid in run.pid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / RUN_LOG
    with open(log_path, "a") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    write_pid_file(out_dir, process.pid)
    return process.pid, log_path


def launch(argv: list[str], out_dir: Path,
           script_path: str | None = None) -> tuple[int, Path, list[str]]:
    """Detach this command: spawn the child and return its pid, log, and command line."""
    child = child_argv(argv, script_path)
    pid, log_path = spawn_detached(child, out_dir)
    return pid, log_path, child


def planned_legs(out_dir: Path) -> list[dict]:
    """Every leg the plan named: {'leg': id, 'reason': skip reason or ''}."""
    path = out_dir / PLAN_FILE
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    planned = [{"leg": item.get("id", ""), "reason": ""} for item in payload.get("legs", [])]
    planned += [{"leg": skip.get("leg", ""), "reason": skip.get("reason", "")}
                for skip in payload.get("skips", [])]
    return planned


def read_block(out_dir: Path, leg_id: str) -> dict | None:
    """The leg's own result block, which is the only thing status trusts."""
    path = out_dir / LEGS_DIR / f"{leg_id}{RESULT_SUFFIX}"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def leg_states(out_dir: Path) -> dict[str, dict]:
    """One status per planned leg: what it wrote, or 'not run' with the plan's reason."""
    states = {}
    for item in planned_legs(out_dir):
        leg_id = item["leg"]
        block = read_block(out_dir, leg_id)
        if block:
            states[leg_id] = {"status": block.get("status", NOT_RUN), "reason": ""}
        else:
            states[leg_id] = {"status": NOT_RUN, "reason": item["reason"]}
    return states
