"""Run generated Python against a test suite in a subprocess.

Portable replacement for the evalplus harness: writes the module and its check
call to a temp directory, runs it with a hard timeout, and reports the result.
Not a security sandbox — a timeout-bounded subprocess. Run untrusted models on
a disposable machine or inside a container.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

TEST_TIMEOUT_S = 10


def run_python_tests(source: str, entry_point: str, test_code: str,
                     timeout_s: int = TEST_TIMEOUT_S) -> tuple[bool, str]:
    """Execute source + test_code and call entry_point. Returns (passed, detail)."""
    module = f"{source}\n\n{test_code}\n\ncheck({entry_point})\n"
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "candidate.py")
        with open(path, "w") as handle:
            handle.write(module)
        try:
            completed = subprocess.run(
                [sys.executable, path],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout_s}s"

    if completed.returncode == 0:
        return True, "ok"
    detail = (completed.stderr or completed.stdout or "").strip().splitlines()
    return False, detail[-1][:200] if detail else f"exit {completed.returncode}"
