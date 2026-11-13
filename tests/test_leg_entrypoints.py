"""Every in-repo leg starts the way the orchestrator starts it.

The orchestrator launches a leg as a script, from the repo root rather than from the
script's own directory. A leg that does not put the repo's package on its path dies on
import and the orchestrator can only record it as a failure. The leg's unit tests do not
see this: they import the module, with a sys.path that some other code already fixed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.config import load_toml  # noqa: E402
from scripts.lib.legs import Leg, load_legs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
START_TIMEOUT_S = 120
IMPORT_FAILURE = "No module named"


def in_repo_legs() -> list[Leg]:
    """Every leg this repo is supposed to run itself, as the standard declares it."""
    standard = load_toml(str(ROOT / "standard.toml"))
    return [leg for leg in load_legs(standard, path=str(ROOT / "legs.toml"))
            if leg.implementation == "in_repo"]


class LegEntrypointTest(unittest.TestCase):
    """The entry script, started cold, in a directory that knows nothing about the repo."""

    def test_every_in_repo_entry_starts(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            for leg in in_repo_legs():
                with self.subTest(leg=leg.id):
                    entry = ROOT / leg.entry
                    self.assertTrue(entry.is_file(), f"{leg.id}: {leg.entry} is missing")
                    result = subprocess.run(
                        [sys.executable, str(entry), "--help"],
                        cwd=elsewhere, capture_output=True, text=True,
                        timeout=START_TIMEOUT_S, check=False)
                    detail = result.stderr.strip().splitlines()[-1:] or [""]
                    self.assertNotIn(IMPORT_FAILURE, result.stderr,
                                     f"{leg.id}: {detail[0]}")
                    self.assertEqual(result.returncode, 0, f"{leg.id}: {detail[0]}")


if __name__ == "__main__":
    unittest.main()
