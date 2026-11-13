"""Test package. Present so `from tests.stub_json import ...` resolves for tooling as well
as at runtime.

The battery's leg default endpoint is a live server, which is right for an operator and
wrong for a test: a leg test that forgets --endpoint would call a real model and pass for
the wrong reason, slowly. Point the default at a dead port for the whole test run, so that
mistake shows up as a refused connection instead.
"""
import os
import sys

DEAD_ENDPOINT = "http://127.0.0.1:1/v1"

# Forced rather than setdefault: the test run must point the default at a dead port, whatever
# the operator's shell happens to export. An ambient value used to win silently, which is the
# one case the guard exists to catch.
os.environ["BATTERY_DEFAULT_ENDPOINT"] = DEAD_ENDPOINT

# `unittest discover -s tests` without `-t .` imports the test modules as top-level names, so
# this package body can run *after* `scripts.lib.harness` has already read the environment and
# bound its default to a live port. Rebinding the loaded module keeps the guard true in either
# import order. The documented command is `unittest discover -s tests -t .`, which imports this
# package first and makes the rebind a no-op.
_harness = sys.modules.get("scripts.lib.harness")
if _harness is not None:
    setattr(_harness, "DEFAULT_ENDPOINT", DEAD_ENDPOINT)
