"""Test package. Present so `from tests.stub_json import ...` resolves for tooling as well
as at runtime.

The battery's leg default endpoint is a live server, which is right for an operator and
wrong for a test: a leg test that forgets --endpoint would call a real model and pass for
the wrong reason, slowly. Point the default at a dead port for the whole test run, so that
mistake shows up as a refused connection instead.
"""
import os

DEAD_ENDPOINT = "http://127.0.0.1:1/v1"
os.environ.setdefault("BATTERY_DEFAULT_ENDPOINT", DEAD_ENDPOINT)
