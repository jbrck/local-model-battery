"""The selftest gate: four checks, one result block, a non-zero exit on failure."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bench_selftest import (  # noqa: E402
    CHECK_NAMES,
    _chat_check,
    _leg_import_check,
    _models_check,
    _thinking_check,
    main,
)
from scripts.lib.harness import LegArgs, leg_config  # noqa: E402
from scripts.lib.model import ModelConfig  # noqa: E402
from tests.stub_json import JsonStub, completion  # noqa: E402

MODEL = "test-model"

PROPS = {"chat_template": "enable_thinking", "default_generation_settings": {"n_ctx": 8192}}
STATIC_PROPS = {"chat_template": "static template"}


class StubCase(unittest.TestCase):
    def start_stub(self, routes: dict) -> JsonStub:
        stub = JsonStub(routes)
        stub._thread.start()
        self.addCleanup(stub.close)
        return stub

    def cfg(self, stub: JsonStub) -> ModelConfig:
        return ModelConfig(name=MODEL, endpoint=stub.endpoint)

    def args(self, stub: JsonStub, output_dir: str) -> LegArgs:
        return replace(LegArgs(model=MODEL, endpoint=stub.endpoint), output_dir=output_dir)


class ModelsCheckTest(StubCase):
    def test_a_listed_model_passes(self):
        stub = self.start_stub({"/v1/models": (200, {"data": [{"id": MODEL}]})})
        ok, note, _ = _models_check(self.cfg(stub))
        self.assertTrue(ok)
        self.assertIn("lists", note)

    def test_a_missing_model_fails(self):
        stub = self.start_stub({"/v1/models": (200, {"data": [{"id": "other"}]})})
        ok, note, _ = _models_check(self.cfg(stub))
        self.assertFalse(ok)
        self.assertIn(MODEL, note)

    def test_a_dead_endpoint_fails(self):
        stub = self.start_stub({})
        ok, note, _ = _models_check(self.cfg(stub))
        self.assertFalse(ok)
        self.assertIn("/v1/models", note)


class ChatCheckTest(StubCase):
    def test_a_one_token_answer_passes_and_a_500_fails(self):
        stub = self.start_stub({"/v1/chat/completions": (200, completion("ok"))})
        ok, _, _ = _chat_check(self.cfg(stub), self.args(stub, "."))
        self.assertTrue(ok)
        dead = self.start_stub({"/v1/chat/completions": (500, {"error": "boom"})})
        ok, note, _ = _chat_check(self.cfg(dead), self.args(dead, "."))
        self.assertFalse(ok)
        self.assertIn("boom", note)


class ThinkingCheckTest(StubCase):
    def test_a_switch_that_moves_passes(self):
        stub = self.start_stub({
            "/props": (200, PROPS),
            "/v1/chat/completions": (200, completion("ok", tokens=50)),
        })
        ok, note = _thinking_check(self.cfg(stub))
        self.assertTrue(ok)
        self.assertIn("took", note)

    def test_a_switch_that_changes_nothing_fails(self):
        stub = self.start_stub({
            "/props": (200, STATIC_PROPS),
            "/v1/chat/completions": (200, completion("same", tokens=200)),
        })
        ok, note = _thinking_check(self.cfg(stub))
        self.assertFalse(ok)
        self.assertIn("did not take", note)

    def test_an_unmeasurable_switch_fails_with_the_reason(self):
        stub = self.start_stub({
            "/props": (200, PROPS),
            "/v1/chat/completions": (500, {"error": "boom"}),
        })
        ok, note = _thinking_check(self.cfg(stub))
        self.assertFalse(ok)
        self.assertIn("unmeasurable", note)


class ImportCheckTest(unittest.TestCase):
    def test_a_broken_leg_is_named_not_crashed_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "good.py")
            bad = os.path.join(tmp, "bad.py")
            with open(good, "w") as handle:
                handle.write("VALUE = 1\n")
            with open(bad, "w") as handle:
                handle.write("raise ImportError('no optional dependency')\n")
            entries = [("good", good), ("bad", bad), ("gone", os.path.join(tmp, "nope.py"))]
            ok, note = _leg_import_check(entries)
            self.assertFalse(ok)
            self.assertIn("no optional dependency", note)
            self.assertIn("gone", note)
            self.assertNotIn("good:", note)

    def test_clean_entries_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "good.py")
            with open(good, "w") as handle:
                handle.write("VALUE = 1\n")
            ok, note = _leg_import_check([("good", good)])
            self.assertTrue(ok)
            self.assertIn("1 in-repo", note)


class GateTest(StubCase):
    def test_a_healthy_endpoint_passes_all_four_checks(self):
        stub = self.start_stub({
            "/props": (200, PROPS),
            "/v1/models": (200, {"data": [{"id": MODEL}]}),
            "/v1/chat/completions": (200, completion("ok", tokens=50)),
        })
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--model", MODEL, "--endpoint", stub.endpoint, "--output-dir", tmp]
            self.assertEqual(main(argv, import_entries=[]), 0)
            with open(os.path.join(tmp, "selftest.result.json")) as handle:
                block = json.load(handle)
        self.assertEqual(sorted(block["checks"]), sorted(CHECK_NAMES))
        self.assertEqual(block["items"], 2)
        self.assertNotIn("gate", block)

    def test_a_broken_endpoint_refuses_the_battery(self):
        stub = self.start_stub({"/v1/models": (200, {"data": [{"id": "other"}]})})
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--model", MODEL, "--endpoint", stub.endpoint, "--output-dir", tmp]
            self.assertEqual(main(argv, import_entries=[]), 1)
            with open(os.path.join(tmp, "selftest.result.json")) as handle:
                block = json.load(handle)
        self.assertEqual(block["gate"], "fail")
        self.assertTrue(block["checks"]["endpoint_models"].startswith("FAIL"))


if __name__ == "__main__":
    unittest.main()
