"""The selftest gate: four checks, one result block, a non-zero exit on failure."""
from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bench_selftest import (  # noqa: E402
    CHECK_NAMES,
    _catalog_entries,
    _chat_check,
    _leg_import_check,
    _model_ids,
    _models_check,
    _thinking_check,
    main,
)
from scripts.lib.harness import LegArgs  # noqa: E402
from scripts.lib.model import ChatResponse, ModelConfig  # noqa: E402
from tests import fixtures  # noqa: E402
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

    def test_a_payload_without_model_entries_lists_nothing(self):
        stub = self.start_stub({"/v1/models": (200, {"data": [42, None, "garbage"]})})
        ok, note, _ = _models_check(self.cfg(stub))
        self.assertFalse(ok)
        self.assertIn("nothing", note)

    def test_malformed_entries_are_skipped_not_crashed_on(self):
        self.assertEqual(_model_ids({"data": [{"id": "a"}, 7, {}]}), ["a", ""])


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


class CatalogEntriesTest(unittest.TestCase):
    """The default entry list comes from the real catalog and points at real files."""

    def test_every_entry_belongs_to_an_implemented_leg(self):
        entries = _catalog_entries()
        self.assertTrue(entries, "the shipped catalog has implemented legs")
        implemented = {leg.id for leg in fixtures.LEGS if leg.implemented}
        self.assertEqual({leg_id for leg_id, _ in entries}, implemented)

    def test_every_entry_path_exists(self):
        for _leg_id, path in _catalog_entries():
            self.assertTrue(os.path.isfile(path), path)


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


class CheckpointTest(unittest.TestCase):
    """A killed selftest keeps its finished checks; the resume re-runs only the rest."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.out_dir = os.path.join(self.tmpdir.name, "results")
        self.calls: list[str] = []

    def fake_models(self, cfg: ModelConfig) -> tuple[bool, str, ChatResponse]:
        self.calls.append("endpoint_models")
        return True, "/v1/models lists test-model", ChatResponse(
            status=200, payload={"data": [{"id": MODEL}]}, elapsed_s=1.0)

    def fake_chat(self, cfg: ModelConfig, args: LegArgs) -> tuple[bool, str, ChatResponse]:
        self.calls.append("one_token_chat")
        return True, "one-token completion answered in 1.0s", ChatResponse(
            status=200, payload=completion("ok", tokens=5), elapsed_s=1.0)

    def fake_thinking(self, cfg: ModelConfig) -> tuple[bool, str]:
        self.calls.append("thinking_switch")
        return True, "thinking switch took"

    def fake_imports(self, entries: list[tuple[str, str]] | None = None) -> tuple[bool, str]:
        self.calls.append("leg_imports")
        return True, "all 0 in-repo leg scripts import"

    def run_main(self, kill_after: str | None = None) -> int:
        checks = {"endpoint_models": "_models_check",
                  "one_token_chat": "_chat_check",
                  "thinking_switch": "_thinking_check",
                  "leg_imports": "_leg_import_check"}
        fakes = {"endpoint_models": self.fake_models,
                 "one_token_chat": self.fake_chat,
                 "thinking_switch": self.fake_thinking,
                 "leg_imports": self.fake_imports}
        with contextlib.ExitStack() as stack:
            for name, attribute in checks.items():
                fake: object = fakes[name]
                if name == kill_after:
                    def killer(*args: object, **kwargs: object) -> object:
                        self.calls.append(name)
                        raise KeyboardInterrupt("run killed")
                    fake = killer
                stack.enter_context(unittest.mock.patch(
                    f"scripts.bench_selftest.{attribute}", fake))
            return main(["--model", MODEL, "--endpoint", "http://127.0.0.1:1/v1",
                         "--output-dir", self.out_dir], import_entries=[])

    def block(self, out_dir: str) -> dict:
        with open(os.path.join(out_dir, "selftest.result.json")) as handle:
            return json.load(handle)

    def cases_dir(self) -> str:
        return os.path.join(self.out_dir, "cases", "selftest")

    def test_a_kill_after_the_first_check_leaves_exactly_one_on_disk(self) -> None:
        with self.assertRaises(KeyboardInterrupt):
            self.run_main(kill_after="one_token_chat")
        self.assertEqual(sorted(os.listdir(self.cases_dir())),
                         ["endpoint_models.json"])

    def test_a_resume_reruns_only_the_missing_checks_and_matches_the_full_run(self) -> None:
        with self.assertRaises(KeyboardInterrupt):
            self.run_main(kill_after="one_token_chat")
        self.calls.clear()
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(self.calls, ["one_token_chat", "thinking_switch",
                                      "leg_imports"])
        resumed = self.block(self.out_dir)

        with tempfile.TemporaryDirectory() as other:
            saved_dir, self.out_dir = self.out_dir, os.path.join(other, "results")
            self.calls.clear()
            try:
                self.assertEqual(self.run_main(), 0)
                full = self.block(self.out_dir)
            finally:
                self.out_dir = saved_dir
        self.assertEqual(resumed, full)


if __name__ == "__main__":
    unittest.main()
