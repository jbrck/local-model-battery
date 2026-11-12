"""The GPQA-Diamond leg: letter extraction, empty answers and cap visibility.

Everything runs against the JsonStub fake endpoint; no network and no model.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.bench_gpqa_diamond import (  # noqa: E402
    extract_letter,
    grade,
    main,
)
from scripts.lib.harness import LegArgs, result_path  # noqa: E402
from scripts.lib.model import ChatResponse  # noqa: E402
from tests.stub_json import JsonStub, completion  # noqa: E402


def args_for(stub: JsonStub, data_path: str, output_dir: str) -> LegArgs:
    return replace(LegArgs(model="stub-model", endpoint=stub.endpoint),
                   data=data_path, output_dir=output_dir, cap=1024, timeout_s=30)


class LetterExtractionTest(unittest.TestCase):
    """The v1 normalization: strip, upper, drop dots, first standalone letter."""

    def test_a_bare_letter(self):
        self.assertEqual(extract_letter("B"), "B")

    def test_a_letter_with_trailing_punctuation(self):
        self.assertEqual(extract_letter("The answer is C."), "C")

    def test_a_boxed_letter(self):
        self.assertEqual(extract_letter(r"\boxed{A}"), "A")

    def test_letters_inside_words_do_not_match(self):
        self.assertEqual(extract_letter("A catalyst speeds the BOXED reaction"), "A")
        self.assertEqual(extract_letter("no standalone letter"), "?")

    def test_the_pinned_gold_format_extracts_its_letter(self):
        self.assertEqual(extract_letter(r"\boxed{D}"), "D")


class GradeTest(unittest.TestCase):
    def answer(self, content: str, reasoning: str | None = None) -> ChatResponse:
        message: dict = {"content": content}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        payload = {"choices": [{"finish_reason": "stop", "message": message}],
                   "usage": {"completion_tokens": 10}}
        return ChatResponse(status=200, payload=payload, elapsed_s=1.0)

    def test_a_matching_letter_is_correct(self):
        correct, predicted, channel = grade(self.answer("B"), "B")
        self.assertTrue(correct)
        self.assertEqual(predicted, "B")
        self.assertEqual(channel, "content")

    def test_a_wrong_letter_is_scored_wrong(self):
        correct, _, _ = grade(self.answer("A"), "C")
        self.assertFalse(correct)

    def test_reasoning_only_with_empty_content_is_empty_not_wrong(self):
        correct, predicted, channel = grade(self.answer("", reasoning="long trace"), "B")
        self.assertIsNone(correct)
        self.assertEqual(predicted, "")
        self.assertEqual(channel, "empty")


class CapHitVisibilityTest(unittest.TestCase):
    """A reply truncated at the cap is scored but the cap hit stays visible."""

    def setUp(self):
        items = [
            {"question": "@@LETTER@@ Which force keeps planets in orbit?",
             "answer": r"\boxed{C}"},
            {"question": "@@CAP@@ Explain the entropy change of the surroundings",
             "answer": r"\boxed{A}"},
        ]
        self.tmpdir = tempfile.TemporaryDirectory()
        data_path = os.path.join(self.tmpdir.name, "items.json")
        with open(data_path, "w") as handle:
            json.dump(items, handle)

        def route(body: dict) -> tuple[int, dict]:
            prompt = body["messages"][1]["content"]
            if "@@LETTER@@" in prompt:
                return 200, completion("C", tokens=10)
            filler = "the entropy change depends on the heat capacity terms. " * 40
            return 200, completion(f"Considering the surroundings, {filler}",
                                   finish_reason="length", tokens=900)

        self.stub = JsonStub(routes=route)
        self.stub._thread.start()
        self.addCleanup(self.stub.close)
        self.data_path = data_path

    def test_a_truncated_reply_scores_and_counts_the_cap_hit(self):
        with tempfile.TemporaryDirectory() as out_dir:
            args = args_for(self.stub, self.data_path, out_dir)
            main(["--model", "stub-model", "--endpoint", self.stub.endpoint,
                  "--data", self.data_path, "--output-dir", out_dir])
            with open(result_path("gpqa_diamond", args)) as handle:
                block = json.load(handle)
            self.assertEqual(block["items"], 2)
            self.assertEqual(block["scored"], 2)
            self.assertEqual(block["correct"], 1)
            self.assertEqual(block["cap_hits"], 1)
            self.assertTrue(any("cap" in note for note in block["notes"]))
            self.assertEqual(block["per_item"][1]["cap_hit"], True)


class MainTest(unittest.TestCase):
    def test_a_clean_run_reports_pass_and_the_block_matches_the_file(self):
        items = [{"question": "Pick one", "answer": r"\boxed{A}"}]
        with tempfile.TemporaryDirectory() as tmp:
            data_path = os.path.join(tmp, "items.json")
            with open(data_path, "w") as handle:
                json.dump(items, handle)
            stub = JsonStub({"/v1/chat/completions": (200, completion("A", tokens=10))})
            stub._thread.start()
            try:
                out_dir = os.path.join(tmp, "results")
                main(["--model", "stub-model", "--endpoint", stub.endpoint,
                      "--data", data_path, "--output-dir", out_dir, "--cap", "1024"])
                with open(result_path("gpqa_diamond",
                                      args_for(stub, data_path, out_dir))) as handle:
                    block = json.load(handle)
                self.assertEqual(block["status"], "pass")
                self.assertEqual(block["correct"], 1)
                self.assertEqual(block["empty_answers"], 0)
                sent = stub.seen[-1]["body"]
                self.assertEqual(sent["max_tokens"], 1024)
                self.assertEqual(sent["messages"][1]["content"], "Pick one")
            finally:
                stub.close()

    def test_limit_runs_only_the_first_n_items(self):
        items = [{"question": "Pick one", "answer": r"\boxed{A}"}] * 3
        with tempfile.TemporaryDirectory() as tmp:
            data_path = os.path.join(tmp, "items.json")
            with open(data_path, "w") as handle:
                json.dump(items, handle)
            stub = JsonStub({"/v1/chat/completions": (200, completion("A", tokens=10))})
            stub._thread.start()
            try:
                out_dir = os.path.join(tmp, "results")
                main(["--model", "stub-model", "--endpoint", stub.endpoint,
                      "--data", data_path, "--output-dir", out_dir, "--limit", "2"])
                with open(result_path("gpqa_diamond",
                                      args_for(stub, data_path, out_dir))) as handle:
                    block = json.load(handle)
                self.assertEqual(block["items"], 2)
            finally:
                stub.close()


if __name__ == "__main__":
    unittest.main()
