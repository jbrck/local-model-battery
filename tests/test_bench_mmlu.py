"""The MMLU leg: letter extraction, cap-hit scoring and the pinned item hash.

Everything runs against the JsonStub fake endpoint; no network and no model.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.bench_mmlu import (  # noqa: E402
    build_prompt,
    items_hash,
    main,
    parse_letter,
    run_mmlu,
)
from scripts.lib.harness import LegArgs, leg_config  # noqa: E402
from tests.stub_json import JsonStub, completion  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "mmlu_test.json")

# Pinned when data/mmlu_test.json was generated from cais/mmlu test split.
PINNED_ITEMS_SHA = "b5ed4c6edc950db6c08a9dff7de42a594824c64185bdad888bc82a067da51dfd"


def load_items() -> list[dict]:
    with open(DATA_PATH) as handle:
        payload = json.load(handle)
    return payload["items"]


def args_for(stub: JsonStub) -> LegArgs:
    return replace(LegArgs(model="stub-model", endpoint=stub.endpoint),
                   cap=4096, timeout_s=30)


class LetterExtractionTest(unittest.TestCase):
    def test_explicit_answer_lines(self):
        self.assertEqual(parse_letter("Answer: C"), "C")
        self.assertEqual(parse_letter("answer: b"), "B")
        self.assertEqual(parse_letter("The answer is (D)."), "D")
        self.assertEqual(parse_letter("Final answer: d"), "D")
        self.assertEqual(parse_letter(r"\boxed{A}"), "A")

    def test_prose_falls_back_to_a_late_letter(self):
        self.assertEqual(parse_letter("I think the best choice is B."), "B")

    def test_last_statement_wins_over_the_first(self):
        self.assertEqual(
            parse_letter("A is wrong; the correct option is C"), "C")

    def test_no_letter_is_none(self):
        self.assertIsNone(parse_letter("no letter here"))
        self.assertIsNone(parse_letter(""))
        self.assertIsNone(parse_letter(None))


class PromptTest(unittest.TestCase):
    def test_prompt_carries_subject_and_numbered_choices(self):
        item = {"id": 0, "subject": "astronomy", "question": "Which one?",
                "choices": ["x", "y", "z", "w"], "answer": "A"}
        prompt = build_prompt(item)
        self.assertIn("about astronomy", prompt)
        self.assertIn("Which one?", prompt)
        for letter in ("A.", "B.", "C.", "D."):
            self.assertIn(f"\n{letter} ", prompt)


class CapHitScoringTest(unittest.TestCase):
    """A truncated reply with no letter still counts as a scored wrong answer."""

    def setUp(self):
        items = [
            {"id": 0, "subject": "astronomy", "question": "@@SENTINEL_A@@ "
             "Which planet is closest to the sun?",
             "choices": ["Venus", "Mercury", "Earth", "Mars"], "answer": "B"},
            {"id": 1, "subject": "astronomy", "question": "@@SENTINEL_B@@ "
             "Which planet has the most moons?",
             "choices": ["Mars", "Venus", "Jupiter", "Saturn"], "answer": "C"},
        ]
        self.tmpdir = tempfile.TemporaryDirectory()
        path = os.path.join(self.tmpdir.name, "items.json")
        with open(path, "w") as handle:
            json.dump({"items": items}, handle)

        def route(body: dict) -> tuple[int, dict]:
            prompt = body["messages"][0]["content"]
            if "@@SENTINEL_A@@" in prompt:
                return 200, completion("Answer: B", tokens=40)
            # Truncated by the cap before any letter appeared; varied prose so
            # it does not also trip the runaway detector.
            filler = " ".join(
                f"Spectroscopic observation number {i} of this system "
                f"resolved no definitive structure yet." for i in range(40))
            return 200, completion(
                f"The orbital mechanics are complex and require {filler}",
                finish_reason="length", tokens=4096)

        self.stub = JsonStub(routes=route)
        self.stub._thread.start()
        self.addCleanup(self.stub.close)
        self.items_path = path

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_truncated_item_counts_as_cap_hit_and_wrong(self):
        args = replace(args_for(self.stub), data=self.items_path)
        with open(args.data) as handle:
            items = json.load(handle)["items"]
        tally = run_mmlu(leg_config(args), items, args)

        self.assertEqual(tally.items, 2)
        self.assertEqual(tally.scored, 2)
        self.assertEqual(tally.correct, 1)     # only the clean answer scored
        self.assertEqual(tally.cap_hits, 1)     # the truncated item hit the cap
        self.assertEqual(tally.status(), "pass")


class MainTest(unittest.TestCase):
    """The entry point: flags in, a scored result block and the item hash out."""

    ITEMS = [
        {"id": 0, "subject": "astronomy", "question": "Which planet is closest to the sun?",
         "choices": ["Venus", "Mercury", "Earth", "Mars"], "answer": "B"},
        {"id": 1, "subject": "astronomy", "question": "Which planet has the most moons?",
         "choices": ["Mars", "Venus", "Jupiter", "Saturn"], "answer": "C"},
    ]

    def test_main_scores_a_small_sample_and_writes_its_block(self):
        def route(body: dict) -> tuple:
            gold = "B" if "closest to the sun" in body["messages"][0]["content"] else "C"
            return 200, completion(f"Answer: {gold}")

        with JsonStub(route) as stub, \
                tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "items.json")
            with open(data, "w") as handle:
                json.dump({"items": self.ITEMS}, handle)
            argv = ["--model", "stub", "--endpoint", stub.endpoint,
                    "--data", data, "--output-dir", tmp, "--limit", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNone(main(argv))
            with open(os.path.join(tmp, "mmlu.result.json")) as handle:
                block = json.load(handle)
        self.assertEqual(block["leg"], "mmlu")
        self.assertEqual(block["items"], 2)
        self.assertEqual(block["correct"], 2)
        self.assertEqual(block["scored"], 2)
        self.assertEqual(block["items_sha"], items_hash(self.ITEMS))

    def test_an_empty_data_file_runs_nothing_and_writes_no_block(self):
        with JsonStub(lambda _body: (200, completion("Answer: A"))) as stub, \
                tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "items.json")
            with open(data, "w") as handle:
                json.dump({"items": []}, handle)
            argv = ["--model", "stub", "--endpoint", stub.endpoint,
                    "--data", data, "--output-dir", tmp]
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(err):
                main(argv)
            self.assertFalse(os.path.exists(os.path.join(tmp, "mmlu.result.json")))
            self.assertIn("no items", err.getvalue())


class ItemsHashTest(unittest.TestCase):
    def test_the_sample_is_pinned_by_hash(self):
        items = load_items()
        self.assertEqual(len(items), 500)
        ids = [item["id"] for item in items]
        self.assertEqual(sorted(ids), ids)          # stable order, never re-shuffled
        self.assertEqual(len(set(ids)), 500)
        for item in items:
            for field in ("question", "subject"):
                self.assertTrue(item[field].strip())
            self.assertIn(item["answer"], ("A", "B", "C", "D"))
            self.assertEqual(len(item["choices"]), 4)
        self.assertEqual(items_hash(items), PINNED_ITEMS_SHA)

    def test_the_data_file_records_its_provenance(self):
        with open(DATA_PATH) as handle:
            payload = json.load(handle)
        self.assertEqual(payload["source"]["split"], "test")
        self.assertEqual(payload["source"]["rows_in_split"], 14042)
        self.assertIn("sha256", payload["sample"]["rule"].lower())
        self.assertIn("items", payload)
