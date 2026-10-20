"""Suite, corpus, compliance, and judge-parsing tests. Stdlib only."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts import validate_suite  # noqa: E402
from scripts.build_scoring_corpus import build_corpus, normalize_answer  # noqa: E402
from scripts.lib.judge import parse_scores  # noqa: E402
from scripts.report_scores import check_compliance, split_copy_fields  # noqa: E402

SUITE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "writing_suite_v1.json")


def load_suite() -> dict:
    with open(SUITE_PATH) as handle:
        return json.load(handle)


class SuiteValidationTest(unittest.TestCase):
    def test_shipped_suite_is_valid(self):
        suite = load_suite()
        self.assertEqual(validate_suite.validate_suite(suite), [])
        self.assertEqual(len(suite["tasks"]), 138)

    def test_missing_rubric_anchor_is_reported(self):
        suite = load_suite()
        broken = json.loads(json.dumps(suite))
        del broken["tasks"][0]["rubric"][0]["anchors"]["5"]
        problems = validate_suite.validate_suite(broken)
        self.assertTrue(any("missing anchor 5" in problem for problem in problems))

    def test_duplicate_task_id_is_reported(self):
        suite = load_suite()
        broken = json.loads(json.dumps(suite))
        broken["tasks"][1]["task_id"] = broken["tasks"][0]["task_id"]
        problems = validate_suite.validate_suite(broken)
        self.assertTrue(any("duplicate task_id" in problem for problem in problems))


class CorpusTest(unittest.TestCase):
    def test_normalize_strips_fences_only(self):
        self.assertEqual(normalize_answer("```\nhello\n```"), "hello")
        self.assertEqual(normalize_answer("  Certainly!  "), "Certainly!")

    def test_corpus_is_blind_and_complete(self):
        records = [
            {"model": "model-a", "task_id": "XP-001", "category": "xpost",
             "answer": "one", "empty": False},
            {"model": "model-b", "task_id": "XP-002", "category": "xpost",
             "answer": "two", "empty": False},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = os.path.join(tmp, "corpus")
            key_path = os.path.join(tmp, "key.json")
            summary = build_corpus(records, corpus_dir, key_path, seed=7)
            self.assertEqual(summary["answers"], 2)

            files = sorted(os.listdir(corpus_dir))
            self.assertEqual(len(files), 2)
            for filename in files:
                with open(os.path.join(corpus_dir, filename)) as handle:
                    payload = json.load(handle)
                self.assertEqual(sorted(payload), ["answer", "task_id"])
                for model in ("model-a", "model-b"):
                    self.assertNotIn(model, json.dumps(payload))

    def test_shuffle_is_reproducible_with_seed(self):
        records = [{"model": "m", "task_id": f"XP-{i:03d}", "category": "xpost",
                    "answer": str(i), "empty": False} for i in range(1, 6)]
        with tempfile.TemporaryDirectory() as tmp:
            first = build_corpus(records, os.path.join(tmp, "a"),
                                 os.path.join(tmp, "key-a.json"), seed=11)
            second = build_corpus(records, os.path.join(tmp, "b"),
                                  os.path.join(tmp, "key-b.json"), seed=11)
            self.assertEqual(first["answers"], second["answers"])
            with open(os.path.join(tmp, "key-a.json")) as handle:
                key_a = json.load(handle)["entries"]
            models = [entry["task_id"] for entry in key_a.values()]
            self.assertEqual(sorted(models), sorted(r["task_id"] for r in records))


class ComplianceTest(unittest.TestCase):
    def test_post_over_limit_fails(self):
        task = {"category": "xpost", "output_contract": {"max_chars": 10}}
        self.assertFalse(check_compliance(task, "this is far too long")["pass"])
        self.assertTrue(check_compliance(task, "short")["pass"])

    def test_thread_checks_each_post(self):
        task = {"category": "xthread", "output_contract": {"max_chars": 10}}
        self.assertTrue(check_compliance(task, "ok one\n---\nok two")["pass"])
        self.assertFalse(check_compliance(task, "ok one\n---\nthis post is too long")["pass"])

    def test_word_bounds(self):
        task = {"category": "summary", "output_contract": {"min_words": 3, "max_words": 5}}
        self.assertTrue(check_compliance(task, "one two three")["pass"])
        self.assertFalse(check_compliance(task, "one two")["pass"])

    def test_copy_fields_parse_from_labels(self):
        fields = split_copy_fields("Headline: Fast Brackets\nDescription: Steel parts, 5-day lead",
                                   ["H1", "D1"])
        self.assertEqual(fields, ["Fast Brackets", "Steel parts, 5-day lead"])

    def test_empty_answer_fails(self):
        task = {"category": "xpost", "output_contract": {"max_chars": 280}}
        self.assertFalse(check_compliance(task, "   ")["pass"])


class JudgeParseTest(unittest.TestCase):
    def test_parses_fenced_json(self):
        raw = 'Here you go:\n```json\n{"scores": [{"criterion": "Hook", "score": 8, "justification": "x"}]}\n```'
        parsed = parse_scores(raw, ["Hook"])
        assert parsed is not None
        self.assertEqual(parsed[0]["score"], 8)

    def test_rejects_wrong_criterion_count(self):
        raw = '{"scores": [{"criterion": "Hook", "score": 8}]}'
        self.assertIsNone(parse_scores(raw, ["Hook", "Structure"]))

    def test_clamps_out_of_range(self):
        raw = '{"scores": [{"criterion": "Hook", "score": 99}]}'
        parsed = parse_scores(raw, ["Hook"])
        assert parsed is not None
        self.assertEqual(parsed[0]["score"], 10)


if __name__ == "__main__":
    unittest.main()
