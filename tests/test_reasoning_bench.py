"""The reasoning leg: trace presence, trace length, and the answer channel."""
import json
import os
import tempfile
import unittest

from scripts.lib.harness import LegArgs, Tally
from scripts.lib.model import ChatResponse
from scripts.reasoning_bench import (answer_channel, main, run_leg, totals,
                                     trace_stats)
from tests.stub_json import JsonStub, completion

ITEMS = [{"id": "one", "prompt": "p1", "expected_answer_pattern": "1"},
         {"id": "two", "prompt": "p2", "expected_answer_pattern": "2"}]


def items_file(tmp: str) -> str:
    path = os.path.join(tmp, "items.json")
    with open(path, "w") as handle:
        json.dump(ITEMS, handle)
    return path


def run_with(routes: dict) -> tuple[Tally, list[dict], JsonStub]:
    stub = JsonStub(routes)
    with stub, tempfile.TemporaryDirectory() as tmp:
        tally, rows = run_leg(LegArgs(model="m", endpoint=stub.endpoint,
                                      data=items_file(tmp)))
    return tally, rows, stub


class ScoringTest(unittest.TestCase):
    def test_the_channel_names_where_the_answer_lived(self):
        both = ChatResponse(status=200, payload=completion("42", "trace"))
        reasoning_only = ChatResponse(status=200, payload=completion("", "trace"))
        empty = ChatResponse(status=200, payload=completion(""))
        self.assertEqual(answer_channel(both), "content")
        self.assertEqual(answer_channel(reasoning_only), "reasoning_content")
        self.assertEqual(answer_channel(empty), "none")

    def test_length_counts_the_reasoning_characters(self):
        traced = ChatResponse(status=200, payload=completion("42", "abcde"))
        plain = ChatResponse(status=200, payload=completion("42"))
        self.assertEqual(trace_stats(traced), (True, 5))
        self.assertEqual(trace_stats(plain), (False, 0))


class RunLegTest(unittest.TestCase):
    def test_a_reasoning_only_model_is_not_scored_empty(self):
        tally, rows, _ = run_with({"/v1/chat/completions": (200, completion("", "trace text"))})
        self.assertEqual((tally.scored, tally.correct), (2, 2))
        self.assertTrue(all(row["channel"] == "reasoning_content" for row in rows))
        self.assertTrue(all(row["trace"] for row in rows))
        self.assertEqual(sum(row["reasoning_len"] for row in rows), 20)

    def test_content_answers_without_a_trace_score_clean(self):
        tally, rows, stub = run_with({"/v1/chat/completions": (200, completion("42"))})
        self.assertEqual((tally.scored, tally.correct), (2, 2))
        self.assertTrue(all(row["channel"] == "content" for row in rows))
        self.assertTrue(all(not row["trace"] for row in rows))
        for seen in stub.seen:
            self.assertNotIn("chat_template_kwargs", seen["body"])

    def test_an_empty_model_scores_zero(self):
        tally, rows, _ = run_with({"/v1/chat/completions": (200, completion(""))})
        self.assertEqual((tally.scored, tally.correct), (2, 0))
        self.assertTrue(all(row["channel"] == "none" for row in rows))

    def test_totals_count_traces_and_channels(self):
        rows = [
            {"trace": True, "reasoning_len": 5, "channel": "content"},
            {"trace": True, "reasoning_len": 9, "channel": "reasoning_content"},
            {"trace": False, "reasoning_len": 0, "channel": "none"},
        ]
        got = totals(rows)
        self.assertEqual(got["with_trace"], 2)
        self.assertEqual(got["total_reasoning_len"], 14)
        self.assertEqual(got["answered_reasoning"], 1)
        self.assertEqual(got["answered_none"], 1)


class MainTest(unittest.TestCase):
    def test_thinking_off_is_refused(self):
        self.assertEqual(main(["--model", "m", "--thinking", "off"]), 2)

    def test_main_writes_the_result_block(self):
        routes = {"/v1/chat/completions": (200, completion("42", "trace"))}
        with JsonStub(routes) as stub:
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "results")
                code = main(["--model", "m", "--endpoint", stub.endpoint,
                             "--data", items_file(tmp), "--output-dir", out])
                self.assertEqual(code, 0)
                with open(os.path.join(out, "reasoning.result.json")) as handle:
                    block = json.load(handle)
        self.assertEqual(block["leg"], "reasoning")
        self.assertEqual(block["correct"], 2)
        self.assertEqual(block["thinking"], "on")
        self.assertEqual(len(block["per_item"]), 2)
        self.assertEqual(block["with_trace"], 2)