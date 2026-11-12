"""Refusal leg tests: separate per-group counts, refuse-everything visibility, seed-7 order."""
from __future__ import annotations

import contextlib
import io
import json
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.harness import parse_leg_args  # noqa: E402
from scripts.refusal_bench import (  # noqa: E402
    PROMPT_SEED,
    classify,
    evaluate,
    main,
    ordered_items,
    reply_text,
)
from tests.stub_json import JsonStub, completion  # noqa: E402

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "refusal_prompts.json"))


def load_items() -> list[dict]:
    with open(DATA) as handle:
        payload = json.load(handle)
    return payload["items"]


def harmful_prompts() -> set[str]:
    return {item["prompt"] for item in load_items() if item["group"] == "harmful"}


def routed_reply(body: dict) -> tuple[int, dict]:
    """Refuse harmful requests, answer edgy and benign ones."""
    prompt = body["messages"][0]["content"]
    if prompt in harmful_prompts():
        return 200, completion("I'm sorry, I can't help with that.", tokens=30)
    return 200, completion("Sure, here is the answer you asked for.", tokens=30)


def run_leg(stub: JsonStub) -> dict:
    args = parse_leg_args("test", default_data=DATA,
                          argv=["--model", "stub", "--endpoint", stub.endpoint,
                                "--data", DATA])
    with contextlib.redirect_stdout(io.StringIO()):
        _tally, extra = evaluate(args)
    return extra


class ClassifierTest(unittest.TestCase):
    def test_a_refusal_phrase_is_refuse(self):
        self.assertEqual(classify("I'm sorry, I can't do that."), "REFUSE")
        self.assertEqual(classify("As an AI assistant, I am unable to help."), "REFUSE")

    def test_a_plain_answer_is_comply(self):
        self.assertEqual(classify("Sure, here is the answer you asked for."), "COMPLY")

    def test_an_empty_reply_is_unclear(self):
        self.assertEqual(classify(""), "UNCLEAR")
        self.assertEqual(classify("   "), "UNCLEAR")


class ReplyTextTest(unittest.TestCase):
    """A verdict is based on the content and, when present, the reasoning joined to it."""

    def test_reasoning_is_joined_onto_the_content(self):
        self.assertEqual(reply_text("Sure thing.", "let me think"), "Sure thing. let me think")

    def test_reasoning_alone_is_the_verdict_text_when_the_content_is_empty(self):
        self.assertEqual(reply_text("", "I'm sorry, I can't help."), "I'm sorry, I can't help.")

    def test_without_reasoning_the_content_stands_alone(self):
        self.assertEqual(reply_text("Sure thing.", ""), "Sure thing.")

    def test_a_refusal_hiding_in_the_reasoning_channel_still_counts(self):
        route = ("/v1/chat/completions",
                 (200, completion("Sure thing!", reasoning="I'm sorry, I can't do that.",
                                  tokens=30)))
        with JsonStub(dict([route])) as stub:
            extra = run_leg(stub)
        self.assertEqual(extra["harmful_refused"], 20)
        self.assertEqual(extra["harmful_answered"], 0)


class CountsTest(unittest.TestCase):
    def test_three_counts_land_in_three_separate_fields(self):
        with JsonStub(routed_reply) as stub:
            extra = run_leg(stub)
        self.assertEqual(extra["harmful_total"], 20)
        self.assertEqual(extra["edgy_total"], 20)
        self.assertEqual(extra["benign_total"], 30)
        self.assertEqual(extra["harmful_refused"], 20)
        self.assertEqual(extra["edgy_answered"], 20)
        self.assertEqual(extra["benign_refused"], 0)

    def test_a_model_that_refuses_everything_shows_70_of_70_not_an_average(self):
        route = ("/v1/chat/completions",
                 (200, completion("I'm sorry, I can't do that.", tokens=30)))
        with JsonStub(dict([route])) as stub:
            extra = run_leg(stub)
        self.assertEqual(extra["harmful_refused"], 20)
        self.assertEqual(extra["benign_refused"], 30)
        self.assertEqual(extra["edgy_refused"], 20)
        self.assertEqual(extra["edgy_answered"], 0)
        self.assertEqual(extra["benign_answered"], 0)


class OrderTest(unittest.TestCase):
    def test_prompt_order_is_seed_7_not_the_default_seed(self):
        items = load_items()
        seeded = list(items)
        random.Random(PROMPT_SEED).shuffle(seeded)
        got = ordered_items(items)
        self.assertEqual([i["prompt"] for i in got], [i["prompt"] for i in seeded])
        default = list(items)
        random.Random(42).shuffle(default)
        self.assertNotEqual([i["prompt"] for i in got], [i["prompt"] for i in default])

    def test_the_requests_reach_the_wire_in_seed_7_order(self):
        with JsonStub(routed_reply) as stub:
            run_leg(stub)
        sent = [entry["body"]["messages"][0]["content"] for entry in stub.seen]
        expected = [i["prompt"] for i in ordered_items(load_items())]
        self.assertEqual(sent, expected)
        self.assertEqual(len(sent), 70)


class MainTest(unittest.TestCase):
    def test_main_runs_a_small_sample_and_writes_the_refusal_block(self):
        with JsonStub(routed_reply) as stub, tempfile.TemporaryDirectory() as tmp:
            argv = ["--model", "stub", "--endpoint", stub.endpoint,
                    "--data", DATA, "--output-dir", tmp, "--limit", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                main(argv)
            with open(os.path.join(tmp, "refusal.result.json")) as handle:
                block = json.load(handle)
        self.assertEqual(block["leg"], "refusal")
        self.assertEqual(block["items"], 2)
        self.assertTrue(block["notes"])
        self.assertIn("seeded with 7", " ".join(block["notes"]))


if __name__ == "__main__":
    unittest.main()