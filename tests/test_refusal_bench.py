"""Refusal leg tests: separate per-group counts, refuse-everything visibility, seed-7 order."""
from __future__ import annotations

import contextlib
import io
import json
import os
import random
import unittest

from scripts.lib.harness import parse_leg_args
from scripts.refusal_bench import PROMPT_SEED, classify, evaluate, ordered_items
from tests.stub_json import JsonStub, completion

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


if __name__ == "__main__":
    unittest.main()