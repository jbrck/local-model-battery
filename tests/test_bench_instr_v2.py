"""The instr_v2 leg on JsonStub: every checker keeps its v1 verdict."""
import json
import os
import tempfile
import unittest

from scripts.bench_instr_v2 import CHECKERS, check
from scripts.bench_instr_v2 import main as bench_main
from scripts.bench_instr_v2 import run_instr
from scripts.lib.harness import LegArgs, leg_config

CASES = [
    ("exact:banana", "BANANA", "banana"),
    ("one_word", "blue", "sky blue"),
    ("two_words", "Paris France", "Paris"),
    ("three_sentences", "A. B. C.", "A. B."),
    ("five_bullets", "- a\n- b\n- c\n- d\n- e", "- a\n- b"),
    ("seven_items", "1, 2, 3, 4, 5, 6, 7", "1, 2, 3"),
    ("all_caps", "WEATHER WARNING", "Weather Warning"),
    ("starts_with", "Although it rained", "It rained, although"),
    ("ends_with", "the night ended in silence.", "silence came early."),
    ("lowercase", "the quick brown fox", "THE QUICK BROWN FOX"),
    ("no_e", "a dog runs", "the dog runs"),
    ("no_commas", "calm seas", "calm, cold seas"),
    ("max20", " ".join(["word"] * 20), " ".join(["word"] * 21)),
    ("min15", " ".join(["word"] * 15), " ".join(["word"] * 14)),
    ("json", '{"city": "Tokyo", "temp_c": 28, "conditions": "Sunny"}', '{"city": "Tokyo"}'),
    ("three_lines", "a\nb\nc", "a\nb"),
    ("no_digits", "math without digits", "room 42"),
    ("exact_phrase", "The eagle lands at dawn", "The eagle lands at noon"),
    ("ends_period", "It rained.", "It rained"),
    ("one_line_three", "Mars Venus Earth", "Mars Venus\nEarth"),
]

ITEMS = [
    {"id": "one_ok", "prompt": "one word color", "checker": "one_word"},
    {"id": "one_bad", "prompt": "one word animal", "checker": "one_word"},
    {"id": "json_ok", "prompt": "json weather", "checker": "json"},
    {"id": "json_bad", "prompt": "json city only", "checker": "json"},
]

REPLIES = {
    "one word color": "blue",
    "one word animal": "a cat",
    "json weather": '{"city": "Tokyo", "temp_c": 28, "conditions": "Sunny"}',
    "json city only": '{"city": "Tokyo"}',
}

BLOCK_KEYS = ("leg", "model", "status", "items", "scored", "correct", "score_pct",
              "tps", "cap_hits", "runaway", "timeout_hits", "failed", "tokens",
              "elapsed_s", "cap", "ctx", "thinking", "seed", "temperature")


class CheckTest(unittest.TestCase):
    def test_every_checker_passes_and_fails_exactly_as_in_v1(self) -> None:
        for checker, good, bad in CASES:
            with self.subTest(checker=checker):
                self.assertTrue(check(checker, good))
                self.assertFalse(check(checker, bad))

    def test_an_unknown_checker_fails(self) -> None:
        self.assertFalse(check("no_such_rule", "anything"))

    def test_an_empty_response_keeps_its_v1_verdict(self) -> None:
        # Vacuous truths in v1 stay vacuous: these five pass on empty output.
        vacuous = ("all_caps", "no_e", "no_commas", "max20", "no_digits")
        for checker in CHECKERS:
            self.assertEqual(check(checker, ""), checker in vacuous, checker)

    def test_every_item_in_the_pinned_data_file_has_a_checker(self) -> None:
        with open("data/instr_v2_tasks.json") as handle:
            items = json.load(handle)
        self.assertEqual(len(items), 20)
        for item in items:
            self.assertIn(item["checker"], CHECKERS)


class RunTest(unittest.TestCase):
    """End to end through JsonStub: the counters and the block they produce."""

    def setUp(self) -> None:
        from tests.stub_json import JsonStub, completion

        def routes(body: dict) -> tuple[int, dict]:
            prompt = body["messages"][0]["content"]
            return 200, completion(REPLIES[prompt])

        self.stub = JsonStub(routes)
        self.stub._thread.start()
        self.addCleanup(self.stub.close)

    @staticmethod
    def write_items() -> str:
        path = os.path.join(tempfile.mkdtemp(), "items.json")
        with open(path, "w") as handle:
            json.dump(ITEMS, handle)
        return path

    def run_once(self, **overrides) -> dict:
        args = LegArgs(model="m", endpoint=self.stub.endpoint,
                       data=self.write_items(), **overrides)
        cfg = leg_config(args)
        with open(args.data) as handle:
            items = json.load(handle)
        tally = run_instr(cfg, items, args)
        return tally.to_result("instr_v2", args)

    def test_counters_reflect_the_stubbed_responses(self) -> None:
        block = self.run_once()
        self.assertEqual(block["items"], 4)
        self.assertEqual(block["scored"], 4)
        self.assertEqual(block["correct"], 2)
        self.assertAlmostEqual(block["score_pct"], 50.0, delta=0.1)
        self.assertEqual(block["cap_hits"], 0)
        self.assertEqual(block["runaway"], 0)
        self.assertEqual(block["timeout_hits"], 0)
        self.assertEqual(block["failed"], 0)
        self.assertEqual(block["status"], "pass")
        self.assertEqual(len(self.stub.seen), 4)

    def test_the_block_carries_the_contract_keys(self) -> None:
        block = self.run_once()
        for key in BLOCK_KEYS:
            self.assertIn(key, block)
        self.assertEqual(block["leg"], "instr_v2")
        self.assertNotIn("errors", block)

    def test_main_writes_the_result_file_and_flags_reach_the_wire(self) -> None:
        out_dir = tempfile.mkdtemp()
        bench_main(["--model", "m", "--endpoint", self.stub.endpoint,
                    "--data", self.write_items(), "--output-dir", out_dir,
                    "--cap", "512", "--thinking", "off"])
        path = os.path.join(out_dir, "instr_v2.result.json")
        with open(path) as handle:
            block = json.load(handle)
        self.assertEqual(block["leg"], "instr_v2")
        self.assertEqual(block["cap"], 512)
        self.assertEqual(block["thinking"], "off")
        self.assertEqual(block["correct"], 2)
        sent = self.stub.seen[-1]["body"]
        self.assertEqual(sent["max_tokens"], 512)
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False})


if __name__ == "__main__":
    unittest.main()
