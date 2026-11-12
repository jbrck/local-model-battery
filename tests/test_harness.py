"""The leg contract: the counters, the runaway rule and the result block."""
import json
import os
import tempfile
import unittest
from dataclasses import replace

from scripts.lib import harness
from scripts.lib.harness import (LegArgs, Tally, is_cap_hit, is_runaway, is_timeout,
                                 leg_config, parse_leg_args, repeated_gram, thinking_body)
from scripts.lib.model import ChatResponse


def answer(text: str, finish: str = "stop", tokens: int = 100) -> ChatResponse:
    """A successful response as the server would send it."""
    payload = {"choices": [{"finish_reason": finish, "message": {"content": text}}],
               "usage": {"completion_tokens": tokens}}
    return ChatResponse(status=200, payload=payload, elapsed_s=2.0)


def failure(error: str) -> ChatResponse:
    return ChatResponse(status=None, payload=None, elapsed_s=1.0, error=error)


# Twelve words repeated: long enough that the runaway rule fires.
LOOP = " ".join(["alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"] * 3)


class ArgsTest(unittest.TestCase):
    def test_defaults_come_from_the_standard(self):
        args = parse_leg_args("test", default_data="data/x.json", argv=["--model", "m"])
        self.assertEqual(args.cap, harness.DEFAULT_CAP)
        self.assertEqual(args.thinking, "on")
        self.assertEqual(args.seed, harness.DEFAULT_SEED)
        self.assertEqual(args.temperature, harness.DEFAULT_TEMPERATURE)
        self.assertEqual(args.data, "data/x.json")

    def test_the_cap_reaches_the_client(self):
        args = replace(LegArgs(model="m"), cap=1024, timeout_s=600)
        cfg = leg_config(args)
        self.assertEqual(cfg.token_limit, 1024)
        self.assertEqual(cfg.timeout_s, 600)

    def test_thinking_off_sends_the_switch_and_on_sends_nothing(self):
        self.assertIsNone(thinking_body("on"))
        self.assertEqual(thinking_body("off"),
                         {"chat_template_kwargs": {"enable_thinking": False}})


class RunawayTest(unittest.TestCase):
    def test_a_loop_is_caught(self):
        self.assertTrue(repeated_gram(LOOP))

    def test_prose_is_not_a_loop(self):
        prose = ("The answer is 42 because the series converges and the remainder term "
                 "vanishes. Working through the algebra gives the same value twice.")
        self.assertFalse(repeated_gram(prose))

    def test_a_short_answer_is_never_a_loop(self):
        self.assertFalse(repeated_gram("the same words"))

    def test_a_response_is_judged_on_its_text(self):
        self.assertTrue(is_runaway(answer(LOOP)))
        self.assertFalse(is_runaway(answer("A short correct answer.")))


class CapHitTest(unittest.TestCase):
    def test_length_with_real_tokens_is_a_cap_hit(self):
        self.assertTrue(is_cap_hit(answer("x" * 50, finish="length", tokens=2000)))

    def test_a_trivial_length_stop_is_not(self):
        self.assertFalse(is_cap_hit(answer("", finish="length", tokens=4)))

    def test_a_clean_stop_is_not(self):
        self.assertFalse(is_cap_hit(answer("done", finish="stop", tokens=2000)))


class TimeoutTest(unittest.TestCase):
    def test_a_socket_timeout_is_counted(self):
        self.assertTrue(is_timeout(failure("TimeoutError: timed out")))

    def test_a_refused_connection_is_not(self):
        self.assertFalse(is_timeout(failure("Connection refused")))


class TallyTest(unittest.TestCase):
    def test_a_clean_run_passes(self):
        tally = Tally()
        tally.record(answer("right"), correct=True)
        tally.record(answer("wrong"), correct=False)
        self.assertEqual(tally.status(), "pass")
        self.assertEqual(tally.scored, 2)
        self.assertEqual(tally.correct, 1)
        self.assertEqual(tally.accuracy, 0.5)

    def test_the_rate_comes_from_tokens_over_seconds(self):
        tally = Tally()
        tally.record(answer("right", tokens=100), correct=True)
        self.assertEqual(tally.tps, 50.0)

    def test_a_cap_hit_is_still_scored(self):
        tally = Tally()
        tally.record(answer("cut off", finish="length", tokens=9000), correct=False)
        self.assertEqual(tally.cap_hits, 1)
        self.assertEqual(tally.scored, 1)
        self.assertEqual(tally.status(), "pass")

    def test_a_loop_makes_the_run_partial(self):
        tally = Tally()
        tally.record(answer(LOOP), correct=False)
        self.assertEqual(tally.runaway, 1)
        self.assertEqual(tally.status(), "partial")

    def test_a_timeout_outranks_everything(self):
        tally = Tally()
        tally.record(answer("right"), correct=True)
        tally.record(failure("read timed out"))
        self.assertEqual(tally.timeout_hits, 1)
        self.assertEqual(tally.status(), "timeout")

    def test_transport_failures_are_kept_for_the_notes(self):
        tally = Tally()
        tally.record(failure("HTTP 500 boom"))
        self.assertEqual(tally.failed, 1)
        self.assertEqual(tally.status(), "fail")
        self.assertIn("HTTP 500 boom", tally.errors)

    def test_an_empty_run_is_not_run(self):
        self.assertEqual(Tally().status(), "not run")

    def test_optional_parts_of_the_block_appear_when_there_is_something_to_say(self):
        args = replace(LegArgs(model="m"), tag="cold-start")
        tally = Tally(notes=["three items came back short"], errors=["HTTP 500 boom"])
        block = tally.to_result("lcb", args, extra={"aborted": True})
        self.assertEqual(block["tag"], "cold-start")
        self.assertEqual(block["notes"], ["three items came back short"])
        self.assertEqual(block["errors"], ["HTTP 500 boom"])
        self.assertTrue(block["aborted"])

    def test_the_block_carries_the_settings_it_ran_under(self):
        args = replace(LegArgs(model="arete-27b"), cap=114688, ctx=131072, thinking="off",
                       tag="pilot")
        tally = Tally()
        tally.record(answer("right"), correct=True)
        block = tally.to_result("lcb", args)
        self.assertEqual(block["leg"], "lcb")
        self.assertEqual(block["cap"], 114688)
        self.assertEqual(block["ctx"], 131072)
        self.assertEqual(block["thinking"], "off")
        self.assertEqual(block["tag"], "pilot")
        self.assertEqual(block["score_pct"], 100.0)
        self.assertNotIn("errors", block)


class FinishTest(unittest.TestCase):
    def test_finish_writes_the_block_the_orchestrator_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = replace(LegArgs(model="m"), output_dir=tmp)
            tally = Tally()
            tally.record(answer("right"), correct=True)
            block = harness.finish("math500", args, tally)
            path = os.path.join(tmp, "math500.result.json")
            self.assertTrue(os.path.exists(path))
            with open(path) as handle:
                self.assertEqual(json.load(handle)["correct"], 1)
            self.assertEqual(block["status"], "pass")


class AskTest(unittest.TestCase):
    """`ask` is where the leg's cap, temperature and thinking setting reach the wire."""

    def setUp(self):
        from tests.stub_json import JsonStub, completion

        self.stub = JsonStub({"/v1/chat/completions": (200, completion("42"))})
        self.stub._thread.start()
        self.addCleanup(self.stub.close)

    def test_the_cap_and_thinking_setting_reach_the_server(self):
        args = replace(LegArgs(model="m", endpoint=self.stub.endpoint), cap=1024, thinking="off")
        response = harness.ask(leg_config(args), [{"role": "user", "content": "hi"}], args)
        self.assertTrue(response.ok)
        sent = self.stub.seen[-1]["body"]
        self.assertEqual(sent["max_tokens"], 1024)
        self.assertEqual(sent["temperature"], 0.0)
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False})

    def test_thinking_on_leaves_the_server_alone(self):
        args = replace(LegArgs(model="m", endpoint=self.stub.endpoint), thinking="on")
        harness.ask(leg_config(args), [{"role": "user", "content": "hi"}], args)
        self.assertNotIn("chat_template_kwargs", self.stub.seen[-1]["body"])


class ItemsTest(unittest.TestCase):
    def test_a_bare_list_and_a_wrapped_list_both_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            bare = os.path.join(tmp, "bare.json")
            wrapped = os.path.join(tmp, "wrapped.json")
            with open(bare, "w") as handle:
                json.dump([{"problem": "1"}, {"problem": "2"}], handle)
            with open(wrapped, "w") as handle:
                json.dump({"items": [{"problem": "1"}, {"problem": "2"}]}, handle)
            self.assertEqual(len(harness.load_items(bare)), 2)
            self.assertEqual(len(harness.load_items(wrapped)), 2)
            self.assertEqual(len(harness.load_items(wrapped, limit=1)), 1)


if __name__ == "__main__":
    unittest.main()
