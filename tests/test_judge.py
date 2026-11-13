"""The judge: the prompt it sees, the reply it must parse, and the pin it runs under.

The judge model is a comparability key, so it belongs in the standard. The endpoint stays
the operator's, because the judge lives wherever their proxy is. The parsing paths run
against the real HTTP stub, because a judge reply is prose first and JSON second.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.judge import (  # noqa: E402
    _excerpt,
    build_judge_prompt,
    describe_contract,
    parse_scores,
    resolve_judge_defaults,
    score_answer,
    setup_judge,
)
from scripts.lib.model import ModelConfig  # noqa: E402
from tests.stub_json import JsonStub, completion  # noqa: E402

PINNED = {"judge": {"model": "nous-glm-5-3-flash", "endpoint": ""}}

TASK = {
    "prompt": "Write a haiku about storage limits.",
    "rubric": [
        {"criterion": "form", "description": "three lines",
         "anchors": {"2": "not three lines", "5": "three lines", "8": "three sharp lines"}},
        {"criterion": "sense", "description": "the images cohere", "anchors": {}},
    ],
}

CRITERIA = [criterion["criterion"] for criterion in TASK["rubric"]]


def judge_reply(*rows: tuple[str, object, str]) -> str:
    """A judge reply in the shape the system prompt demands."""
    return json.dumps({"scores": [{"criterion": name, "score": value, "justification": why}
                                  for name, value, why in rows]})


class JudgeDefaultsTest(unittest.TestCase):
    def test_the_standard_pins_the_model_when_the_operator_is_silent(self):
        model, endpoint = resolve_judge_defaults("", "", "http://run:4000/v1", PINNED)
        self.assertEqual(model, "nous-glm-5-3-flash")
        self.assertEqual(endpoint, "http://run:4000/v1")

    def test_the_operator_beats_the_pin(self):
        resolved = resolve_judge_defaults("mine", "http://judge/v1", "http://run/v1", PINNED)
        self.assertEqual(resolved, ("mine", "http://judge/v1"))

    def test_a_pinned_judge_endpoint_wins_over_the_run_endpoint(self):
        pin = {"judge": {"model": "j", "endpoint": "http://judge/v1"}}
        self.assertEqual(resolve_judge_defaults("", "", "http://run/v1", pin)[1],
                         "http://judge/v1")

    def test_a_standard_with_no_judge_section_leaves_the_judge_unset(self):
        model, endpoint = resolve_judge_defaults("", "", "http://run/v1", {})
        self.assertEqual(model, "")
        self.assertEqual(endpoint, "http://run/v1")

    def test_the_run_endpoint_is_not_used_when_the_operator_named_a_judge(self):
        resolved = resolve_judge_defaults("", "http://judge/v1", "http://run/v1", PINNED)
        self.assertEqual(resolved, ("nous-glm-5-3-flash", "http://judge/v1"))

    def test_the_judge_is_configured_to_score_deterministically(self):
        judge = setup_judge("deckard-40b", "http://judge/v1")
        self.assertEqual(judge.name, "deckard-40b")
        self.assertEqual(judge.temperature, 0.0)
        self.assertEqual(judge.endpoint, "http://judge/v1")


class ContractTest(unittest.TestCase):
    def test_word_bounds_read_as_one_clause_each(self):
        self.assertEqual(describe_contract({"min_words": 40, "max_words": 120}),
                         "at least 40 words, at most 120 words")

    def test_a_flat_character_limit_is_one_clause(self):
        self.assertEqual(describe_contract({"max_chars": 500}),
                         "at most 500 characters")

    def test_a_per_field_character_limit_names_each_field(self):
        self.assertEqual(describe_contract({"max_chars": {"answer": 500, "notes": 200}}),
                         "answer at most 500 characters, notes at most 200 characters")

    def test_an_empty_contract_states_that_it_is_empty(self):
        self.assertEqual(describe_contract({}), "none stated")

    def test_clauses_are_joined_in_the_order_they_are_checked(self):
        contract = {"min_words": 10, "max_chars": {"answer": 300}}
        self.assertEqual(describe_contract(contract),
                         "at least 10 words, answer at most 300 characters")


class PromptTest(unittest.TestCase):
    def test_the_prompt_carries_the_task_the_rubric_and_the_blind_answer(self):
        prompt = build_judge_prompt(TASK, "tokens run out\nbefore the point")
        self.assertIn("TASK AS GIVEN TO THE WRITER:\nWrite a haiku", prompt)
        self.assertIn("- form: three lines", prompt)
        self.assertIn("2 = not three lines", prompt)
        self.assertNotIn("deckard", prompt)
        self.assertIn("ANSWER TO SCORE:\n<<<\ntokens run out\nbefore the point\n>>>", prompt)


class ParseScoresTest(unittest.TestCase):
    def test_a_clean_reply_yields_clamped_rows(self):
        parsed = parse_scores(judge_reply(("form", 8, "three lines"), ("sense", 5, "coheres")),
                              CRITERIA)
        self.assertEqual(parsed, [
            {"criterion": "form", "score": 8, "justification": "three lines"},
            {"criterion": "sense", "score": 5, "justification": "coheres"},
        ])

    def test_a_reply_wrapped_in_a_markdown_fence_still_parses(self):
        fenced = "```json\n" + judge_reply(("form", 7, "ok"), ("sense", 6, "ok")) + "\n```"
        parsed = parse_scores(fenced, CRITERIA)
        self.assertEqual([row["score"] for row in parsed or []], [7, 6])

    def test_scores_outside_the_scale_are_clamped_into_it(self):
        low = parse_scores(judge_reply(("form", 0, ""), ("sense", -3, "")), CRITERIA)
        high = parse_scores(judge_reply(("form", 42, ""), ("sense", 10, "")), CRITERIA)
        self.assertEqual([row["score"] for row in low or []], [1, 1])
        self.assertEqual([row["score"] for row in high or []], [10, 10])

    def test_a_missing_criterion_name_falls_back_to_the_rubric(self):
        raw = json.dumps({"scores": [{"score": 7, "justification": "ok"},
                                     {"score": 6, "justification": "ok"}]})
        parsed = parse_scores(raw, CRITERIA)
        self.assertEqual([row["criterion"] for row in parsed or []], ["form", "sense"])

    def test_an_empty_reply_is_unusable(self):
        self.assertIsNone(parse_scores("", CRITERIA))
        self.assertIsNone(parse_scores(None, CRITERIA))

    def test_a_reply_without_braces_is_unusable(self):
        self.assertIsNone(parse_scores("I cannot score this answer.", CRITERIA))

    def test_braces_that_do_not_wrap_json_are_unusable(self):
        self.assertIsNone(parse_scores("the { was unclosed", CRITERIA))

    def test_braces_that_wrap_broken_json_are_unusable(self):
        self.assertIsNone(parse_scores("{scores: [1, 2]}", CRITERIA))

    def test_a_score_array_of_the_wrong_length_is_unusable(self):
        short = judge_reply(("form", 7, "ok"))
        self.assertIsNone(parse_scores(short, CRITERIA))

    def test_a_scores_key_that_is_not_a_list_is_unusable(self):
        self.assertIsNone(parse_scores('{"scores": "8 and 6"}', CRITERIA))

    def test_a_non_numeric_score_is_unusable(self):
        raw = judge_reply(("form", "high", "ok"), ("sense", 6, "ok"))
        self.assertIsNone(parse_scores(raw, CRITERIA))

    def test_a_missing_score_is_unusable(self):
        raw = json.dumps({"scores": [{"criterion": "form", "justification": "ok"},
                                     {"criterion": "sense", "score": 6}]})
        self.assertIsNone(parse_scores(raw, CRITERIA))


class ScoreAnswerTest(unittest.TestCase):
    """score_answer against a live stub, because the reply arrives as prose over HTTP."""

    def setUp(self):
        self.judge = ModelConfig(name="stub-judge", token_limit=1200)

    def score_with(self, content: str, status: int = 200) -> dict:
        routes: dict[str, tuple[int, object]] = {
            "/v1/chat/completions": (status, completion(content))}
        with JsonStub(routes) as stub:
            self.judge.endpoint = stub.endpoint
            result = score_answer(self.judge, TASK, "a haiku of three lines")
        return result

    def test_a_good_reply_is_scored_and_averaged(self):
        result = self.score_with(judge_reply(("form", 8, "three lines"), ("sense", 5, "coheres")))
        self.assertEqual(result["judge_error"], None)
        self.assertFalse(result["empty"])
        self.assertEqual(result["mean"], 6.5)
        self.assertEqual(result["scores"][0]["criterion"], "form")

    def test_a_reply_that_is_not_json_is_reported_with_an_excerpt(self):
        result = self.score_with("I cannot score this answer in JSON.")
        self.assertEqual(result["scores"], [])
        self.assertIsNone(result["mean"])
        self.assertIn("I cannot score", result["judge_error"])

    def test_a_dead_endpoint_reports_the_transport_failure(self):
        result = self.score_with("unused", status=500)
        self.assertEqual(result["scores"], [])
        self.assertIn("500", result["judge_error"])

    def test_a_thinking_judge_is_named_instead_of_scored(self):
        routes: dict[str, tuple[int, object]] = {
            "/v1/chat/completions": (200, completion(
                "", reasoning="Let me analyze the answer against the rubric. " * 40,
                tokens=1200, finish_reason="length"))}
        with JsonStub(routes) as stub:
            self.judge.endpoint = stub.endpoint
            result = score_answer(self.judge, TASK, "a haiku of three lines")
        self.assertEqual(result["scores"], [])
        self.assertIsNone(result["mean"])
        self.assertIn("reasoning channel", result["judge_error"])
        self.assertIn("1200", result["judge_error"])

    def test_every_judge_call_asks_for_reasoning_off_at_temperature_zero(self):
        routes: dict[str, tuple[int, object]] = {
            "/v1/chat/completions": (200, completion(
                judge_reply(("form", 8, "three lines"), ("sense", 5, "coheres"))))}
        with JsonStub(routes) as stub:
            self.judge.endpoint = stub.endpoint
            score_answer(self.judge, TASK, "a haiku of three lines")
            sent = stub.seen[-1]["body"]
        self.assertEqual(sent["reasoning_effort"], "none")
        self.assertEqual(sent["temperature"], 0.0)

    def test_an_answer_with_no_content_takes_the_floor_without_a_call(self):
        routes: dict[str, tuple[int, object]] = {
            "/v1/chat/completions": (200, completion("should not be read"))}
        with JsonStub(routes) as stub:
            self.judge.endpoint = stub.endpoint
            result = score_answer(self.judge, TASK, "   \n\t")
        self.assertTrue(result["empty"])
        self.assertEqual(result["mean"], 1.0)
        self.assertEqual([row["score"] for row in result["scores"]], [1, 1])
        self.assertFalse([seen for seen in stub.seen if "chat/completions" in seen["path"]])


class ExcerptTest(unittest.TestCase):
    def test_no_reply_says_so(self):
        self.assertEqual(_excerpt(None), "no reply")
        self.assertEqual(_excerpt(""), "no reply")

    def test_whitespace_collapses_and_the_excerpt_stays_short(self):
        raw = "line one\n\nline two\twith a tab   and spaces"
        self.assertEqual(_excerpt(raw), "line one line two with a tab and spaces")
        self.assertEqual(len(_excerpt("x" * 500)), 200)


if __name__ == "__main__":
    unittest.main()
