"""The writing leg on JsonStub: generation always runs, scoring only with a judge."""
import json
import os
import tempfile
import unittest
from unittest import mock

from scripts.bench_writing_generate import (
    JUDGE_API_KEY_ENV,
    build_user_message,
    main as bench_main,
    parse_args,
    resolve_judge_key,
    scoring_extra,
    unscored_judge,
)


def task(task_id: str, category: str, prompt: str, source: str | None = None) -> dict:
    return {
        "task_id": task_id, "category": category, "prompt": prompt,
        "source_text": source, "output_contract": {},
        "rubric": [{"criterion": "craft", "description": "well made",
                    "anchors": {"2": "poor", "5": "fair", "8": "strong"}}],
    }


def write_suite(path: str) -> list[dict]:
    tasks = [
        task("T1", "xpost", "Write a post about boxes."),
        task("T2", "memo", "Summarise the memo.", source="MEMO BODY TEXT"),
        task("T3", "xpost", "Write a post about pens."),
    ]
    with open(path, "w") as handle:
        json.dump({"suite_version": "1.0.0", "tasks": tasks}, handle)
    return tasks


def generation_routes(body: dict) -> tuple[int, dict]:
    from tests.stub_json import completion

    content = f"ANSWER FOR {body['messages'][0]['content'][:22]}"
    return 200, completion(content)


def judge_routes(body: dict) -> tuple[int, dict]:
    from tests.stub_json import completion

    verdict = {"scores": [{"criterion": "craft", "score": 7, "justification": "fine"}]}
    return 200, completion(json.dumps(verdict))


class LegCase(unittest.TestCase):
    """Shared plumbing: one stub for generation, one for the judge, one suite file."""

    def setUp(self) -> None:
        from tests.stub_json import JsonStub

        self.gen = JsonStub(generation_routes)
        self.gen._thread.start()
        self.addCleanup(self.gen.close)
        self.suite = os.path.join(tempfile.mkdtemp(), "suite.json")
        write_suite(self.suite)
        self.out = os.path.join(tempfile.mkdtemp(), "legs")

    def run_leg(self, *extra: str) -> int:
        return bench_main(["--model", "m", "--endpoint", self.gen.endpoint,
                           "--data", self.suite, "--output-dir", self.out, *extra])

    def sample(self, task_id: str) -> dict:
        with open(os.path.join(self.out, "m", f"{task_id}.json")) as handle:
            return json.load(handle)

    def block(self) -> dict:
        with open(os.path.join(self.out, "writing.result.json")) as handle:
            return json.load(handle)


class MessageTest(unittest.TestCase):
    def test_a_source_rides_along_when_the_prompt_lacks_it(self):
        message = build_user_message(task("T", "c", "Do the thing.", source="THE SOURCE"))
        self.assertEqual(message, "Do the thing.\n\nSOURCE TEXT:\nTHE SOURCE")

    def test_a_source_already_in_the_prompt_is_not_duplicated(self):
        prompt = "Here is the text: SHARED. Now write."
        self.assertEqual(build_user_message(task("T", "c", prompt, source="SHARED")), prompt)

    def test_a_missing_source_leaves_the_prompt_alone(self):
        self.assertEqual(build_user_message(task("T", "c", "Just write.")), "Just write.")


class NoJudgeTest(LegCase):
    def test_generation_runs_and_the_block_reports_not_scored(self):
        code = self.run_leg()
        self.assertEqual(code, 0)
        for task_id in ("T1", "T2", "T3"):
            record = self.sample(task_id)
            self.assertTrue(record["response"].startswith("ANSWER FOR"))
            self.assertFalse(record["empty"])
            self.assertEqual(record["model"], "m")
            self.assertEqual(record["suite_version"], "1.0.0")
        block = self.block()
        self.assertEqual(block["status"], "not scored")
        self.assertIn("judge", block["reason"])
        self.assertEqual(block["items"], 3)
        self.assertEqual(block["scored"], 0)
        self.assertEqual(block["correct"], 0)
        self.assertEqual(len(self.gen.seen), 3)

    def test_the_generation_settings_reach_the_wire(self):
        self.run_leg("--cap", "512")
        sent = self.gen.seen[-1]["body"]
        self.assertEqual(sent["max_tokens"], 512)
        self.assertEqual(sent["temperature"], 0.7)
        self.assertNotIn("chat_template_kwargs", sent)
        self.assertIn("SOURCE TEXT", self.gen.seen[1]["body"]["messages"][0]["content"])

    def test_the_standard_cap_and_temperature_are_the_defaults(self):
        self.run_leg()
        sent = self.gen.seen[0]["body"]
        self.assertEqual(sent["max_tokens"], 16384)
        self.assertEqual(sent["temperature"], 0.7)

    def test_category_and_limit_select_the_tasks(self):
        self.run_leg("--category", "xpost")
        self.assertEqual(self.block()["items"], 2)
        self.run_leg("--limit", "1")
        self.assertEqual(self.block()["items"], 1)

    def test_skip_existing_reuses_the_samples_on_disk(self):
        self.run_leg()
        self.run_leg("--skip-existing")
        self.assertEqual(len(self.gen.seen), 3)
        self.assertEqual(self.block()["items"], 3)

    def test_a_judge_without_an_endpoint_is_not_scored(self):
        self.run_leg("--judge-model", "judge-m")
        self.assertEqual(self.block()["status"], "not scored")
        self.assertIn("endpoint", self.block()["reason"])


class JudgeTest(LegCase):
    def setUp(self) -> None:
        from tests.stub_json import JsonStub

        super().setUp()
        self.judge = JsonStub(judge_routes)
        self.judge._thread.start()
        self.addCleanup(self.judge.close)

    def test_the_judge_path_scores_every_sample(self):
        code = self.run_leg("--judge-model", "judge-m", "--judge-endpoint", self.judge.endpoint)
        self.assertEqual(code, 0)
        block = self.block()
        self.assertEqual(block["status"], "pass")
        self.assertEqual(block["judge_model"], "judge-m")
        self.assertEqual(block["judge_endpoint"], self.judge.endpoint)
        self.assertEqual(block["judge_scored"], 3)
        self.assertEqual(block["judge_errors"], 0)
        self.assertEqual(block["scores"], {"T1": 7.0, "T2": 7.0, "T3": 7.0})
        self.assertEqual(block["mean_score"], 7.0)
        self.assertEqual(self.sample("T2")["judgement"]["mean"], 7.0)
        self.assertEqual(len(self.judge.seen), 3)
        self.assertEqual(len(self.gen.seen), 3)


class SelfJudgeTest(LegCase):
    def test_a_judge_that_is_the_model_under_test_is_refused(self):
        code = self.run_leg("--judge-model", "m", "--judge-endpoint", self.gen.endpoint)
        self.assertEqual(code, 0)
        block = self.block()
        self.assertEqual(block["status"], "not scored")
        self.assertIn("model under test", block["reason"])
        self.assertEqual(len(self.gen.seen), 3)


def judge_key_argv(*argv: str) -> list[str]:
    """A full leg command line, so parse_args resolves the key the way a run would."""
    return ["--model", "m", "--endpoint", "http://run/v1", "--data", "suite.json",
            "--output-dir", "out", *argv]


class JudgeKeyTest(unittest.TestCase):
    """The judge key falls back to the run's key, because the judge sits behind it."""

    def test_the_flag_beats_the_env_var_and_the_run_key(self):
        with mock.patch.dict(os.environ, {JUDGE_API_KEY_ENV: "from-env"}):
            options = parse_args(judge_key_argv("--judge-api-key", "from-flag",
                                                "--api-key", "run-key"))
        self.assertEqual(options.judge_api_key, "from-flag")

    def test_the_env_var_is_used_when_the_flag_is_absent(self):
        with mock.patch.dict(os.environ, {JUDGE_API_KEY_ENV: "from-env"}):
            options = parse_args(judge_key_argv("--api-key", "run-key"))
        self.assertEqual(options.judge_api_key, "from-env")

    def test_the_run_key_is_the_fallback(self):
        with mock.patch.dict(os.environ, {JUDGE_API_KEY_ENV: ""}):
            options = parse_args(judge_key_argv("--api-key", "run-key"))
        self.assertEqual(options.judge_api_key, "run-key")

    def test_a_run_with_no_keys_at_all_asks_for_none(self):
        with mock.patch.dict(os.environ, {JUDGE_API_KEY_ENV: ""}):
            options = parse_args(judge_key_argv())
        self.assertEqual(options.judge_api_key, "not-needed")

    def test_the_resolver_is_a_plain_precedence_chain(self):
        with mock.patch.dict(os.environ, {JUDGE_API_KEY_ENV: ""}):
            self.assertEqual(resolve_judge_key("flag", "run"), "flag")
            self.assertEqual(resolve_judge_key("", "run"), "run")
            self.assertEqual(resolve_judge_key("", ""), "not-needed")


class UnscoredJudgeTest(unittest.TestCase):
    """A judge that parsed nothing is no score, not a zero."""

    def extra(self, *errors: str) -> dict:
        results = {f"T{index}": {"mean": None, "judge_error": error}
                   for index, error in enumerate(errors)}
        return scoring_extra(results, "judge-m", "http://judge/v1")

    def test_the_first_error_is_carried_for_the_report(self):
        extra = self.extra("first failure", "second failure")
        self.assertEqual(extra["judge_errors"], 2)
        self.assertEqual(extra["judge_first_error"], "first failure")
        self.assertEqual(extra["judge_scored"], 0)

    def test_a_judge_that_scored_nothing_reports_unscored(self):
        block = unscored_judge(self.extra("the judge answered only on the reasoning channel"), 3)
        self.assertEqual(block["status"], "not scored")
        self.assertIn("no parseable score", block["reason"])
        self.assertIn("reasoning channel", block["reason"])
        self.assertNotIn("mean_score", block)
        self.assertEqual(block["judge_scored"], 0)

    def test_a_partial_judge_keeps_the_scores_it_has(self):
        results = {"T1": {"mean": 7.0, "judge_error": None},
                   "T2": {"mean": None, "judge_error": "boom"}}
        extra = scoring_extra(results, "judge-m", "http://judge/v1")
        self.assertEqual(extra["judge_scored"], 1)
        self.assertEqual(extra["judge_errors"], 1)
        self.assertEqual(extra["mean_score"], 7.0)


def thinking_judge_routes(body: dict) -> tuple[int, dict]:
    """A judge that spends the whole budget reasoning and never emits JSON."""
    from tests.stub_json import completion

    return 200, completion("", reasoning="Let me analyze the answer against the rubric. " * 40,
                           tokens=1200, finish_reason="length")


class ThinkingJudgeTest(LegCase):
    """End to end: the failure that used to look like a passing zero."""

    def setUp(self) -> None:
        from tests.stub_json import JsonStub

        super().setUp()
        self.judge = JsonStub(thinking_judge_routes)
        self.judge._thread.start()
        self.addCleanup(self.judge.close)

    def test_the_run_reports_unscored_with_the_reason(self):
        code = self.run_leg("--judge-model", "judge-m", "--judge-endpoint", self.judge.endpoint,
                            "--api-key", "run-key")
        self.assertEqual(code, 0)
        block = self.block()
        self.assertEqual(block["status"], "not scored")
        self.assertIn("no parseable score", block["reason"])
        self.assertIn("3 judge errors", block["reason"])
        self.assertNotIn("mean_score", block)
        self.assertEqual(self.sample("T1")["response"][:10], "ANSWER FOR")

    def test_the_judge_request_carries_reasoning_off_and_the_run_key(self):
        self.run_leg("--judge-model", "judge-m", "--judge-endpoint", self.judge.endpoint,
                     "--api-key", "run-key")
        sent = self.judge.seen[0]
        self.assertEqual(sent["body"]["reasoning_effort"], "none")
        headers = {key.lower(): value for key, value in sent["headers"].items()}
        self.assertEqual(headers["authorization"], "Bearer run-key")


if __name__ == "__main__":
    unittest.main()
