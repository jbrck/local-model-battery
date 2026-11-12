"""IFBench leg: strict and loose counting, cap-hit handling, missing-verifier reporting.

Runs against JsonStub with an injected fake checker: no real model, no ifbench
package. The fake decides pass/fail from substrings in the response, so every
combination is scriptable.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.bench_ifbench import (  # noqa: E402
    Checker,
    classify_probe_failure,
    loose_variants,
    main,
    parse_probe,
    resolve_checker,
    run_items,
)
from scripts.lib.harness import LegArgs, leg_config  # noqa: E402


class StubChecker(Checker):
    """The injected fake verifier: substrings in the response decide the flags."""

    STRICT_OK = "strict-ok"
    LOOSE_OK = "loose-ok"

    def check(self, prompt: str, instruction_ids: list[str],
              kwargs_list: list[dict], response: str) -> tuple:
        strict_hit = self.STRICT_OK in response
        loose_hit = self.LOOSE_OK in response
        return ([strict_hit] * len(instruction_ids),
                [loose_hit] * len(instruction_ids))


def stub_route(body: dict) -> tuple[int, dict]:
    """Answer from the prompt, so each item can demand its own outcome."""
    from tests.stub_json import completion

    marker = body["messages"][0]["content"]
    if "expect-strict-loose" in marker:
        content = "strict-ok and loose-ok both present."
    elif "expect-loose" in marker:
        content = "loose-ok only, strict missing."
    else:
        content = "neither marker present."
    return 200, completion(content, tokens=40)


def cap_route(body: dict) -> tuple[int, dict]:
    """A capped answer: the server stops for length with real tokens behind it."""
    from tests.stub_json import completion

    return 200, completion("strict-ok loose-ok", tokens=9000,
                           finish_reason="length")


def sample_items() -> list[dict]:
    return [
        {"key": "0", "prompt": "item expect-strict-loose",
         "instruction_id_list": ["count:keywords_multiple"],
         "kwargs": [{"keyword1": "a"}]},
        {"key": "1", "prompt": "item expect-strict-loose",
         "instruction_id_list": ["detectable_format:title"],
         "kwargs": [{}]},
        {"key": "2", "prompt": "item expect-loose",
         "instruction_id_list": ["punctuation:no_comma"],
         "kwargs": [{}]},
        {"key": "3", "prompt": "item expect-nothing",
         "instruction_id_list": ["change_case:english_lowercase"],
         "kwargs": [{}]},
    ]


def make_args(endpoint: str, out_dir: str) -> LegArgs:
    return replace(LegArgs(model="m", endpoint=endpoint), cap=1024,
                   output_dir=out_dir)


class RunItemsTest(unittest.TestCase):
    def setUp(self):
        from tests.stub_json import JsonStub

        self.stub = JsonStub(stub_route)
        self.stub._thread.start()
        self.addCleanup(self.stub.close)
        self.out = tempfile.TemporaryDirectory()
        self.addCleanup(self.out.cleanup)
        self.args = make_args(self.stub.endpoint, self.out.name)

    def test_strict_and_loose_counts_come_from_the_checker(self):
        summary = run_items(sample_items(), leg_config(self.args), self.args,
                            StubChecker())
        tally = summary.tally
        self.assertEqual(tally.scored, 4)
        self.assertEqual(summary.strict_prompts, 2)
        self.assertEqual(summary.loose_prompts, 3)
        self.assertEqual(tally.correct, 2)
        self.assertEqual(summary.strict_instr_pass, 2)
        self.assertEqual(summary.strict_instr_total, 4)
        self.assertEqual(summary.loose_instr_pass, 3)
        self.assertEqual(summary.loose_instr_total, 4)
        self.assertEqual(tally.status(), "pass")

    def test_a_capped_answer_is_still_scored_and_counted(self):
        from tests.stub_json import JsonStub

        capped = JsonStub(cap_route)
        capped._thread.start()
        self.addCleanup(capped.close)
        args = make_args(capped.endpoint, self.out.name)
        summary = run_items(sample_items()[:2], leg_config(args), args,
                            StubChecker())
        self.assertEqual(summary.tally.cap_hits, 2)
        self.assertEqual(summary.tally.scored, 2)
        self.assertEqual(summary.strict_prompts, 2)
        self.assertEqual(summary.tally.status(), "pass")


class VerifierResolutionTest(unittest.TestCase):
    def test_a_missing_interpreter_reports_a_reason(self):
        env, reason = resolve_checker("/no/such/python-ifbench", "/tmp/nltk-none")
        self.assertIsNone(env)
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("does not exist", reason)

    def test_a_missing_nltk_corpus_reports_a_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = os.path.join(tmp, "fake_python")
            with open(fake, "w") as handle:
                handle.write("#!/usr/bin/env python3\n"
                             "import sys\n"
                             "sys.stdout.write('{\"ok\": false, \"error\": "
                             "\"LookupError: wordnet resource not found\"}\\n')\n")
            os.chmod(fake, 0o755)
            env, reason = resolve_checker(fake, os.path.join(tmp, "nltk-none"))
        self.assertIsNone(env)
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("NLTK corpus missing", reason)


class NotRunIntegrationTest(unittest.TestCase):
    def test_a_missing_verifier_writes_a_not_run_block(self):
        with tempfile.TemporaryDirectory() as out:
            argv = ["--model", "m", "--output-dir", out,
                    "--data", os.path.join(out, "items.json"),
                    "--ifbench-python", "/no/such/python-ifbench"]
            with open(os.path.join(out, "items.json"), "w") as handle:
                json.dump(sample_items(), handle)
            code = main(argv)
            self.assertEqual(code, 2)
            path = os.path.join(out, "ifbench.result.json")
            self.assertTrue(os.path.exists(path))
            with open(path) as handle:
                block = json.load(handle)
            self.assertEqual(block["status"], "not run")
            self.assertIn("does not exist", block["notes"][0])


FAKE_WORKER = (
    "#!/usr/bin/env python3\n"
    "import json, sys\n"
    "args = sys.argv\n"
    "cmd = args[args.index('-c') + 1] if '-c' in args else ''\n"
    "if 'def main()' in cmd:\n"
    "    print(json.dumps({'id': 'ready', 'registry': 2}), flush=True)\n"
    "    for line in sys.stdin:\n"
    "        req = json.loads(line)\n"
    "        print(json.dumps({'id': req['id'], 'strict': [True],\n"
    "                          'loose': [True]}), flush=True)\n"
    "else:\n"
    "    print(json.dumps({'ok': True, 'registry': 2}))\n"
)


def write_fake_python(directory: str) -> str:
    path = os.path.join(directory, "fake_python")
    with open(path, "w") as handle:
        handle.write(FAKE_WORKER)
    os.chmod(path, 0o755)
    return path


class FakeWorkerTest(unittest.TestCase):
    """The real client protocol exercised against a fake interpreter."""

    def setUp(self):
        from tests.stub_json import JsonStub, completion

        self.stub = JsonStub(lambda body: (200,
                                           completion("plain answer here.")))
        self.stub._thread.start()
        self.addCleanup(self.stub.close)
        self.out = tempfile.TemporaryDirectory()
        self.addCleanup(self.out.cleanup)
        self.args = make_args(self.stub.endpoint, self.out.name)

    def test_a_resolved_checker_runs_the_real_protocol_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = write_fake_python(tmp)
            checker, reason = resolve_checker(fake, os.path.join(tmp, "nl"))
        self.assertIsNone(reason)
        assert checker is not None
        try:
            summary = run_items(sample_items(), leg_config(self.args),
                                self.args, checker)
        finally:
            checker.close()
        self.assertEqual(summary.tally.scored, 4)
        self.assertEqual(summary.strict_prompts, 4)
        self.assertEqual(summary.loose_prompts, 4)

    def test_a_worker_that_never_handshakes_reports_a_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = os.path.join(tmp, "mute_python")
            with open(fake, "w") as handle:
                handle.write("#!/usr/bin/env python3\n"
                             "import sys\n"
                             "args = sys.argv\n"
                             "cmd = args[args.index('-c') + 1] "
                             "if '-c' in args else ''\n"
                             "if 'def main()' in cmd:\n"
                             "    sys.exit(0)\n"
                             "else:\n"
                             "    print('{\"ok\": true, \"registry\": 2}')\n")
            os.chmod(fake, 0o755)
            env, reason = resolve_checker(fake, os.path.join(tmp, "nl"))
        self.assertIsNone(env)
        assert reason is not None
        self.assertIn("worker failed to start", reason)

    def test_main_runs_end_to_end_and_writes_the_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = write_fake_python(tmp)
            data = os.path.join(tmp, "items.json")
            with open(data, "w") as handle:
                json.dump(sample_items(), handle)
            argv = ["--model", "m", "--endpoint", self.stub.endpoint,
                    "--output-dir", tmp,
                    "--data", data, "--ifbench-python", fake,
                    "--limit", "2"]
            code = main(argv)
            self.assertEqual(code, 0)
            path = os.path.join(tmp, "ifbench.result.json")
            with open(path) as handle:
                block = json.load(handle)
            self.assertEqual(block["status"], "pass")
            self.assertEqual(block["loose_prompt_correct"], 2)
            self.assertEqual(block["strict_prompt_correct"], 2)


class ClassifierUnitTest(unittest.TestCase):
    """The probe parser and failure classifier in isolation."""

    def test_a_garbage_probe_result_is_a_readable_reason(self):
        reason = parse_probe("not json at all")
        self.assertIsInstance(reason, str)

    def test_a_registry_error_is_classified_as_such(self):
        reason = classify_probe_failure(
            {"error": "instructions_registry is empty"}, "/x", "/n")
        self.assertIn("instructions_registry", reason)

    def test_a_generic_import_error_names_the_package(self):
        reason = classify_probe_failure(
            {"error": "ImportError: No module named 'ifbench'"}, "/x", "/n")
        self.assertIn("no usable 'ifbench' package", reason)


class LooseVariantsTest(unittest.TestCase):
    def test_the_transform_matches_the_reference_order(self):
        variants = loose_variants("*one\n*two\n*three")
        self.assertEqual(variants[:3],
                         ["*one\n*two\n*three", "one\ntwo\nthree", "*two\n*three"])
        self.assertEqual(variants[3], "*one\n*two")
        self.assertEqual(variants[4], "*two")
        self.assertEqual(variants[5:], ["two\nthree", "one\ntwo", "two"])


if __name__ == "__main__":
    unittest.main()
