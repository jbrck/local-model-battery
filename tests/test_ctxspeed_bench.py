"""The context-depth leg: the ladder, the validity rules, the table, and a run against a stub."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from scripts.ctxspeed_bench import (BISECT_RESOLUTION, DEFAULT_RUNGS, MIN_DECODE_TOKENS,
                                    Calibration, Failure, Options, calibrate, ceiling_block, clipped,
                                    copy_ratio, count_trial, deepest_retention, depth_error,
                                    failure_kind, ladder, main, parse_args, per_depth, plan_climb,
                                    pooled_decode_tps, retention, run_leg, run_trial,
                                    server_rate, split_csv, trial_row, trial_seed, WINDOW_SLACK)
from scripts.lib.harness import LegArgs, Tally, leg_config
from scripts.lib.model import ChatResponse
from scripts.lib.stream import StreamTiming
from tests.stub_json import completion
from tests.stub_sse import SseStub, prompt_of, prompt_tokens

CORPUS = [f"unit {index:04d} " + "alpha beta gamma delta " * 3 for index in range(1500)]
CAP = MIN_DECODE_TOKENS + 32


def args_for(stub: SseStub, tmp: str, **overrides) -> LegArgs:
    fields: dict = dict(model="m", endpoint=stub.endpoint, output_dir=tmp, cap=CAP, ctx=65536,
                        timeout_s=30)
    fields.update(overrides)
    # `**fields` erases the per-parameter types, which the checker cannot follow.
    return LegArgs(**fields)  # type: ignore[arg-type]


def timing(prompt: int = 1000, tokens: int = 200, ttft: float = 2.0, decode: float = 4.0,
           finish: str = "length", text: str = "w ") -> StreamTiming:
    payload = completion(text, tokens=tokens, finish_reason=finish)
    payload["usage"]["prompt_tokens"] = prompt
    return StreamTiming(response=ChatResponse(status=200, payload=payload, elapsed_s=ttft + decode),
                        ttft_s=ttft, decode_s=decode, usage_reported=True)


class LadderTest(unittest.TestCase):
    def test_the_default_ladder_reaches_every_depth_worth_asking_about(self):
        for depth in (131072, 262144, 524288, 1048576):
            self.assertIn(depth, DEFAULT_RUNGS)

    def test_the_default_ladder_is_ascending(self):
        self.assertEqual(list(DEFAULT_RUNGS), sorted(DEFAULT_RUNGS))

    def test_no_depths_asked_gives_the_default_ladder(self):
        self.assertEqual(ladder(()), list(DEFAULT_RUNGS))

    def test_asked_depths_are_sorted_and_deduplicated_not_trimmed_to_a_window(self):
        self.assertEqual(ladder((900000, 8192, 900000, 32768)), [8192, 32768, 900000])


class ParseArgsTest(unittest.TestCase):
    BASE = ["--model", "m", "--endpoint", "http://x/v1"]

    def test_defaults_run_both_tasks_with_two_trials_and_an_automatic_ladder(self):
        _, options = parse_args(self.BASE)
        self.assertEqual((options.depths, options.trials, options.tasks),
                         ((), 2, ("prose", "copy")))

    def test_flags_override_the_defaults_and_tolerate_spacing(self):
        args, options = parse_args([*self.BASE, "--depths", "512, 8192,", "--trials", "3",
                                    "--tasks", "copy", "--ctx", "4096"])
        self.assertEqual((options.depths, options.trials, options.tasks),
                         ((512, 8192), 3, ("copy",)))
        self.assertEqual(args.ctx, 4096)

    def test_zero_trials_is_raised_to_one(self):
        self.assertEqual(parse_args([*self.BASE, "--trials", "0"])[1].trials, 1)

    def test_an_unknown_task_is_refused(self):
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            parse_args([*self.BASE, "--tasks", "prose,poetry"])
        self.assertIn("poetry", err.getvalue())

    def test_split_csv_drops_blanks(self):
        self.assertEqual(split_csv(" a, ,b ,"), ["a", "b"])


class CopyRatioTest(unittest.TestCase):
    CONTEXT = " ".join(f"w{index}" for index in range(100))

    def test_a_verbatim_copy_scores_one(self):
        self.assertEqual(copy_ratio(self.CONTEXT, " ".join(f"w{i}" for i in range(50, 100))), 1.0)

    def test_unrelated_text_scores_zero(self):
        self.assertEqual(copy_ratio(self.CONTEXT, " ".join(["printing press"] * 30)), 0.0)

    def test_output_shorter_than_one_gram_scores_zero(self):
        self.assertEqual(copy_ratio(self.CONTEXT, "w1 w2"), 0.0)

    def test_half_copied_scores_between(self):
        copied = " ".join(f"w{i}" for i in range(20))
        invented = " ".join(f"x{i}" for i in range(20))
        self.assertTrue(0.3 < copy_ratio(self.CONTEXT, copied + " " + invented) < 0.7)


class TrialRowTest(unittest.TestCase):
    def test_a_trial_inside_the_depth_tolerance_is_valid(self):
        row = trial_row("prose", 1000, 0, "p", "c", timing(prompt=1050))
        self.assertTrue(row["valid"])
        self.assertEqual(row["depth_error_pct"], 5.0)
        self.assertEqual(row["decode_tps"], 49.8)  # (200 - 1) / 4.0
        self.assertEqual(row["prefill_tps"], 525.0)  # 1050 / 2.0

    def test_a_prompt_that_landed_far_from_its_depth_is_invalid(self):
        self.assertFalse(trial_row("prose", 1000, 0, "p", "c", timing(prompt=700))["valid"])

    def test_a_window_too_short_to_call_a_rate_is_invalid(self):
        short = timing(tokens=MIN_DECODE_TOKENS - 1)
        self.assertFalse(trial_row("prose", 1000, 0, "p", "c", short)["valid"])

    def test_a_failed_request_is_invalid(self):
        failed = StreamTiming(response=ChatResponse(status=500, error="HTTP 500"))
        row = trial_row("prose", 1000, 0, "p", "c", failed)
        self.assertFalse(row["valid"])
        self.assertEqual(row["error"], "HTTP 500")

    def test_no_prompt_count_does_not_invalidate_a_trial_on_depth(self):
        self.assertIsNone(depth_error(1000, 0))
        self.assertTrue(trial_row("prose", 1000, 0, "p", "c", timing(prompt=0))["valid"])

    def test_trials_and_tasks_get_distinct_seeds(self):
        args = LegArgs(model="m", seed=42)
        seeds = {trial_seed(args, trial, task) for trial in range(5) for task in range(2)}
        self.assertEqual(len(seeds), 10)


class RateTest(unittest.TestCase):
    def test_the_servers_own_rate_is_kept_for_cross_checking(self):
        reported = timing()
        reported.server_timings = {"predicted_per_second": 52.46}
        self.assertEqual(server_rate(reported), 52.5)
        self.assertEqual(server_rate(timing()), 0.0)

    def test_retention_treats_a_zero_baseline_as_the_first_line(self):
        self.assertEqual((retention(80.0, 0.0), retention(40.0, 80.0), retention(0.0, 0.0)),
                         (1.0, 0.5, 0.0))


class CountTrialTest(unittest.TestCase):
    def test_the_designed_length_stop_is_not_a_cap_hit(self):
        tally = Tally()
        count_trial(tally, timing(finish="length", tokens=200), valid=True)
        self.assertEqual((tally.items, tally.scored, tally.correct, tally.cap_hits), (1, 1, 1, 0))

    def test_an_invalid_trial_is_scored_against(self):
        tally = Tally()
        count_trial(tally, timing(), valid=False)
        self.assertEqual((tally.scored, tally.correct), (1, 0))


def row(task: str, depth: int, tps: float, valid: bool = True, tokens: int = 200) -> dict:
    return {"task": task, "depth": depth, "valid": valid, "decode_tps": tps, "ttft_s": 1.0,
            "prefill_tps": 100.0, "prompt_tokens": depth, "copy_ratio": 0.5,
            "completion_tokens": tokens, "decode_s": (tokens - 1) / tps if tps else 0.0}


class TableTest(unittest.TestCase):
    ROWS = [row("prose", 1000, 100.0), row("prose", 1000, 110.0), row("prose", 8000, 60.0),
            row("prose", 8000, 0.0, valid=False), row("copy", 1000, 200.0), row("copy", 8000, 100.0)]

    def test_each_depth_reports_the_median_of_its_valid_trials(self):
        table = {(line["task"], line["depth"]): line for line in per_depth(self.ROWS)}
        self.assertEqual(table[("prose", 1000)]["decode_tps"], 105.0)
        self.assertEqual(table[("prose", 8000)]["decode_tps"], 60.0)
        self.assertEqual((table[("prose", 8000)]["trials"], table[("prose", 8000)]["valid"]), (2, 1))

    def test_retention_is_measured_against_the_shallowest_valid_depth_of_the_same_task(self):
        table = {(line["task"], line["depth"]): line for line in per_depth(self.ROWS)}
        self.assertEqual(table[("prose", 1000)]["retention"], 1.0)
        self.assertEqual(table[("prose", 8000)]["retention"], round(60.0 / 105.0, 3))
        self.assertEqual(table[("copy", 8000)]["retention"], 0.5)

    def test_a_depth_with_no_valid_trial_reports_zero_not_a_borrowed_number(self):
        table = per_depth([row("prose", 1000, 100.0), row("prose", 8000, 50.0, valid=False)])
        self.assertEqual(table[1]["decode_tps"], 0.0)
        self.assertEqual(table[1]["valid"], 0)

    def test_the_headline_retention_reads_the_deepest_valid_depth(self):
        deepest = deepest_retention(per_depth(self.ROWS))
        self.assertEqual(deepest["copy"], {"depth": 8000, "retention": 0.5})

    def test_pooled_rate_is_tokens_over_seconds_not_a_mean_of_rates(self):
        pooled = [row("prose", 1000, 100.0, tokens=101), row("prose", 8000, 10.0, tokens=101)]
        self.assertEqual(pooled_decode_tps(pooled), round(200 / (1.0 + 10.0), 1))

    def test_pooled_rate_ignores_invalid_trials(self):
        self.assertEqual(pooled_decode_tps([row("prose", 1000, 50.0, valid=False)]), 0.0)


class RunLegTest(unittest.TestCase):
    def run_stub(self, stub: SseStub, **options) -> tuple:
        with stub, tempfile.TemporaryDirectory() as tmp:
            return run_leg(args_for(stub, tmp), Options(**options), corpus=CORPUS)

    def test_decode_rate_falls_with_depth_and_the_table_shows_it(self):
        stub = SseStub(token_s=0.002, token_s_per_ktok=0.002)
        with contextlib.redirect_stdout(io.StringIO()):
            tally, rows, facts = self.run_stub(stub, depths=(500, 4000), trials=1, tasks=("prose",))
        table = {line["depth"]: line for line in per_depth(rows)}
        self.assertEqual(facts["depths"], [500, 4000])
        self.assertTrue(all(item["valid"] for item in rows), rows)
        self.assertGreater(table[500]["decode_tps"], table[4000]["decode_tps"])
        self.assertLess(table[4000]["retention"], 0.8)
        self.assertEqual((tally.items, tally.cap_hits), (2, 0))

    def test_prefill_time_lands_in_time_to_first_token(self):
        stub = SseStub(prefill_s_per_ktok=0.2, token_s=0.001)
        with contextlib.redirect_stdout(io.StringIO()):
            _, rows, _ = self.run_stub(stub, depths=(500, 4000), trials=1, tasks=("prose",))
        shallow, deep = rows
        self.assertGreater(deep["ttft_s"], shallow["ttft_s"] + 0.4)

    def test_the_copy_task_reads_back_as_a_copy_and_prose_does_not(self):
        with contextlib.redirect_stdout(io.StringIO()):
            _, rows, _ = self.run_stub(SseStub(), depths=(2000,), trials=1)
        ratios = {item["task"]: item["copy_ratio"] for item in rows}
        self.assertGreater(ratios["copy"], 0.9)
        self.assertLess(ratios["prose"], 0.1)

    def test_every_trial_gets_different_filler(self):
        stub = SseStub()
        with contextlib.redirect_stdout(io.StringIO()):
            _, rows, _ = self.run_stub(stub, depths=(1000,), trials=3, tasks=("prose",))
        self.assertEqual(len({item["prompt_sha16"] for item in rows}), 3)

    def test_the_depth_is_calibrated_from_the_servers_own_token_count(self):
        with SseStub() as stub, tempfile.TemporaryDirectory() as tmp:
            args = args_for(stub, tmp)
            calibration = calibrate(leg_config(args), args, CORPUS)
        self.assertAlmostEqual(calibration.chars_per_token, 4.0, delta=0.4)
        self.assertEqual(calibration.note, "")
        self.assertTrue(all(tokens > 20 for tokens in calibration.overhead.values()))

    def test_a_server_without_a_prompt_count_falls_back_and_says_so(self):
        with SseStub(send_usage=False) as stub, tempfile.TemporaryDirectory() as tmp:
            args = args_for(stub, tmp)
            calibration = calibrate(leg_config(args), args, CORPUS)
        self.assertEqual(calibration.chars_per_token, 3.5)
        self.assertIn("estimate", calibration.note)
        self.assertEqual(set(calibration.overhead.values()), {0})

    def test_a_shallow_rung_lands_on_its_depth_despite_the_wrapper(self):
        with contextlib.redirect_stdout(io.StringIO()):
            _, rows, facts = self.run_stub(SseStub(), depths=(300,), trials=1)
        self.assertTrue(all(item["valid"] for item in rows), rows)
        self.assertTrue(all(abs(item["depth_error_pct"]) <= 5 for item in rows), rows)
        self.assertTrue(all(tokens > 0 for tokens in facts["wrapper_tokens"].values()))

    def climb(self, stub: SseStub, depths: tuple[int, ...]) -> tuple:
        with contextlib.redirect_stdout(io.StringIO()):
            return self.run_stub(stub, depths=depths, trials=1, tasks=("prose",))

    def test_a_refused_prompt_ends_the_climb_and_names_the_window(self):
        tally, rows, facts = self.climb(SseStub(window=20000), (500, 8000, 65536, 200000))
        ceiling = facts["ceiling"]
        self.assertEqual(ceiling["limit"], "rejected")
        self.assertGreater(ceiling["first_failed_depth"], 20000)
        self.assertEqual(facts["unreached_depths"], [200000])
        self.assertIn("window", ceiling["meaning"])
        self.assertEqual(tally.status(), "pass")

    def test_the_bisect_lands_within_its_resolution_of_the_window(self):
        _, rows, facts = self.climb(SseStub(window=20000), (500, 8000, 65536))
        ceiling = facts["ceiling"]
        self.assertLessEqual(ceiling["deepest_prompt_tokens_ok"], 20000)
        self.assertGreater(ceiling["deepest_prompt_tokens_ok"], 20000 - 2 * BISECT_RESOLUTION)
        self.assertLessEqual(ceiling["first_failed_depth"] - ceiling["deepest_depth_ok"],
                             BISECT_RESOLUTION)

    def test_a_clipped_prompt_is_reported_as_truncation_not_a_good_rung(self):
        _, _, facts = self.climb(SseStub(window=12000, over="clip"), (500, 4000, 65536))
        ceiling = facts["ceiling"]
        self.assertEqual(ceiling["limit"], "truncated")
        self.assertLessEqual(ceiling["deepest_prompt_tokens_ok"], 12000)

    def test_a_dropped_connection_stops_the_climb_without_a_bisect(self):
        stub = SseStub(window=12000, over="drop")
        _, rows, facts = self.climb(stub, (500, 4000, 65536, 200000))
        ceiling = facts["ceiling"]
        self.assertEqual((ceiling["limit"], ceiling["deepest_depth_ok"],
                          ceiling["first_failed_depth"]), ("server_failure", 4000, 65536))
        self.assertEqual(facts["unreached_depths"], [200000])

    def test_a_climb_that_never_fails_says_the_top_rung_is_a_floor(self):
        _, rows, facts = self.climb(SseStub(), (500, 4000))
        ceiling = facts["ceiling"]
        self.assertEqual((ceiling["limit"], ceiling["deepest_depth_ok"],
                          ceiling["first_failed_depth"]), ("ladder_top", 4000, None))
        self.assertIn("floor", ceiling["meaning"])

    def test_a_server_that_refuses_the_first_rung_is_a_failed_leg(self):
        tally, rows, facts = self.climb(SseStub(window=100), (500, 4000))
        self.assertEqual((rows, facts["ceiling"]["deepest_depth_ok"]), ([], 0))
        self.assertEqual(tally.status(), "fail")

    def test_a_reported_window_keeps_the_leg_from_sending_a_doomed_rung(self):
        stub = SseStub(window=20000, report_window=True)
        tally, rows, facts = self.climb(stub, (500, 8000, 65536, 200000))
        ceiling = facts["ceiling"]
        self.assertEqual(facts["served_window"], 20000)
        self.assertEqual(ceiling["limit"], "served_window")
        self.assertEqual(ceiling["skipped_beyond_window"], [65536, 200000])
        self.assertIsNone(ceiling["first_failed_depth"])
        self.assertIn("relaunch", ceiling["meaning"])
        sizes = [prompt_tokens(prompt_of(body)) for body in stub.seen]
        self.assertLessEqual(max(sizes), 20000)
        self.assertEqual(tally.status(), "pass")

    def test_the_top_rung_runs_near_the_window_edge(self):
        stub = SseStub(window=20000, report_window=True)
        _, _, facts = self.climb(stub, (500, 8000, 65536))
        landed = facts["ceiling"]["deepest_prompt_tokens_ok"]
        self.assertGreater(landed, 17000)
        self.assertLessEqual(landed, 20000)

    def test_ignore_window_sends_every_rung_anyway(self):
        stub = SseStub(window=20000, report_window=True)
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, facts = self.run_stub(stub, depths=(500, 8000, 65536), trials=1,
                                        tasks=("prose",), ignore_window=True)
        self.assertEqual(facts["ceiling"]["limit"], "rejected")
        self.assertEqual(facts["ceiling"]["skipped_beyond_window"], [])

    def test_filler_past_one_corpus_pass_is_built_by_extra_passes(self):
        _, rows, _ = self.climb(SseStub(), (60000,))
        self.assertTrue(rows)
        self.assertTrue(all(item["valid"] for item in rows), rows)


class RunTrialResizeTest(unittest.TestCase):
    """A prompt that lands off its depth is resized once and the fix is kept."""

    ACTUAL_CHARS_PER_TOKEN = 3.0
    WRAPPER = 50

    def fake_stream(self, calls: list[int]):
        def stream(cfg, messages, **kwargs):
            prompt = messages[0]["content"]
            calls.append(len(prompt))
            counted = self.WRAPPER + int(len(prompt) / self.ACTUAL_CHARS_PER_TOKEN)
            return timing(prompt=counted)
        return stream

    def send(self, calibration: Calibration, trials: int = 2) -> tuple[list[dict], list[int]]:
        calls: list[int] = []
        rows = []
        args = LegArgs(model="m", cap=CAP)
        with mock.patch("scripts.ctxspeed_bench.stream_chat", self.fake_stream(calls)), \
                mock.patch("scripts.ctxspeed_bench.build_prompt",
                           lambda context, task: "x" * self.WRAPPER * 3 + context):
            for trial in range(trials):
                row, _ = run_trial(None, args, CORPUS, calibration, "prose", 4000, trial)
                rows.append(row)
        return rows, calls

    def test_a_prompt_that_lands_off_its_depth_is_resized_and_sent_again(self):
        calibration = Calibration(chars_per_token=4.0, overhead={"prose": self.WRAPPER})
        rows, calls = self.send(calibration, trials=1)
        self.assertEqual(len(calls), 2)
        self.assertTrue(rows[0]["valid"])
        self.assertLess(abs(rows[0]["depth_error_pct"]), 5)

    def test_the_resize_is_kept_for_the_depths_later_trials(self):
        calibration = Calibration(chars_per_token=4.0, overhead={"prose": self.WRAPPER})
        rows, calls = self.send(calibration, trials=2)
        self.assertEqual(len(calls), 3)
        self.assertIn(4000, calibration.scale)
        self.assertTrue(all(row["valid"] for row in rows))

    def test_a_prompt_that_lands_on_its_depth_is_sent_once(self):
        calibration = Calibration(chars_per_token=3.0, overhead={"prose": self.WRAPPER})
        _, calls = self.send(calibration, trials=1)
        self.assertEqual(len(calls), 1)

    def test_a_prompt_the_server_keeps_clipping_is_still_short_after_the_resize(self):
        calibration = Calibration(chars_per_token=3.0, overhead={"prose": self.WRAPPER})
        clip = lambda cfg, messages, **kwargs: timing(prompt=1000)
        with mock.patch("scripts.ctxspeed_bench.stream_chat", clip):
            row, _ = run_trial(None, LegArgs(model="m", cap=CAP), CORPUS, calibration,
                               "prose", 4000, 0)
        self.assertTrue(clipped(row))


class PlanClimbTest(unittest.TestCase):
    RUNGS = [512, 8192, 131072, 262144, 524288, 1048576]

    def test_with_no_window_every_rung_is_sent(self):
        self.assertEqual(plan_climb(self.RUNGS, None, 2000), (self.RUNGS, []))

    def test_rungs_past_the_window_are_held_back(self):
        sent, beyond = plan_climb(self.RUNGS, 245760, 2000)
        self.assertEqual(beyond, [262144, 524288, 1048576])
        self.assertEqual(sent[:3], [512, 8192, 131072])

    def test_a_top_rung_lands_at_the_usable_edge_of_the_window(self):
        sent, _ = plan_climb(self.RUNGS, 245760, 2000)
        self.assertEqual(sent[-1], int((245760 - 2000) * WINDOW_SLACK))
        self.assertEqual(sent, sorted(sent))

    def test_a_window_past_the_top_rung_holds_nothing_back_and_adds_no_rung(self):
        self.assertEqual(plan_climb(self.RUNGS, 2_000_000, 2000), (self.RUNGS, []))

    def test_a_window_smaller_than_the_reserve_sends_nothing(self):
        sent, beyond = plan_climb(self.RUNGS, 1000, 2000)
        self.assertEqual(sent, [])
        self.assertEqual(beyond, self.RUNGS)


class FailureKindTest(unittest.TestCase):
    def test_a_client_error_is_a_refusal(self):
        self.assertEqual(failure_kind(ChatResponse(status=400, error="HTTP 400")), "rejected")

    def test_a_timeout_is_a_hardware_stop(self):
        self.assertEqual(failure_kind(ChatResponse(error="timed out")), "timeout")

    def test_no_status_and_no_timeout_is_a_server_failure(self):
        self.assertEqual(failure_kind(ChatResponse(error="connection reset")), "server_failure")
        self.assertEqual(failure_kind(ChatResponse(status=503, error="HTTP 503")), "server_failure")


class ClippedTest(unittest.TestCase):
    def test_an_undershoot_past_the_tolerance_is_a_clip(self):
        self.assertTrue(clipped({"depth_error_pct": -12.0}))

    def test_drift_inside_the_tolerance_is_not(self):
        self.assertFalse(clipped({"depth_error_pct": -4.0}))
        self.assertFalse(clipped({"depth_error_pct": 30.0}))

    def test_an_unknown_error_is_not(self):
        self.assertFalse(clipped({"depth_error_pct": None}))


class CeilingBlockTest(unittest.TestCase):
    def test_the_block_carries_the_failure_and_the_deepest_landed_prompt(self):
        rows = [{"valid": True, "prompt_tokens": 9000}, {"valid": False, "prompt_tokens": 99000}]
        block = ceiling_block(rows, 9000, Failure(20000, "rejected", "HTTP 400", 400))
        self.assertEqual((block["deepest_depth_ok"], block["deepest_prompt_tokens_ok"],
                          block["first_failed_depth"], block["limit"]),
                         (9000, 9000, 20000, "rejected"))


class MainTest(unittest.TestCase):
    def run_main(self, stub: SseStub, extra: list[str]) -> tuple[int, dict, str]:
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--model", "m", "--endpoint", stub.endpoint, "--output-dir", tmp,
                    "--cap", str(CAP), "--timeout", "30", *extra]
            out = io.StringIO()
            with stub, contextlib.redirect_stdout(out):
                code = main(argv)
            path = os.path.join(tmp, "ctxspeed.result.json")
            block = {}
            if os.path.exists(path):
                with open(path) as handle:
                    block = json.load(handle)
        return code, block, out.getvalue()

    def test_a_clean_run_writes_the_result_block_the_orchestrator_reads(self):
        code, block, out = self.run_main(
            SseStub(token_s=0.001), ["--ctx", "65536", "--depths", "500,2000", "--trials", "1"])
        self.assertEqual(code, 0)
        self.assertEqual((block["leg"], block["status"], block["items"]), ("ctxspeed", "pass", 4))
        self.assertEqual(len(block["per_depth"]), 4)
        self.assertEqual(len(block["per_trial"]), 4)
        self.assertGreater(block["tps"], 0)
        self.assertRegex(block["items_sha"], r"^[0-9a-f]{16}$")
        self.assertIn("retention_at_deepest", block)
        self.assertIn("@@RESULT ", out)

    def test_a_run_with_an_invalid_trial_is_partial_not_pass(self):
        code, block, _ = self.run_main(
            SseStub(), ["--ctx", "65536", "--depths", "500", "--trials", "1", "--cap", "40"])
        self.assertEqual(block["status"], "partial")
        self.assertEqual(code, 1)

    def test_a_dead_endpoint_fails_instead_of_passing_empty(self):
        code, block, _ = self.run_main(
            SseStub(status=500), ["--ctx", "65536", "--depths", "500", "--trials", "1"])
        self.assertEqual(code, 1)
        self.assertIn(block["status"], ("fail", "partial"))
        self.assertEqual(block["tps"], 0.0)

    def test_no_window_flag_is_needed_the_leg_finds_the_window_itself(self):
        code, block, _ = self.run_main(
            SseStub(window=20000),
            ["--depths", "500,8000,65536", "--trials", "1", "--tasks", "prose"])
        self.assertEqual(code, 0)
        self.assertEqual(block["ceiling"]["limit"], "rejected")
        self.assertLessEqual(block["ceiling"]["deepest_prompt_tokens_ok"], 20000)


if __name__ == "__main__":
    unittest.main()
