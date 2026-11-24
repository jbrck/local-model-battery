#!/usr/bin/env python3
"""Context-depth decode bench: tokens per second as the prompt grows, prose and copy.

Every other leg measures speed averaged over short prompts. This one holds the request
shape fixed and moves only the depth: how many tokens are already in context when decode
starts. Each rung sends real, non-repeating filler of that depth, then asks for a fixed
window of output, and times the stream so prefill and decode are reported apart.

Two workloads, because a speculative drafter behaves differently on each:
    prose   free writing unrelated to the filler. The hard case for a drafter.
    copy    a verbatim copy of the end of the filler. The easy case: the answer is in context.
`copy_ratio` per trial says how much of the output was really lifted from the context, so
a model that ignores the copy instruction shows up as a prose-like row instead of passing.

One run is one engine configuration. Drafter on versus off is two runs on two server
configurations, compared by their `per_depth` tables: a leg cannot restart the server.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.corpus import (  # noqa: E402
    build_context, corpus_chars, corpus_hash, load_corpus)
from scripts.lib.harness import (  # noqa: E402
    LegArgs, Tally, add_leg_args, finish, leg_config, thinking_body)
from scripts.lib.stream import StreamTiming, stream_chat  # noqa: E402

LEG_ID = "ctxspeed"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_RUNGS = (512, 8192, 32768, 65536, 131072, 200000)
DEFAULT_TASKS = ("prose", "copy")
DEFAULT_TRIALS = 2
# The top rung is this share of the room left after the decode window, because the
# characters-per-token estimate is good to a few percent and the server rejects overflow.
TOP_RUNG_SHARE = 0.9
TOP_RUNG_STEP = 1024
# A requested depth must still fit with this much slack for the estimate's error.
FIT_SLACK = 1.05
# A trial counts when the prompt landed within this share of its target depth.
DEPTH_TOLERANCE = 0.10
# Fewer output tokens than this is too short a window to call a rate.
MIN_DECODE_TOKENS = 128
PROBE_CHARS = 16000
# A streamed request sends its first byte only after prefill, so the socket timeout is the
# time to first token. The per-case timeout the planner derives is sized for decode and
# would clip a deep prefill on a slow host, which is a hidden cap by another name.
PREFILL_TIMEOUT_FLOOR_S = 3600
FALLBACK_CHARS_PER_TOKEN = 3.5
WARMUP_TOKENS = 64
COPY_GRAM = 6
SEED_STRIDE_TRIAL = 101
SEED_STRIDE_TASK = 7

TASKS = {
    "prose": ("The reference text above is unrelated to this request. Ignore it. Write a long, "
              "continuous essay in plain prose, with no lists and no headings, on how the "
              "printing press changed the way knowledge spread through Europe. Keep writing."),
    "copy": ("Reproduce the final 400 words of the reference text above, word for word. "
             "Output only the copied text, with no commentary."),
}
TASKS_ORDER = tuple(TASKS)


@dataclass
class Options:
    """The flags this leg adds to the shared set."""

    depths: tuple[int, ...] = ()
    trials: int = DEFAULT_TRIALS
    tasks: tuple[str, ...] = DEFAULT_TASKS


def split_csv(text: str) -> list[str]:
    """The non-empty, stripped items of a comma-separated flag."""
    return [item.strip() for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    """The shared leg flags plus --depths, --trials and --tasks."""
    parser = argparse.ArgumentParser(description="Decode speed against context depth")
    add_leg_args(parser)
    parser.add_argument("--depths", default="",
                        help="comma-separated prompt depths in tokens; default is a ladder "
                             "up to the served window")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help="requests per depth and task, each on different filler")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                        help="comma-separated: prose, copy")
    return parser


def parse_args(argv: list[str] | None = None) -> tuple[LegArgs, Options]:
    """Split the parsed flags into the shared leg arguments and this leg's options."""
    parser = build_parser()
    values = vars(parser.parse_args(argv))
    options = Options(depths=tuple(int(item) for item in split_csv(values.pop("depths"))),
                      trials=max(1, values.pop("trials")),
                      tasks=tuple(split_csv(values.pop("tasks"))))
    unknown = [name for name in options.tasks if name not in TASKS]
    if unknown:
        parser.error(f"unknown task {unknown}; choose from {sorted(TASKS)}")
    return LegArgs(**values), options


def requested_ladder(room: int, requested: tuple[int, ...]) -> tuple[list[int], list[int]]:
    """(asked depths that fit, asked depths that cannot), each ascending."""
    asked = sorted(set(requested))
    fits = [depth for depth in asked if depth * FIT_SLACK <= room]
    return fits, [depth for depth in asked if depth not in fits]


def default_ladder(room: int) -> list[int]:
    """The standard rungs that fit, plus a top rung just under the window when there is space."""
    top = int(room * TOP_RUNG_SHARE) // TOP_RUNG_STEP * TOP_RUNG_STEP
    rungs = [depth for depth in DEFAULT_RUNGS if depth <= top]
    if rungs and top >= rungs[-1] * 1.25:
        rungs.append(top)
    return rungs


def ladder(ctx: int, decode_tokens: int, requested: tuple[int, ...]) -> tuple[list[int], list[int]]:
    """(depths to run, depths skipped because they cannot fit the served window)."""
    room = ctx - decode_tokens
    if requested:
        return requested_ladder(room, requested)
    return default_ladder(room), []


def build_prompt(context: str, task: str) -> str:
    """The user turn: the filler as reference text, then the task."""
    return f"Reference text:\n\n{context}\n\n---\n\n{TASKS[task]}"


def trial_seed(args: LegArgs, trial: int, task_index: int) -> int:
    """A different filler for every trial and task, so no request can reuse another's prefix."""
    return args.seed + trial * SEED_STRIDE_TRIAL + task_index * SEED_STRIDE_TASK


def copy_ratio(context: str, output: str, gram: int = COPY_GRAM) -> float:
    """The share of the output's word n-grams that also occur in the context."""
    words = output.split()
    if len(words) < gram:
        return 0.0
    reference = context.split()
    seen = {" ".join(reference[i:i + gram]) for i in range(len(reference) - gram + 1)}
    grams = [" ".join(words[i:i + gram]) for i in range(len(words) - gram + 1)]
    return sum(1 for item in grams if item in seen) / len(grams)


def probe_prompt_tokens(cfg, args: LegArgs, prompt: str) -> int:
    """Prompt tokens as the server counts them, from a one-token request."""
    timing = stream_chat(cfg, [{"role": "user", "content": prompt}], max_tokens=1,
                         temperature=args.temperature, extra_body=thinking_body(args.thinking))
    return timing.prompt_tokens


@dataclass
class Calibration:
    """How the server counts this filler, and what the prompt wrapper costs before any filler."""

    chars_per_token: float
    overhead: dict[str, int]
    note: str = ""


def calibrate(cfg, args: LegArgs, corpus: list[str], tasks: tuple[str, ...] = DEFAULT_TASKS
              ) -> Calibration:
    """Measure characters per token of the filler and the wrapper's fixed token cost per task.

    The wrapper (chat template plus task text) is the same size at every depth. At a shallow
    rung it is a visible share of the prompt, so the filler is sized to depth minus overhead.
    """
    overhead = {task: probe_prompt_tokens(cfg, args, build_prompt("", task)) for task in tasks}
    context = build_context(corpus, PROBE_CHARS, args.seed)
    full = probe_prompt_tokens(cfg, args, build_prompt(context, tasks[0]))
    tokens = full - overhead[tasks[0]]
    if overhead[tasks[0]] <= 0 or tokens <= 0:
        return Calibration(FALLBACK_CHARS_PER_TOKEN, {task: 0 for task in tasks},
                           "the server reports no prompt token count, so depth is an estimate "
                           "at 3.5 characters per token")
    return Calibration(len(context) / tokens, overhead)


def warm_up(cfg, args: LegArgs) -> None:
    """One short throwaway generation, so the first timed trial does not pay for cold kernels."""
    stream_chat(cfg, [{"role": "user", "content": build_prompt("", "prose")}],
                max_tokens=WARMUP_TOKENS, temperature=args.temperature,
                extra_body=thinking_body(args.thinking))


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def depth_error(target: int, prompt_tokens: int) -> float | None:
    """How far the prompt landed from its target depth, as a share. None when unknowable."""
    if prompt_tokens <= 0:
        return None
    return (prompt_tokens - target) / target


def trial_valid(timing: StreamTiming, error: float | None) -> bool:
    """A trial counts when it succeeded, filled the decode window and landed on its depth."""
    response = timing.response
    on_depth = error is None or abs(error) <= DEPTH_TOLERANCE
    return (response.ok and response.completion_tokens >= MIN_DECODE_TOKENS
            and timing.decode_tps > 0 and on_depth)


def server_rate(timing: StreamTiming) -> float:
    """The server's own decode rate when it reports one, kept to cross-check the measured rate."""
    return round(float(timing.server_timings.get("predicted_per_second") or 0.0), 1)


def trial_row(task: str, target: int, trial: int, prompt: str, context: str,
              timing: StreamTiming) -> dict:
    """Everything one trial measured, plus whether it counts toward the depth's median."""
    response = timing.response
    error = depth_error(target, timing.prompt_tokens)
    prefill_tps = timing.prompt_tokens / timing.ttft_s if timing.ttft_s > 0 else 0.0
    return {
        "task": task, "depth": target, "trial": trial, "valid": trial_valid(timing, error),
        "prompt_tokens": timing.prompt_tokens,
        "depth_error_pct": None if error is None else round(error * 100, 1),
        "completion_tokens": response.completion_tokens,
        "ttft_s": round(timing.ttft_s, 3), "decode_s": round(timing.decode_s, 3),
        "decode_tps": round(timing.decode_tps, 1), "prefill_tps": round(prefill_tps, 1),
        "server_decode_tps": server_rate(timing),
        "usage_reported": timing.usage_reported,
        "copy_ratio": round(copy_ratio(context, response.content or response.reasoning), 3),
        "finish": response.finish_reason, "prompt_sha16": sha16(prompt),
        "error": response.error,
    }


def count_trial(tally: Tally, timing: StreamTiming, valid: bool) -> None:
    """Count one trial. The decode window ends every trial at `length` by design, so that
    stop is not a cap hit: cap_hits keeps meaning the cap cut real work."""
    cap_hits = tally.cap_hits
    tally.record(timing.response, correct=valid)
    tally.cap_hits = cap_hits


def run_trial(cfg, args: LegArgs, corpus: list[str], calibration: Calibration,
              task: str, depth: int, trial: int) -> tuple[dict, StreamTiming]:
    """Build the filler for one rung, send it, and measure the stream."""
    seed = trial_seed(args, trial, TASKS_ORDER.index(task))
    filler_tokens = max(depth - calibration.overhead.get(task, 0), 0)
    context = build_context(corpus, int(filler_tokens * calibration.chars_per_token), seed)
    prompt = build_prompt(context, task)
    timing = stream_chat(cfg, [{"role": "user", "content": prompt}], max_tokens=args.cap,
                         temperature=args.temperature, extra_body=thinking_body(args.thinking))
    return trial_row(task, depth, trial, prompt, context, timing), timing


def run_leg(args: LegArgs, options: Options,
            corpus: list[str] | None = None) -> tuple[Tally, list[dict], dict]:
    """Walk the ladder shallow to deep. Returns the tally, the trial rows and the run facts."""
    cfg = leg_config(args)
    cfg.timeout_s = max(args.timeout_s, PREFILL_TIMEOUT_FLOOR_S)
    corpus = corpus if corpus is not None else load_corpus(ROOT)
    depths, skipped = ladder(args.ctx, args.cap, options.depths)
    facts: dict = {"depths": depths, "skipped_depths": skipped,
                   "corpus_chars": corpus_chars(corpus), "items_sha": corpus_hash(corpus)}
    tally, rows = Tally(), []
    if not depths:
        return tally, rows, facts
    warm_up(cfg, args)
    calibration = calibrate(cfg, args, corpus, options.tasks)
    facts["chars_per_token"] = round(calibration.chars_per_token, 3)
    facts["wrapper_tokens"] = calibration.overhead
    if calibration.note:
        tally.notes.append(calibration.note)
    for depth in depths:
        for trial in range(options.trials):
            for task in options.tasks:
                row, timing = run_trial(cfg, args, corpus, calibration, task, depth, trial)
                count_trial(tally, timing, row["valid"])
                rows.append(row)
                print(f"  {task:<5} depth {depth:>6} trial {trial + 1}: "
                      f"{row['decode_tps']:>6} tok/s decode, ttft {row['ttft_s']}s, "
                      f"prompt {row['prompt_tokens']}{'' if row['valid'] else '  (invalid)'}",
                      flush=True)
    return tally, rows, facts


def median(values: list[float]) -> float:
    return round(statistics.median(values), 1) if values else 0.0


MEDIAN_KEYS = ("ttft_s", "prefill_tps", "copy_ratio")


def retention(tps: float, baseline: float) -> float:
    """Rate over the task's baseline rate. A zero baseline makes the line its own baseline."""
    reference = baseline or tps
    return round(tps / reference, 3) if reference else 0.0


def depth_line(task: str, depth: int, group: list[dict], baseline: float) -> dict:
    """One table line: the medians over a depth's valid trials."""
    good = [item for item in group if item["valid"]]
    tps = median([item["decode_tps"] for item in good])
    medians = {key: median([item[key] for item in good]) for key in MEDIAN_KEYS}
    return {"task": task, "depth": depth, "trials": len(group), "valid": len(good),
            "decode_tps": tps, "retention": retention(tps, baseline),
            "prompt_tokens": int(median([item["prompt_tokens"] for item in good])), **medians}


def task_lines(task: str, rows: list[dict]) -> list[dict]:
    """The table lines of one task, shallow to deep. The first depth with a valid trial is the
    baseline every deeper line is measured against."""
    lines: list[dict] = []
    baseline = 0.0
    for depth in sorted({item["depth"] for item in rows}):
        group = [item for item in rows if item["depth"] == depth]
        line = depth_line(task, depth, group, baseline)
        baseline = baseline or line["decode_tps"]
        lines.append(line)
    return lines


def per_depth(rows: list[dict]) -> list[dict]:
    """One line per task and depth, medians over the valid trials, with retention against the
    task's shallowest valid depth so the curve reads as a ratio as well as a rate."""
    tasks = sorted({item["task"] for item in rows}, key=TASKS_ORDER.index)
    return [line for task in tasks
            for line in task_lines(task, [item for item in rows if item["task"] == task])]


def pooled_decode_tps(rows: list[dict]) -> float:
    """Decode tokens over decode seconds across every valid trial."""
    good = [row for row in rows if row["valid"]]
    seconds = sum(row["decode_s"] for row in good)
    return round(sum(row["completion_tokens"] - 1 for row in good) / seconds, 1) if seconds else 0.0


def deepest_retention(table: list[dict]) -> dict:
    """Per task: decode rate at the deepest valid rung over the shallowest valid rung."""
    out: dict = {}
    for task in {line["task"] for line in table}:
        lines = [line for line in table if line["task"] == task and line["valid"]]
        if lines:
            out[task] = {"depth": lines[-1]["depth"], "retention": lines[-1]["retention"]}
    return out


def main(argv: list[str] | None = None) -> int:
    args, options = parse_args(argv)
    if args.ctx <= 0 and not options.depths:
        print("--ctx is the served window and the ladder is built from it; pass --ctx or "
              "--depths", file=sys.stderr)
        return 2
    tally, rows, facts = run_leg(args, options)
    table = per_depth(rows)
    extra = {"per_trial": rows, "per_depth": table, **facts,
             "tps": pooled_decode_tps(rows), "decode_tps_pooled": pooled_decode_tps(rows),
             "retention_at_deepest": deepest_retention(table)}
    if tally.status() == "pass" and any(not row["valid"] for row in rows):
        extra["status"] = "partial"
    finish(LEG_ID, args, tally, extra)
    return 0 if any(row["valid"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
