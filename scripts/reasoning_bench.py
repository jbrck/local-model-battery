#!/usr/bin/env python3
"""Reasoning bench: whether the model emits a trace, and how long it is."""
from __future__ import annotations

import sys

from scripts.lib.harness import (LegArgs, Tally, ask, finish, leg_config, load_items,
                                 parse_leg_args)
from scripts.lib.model import ChatResponse

LEG_ID = "reasoning"
DEFAULT_DATA = "data/reasoning_tasks.json"


def answer_channel(response: ChatResponse) -> str:
    """Which channel carried the answer: content, reasoning_content, or none."""
    if response.content:
        return "content"
    if response.reasoning:
        return "reasoning_content"
    return "none"


def trace_stats(response: ChatResponse) -> tuple[bool, int]:
    """(trace present, trace length in characters)."""
    trace = response.reasoning
    return bool(trace), len(trace)


def run_leg(args: LegArgs) -> tuple[Tally, list[dict]]:
    """Ask every item with thinking on and record trace presence and length."""
    tally = Tally()
    rows: list[dict] = []
    cfg = leg_config(args)
    items = load_items(args.data, args.limit)
    for index, item in enumerate(items, start=1):
        response = ask(cfg, [{"role": "user", "content": item["prompt"]}], args)
        has_trace, trace_len = trace_stats(response)
        channel = answer_channel(response)
        tally.record(response, correct=channel != "none")
        rows.append({
            "id": item["id"],
            "trace": has_trace,
            "reasoning_len": trace_len,
            "channel": channel,
            "completion_tokens": response.completion_tokens,
        })
        print(f"  [{index}/{len(items)}] {item['id']}: {channel}, trace {trace_len}",
              flush=True)
    return tally, rows


def totals(rows: list[dict]) -> dict:
    """Leg-wide counts over the per-item rows."""
    with_trace = 0
    reasoning_len = 0
    channels = {"content": 0, "reasoning_content": 0, "none": 0}
    for row in rows:
        if row["trace"]:
            with_trace += 1
        reasoning_len += row["reasoning_len"]
        channels[row["channel"]] += 1
    return {
        "with_trace": with_trace,
        "total_reasoning_len": reasoning_len,
        "answered_content": channels["content"],
        "answered_reasoning": channels["reasoning_content"],
        "answered_none": channels["none"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_leg_args("Reasoning bench: trace presence and trace length",
                          DEFAULT_DATA, argv)
    if args.thinking != "on":
        print("the reasoning leg measures thinking and is meaningless with --thinking off",
              file=sys.stderr)
        return 2
    tally, rows = run_leg(args)
    finish(LEG_ID, args, tally, {"per_item": rows, **totals(rows)})
    return 0


if __name__ == "__main__":
    sys.exit(main())