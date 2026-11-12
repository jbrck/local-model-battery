#!/usr/bin/env python3
"""GPQA-Diamond: graduate-level science questions, graded by answer letter."""
from __future__ import annotations

import re
import sys

from scripts.lib.harness import (LegArgs, Tally, ask, finish, is_cap_hit,
                                 leg_config, load_items, parse_leg_args)
from scripts.lib.model import ChatResponse

LEG_ID = "gpqa_diamond"
DEFAULT_DATA = "data/gpqa_diamond.json"
SYSTEM = ("Answer the multiple-choice question. Reply with ONLY the letter of "
          "the correct answer (A, B, C, or D). No explanation.")
LETTER_PATTERN = r"\b([ABCD])\b"


def extract_letter(text: str) -> str:
    """The first standalone A-D letter after v1 normalization, or '?'."""
    normalized = text.strip().upper().replace(".", "")
    match = re.search(LETTER_PATTERN, normalized)
    return match.group(1) if match else "?"


def grade(response: ChatResponse, gold: str) -> tuple[bool | None, str, str]:
    """Score one reply: (correct, predicted letter, channel).

    A reply that only thinks, with empty content, is empty, not wrong.
    """
    text = response.content
    if not text:
        return None, "", "empty"
    predicted = extract_letter(text)
    return predicted == gold, predicted, "content"


def run_leg(args: LegArgs) -> tuple[Tally, list[dict]]:
    """Ask every item and tally the letter matches."""
    tally = Tally()
    rows: list[dict] = []
    cfg = leg_config(args)
    items = load_items(args.data, args.limit)
    for index, item in enumerate(items, start=1):
        gold = extract_letter(item["answer"])
        response = ask(cfg, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": item["question"]},
        ], args)
        correct, predicted, channel = grade(response, gold)
        tally.record(response, correct=correct)
        rows.append({
            "id": index - 1,
            "gold": gold,
            "predicted": predicted,
            "passed": correct,
            "channel": channel,
            "cap_hit": response.ok and is_cap_hit(response),
            "completion_tokens": response.completion_tokens,
        })
        mark = "?" if correct is None else ("PASS" if correct else "FAIL")
        print(f"  [{index}/{len(items)}] {mark} (gold={gold}, got={predicted or '?'})"
              f" ({response.elapsed_s:.1f}s)", flush=True)
    if tally.cap_hits:
        tally.notes.append(
            f"{tally.cap_hits} answers stopped at the cap and their letters come "
            "from truncated text")
    return tally, rows


def totals(rows: list[dict]) -> dict:
    """Leg-wide counts the orchestrator can compare across runs."""
    return {
        "empty_answers": sum(1 for row in rows if row["channel"] == "empty"),
        "cap_hits": sum(1 for row in rows if row["cap_hit"]),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_leg_args("GPQA-Diamond: graduate-level science QA", DEFAULT_DATA, argv)
    print(f"GPQA-Diamond [{args.model}]", flush=True)
    tally, rows = run_leg(args)
    finish(LEG_ID, args, tally, {"per_item": rows, **totals(rows)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
