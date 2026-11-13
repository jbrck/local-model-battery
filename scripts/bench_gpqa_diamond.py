#!/usr/bin/env python3
"""GPQA-Diamond: graduate-level science questions, graded by answer letter."""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.harness import (  # noqa: E402
    LegArgs, Tally, ask, case_record, finish, is_cap_hit,
    leg_config, load_case, load_items, parse_leg_args,
    replay_response, write_case)
from scripts.lib.model import ChatResponse  # noqa: E402

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


def _replay_saved(saved: dict, index: int, total: int, rows: list[dict],
                  tally: Tally) -> None:
    """Count a checkpointed case again, without re-asking the model."""
    row = saved["row"]
    rows.append(row)
    tally.record(replay_response(saved), correct=saved.get("correct"))
    print(f"  [{index}/{total}] reused (gold={row['gold']}, "
          f"got={row['predicted'] or '?'})", flush=True)


def _fresh_case(cfg, args: LegArgs, item: dict, case_id: str, index: int, total: int,
                rows: list[dict], tally: Tally) -> None:
    """Ask one item, grade its letter, checkpoint the case and print its line."""
    gold = extract_letter(item["answer"])
    response = ask(cfg, [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": item["question"]},
    ], args)
    correct, predicted, channel = grade(response, gold)
    tally.record(response, correct=correct)
    row = {
        "id": index - 1,
        "gold": gold,
        "predicted": predicted,
        "passed": correct,
        "channel": channel,
        "cap_hit": response.ok and is_cap_hit(response),
        "completion_tokens": response.completion_tokens,
    }
    rows.append(row)
    write_case(args, LEG_ID, case_id,
               case_record(case_id, response, correct=correct, row=row))
    mark = "?" if correct is None else ("PASS" if correct else "FAIL")
    print(f"  [{index}/{total}] {mark} (gold={gold}, got={predicted or '?'})"
          f" ({response.elapsed_s:.1f}s)", flush=True)


def _cap_note(tally: Tally) -> None:
    """Cap hits change the letters, so the tally has to say so."""
    if tally.cap_hits:
        tally.notes.append(
            f"{tally.cap_hits} answers stopped at the cap and their letters come "
            "from truncated text")


def run_leg(args: LegArgs) -> tuple[Tally, list[dict]]:
    """Ask every item, checkpoint each case and tally the letter matches."""
    tally = Tally()
    rows: list[dict] = []
    cfg = leg_config(args)
    items = load_items(args.data, args.limit)
    for index, item in enumerate(items, start=1):
        case_id = f"{index - 1:05d}"
        saved = load_case(args, LEG_ID, case_id)
        if saved is not None:
            _replay_saved(saved, index, len(items), rows, tally)
            continue
        _fresh_case(cfg, args, item, case_id, index, len(items), rows, tally)
    _cap_note(tally)
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
