#!/usr/bin/env python3
"""Instruction-following battery v2: one mechanical constraint per item.

20 items in data/instr_v2_tasks.json, each naming a checker; grading is a
table lookup over the exact per-constraint rules this leg has always used.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.lib import harness  # noqa: E402
from scripts.lib.harness import (  # noqa: E402
    LegArgs,
    Tally,
    case_record,
    load_case,
    replay_response,
    write_case,
)

LEG_ID = "instr_v2"
DEFAULT_DATA = "data/instr_v2_tasks.json"
JSON_KEYS = ("city", "temp_c", "conditions")
SENTENCE_SPLIT = re.compile(r"[.!?]+")


def _exact_banana(text: str) -> bool:
    return text == "BANANA"


def _one_word(text: str) -> bool:
    return len(text.split()) == 1


def _two_words(text: str) -> bool:
    return len(text.split()) == 2


def _three_sentences(text: str) -> bool:
    sentences = [s for s in SENTENCE_SPLIT.split(text) if s.strip()]
    return len(sentences) == 3


def _five_bullets(text: str) -> bool:
    bullets = [line for line in text.split("\n") if line.strip().startswith("-")]
    return len(bullets) == 5


def _seven_items(text: str) -> bool:
    items = [item.strip() for item in text.split(",")]
    return len(items) == 7


def _all_caps(text: str) -> bool:
    return text == text.upper()


def _starts_with(text: str) -> bool:
    return text.lower().startswith("although")


def _ends_with(text: str) -> bool:
    return text.lower().rstrip(".").endswith("silence")


def _lowercase(text: str) -> bool:
    return text == text.lower() and "the quick brown fox" in text.lower()


def _no_e(text: str) -> bool:
    return "e" not in text.lower()


def _no_commas(text: str) -> bool:
    return "," not in text


def _max20(text: str) -> bool:
    return len(text.split()) <= 20


def _min15(text: str) -> bool:
    return len(text.split()) >= 15


def _json(text: str) -> bool:
    try:
        obj = json.loads(text)
        return all(key in obj for key in JSON_KEYS)
    except (json.JSONDecodeError, TypeError):
        return False


def _three_lines(text: str) -> bool:
    lines = [line for line in text.split("\n") if line.strip()]
    return len(lines) == 3


def _no_digits(text: str) -> bool:
    return not any(char.isdigit() for char in text)


def _exact_phrase(text: str) -> bool:
    return text.rstrip(".") == "The eagle lands at dawn"


def _ends_period(text: str) -> bool:
    return text.rstrip().endswith(".")


def _one_line_three(text: str) -> bool:
    words = text.split()
    return len(words) == 3 and not text.startswith("-") and "\n" not in text


CHECKERS = {
    "exact:banana": _exact_banana,
    "one_word": _one_word,
    "two_words": _two_words,
    "three_sentences": _three_sentences,
    "five_bullets": _five_bullets,
    "seven_items": _seven_items,
    "all_caps": _all_caps,
    "starts_with": _starts_with,
    "ends_with": _ends_with,
    "lowercase": _lowercase,
    "no_e": _no_e,
    "no_commas": _no_commas,
    "max20": _max20,
    "min15": _min15,
    "json": _json,
    "three_lines": _three_lines,
    "no_digits": _no_digits,
    "exact_phrase": _exact_phrase,
    "ends_period": _ends_period,
    "one_line_three": _one_line_three,
}


def check(checker: str, response: str) -> bool:
    """Grade one response against its named constraint. Unknown checkers fail."""
    rule = CHECKERS.get(checker)
    return bool(rule and rule(response.strip()))


def items_hash(items: list[dict]) -> str:
    """The comparability stamp over exactly these items."""
    canonical = json.dumps(items, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def run_instr(cfg, items: list[dict], args: LegArgs) -> Tally:
    """One ask per item, graded by the item's checker, checkpointed and counted."""
    tally = Tally()
    for index, item in enumerate(items):
        case_id = f"{index:05d}"
        saved = load_case(args, LEG_ID, case_id)
        if saved is not None:
            tally.record(replay_response(saved), correct=saved.get("correct"))
            print(f"  {item['id']:20s} reused", flush=True)
            continue
        messages = [{"role": "user", "content": item["prompt"]}]
        response = harness.ask(cfg, messages, args)
        if response.ok:
            passed = check(item["checker"], response.answer() or "")
            mark = "\u2713" if passed else "\u2717"
        else:
            passed = None
            mark = "!"
        tally.record(response, correct=passed)
        write_case(args, LEG_ID, case_id,
                   case_record(case_id, response, correct=passed, item=item["id"]))
        print(f"  {item['id']:20s} {mark}", flush=True)
    return tally


def main(argv: list[str] | None = None) -> None:
    args = harness.parse_leg_args(
        "Instruction-following v2: one mechanical constraint per item.",
        default_data=DEFAULT_DATA, argv=argv)
    items = harness.load_items(args.data, limit=args.limit)
    if not items:
        return
    cfg = harness.leg_config(args)
    tally = run_instr(cfg, items, args)
    harness.finish(LEG_ID, args, tally, extra={"items_sha": items_hash(items)})


if __name__ == "__main__":
    main()
