#!/usr/bin/env python3
"""MATH-500: symbolic math reasoning on the shared leg contract.

Reads problems from the items file (default data/math500.json), sends each to
the endpoint through ask(), grades with sympy equivalence against the reference
answer, and writes one result block for the orchestrator.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import SympifyError, simplify  # noqa: E402

from scripts.lib.harness import (  # noqa: E402
    LegArgs,
    Tally,
    ask,
    case_record,
    finish,
    leg_config,
    load_case,
    load_items,
    parse_leg_args,
    replay_response,
    write_case,
)

SYSTEM_PROMPT = ("Answer directly and concisely. No preamble. No meta-commentary. "
                 "No sign-off. Stay on topic; stop when the answer is complete.")
LEG_ID = "math500"


# ---------- grading ----------


def _scan_braces(text: str, brace: int) -> int | None:
    """The index just past the matching close brace, or None when unbalanced."""
    depth = 1
    i = brace + 1
    while depth > 0 and i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return i if depth == 0 else None


def _extract_boxed(text: str) -> str | None:
    """Extract the content inside the LAST \\boxed{}."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    brace = text.find("{", idx)
    if brace == -1:
        return None
    end = _scan_braces(text, brace)
    return text[brace + 1 : end - 1] if end else None


def _extract_answer(text: str) -> str:
    """Try to extract a boxed answer, else return the cleaned text."""
    boxed = _extract_boxed(text)
    if boxed:
        return boxed.strip()
    text = text.strip()
    # Last line heuristics
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        last = lines[-1]
        for prefix in ["answer is", "answer:", "result:", "= "]:
            if prefix in last.lower():
                return last.split(prefix)[-1].strip()
        return last
    return text


def _grade_answer(gold: str, pred: str) -> bool:
    """Compare gold and predicted answers using sympy simplification."""
    gold = _extract_answer(gold.strip())
    pred = _extract_answer(pred.strip())

    if not pred or not gold:
        return False
    if pred.lower() == gold.lower():
        return True
    try:
        return bool(simplify(gold) == simplify(pred))
    except SympifyError:
        return gold.strip().lower() == pred.strip().lower()


# ---------- main ----------


def run_math500(args: LegArgs) -> Tally:
    """One ask per item; grade what comes back, checkpoint it and tally it."""
    cfg = leg_config(args)
    items = load_items(args.data, args.limit)
    tally = Tally()
    for index, item in enumerate(items):
        case_id = f"{index:05d}"
        saved = load_case(args, LEG_ID, case_id)
        if saved is not None:
            tally.record(replay_response(saved), correct=saved.get("correct"))
            print(f"  [{tally.items}/{len(items)}] reused ({case_id})", flush=True)
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["problem"]},
        ]
        response = ask(cfg, messages, args)
        passed: bool | None = None
        if response.ok:
            passed = _grade_answer(item["answer"], response.answer() or "")
            mark = "\u2713" if passed else "\u2717"
        else:
            mark = "!"
        tally.record(response, correct=passed)
        write_case(args, LEG_ID, case_id, case_record(case_id, response, correct=passed))
        print(f"  [{tally.items}/{len(items)}] {mark} ({response.elapsed_s:.1f}s)",
              flush=True)
    return tally


def main(argv: list[str] | None = None) -> None:
    args = parse_leg_args("MATH-500 benchmark", default_data="data/math500.json",
                          argv=argv)
    print(f"MATH-500 [{args.model}] cap {args.cap} thinking {args.thinking}",
          flush=True)
    finish(LEG_ID, args, run_math500(args))


if __name__ == "__main__":
    main()
