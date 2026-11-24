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

# What sympy can raise on text that is not an expression: its own wrapped error, the
# ValueError it re-raises from parse_expr, and the TypeError a bad call produces.
GRADER_ERRORS = (SympifyError, ValueError, TypeError, AttributeError, SyntaxError)

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


def _extract_delimited(text: str, open_delim: str,
                       close_delim: str) -> str | None:
    """The content inside the LAST pair of a math delimiter (\\\\(...\\\\) or $...$).

    The close delimiter is searched from the open onward so a stray open with no
    close returns None rather than reaching past it.
    """
    idx = text.rfind(open_delim)
    if idx == -1:
        return None
    end = text.find(close_delim, idx + len(open_delim))
    if end == -1:
        return None
    return text[idx + len(open_delim) : end]


def _latex_simplify(text: str) -> str:
    """Bring \"\\\"-escaped LaTeX math to a plain-text-ish form sympy can read.

    Applied to BOTH the gold and predicted sides, so a control-word that stays
    unresolved cancels out. Only the tokens that appear in model final answers
    are rewritten; everything unknown is left verbatim, and a \\frac{a}{b} that
    is not converted cannot be simplified, so both sides fall to the string
    compare rather than being mismatched by a partial conversion.
    """
    stage = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    stage = stage.replace("\\,", " ").replace("\\:", " ").replace("\\;", " ")
    stage = stage.replace("\\left(", "(").replace("\\right)", ")")
    stage = stage.replace("\\left[", "[").replace("\\right]", "]")
    stage = stage.replace("\\cdot", "*").replace("\\times", "*")
    stage = stage.replace("\\pi", "pi").replace("\\theta", "theta")
    stage = stage.replace("\\ ", "").replace("\\quad", " ").replace("\\qquad", " ")
    return stage


def _extract_answer(text: str) -> str:
    """Try to extract a boxed or inline-math answer, else return the cleaned text."""
    boxed = _extract_boxed(text)
    if boxed:
        boxed = _latex_simplify(boxed)
        return boxed.strip()
    # DeepSeek-style final answers are wrapped in \\(...\\) or $...$ (which the
    # \\boxed{} scanner never sees). Keep the raw inline math when its delimiters
    # are present, then let the trailing-text fallback handle prose around it.
    for open_delim, close_delim in (("\\(", "\\)"), ("$", "$")):
        delimited = _extract_delimited(text, open_delim, close_delim)
        if delimited is not None:
            return _latex_simplify(delimited).strip()
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
    gold = _latex_simplify(_extract_answer(gold.strip()))
    pred = _latex_simplify(_extract_answer(pred.strip()))

    if not pred or not gold:
        return False
    if pred.lower() == gold.lower():
        return True
    try:
        return bool(simplify(gold) == simplify(pred))
    except GRADER_ERRORS:
        # sympy raises more than SympifyError on text it cannot read as an expression.
        # A gold like "(a+5)(b+2)" parses as a call on an Add and raises ValueError, and
        # that escaped the old handler and killed the whole leg at case 103 of 500.
        # An answer the grader cannot parse is simply not equal; it never ends the leg.
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
