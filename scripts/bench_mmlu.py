#!/usr/bin/env python3
"""MMLU (generative): broad knowledge on a pinned 500-item sample of the test split.

Reads data/mmlu_test.json and asks one multiple-choice question at a time,
scoring the answer letter against the reference. The sample was generated once
from cais/mmlu config \"all\", split \"test\", sorted by item hash rather than by
re-shuffle at run time; rerunning this leg always sees the same items in the same
order because they are committed in the data file.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.lib import harness  # noqa: E402
from scripts.lib.model import ModelConfig  # noqa: E402

LEG_ID = "mmlu"
LETTERS = ("A", "B", "C", "D")
DEFAULT_DATA = "data/mmlu_test.json"

# The prompt format inherited from the reference box runs this leg was ported from.
PROMPT_TEMPLATE = (
    "The following is a multiple choice question about {subject}. "
    "Think step by step, then give your final answer.\n\n"
    "{question}\n{choices}\n\n"
    "End with a line exactly like: 'Answer: <letter>'"
)

# Explicit final-answer phrasings first, last match wins within each phrasing;
# then any standalone uppercase letter as the fallback for unhelpful prose.
ANSWER_PATTERNS = (
    re.compile(r"answer\s*[:\\\-]?\s*\(?([A-Da-d])\)?"),
    re.compile(r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}"),
    re.compile(r"final answer\s*(?:is|:)\s*\(?([A-Da-d])\)?"),
    re.compile(r"correct (?:option|choice|answer)\s*(?:is|:)\s*\(?([A-Da-d])\)?"),
)
STANDALONE_LETTER = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")


def build_prompt(item: dict) -> str:
    """Render one item as the prompt sent to the endpoint."""
    choices = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(item["choices"]))
    return PROMPT_TEMPLATE.format(subject=item["subject"],
                                  question=item["question"], choices=choices)


def parse_letter(text: str | None) -> str | None:
    """Extract the answer letter A-D from model output."""
    if not text:
        return None
    for pattern in ANSWER_PATTERNS:
        found = pattern.findall(text)
        if found:
            return found[-1].upper()
    found = STANDALONE_LETTER.findall(text)
    return found[-1].upper() if found else None


def items_hash(items: list[dict]) -> str:
    """The comparability stamp over exactly these items."""
    canonical = json.dumps(items, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def run_mmlu(cfg: ModelConfig, items: list[dict], args) -> harness.Tally:
    """Score every item once and report progress while resuming-friendly."""
    total = len(items)
    tally = harness.Tally()
    for index, item in enumerate(items):
        messages = [{"role": "user", "content": build_prompt(item)}]
        response = harness.ask(cfg, messages, args)
        gold = item["answer"]
        if response.ok:
            predicted = parse_letter(response.content)
            tally.record(response,
                         correct=(predicted == gold) if predicted else False)
            print(f"[{index + 1}/{total}] gold={gold} got={predicted} "
                  f"{tally.correct}/{tally.scored}", flush=True)
        else:
            tally.record(response)
            print(f"[{index + 1}/{total}] skipped ({response.error})",
                  flush=True)
    return tally


def main(argv: list[str] | None = None) -> None:
    """Run the leg: parse flags, score the items, write the result block."""
    args = harness.parse_leg_args(
        "MMLU (generative): broad knowledge on a pinned 500-item sample.",
        default_data=DEFAULT_DATA, argv=argv)
    cfg = harness.leg_config(args)
    items = harness.load_items(args.data, limit=args.limit)
    if not items:
        return
    tally = run_mmlu(cfg, items, args)
    harness.finish(LEG_ID, args, tally, extra={"items_sha": items_hash(items)})


if __name__ == "__main__":
    main()
