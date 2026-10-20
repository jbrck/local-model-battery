"""Suite schema validation for a writing battery suite file.

Checks structure, rubric shape, and the category-specific rules the suite was
authored under. Exits non-zero on any violation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

CRITERIA_MIN = 4
CRITERIA_MAX = 6
ANCHOR_LEVELS = ("2", "5", "8")
PROMPT_MIN_CHARS = 40

REQUIRED_CATEGORY_IDS = ("xpost", "xthread", "long_form", "revision",
                         "summary", "craft", "persuasion", "copy")

# Meta-words that must not appear in a task prompt: they leak the scoring frame.
LEAK_PATTERN = re.compile(r"\b(rubric|grading rubric|judge)\b", re.I)


def validate_suite(suite: dict) -> list[str]:
    """Return a list of problems. Empty list means the suite is valid."""
    problems: list[str] = []
    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return ["suite has no tasks array"]

    if suite.get("task_count") not in (None, len(tasks)):
        problems.append(
            f"task_count says {suite['task_count']}, tasks array has {len(tasks)}"
        )

    seen_ids: set[str] = set()
    counts: dict[str, int] = {}
    for task in tasks:
        problems.extend(_validate_task(task, seen_ids, counts))

    declared = suite.get("categories") or {}
    for category, expected in declared.items():
        actual = counts.get(category, 0)
        if actual != expected:
            problems.append(f"category {category}: {actual} tasks, header says {expected}")
    for category in REQUIRED_CATEGORY_IDS:
        if counts.get(category, 0) == 0:
            problems.append(f"category {category} has no tasks")
    return problems


def _validate_task(task: dict, seen_ids: set[str], counts: dict[str, int]) -> list[str]:
    problems: list[str] = []
    task_id = str(task.get("task_id", "?"))
    category = task.get("category", "")

    if not re.fullmatch(r"[A-Z]+-\d{3}", task_id):
        problems.append(f"{task_id}: bad task_id format")
    if task_id in seen_ids:
        problems.append(f"{task_id}: duplicate task_id")
    seen_ids.add(task_id)

    if category not in REQUIRED_CATEGORY_IDS:
        problems.append(f"{task_id}: unknown category {category!r}")
    counts[category] = counts.get(category, 0) + 1

    prompt = task.get("prompt") or ""
    if len(prompt.strip()) < PROMPT_MIN_CHARS:
        problems.append(f"{task_id}: prompt missing or too short")
    if LEAK_PATTERN.search(prompt):
        problems.append(f"{task_id}: prompt leaks a scoring word")

    contract = task.get("output_contract")
    if not isinstance(contract, dict):
        problems.append(f"{task_id}: missing output_contract")
    else:
        problems.extend(_validate_contract(task_id, category, contract))

    source_text = task.get("source_text")
    if source_text is not None and not isinstance(source_text, str):
        problems.append(f"{task_id}: source_text must be a string or null")
    elif category in ("revision", "summary") and not (source_text or "").strip():
        problems.append(f"{task_id}: {category} task needs source_text")

    rubric = task.get("rubric")
    if not isinstance(rubric, list):
        problems.append(f"{task_id}: missing rubric")
        return problems
    if not CRITERIA_MIN <= len(rubric) <= CRITERIA_MAX:
        problems.append(f"{task_id}: {len(rubric)} criteria, need {CRITERIA_MIN}-{CRITERIA_MAX}")
    for criterion in rubric:
        problems.extend(_validate_criterion(task_id, criterion))

    if category == "xpost" and rubric:
        if "hook" not in str(rubric[0].get("criterion", "")).lower():
            problems.append(f"{task_id}: first xpost criterion is not the hook")
    if category == "long_form" and not re.search(r"\d+\s*words?", prompt, re.I):
        problems.append(f"{task_id}: long_form prompt states no word count")
    if category == "xthread" and "---" not in prompt:
        problems.append(f"{task_id}: xthread prompt lacks '---' separator")
    return problems


def _validate_contract(task_id: str, category: str, contract: dict) -> list[str]:
    problems: list[str] = []
    for key in ("min_words", "max_words"):
        value = contract.get(key)
        if not isinstance(value, int) or value < 0:
            problems.append(f"{task_id}: output_contract.{key} must be a non-negative int")
    max_chars = contract.get("max_chars")
    if category == "copy":
        if not isinstance(max_chars, dict) or not max_chars:
            problems.append(f"{task_id}: copy contract needs per-field max_chars")
        elif not all(isinstance(v, int) and v > 0 for v in max_chars.values()):
            problems.append(f"{task_id}: copy max_chars values must be positive ints")
    elif max_chars is not None and (not isinstance(max_chars, int) or max_chars <= 0):
        problems.append(f"{task_id}: max_chars must be a positive int or null")
    return problems


def _validate_criterion(task_id: str, criterion: dict) -> list[str]:
    problems: list[str] = []
    name = criterion.get("criterion", "?")
    if not str(name).strip():
        problems.append(f"{task_id}: criterion with no name")
    if not str(criterion.get("description", "")).strip():
        problems.append(f"{task_id}: criterion {name!r} has no description")
    anchors = criterion.get("anchors") or {}
    for level in ANCHOR_LEVELS:
        if not str(anchors.get(level, "")).strip():
            problems.append(f"{task_id}: criterion {name!r} missing anchor {level}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a writing suite JSON file")
    parser.add_argument("suite", nargs="?", default="data/writing_suite_v1.json")
    args = parser.parse_args()

    with open(args.suite) as handle:
        suite = json.load(handle)

    problems = validate_suite(suite)
    if problems:
        print(f"FAIL: {len(problems)} problem(s) in {args.suite}")
        for problem in problems[:40]:
            print(f"  - {problem}")
        return 1

    counts = {}
    for task in suite["tasks"]:
        counts[task["category"]] = counts.get(task["category"], 0) + 1
    summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    print(f"PASS: {len(suite['tasks'])} tasks ({summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
