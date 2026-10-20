"""Join blind scores back to models and report the table.

Runs only after scoring is locked. Mechanical contract checks (word counts,
character caps) are counted here, not by the judge.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

DEFAULT_SCORES = ""
DEFAULT_KEY = "scoring/key.json"
DEFAULT_CORPUS_DIR = "scoring/corpus"
DEFAULT_SUITE = "data/writing_suite_v1.json"

THREAD_SEPARATOR = re.compile(r"\n\s*---\s*\n")
LABEL_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z ]{0,24}):\s*(.*)$")


def split_thread_posts(answer: str) -> list[str]:
    """Split an X thread answer into posts on the '---' separator."""
    return [post.strip() for post in THREAD_SEPARATOR.split(answer) if post.strip()]


def split_copy_fields(answer: str, field_names: list[str]) -> list[str] | None:
    """Split labelled ad copy into its fields. Returns None if unparseable."""
    labelled: list[str] = []
    for line in answer.splitlines():
        match = LABEL_LINE.match(line)
        if match and match.group(2).strip():
            labelled.append(match.group(2).strip())
    if len(labelled) >= len(field_names):
        return labelled[: len(field_names)]

    bare = [line.strip() for line in answer.splitlines() if line.strip()]
    bare = [line for line in bare if not LABEL_LINE.match(line)]
    if len(bare) >= len(field_names):
        return bare[: len(field_names)]
    return None


def check_compliance(task: dict, answer: str) -> dict:
    """Check the hard contract. Returns {'pass': True|False|None, 'detail': str}."""
    contract = task.get("output_contract") or {}
    category = task.get("category")
    text = answer.strip()
    if not text:
        return {"pass": False, "detail": "empty answer"}

    problems: list[str] = []
    max_chars = contract.get("max_chars")

    if isinstance(max_chars, int):
        if category == "xthread":
            posts = split_thread_posts(text)
            over = [index + 1 for index, post in enumerate(posts) if len(post) > max_chars]
            if over:
                problems.append(f"posts over {max_chars} chars: {over}")
        elif len(text) > max_chars:
            problems.append(f"{len(text)} chars > {max_chars}")
    elif isinstance(max_chars, dict):
        fields = split_copy_fields(text, list(max_chars.keys()))
        if fields is None:
            return {"pass": None, "detail": "could not parse copy fields"}
        for name, value, limit in zip(max_chars.keys(), fields, max_chars.values()):
            if len(value) > limit:
                problems.append(f"{name} {len(value)} chars > {limit}")

    words = len(text.split())
    if contract.get("min_words") and words < contract["min_words"]:
        problems.append(f"{words} words < {contract['min_words']}")
    if contract.get("max_words") and words > contract["max_words"]:
        problems.append(f"{words} words > {contract['max_words']}")

    if problems:
        return {"pass": False, "detail": "; ".join(problems)}
    return {"pass": True, "detail": "ok"}


def load_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def build_report(scores_path: str, key_path: str, corpus_dir: str, suite_path: str) -> dict:
    """Join scores to models and aggregate. Returns the report payload."""
    scored = load_json(scores_path)
    key = load_json(key_path)["entries"]
    suite = load_json(suite_path)
    tasks_by_id = {task["task_id"]: task for task in suite["tasks"]}

    per_model: dict[str, dict] = {}
    for opaque_id, score in scored["scores"].items():
        entry = key.get(opaque_id)
        if entry is None:
            raise SystemExit(f"score {opaque_id} has no key entry — wrong key file?")
        model = entry["model"]
        task = tasks_by_id[entry["task_id"]]

        corpus_path = os.path.join(corpus_dir, f"{opaque_id}.json")
        answer = ""
        if os.path.exists(corpus_path):
            answer = load_json(corpus_path).get("answer", "")
        compliance = check_compliance(task, answer)

        bucket = per_model.setdefault(model, {
            "answered": 0, "empty": 0, "judge_errors": 0,
            "sum_mean": 0.0, "scored": 0,
            "by_category": {}, "by_criterion": {},
            "compliance_pass": 0, "compliance_fail": 0, "compliance_unparsed": 0,
        })
        bucket["answered"] += 1
        bucket["empty"] += int(score.get("empty", False))
        bucket["judge_errors"] += int(bool(score.get("judge_error")))

        if compliance["pass"] is True:
            bucket["compliance_pass"] += 1
        elif compliance["pass"] is False:
            bucket["compliance_fail"] += 1
        else:
            bucket["compliance_unparsed"] += 1

        if score.get("mean") is None:
            continue
        bucket["scored"] += 1
        bucket["sum_mean"] += score["mean"]
        category = entry["category"]
        bucket["by_category"].setdefault(category, []).append(score["mean"])
        for criterion in score.get("scores", []):
            bucket["by_criterion"].setdefault(criterion["criterion"], []).append(
                criterion["score"])

    rows = []
    for model, bucket in per_model.items():
        total_checks = (bucket["compliance_pass"] + bucket["compliance_fail"])
        rows.append({
            "model": model,
            "mean_score": round(bucket["sum_mean"] / bucket["scored"], 2)
            if bucket["scored"] else None,
            "scored": bucket["scored"],
            "answered": bucket["answered"],
            "empty": bucket["empty"],
            "judge_errors": bucket["judge_errors"],
            "compliance_pass_pct": round(
                100 * bucket["compliance_pass"] / total_checks, 1) if total_checks else None,
            "compliance_unparsed": bucket["compliance_unparsed"],
            "by_category": {category: round(sum(values) / len(values), 2)
                            for category, values in sorted(bucket["by_category"].items())},
            "by_criterion": {name: round(sum(values) / len(values), 2)
                             for name, values in sorted(bucket["by_criterion"].items())},
        })
    rows.sort(key=lambda row: (row["mean_score"] is None, -(row["mean_score"] or 0)))

    return {
        "suite_version": scored.get("suite_version"),
        "judge_model": scored.get("judge_model"),
        "scored_at": scored.get("scored_at"),
        "models": rows,
    }


def render_markdown(report: dict) -> str:
    """Render the summary table as markdown."""
    lines = [
        f"# Writing suite v{report['suite_version']} — scores",
        "",
        f"Judge: `{report['judge_model']}` · scored {report['scored_at']} · blind "
        f"(opaque IDs, no model names in judge input)",
        "",
        "| Model | Mean | Scored | Empty | Compliance | Judge errors |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["models"]:
        mean = f"{row['mean_score']:.2f}" if row["mean_score"] is not None else "—"
        compliance = "—" if row["compliance_pass_pct"] is None else \
            f"{row['compliance_pass_pct']:.0f}%"
        lines.append(
            f"| {row['model']} | {mean} | {row['scored']}/{row['answered']} | "
            f"{row['empty']} | {compliance} | {row['judge_errors']} |"
        )

    lines.extend(["", "## Mean by category", ""])
    categories = sorted({category for row in report["models"]
                         for category in row["by_category"]})
    lines.append("| Model | " + " | ".join(categories) + " |")
    lines.append("|---" * (len(categories) + 1) + "|")
    for row in report["models"]:
        cells = [str(row["by_category"].get(category, "—")) for category in categories]
        lines.append(f"| {row['model']} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "Scores are 1-10 per rubric criterion against anchored descriptions. "
        "Compliance is counted mechanically from the answer text, not scored by the "
        "judge. Empty answers take the floor score.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Join blind scores to models")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--out", default=None, help="Markdown output path")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of markdown")
    args = parser.parse_args()

    report = build_report(args.scores, args.key, args.corpus_dir, args.suite)
    markdown = render_markdown(report)

    if args.out:
        with open(args.out, "w") as handle:
            handle.write(markdown)
        print(f"Report written to {args.out}")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
