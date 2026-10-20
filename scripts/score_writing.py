"""Phase C: score a blind corpus against the suite rubrics.

Reads only the corpus (opaque IDs + answers + task IDs) and the suite. It never
reads the key file, so model identities cannot influence scoring.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.lib.judge import score_answer, setup_judge  # noqa: E402
from scripts.lib.model import verify_endpoint  # noqa: E402

DEFAULT_CORPUS_DIR = "scoring/corpus"
DEFAULT_SUITE = "data/writing_suite_v1.json"


def load_corpus(corpus_dir: str) -> list[dict]:
    """Read every opaque-ID answer file in the corpus directory."""
    entries = []
    for filename in sorted(os.listdir(corpus_dir)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(corpus_dir, filename)) as handle:
            payload = json.load(handle)
        if "task_id" not in payload:
            raise SystemExit(f"{filename}: not a corpus file (no task_id)")
        entries.append({
            "opaque_id": filename[:-len(".json")],
            "task_id": payload["task_id"],
            "answer": payload.get("answer") or "",
        })
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Blind-score a writing corpus")
    parser.add_argument("--judge-model", required=True,
                        help="Judge model alias on the endpoint. No default.")
    parser.add_argument("--judge-endpoint", required=True,
                        help="OpenAI-compatible endpoint serving the judge")
    parser.add_argument("--judge-api-key", default="not-needed")
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--out", default=None, help="Scores output path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse scores from --out if present and resume")
    args = parser.parse_args()

    with open(args.suite) as handle:
        suite = json.load(handle)
    tasks_by_id = {task["task_id"]: task for task in suite["tasks"]}

    entries = load_corpus(args.corpus_dir)
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        print(f"No corpus files in {args.corpus_dir}.", file=sys.stderr)
        return 1

    unknown = [entry["task_id"] for entry in entries if entry["task_id"] not in tasks_by_id]
    if unknown:
        print(f"ERROR: corpus references tasks absent from the suite: {unknown[:5]}",
              file=sys.stderr)
        return 1

    judge = setup_judge(args.judge_model, args.judge_endpoint, args.judge_api_key)
    if not verify_endpoint(args.judge_endpoint, args.judge_model, args.judge_api_key):
        print(f"ERROR: judge endpoint {args.judge_endpoint} does not list "
              f"'{args.judge_model}'", file=sys.stderr)
        return 1

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.corpus_dir)),
        f"scores-{time.strftime('%Y-%m-%d-%H%M%S')}.json",
    )
    scores: dict[str, dict] = {}
    if args.skip_existing and os.path.exists(out_path):
        with open(out_path) as handle:
            scores = json.load(handle).get("scores", {})

    timer = time.time()
    errors = 0
    for index, entry in enumerate(entries, start=1):
        if entry["opaque_id"] in scores:
            continue
        task = tasks_by_id[entry["task_id"]]
        result = score_answer(judge, task, entry["answer"])
        scores[entry["opaque_id"]] = {"task_id": entry["task_id"], **result}
        if result["judge_error"]:
            errors += 1
            status = f"JUDGE ERROR ({result['judge_error'][:40]})"
        else:
            status = f"mean {result['mean']}"
        print(f"  [{index}/{len(entries)}] {entry['opaque_id']} "
              f"{entry['task_id']:8s} {status}")
        sys.stdout.flush()
        with open(out_path, "w") as handle:
            _write_scores(handle, suite, args.judge_model, scores)

    with open(out_path, "w") as handle:
        _write_scores(handle, suite, args.judge_model, scores)

    print(f"\nScored {len(scores)} answers, {errors} judge error(s), "
          f"{int(time.time() - timer)}s")
    print(f"Scores: {out_path}")
    print("Join scores to models: python scripts/report_scores.py --scores " + out_path)
    return 0 if errors == 0 else 2


def _write_scores(handle, suite: dict, judge_model: str, scores: dict) -> None:
    """Serialise the score file, including provenance for the audit trail."""
    json.dump({
        "suite_version": suite.get("suite_version"),
        "judge_model": judge_model,
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scoring": "blind: corpus carries opaque IDs only",
        "scores": scores,
    }, handle, indent=2)


if __name__ == "__main__":
    sys.exit(main())
