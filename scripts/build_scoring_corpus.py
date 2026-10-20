"""Phase C prep: turn per-model answer files into a blind scoring corpus.

Writes a flat directory of opaque-ID files (answer + task ID only) plus a key
file that lives outside that directory. The scoring step must never read the key.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import secrets
import sys

DEFAULT_ANSWERS_DIR = "answers"
DEFAULT_CORPUS_DIR = "scoring/corpus"
DEFAULT_KEY_PATH = "scoring/key.json"


def normalize_answer(text: str) -> str:
    """Strip whitespace and markdown fences. Prose habits stay; they are the answer."""
    stripped = text.strip()
    stripped = re.sub(r"^```[a-zA-Z]*\n", "", stripped)
    stripped = re.sub(r"\n```$", "", stripped)
    return stripped.strip()


def collect_answers(answers_dir: str, canonical_task_ids: set[str]) -> list[dict]:
    """Read every <answers_dir>/<model>/<task_id>.json into flat records."""
    records: list[dict] = []
    for entry in sorted(os.listdir(answers_dir)):
        model_dir = os.path.join(answers_dir, entry)
        if not os.path.isdir(model_dir):
            continue
        for filename in sorted(os.listdir(model_dir)):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(model_dir, filename)) as handle:
                record = json.load(handle)
            task_id = record.get("task_id")
            if task_id not in canonical_task_ids:
                raise SystemExit(f"{model_dir}/{filename}: task_id {task_id!r} not in suite")
            response = record.get("response")
            records.append({
                "model": entry,
                "task_id": task_id,
                "category": record.get("category"),
                "answer": normalize_answer(response) if isinstance(response, str) else "",
                "empty": response is None or not str(response).strip(),
            })
    return records


def build_corpus(records: list[dict], corpus_dir: str, key_path: str,
                 seed: int | None = None) -> dict:
    """Write the blind corpus and the key. Returns a summary dict."""
    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(key_path)), exist_ok=True)

    order = list(range(len(records)))
    random.Random(seed if seed is not None else secrets.randbits(32)).shuffle(order)

    key: dict[str, dict] = {}
    for position, index in enumerate(order):
        record = records[index]
        opaque_id = f"{position:04d}-{secrets.token_hex(6)}"
        with open(os.path.join(corpus_dir, f"{opaque_id}.json"), "w") as handle:
            json.dump({"task_id": record["task_id"], "answer": record["answer"]},
                      handle, indent=2)
        key[opaque_id] = {"model": record["model"], "task_id": record["task_id"],
                          "category": record["category"], "empty": record["empty"]}

    with open(key_path, "w") as handle:
        json.dump({"seed": seed, "entries": key}, handle, indent=2)

    return {
        "corpus_dir": corpus_dir,
        "key_path": key_path,
        "answers": len(key),
        "models": sorted({entry["model"] for entry in records}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the blind scoring corpus")
    parser.add_argument("--suite", default="data/writing_suite_v1.json")
    parser.add_argument("--answers-dir", default=DEFAULT_ANSWERS_DIR)
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--key", default=DEFAULT_KEY_PATH,
                        help="Key file path. Must sit outside --corpus-dir.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Shuffle seed, for an auditable re-run")
    args = parser.parse_args()

    with open(args.suite) as handle:
        suite = json.load(handle)
    task_ids = {task["task_id"] for task in suite["tasks"]}

    if os.path.abspath(args.key).startswith(os.path.abspath(args.corpus_dir) + os.sep):
        print("ERROR: the key file must not live inside the corpus directory.",
              file=sys.stderr)
        return 1

    records = collect_answers(args.answers_dir, task_ids)
    if not records:
        print(f"No answers found under {args.answers_dir}.", file=sys.stderr)
        return 1

    summary = build_corpus(records, args.corpus_dir, args.key, args.seed)
    print(f"Corpus: {summary['answers']} answers from {len(summary['models'])} model(s)")
    for model in summary["models"]:
        count = sum(1 for record in records if record["model"] == model)
        print(f"  {model}: {count}")
    print(f"Blind corpus: {summary['corpus_dir']}")
    print(f"Key (scoring must not read this): {summary['key_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
