"""Dataset loading and the resumable partial-jsonl store for the LiveCodeBench leg."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.adapters.lcb import LcbArgs


def load_problems(path: str, limit: int, seed: int) -> list[dict]:
    """The dataset's problems in the legacy order: sort, seeded shuffle, limit, sort."""
    with open(path) as handle:
        raw = [json.loads(line) for line in handle if line.strip()]
    raw.sort(key=lambda problem: str(problem["question_id"]))
    random.Random(seed).shuffle(raw)
    if limit:
        raw = raw[:limit]
    raw.sort(key=lambda problem: str(problem["question_id"]))
    return raw


def items_sha(problems: list[dict]) -> str:
    """The comparability stamp over exactly these problems, in run order."""
    ids = [str(problem["question_id"]) for problem in problems]
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode()).hexdigest()


def partial_path(args: "LcbArgs") -> str:
    """The resumable generation log for this model and output directory."""
    return os.path.join(args.output_dir, f"{args.model}-lcb.partial.jsonl")


def load_done(path: str) -> dict[str, dict]:
    """Every generation already in the partial jsonl, keyed by question id."""
    if not os.path.exists(path):
        return {}
    done: dict[str, dict] = {}
    with open(path) as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[str(record.get("question_id"))] = record
    return done


def append_record(path: str, record: dict) -> None:
    """Append one generation, creating the output directory on the way."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")


def write_aborted(args: "LcbArgs", reason: str, done: dict[str, dict],
                  elapsed_s: float) -> str:
    """Persist why generation stopped, so a truncated leg is never read as a full one."""
    path = partial_path(args)[: -len(".partial.jsonl")] + ".aborted.json"
    payload = {"aborted": True, "reason": reason, "model": args.model,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "elapsed_s": round(elapsed_s, 1),
               "budget_hours": args.max_hours, "generated": len(done),
               "with_code": sum(1 for record in done.values() if record.get("code"))}
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)
    return path
