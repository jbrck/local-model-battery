#!/usr/bin/env python3
"""HumanEval+: code generation on the shared leg contract.

Reads problems from the items file (default data/humaneval_plus.json), asks the
model to complete each function, executes the candidate against the bundled
test suite in a temp dir, and writes one result block for the orchestrator.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.code_exec import run_python_tests  # noqa: E402
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
from scripts.lib.model import ChatResponse  # noqa: E402

SYSTEM = """You are an expert Python developer. Write a solution for the given problem.
Output ONLY the code, no explanation, no markdown formatting, no imports.
The solution must be a standalone Python function."""

CODE_FENCE = "```"
LEG_ID = "humaneval_plus"


def _trim_head(lines: list[str]) -> None:
    while lines and not lines[0].strip():
        lines.pop(0)


def _trim_tail(lines: list[str]) -> None:
    while lines and not lines[-1].strip():
        lines.pop()


def _pop_fence(lines: list[str], index: int) -> None:
    if lines and lines[index].strip().startswith(CODE_FENCE):
        lines.pop(index)


def strip_fences(text: str) -> str:
    """Remove a markdown code fence, preserving the code's own indentation."""
    lines = text.splitlines()
    _trim_head(lines)
    _pop_fence(lines, 0)
    _trim_tail(lines)
    _pop_fence(lines, -1)
    _trim_tail(lines)
    return "\n".join(lines)


def candidate_source(imports: str, prompt: str, completion: str) -> str:
    """The module the grader executes: imports, then the signature, then the answer."""
    return "\n".join(part for part in (imports, prompt, completion) if part.strip())


def grade_completion(source: str, entry_point: str, test: str) -> tuple[bool | None, str]:
    """Execute the candidate against its tests. None means the run never scored."""
    ok, detail = run_python_tests(source, entry_point, test)
    if "timeout" in detail.lower():
        return None, detail
    return ok, detail


def messages_for(item: dict) -> list[dict]:
    """The v1 prompt: the system instruction, then the problem statement."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": item["prompt"]},
    ]


def _grade_item(response: ChatResponse, item: dict) -> tuple[bool | None, str]:
    """Grade one answer. None means the candidate run never scored."""
    completion = strip_fences(response.answer() or "")
    imports = item.get("imports") or ""
    source = candidate_source(imports, item["prompt"], completion)
    return grade_completion(source, item["entry_point"], item["test"])


def _tally_item(tally: Tally, response: ChatResponse, item: dict,
                total: int) -> tuple[bool | None, dict]:
    """Record one case in the tally and print its line.

    Returns the correctness it was counted with and the extras its checkpoint needs.
    """
    if not response.ok:
        tally.record(response)
        print(f"  [{tally.items}/{total}] ! ({response.elapsed_s:.1f}s)"
              f" {response.error or ''}", flush=True)
        return None, {}
    correct, detail = _grade_item(response, item)
    extras: dict = {}
    if correct is None:
        # The candidate never scored: counted as wrong, named in the notes.
        extras = {"exec_timeout": True, "note": f"{item['task_id']}: execution {detail}"}
        correct = False
        tally.record(response, correct=correct)
        tally.timeout_hits += 1
        tally.notes.append(extras["note"])
    else:
        tally.record(response, correct=correct)
    mark = "PASS" if correct else "FAIL"
    print(f"  [{tally.items}/{total}] {mark} ({response.elapsed_s:.1f}s)"
          + ("" if correct else f" {detail}"), flush=True)
    return correct, extras


def _replay_item(tally: Tally, saved: dict, total: int, case_id: str) -> None:
    """Count one checkpointed case again, without re-asking or re-executing it."""
    tally.record(replay_response(saved), correct=saved.get("correct"))
    if saved.get("exec_timeout"):
        tally.timeout_hits += 1
        tally.notes.append(saved.get("note") or "")
    print(f"  [{tally.items}/{total}] reused ({case_id})", flush=True)


def run_humaneval(args: LegArgs) -> Tally:
    """One ask per item; execute what comes back, checkpoint it and tally it."""
    cfg = leg_config(args)
    items = load_items(args.data, args.limit)
    tally = Tally()
    for index, item in enumerate(items):
        case_id = f"{index:05d}"
        saved = load_case(args, LEG_ID, case_id)
        if saved is not None:
            _replay_item(tally, saved, len(items), item.get("task_id") or case_id)
            continue
        response = ask(cfg, messages_for(item), args)
        correct, extras = _tally_item(tally, response, item, len(items))
        write_case(args, LEG_ID, case_id,
                   case_record(case_id, response, correct=correct, **extras))
    return tally


def main(argv: list[str] | None = None) -> None:
    args = parse_leg_args("HumanEval+ benchmark",
                          default_data="data/humaneval_plus.json", argv=argv)
    print(f"HumanEval+ [{args.model}] cap {args.cap} thinking {args.thinking}",
          flush=True)
    finish(LEG_ID, args, run_humaneval(args))


if __name__ == "__main__":
    main()
