#!/usr/bin/env python3
"""Selftest gate: refuse the battery in its first half minute when it cannot run.

Checks four things: the endpoint lists the requested model, a one-token chat
completion answers, the thinking switch takes, and every in-repo leg script
imports. Each check gets its own line in the result block; any failed check
exits non-zero with a readable reason.
"""
from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib import server_probe  # noqa: E402
from scripts.lib.config import load_toml  # noqa: E402
from scripts.lib.harness import (  # noqa: E402
    LegArgs,
    Tally,
    ask,
    finish,
    leg_config,
    parse_leg_args,
)
from scripts.lib.legs import load_legs  # noqa: E402
from scripts.lib.model import ChatResponse, ModelConfig, get_json  # noqa: E402

SELFTEST_LEG_ID = "selftest"
ONE_TOKEN_CAP = 1
IMPORT_TIMEOUT_S = 60
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK_NAMES = ("endpoint_models", "one_token_chat", "thinking_switch", "leg_imports")

_IMPORT_SNIPPET = (
    "import importlib.util, sys\n"
    "sys.path.insert(0, sys.argv[2])\n"
    "spec = importlib.util.spec_from_file_location('leg_entry', sys.argv[1])\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(module)\n"
)


def _label(ok: bool, note: str) -> str:
    return f"pass: {note}" if ok else f"FAIL: {note}"


def _model_ids(payload: dict) -> list[str]:
    """The model ids a /v1/models payload lists, skipping malformed entries."""
    entries = [entry for entry in payload.get("data") or [] if isinstance(entry, dict)]
    return [str(entry.get("id", "")) for entry in entries]


def _models_check(cfg: ModelConfig) -> tuple[bool, str, ChatResponse]:
    status, payload, error = get_json(cfg.models_url, cfg.api_key)
    response = ChatResponse(status=status, payload=payload, error=error)
    if error or not isinstance(payload, dict):
        return False, f"/v1/models did not answer: {error or 'no payload'}", response
    ids = _model_ids(payload)
    if not any(cfg.name in model_id for model_id in ids):
        listed = ", ".join(ids[:5]) or "nothing"
        return False, f"model {cfg.name!r} not listed by /v1/models ({listed})", response
    return True, f"/v1/models lists {cfg.name}", response


def _chat_check(cfg: ModelConfig, args: LegArgs) -> tuple[bool, str, ChatResponse]:
    messages = [{"role": "user", "content": "Reply with the single word ok."}]
    response = ask(cfg, messages, args, cap=ONE_TOKEN_CAP)
    if not response.ok:
        reason = response.error or f"HTTP {response.status}"
        return False, f"one-token completion failed: {reason}", response
    return True, f"one-token completion answered in {response.elapsed_s:.1f}s", response


def _thinking_check(cfg: ModelConfig) -> tuple[bool, str]:
    facts = server_probe.probe_server(cfg, run_thinking=True, run_tools=False)
    if not facts.reachable:
        return False, "endpoint unreachable during the thinking probe"
    if facts.thinking_switch is None:
        return False, "thinking switch unmeasurable: " + "; ".join(facts.errors)
    counts = f"on {facts.thinking_on_tokens} tokens, off {facts.thinking_off_tokens}"
    if not facts.thinking_switch:
        return False, f"thinking switch did not take ({counts})"
    return True, f"thinking switch took ({counts})"


def _import_one(entry_path: str) -> str | None:
    command = [sys.executable, "-c", _IMPORT_SNIPPET, entry_path, REPO_ROOT]
    try:
        proc = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=IMPORT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return f"{entry_path}: import timed out after {IMPORT_TIMEOUT_S}s"
    if proc.returncode == 0:
        return None
    lines = [line for line in (proc.stderr or "").strip().splitlines() if line]
    return f"{entry_path}: {lines[-1] if lines else 'no error output'}"


def _catalog_entries() -> list[tuple[str, str]]:
    standard = load_toml(os.path.join(REPO_ROOT, "standard.toml"))
    return [(leg.id, os.path.join(REPO_ROOT, leg.entry))
            for leg in load_legs(standard) if leg.implemented]


def _leg_import_check(entries: list[tuple[str, str]] | None = None) -> tuple[bool, str]:
    checked = entries if entries is not None else _catalog_entries()
    problems = []
    for leg_id, entry_path in checked:
        if not os.path.isfile(entry_path):
            problems.append(f"{leg_id}: entry script {entry_path} is missing")
            continue
        problem = _import_one(entry_path)
        if problem:
            problems.append(f"{leg_id}: {problem}")
    if problems:
        return False, "; ".join(problems)
    return True, f"all {len(checked)} in-repo leg scripts import"


def main(argv: list[str] | None = None,
         import_entries: list[tuple[str, str]] | None = None) -> int:
    args = parse_leg_args("endpoint and harness selftest", argv=argv)
    cfg = leg_config(args)
    tally = Tally()
    checks: dict[str, str] = {}
    ok_models, note_models, models_response = _models_check(cfg)
    checks["endpoint_models"] = _label(ok_models, note_models)
    tally.record(models_response, correct=ok_models)
    ok_chat, note_chat, chat_response = _chat_check(cfg, args)
    checks["one_token_chat"] = _label(ok_chat, note_chat)
    tally.record(chat_response, correct=ok_chat)
    checks["thinking_switch"] = _label(*_thinking_check(cfg))
    ok_imports, note_imports = _leg_import_check(import_entries)
    checks["leg_imports"] = _label(ok_imports, note_imports)
    gate_ok = all(checks[name].startswith("pass") for name in CHECK_NAMES)
    extra: dict = {"checks": checks}
    if not gate_ok:
        extra["gate"] = "fail"
    finish(SELFTEST_LEG_ID, args, tally, extra=extra)
    if not gate_ok:
        print("selftest gate failed:", file=sys.stderr)
        for name in CHECK_NAMES:
            if not checks[name].startswith("pass"):
                print(f"  {name}: {checks[name][6:]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
