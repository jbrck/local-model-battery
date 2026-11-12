"""HermesAgent-20 leg: 20 sandboxed agentic scenarios run and scored by benchlocal-cli."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from scripts.lib.harness import (LegArgs, Tally, add_leg_args, finish,
                                 parse_leg_args)

LEG_ID = "hermesagent20"
PACK_ID = "hermesagent-20"
RUNNER = "/mnt/data/benchmark-venv/bin/benchlocal-cli"
SANDBOX_FLAG = "--enable-sandboxed-packs"
TIMEOUT_MODES = frozenset({"timeout", "agent_runner_timeout"})
INFRA_MODES = frozenset({"http_error", "server_error", "stall"})
RUN_SUFFIX = "-run.json"
JOURNAL_SUFFIX = ".partial.jsonl"


def journal_path(save_json: str) -> str:
    return save_json + JOURNAL_SUFFIX


def thinking_flag(mode: str) -> str:
    return "--no-thinking" if mode == "off" else "--enable-thinking"


def build_command(args: LegArgs, save_json: str, resume: str | None) -> list[str]:
    command = [
        RUNNER, "run",
        "--pack", PACK_ID,
        SANDBOX_FLAG,
        "--endpoint", args.endpoint,
        "--model", args.model,
        "--api-key", args.api_key,
        "--max-tokens", str(args.cap),
        "--thinking-max-tokens", str(args.cap),
        "--temperature", str(args.temperature),
        "--timeout-per-case", str(args.timeout_s),
        thinking_flag(args.thinking),
        "--save-json", save_json,
        "--incremental",
        "--sandbox-log-dir", os.path.join(args.output_dir, "sandbox-logs"),
    ]
    if resume:
        command += ["--resume", resume]
    return command


def _pack_from_journal(records: list[dict]) -> dict:
    scenarios = [record["scenario"] for record in records
                 if record.get("pack", {}).get("pack_id") == PACK_ID
                 and isinstance(record.get("scenario"), dict)]
    meta = records[0].get("pack", {}) if records else {}
    return {"pack_id": PACK_ID, "version": meta.get("version"),
            "catalog_scenario_count": meta.get("catalog_scenario_count"),
            "scenarios": scenarios}


def load_run(save_json: str) -> dict | None:
    """The run as the runner saved it, or rebuilt from the partial journal, or None."""
    if os.path.isfile(save_json):
        with open(save_json) as handle:
            return json.load(handle)
    journal = journal_path(save_json)
    if os.path.isfile(journal):
        records = []
        with open(journal) as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        if records:
            return {"packs": [_pack_from_journal(records)]}
    return None


def pack_of(run_data: dict) -> dict:
    for pack in run_data.get("packs", []):
        if pack.get("pack_id") == PACK_ID:
            return pack
    return {}


def run_is_complete(pack: dict) -> bool:
    catalog = int(pack.get("catalog_scenario_count") or 0)
    return catalog > 0 and len(pack.get("scenarios", [])) >= catalog


def note_failure_mode(tally: Tally, name: str, mode: str) -> bool:
    """True when the case still reaches the scorer."""
    if mode in TIMEOUT_MODES:
        tally.timeout_hits += 1
    elif mode in INFRA_MODES:
        tally.failed += 1
    else:
        if mode == "token_limit":
            tally.runaway += 1
        return True
    tally.errors.append(f"{name}: {mode}")
    return False


def record_scenario(tally: Tally, scenario: dict) -> None:
    tally.items += 1
    name = str(scenario.get("id") or "?")
    mode = str(scenario.get("failure_mode") or "")
    tally.tokens += int(scenario.get("tokens_completion") or 0)
    tally.seconds += float(scenario.get("latency_seconds") or 0.0)
    if not note_failure_mode(tally, name, mode):
        return
    tally.scored += 1
    tally.correct += int(bool(scenario.get("passed")))
    if not scenario.get("passed"):
        tally.errors.append(f"{name}: {mode or 'unspecified'}")


def score_pack(pack: dict, tally: Tally) -> str:
    for scenario in pack.get("scenarios", []):
        record_scenario(tally, scenario)
    return str(pack.get("version") or "unknown")


def main(argv: list[str] | None = None) -> int:
    args = parse_leg_args("HermesAgent-20: 20 agentic scenarios in a Docker sandbox",
                          argv=argv)
    save_json = os.path.join(args.output_dir, f"{LEG_ID}{RUN_SUFFIX}")
    tally = Tally()
    extra: dict = {}
    run_data = load_run(save_json) or {}
    pack = pack_of(run_data)
    if run_is_complete(pack):
        print(f"complete run found at {save_json}; reporting it without a rerun")
    else:
        resume = journal_path(save_json) if os.path.isfile(journal_path(save_json)) else None
        code = subprocess.run(build_command(args, save_json, resume)).returncode
        if code != 0:
            tally.notes.append(f"runner exited {code}; scored what was written")
        run_data = load_run(save_json) or {}
        pack = pack_of(run_data)
    if not pack:
        tally.notes.append("no hermesagent-20 output to score")
    else:
        extra["harness_version"] = score_pack(pack, tally)
        extra["case_timeout"] = args.timeout_s
    finish(LEG_ID, args, tally, extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
