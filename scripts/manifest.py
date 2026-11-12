"""The run's paper trail: what was configured, and what the run turned out to be.

    python3 scripts/manifest.py close --out runs/2026-11-12-x

`start` writes MANIFEST.json next to the plan and refuses to continue a directory whose
configuration moved. `close` writes NOTES.md, ending in the comparability stamp, so two runs
can be told apart or matched without reading the numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.lib.config import repo_path  # noqa: E402
from scripts.lib.env_probe import HostFacts  # noqa: E402
from scripts.lib.plan_run import BATTERY_VERSION, RunPlan  # noqa: E402
from scripts.lib.server_probe import ServerFacts  # noqa: E402

MANIFEST_FILE = "MANIFEST.json"
NOTES_FILE = "NOTES.md"
RESULTS_FILE = "results.json"
PREFLIGHT_FILE = "PREFLIGHT.json"

# Environment names worth recording: where the external data lives, and how the engine sees
# the GPUs. Nothing beyond the prefix allowlist is read, and secret-shaped names never are.
ENV_PREFIXES = ("LCB_", "NLTK_", "BATTERY_", "CUDA_", "HF_")
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")

# Pilot-derived, so a resume on the same configuration would look different for no reason.
VOLATILE_DECISIONS = ("case_timeout",)


class ManifestMismatch(Exception):
    """A run directory that cannot be continued under the current configuration."""


def _digest(payload: object) -> str:
    """A stable md5 over a JSON-serializable structure."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.md5(blob).hexdigest()


def env_snapshot(environ: dict | None = None) -> dict:
    """The data-location environment, minus anything that looks like a credential."""
    source = environ if environ is not None else _os_environ()
    snapshot = {}
    for name, value in sorted(source.items()):
        if not name.startswith(ENV_PREFIXES):
            continue
        if any(marker in name.upper() for marker in SECRET_MARKERS):
            continue
        snapshot[name] = value
    return snapshot


def _os_environ() -> dict:
    import os
    return dict(os.environ)


def harness_digests(plan: RunPlan) -> dict:
    """md5 of each leg's entry script, resolved inside the repo. Missing files are recorded
    as an empty digest rather than omitted: a pending port is a fact about the run."""
    digests = {}
    for planned in plan.legs:
        entry = planned.leg.entry
        path = repo_path(entry)
        digests[entry] = _file_md5(path) if path.is_file() else ""
    return digests


def _file_md5(path: Path) -> str:
    from scripts.lib.config import file_md5
    return file_md5(path)


def config_fingerprint(plan: RunPlan) -> dict:
    """The parts of a plan that invalidate a resume when they move."""
    return {
        "standard_md5": plan.standard_md5,
        "profile_md5": plan.profile_md5,
        "model": {"id": plan.model, "weights_gb": round(plan.fit.weights_gb, 2),
                  "kv_bytes_per_token": plan.fit.kv_bytes_per_token,
                  "tier": plan.fit.tier, "max_ctx": plan.fit.max_ctx,
                  "vram_free_gb": round(plan.fit.vram_free_gb, 2)},
        "legs": {item.id: {"cap": item.cap, "ctx": item.ctx, "thinking": item.thinking}
                 for item in plan.legs},
        "decisions": {item.variable: item.chosen for item in plan.decisions
                      if item.variable not in VOLATILE_DECISIONS},
        "skips": sorted(skip.leg for skip in plan.skips),
    }


def build_manifest(plan: RunPlan, host: HostFacts, server: ServerFacts) -> dict:
    """Everything needed to say later exactly what this run was."""
    config = config_fingerprint(plan)
    return {
        "battery_version": BATTERY_VERSION,
        "standard_version": plan.standard_version,
        "standard_md5": plan.standard_md5,
        "profile_path": plan.profile_path,
        "profile_md5": plan.profile_md5,
        "model": plan.model,
        "config_md5": _digest(config),
        "config": config,
        "plan_md5": _digest(plan.to_dict()),
        "harness_md5": harness_digests(plan),
        "serve_cmdline": str(plan.to_dict().get("serve_cmdline") or "") or _serve_hint(server),
        "props": server.props if isinstance(server.props, dict) else {},
        "env": env_snapshot(),
        "host": {
            "gpus": [gpu.to_dict() for gpu in host.gpus],
            "ram_gb": host.ram_total_gb, "ram_available_gb": host.ram_available_gb,
            "cpu": host.cpu_model, "cpu_threads": host.cpu_threads,
            "tools": {tool.name: tool.version for tool in host.tools},
        },
        "pilot_tps": plan.stamp.get("pilot_tps", ""),
        "stamp_line": plan.stamp_line,
        "stamp": plan.stamp,
        "fit": plan.fit.to_dict(),
    }


def _serve_hint(server: ServerFacts) -> str:
    """The endpoint's own report of how it was started, when there is one."""
    model_path = (server.props or {}).get("model_path") if isinstance(server.props, dict) else None
    if model_path:
        return f"served from {model_path} (n_ctx {server.ctx}, {server.slots} slots)"
    return "not reported by the endpoint"


def start(out_dir: Path, plan: RunPlan, host: HostFacts, server: ServerFacts,
          resume: bool = False, manifest: dict | None = None) -> Path:
    """Write MANIFEST.json, refusing to continue a directory whose configuration moved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = manifest if manifest is not None else build_manifest(plan, host, server)
    path = out_dir / MANIFEST_FILE
    if path.exists():
        previous = json.loads(path.read_text())
        if not resume:
            raise ManifestMismatch(f"{out_dir} already has a manifest; pass resume to continue")
        for key, label in (("standard_md5", "standard"), ("profile_md5", "profile"),
                           ("config_md5", "configuration")):
            if previous.get(key) != payload.get(key):
                raise ManifestMismatch(f"{label} changed since this run started "
                                       f"({previous.get(key)} -> {payload.get(key)})")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def start_allowed(out_dir: Path, plan: RunPlan) -> tuple[bool, str]:
    """Why the orchestrator may or may not start: the gate ran, and the config still holds."""
    gate = out_dir / PREFLIGHT_FILE
    if not gate.is_file():
        return False, f"{gate} is missing: run scripts/preflight.py first"
    payload = json.loads(gate.read_text())
    if not payload.get("ok"):
        return False, "preflight recorded a blocking miss"
    if payload.get("standard_md5") != plan.standard_md5:
        return False, (f"the standard moved since preflight "
                       f"({payload.get('standard_md5')} -> {plan.standard_md5})")
    return True, "preflight passed and the standard is unchanged"


def _rate(value: object) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) and value else "-"


def _totals(plan: RunPlan, results: dict) -> dict:
    """Sum the counters the orchestrator reports, ignoring anything that is not a number."""
    totals = {"items": 0, "cap_hits": 0, "runaway": 0, "timeout_hits": 0}
    for planned in plan.legs:
        stats = results.get(planned.id, {})
        for key in totals:
            value = stats.get(key, 0)
            if isinstance(value, int):
                totals[key] += value
    return totals


def _row(planned, stats: dict) -> str:
    return (f"| {planned.id} | {stats.get('status', 'not run')} | {stats.get('items', 0)} | "
            f"{_rate(stats.get('tps'))} | {stats.get('cap_hits', 0)} | "
            f"{stats.get('runaway', 0)} | {stats.get('timeout_hits', 0)} |")


def _deviation_lines(plan: RunPlan) -> list[str]:
    if not plan.deviations:
        return ["", "Nothing was moved off the standard: these numbers compare with every "
                    "other ON-STANDARD run at this version."]
    return ["", "Off the standard, deliberately:"] + [
        f"- {item.variable}: {item.chosen} (standard {item.standard_value}) - {item.consequence}"
        for item in plan.deviations]


def notes(plan: RunPlan, results: dict) -> str:
    """The run's own account of itself, ending in the stamp."""
    lines = [
        f"# {plan.model} on the portable battery {plan.standard_version}",
        "",
        f"Stamp: `{plan.stamp_line}`",
        f"Standard md5: {plan.standard_md5}  Profile md5: {plan.profile_md5}",
        "",
        "| leg | status | items | tok/s | cap hits | runaways | timeouts |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [_row(planned, results.get(planned.id, {})) for planned in plan.legs]
    totals = _totals(plan, results)
    lines += [
        "",
        f"Totals: {totals['items']} items, {totals['cap_hits']} cap hits, "
        f"{totals['runaway']} runaways, {totals['timeout_hits']} timeouts.",
        f"Skipped: {', '.join(skip.leg for skip in plan.skips) or 'none'}",
    ]
    lines += [f"- {skip.leg}: {skip.reason}" for skip in plan.skips]
    lines += _deviation_lines(plan)
    lines += [f"- warning: {warning}" for warning in plan.warnings]
    return "\n".join(lines) + "\n"


def close(out_dir: Path, plan: RunPlan, results: dict | None = None) -> Path:
    """Write NOTES.md, reading the orchestrator's results file when none is passed in."""
    loaded = results
    if loaded is None:
        path = out_dir / RESULTS_FILE
        loaded = json.loads(path.read_text()) if path.is_file() else {}
    target = out_dir / NOTES_FILE
    target.write_text(notes(plan, loaded))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("close", "show"))
    parser.add_argument("--out", required=True, help="run directory")
    args = parser.parse_args(argv)
    out_dir = Path(args.out).expanduser()
    if args.action == "show":
        payload = json.loads((out_dir / MANIFEST_FILE).read_text())
        print(f"{payload['model']} stamp {payload['stamp_line']}")
        print(f"config {payload['config_md5']} plan {payload['plan_md5']}")
        return 0
    plan_path = out_dir / "run-plan.json"
    if not plan_path.is_file():
        print(f"{plan_path} is missing: the plan has to be written before the notes",
              file=sys.stderr)
        return 1
    from scripts.lib.plan_from_dict import plan_from_dict
    plan = plan_from_dict(json.loads(plan_path.read_text()))
    print(f"wrote {close(out_dir, plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
