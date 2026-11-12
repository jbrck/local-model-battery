"""The gate between a plan and a run: nothing starts until every check passes.

    python3 scripts/preflight.py --profile profiles/x.toml --endpoint URL \\
        --model served-name --out runs/2026-11-12-x [--resume]

Runs the same probes the planner ran and re-checks them against the plan, because the host
moves: a model loaded between planning and starting invalidates the fit. Every check is
recorded in PREFLIGHT.json. A blocking miss exits non-zero and names the miss.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.lib.config import load_toml  # noqa: E402
from scripts.lib.env_probe import HostFacts, probe_host  # noqa: E402
from scripts.lib.fit import RAM_HEADROOM_GB  # noqa: E402
from scripts.lib.legs import load_legs  # noqa: E402
from scripts.lib.model import ModelConfig  # noqa: E402
from scripts.lib.pilot import PilotResult, run_pilot  # noqa: E402
from scripts.lib.plan_run import BATTERY_VERSION, RunPlan, plan_run  # noqa: E402
from scripts.lib.prereqs import Judge  # noqa: E402
from scripts.lib.server_probe import ServerFacts, probe_server  # noqa: E402
from scripts.lib.variables import load_variables  # noqa: E402

STANDARD_FILE = "standard.toml"
PREFLIGHT_FILE = "PREFLIGHT.json"
SWAP_TOLERANCE_GB = 1.0


@dataclass(frozen=True)
class Check:
    """One thing that has to hold before the battery starts."""

    name: str
    ok: bool
    detail: str
    hard: bool = True

    @property
    def blocks(self) -> bool:
        return self.hard and not self.ok

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "hard": self.hard, "detail": self.detail}


def check_data(plan: RunPlan) -> list[Check]:
    """Every runnable leg's prerequisites, re-resolved at the moment of starting."""
    checks = []
    for planned in plan.legs:
        missing = [item for item in planned.dependencies if not item.available]
        detail = ("all present" if not missing
                  else "; ".join(f"{item.name}: {item.detail}" for item in missing))
        checks.append(Check(f"data:{planned.id}", not missing, detail))
    return checks


def _value(plan: RunPlan, variable: str) -> str:
    rows = [item.chosen for item in plan.decisions if item.variable == variable]
    return rows[0] if rows else ""


def _standard(plan: RunPlan, variable: str) -> str:
    rows = [item.standard_value for item in plan.decisions if item.variable == variable]
    return rows[0] if rows else ""


def _reach_check(server: ServerFacts) -> Check:
    return Check("endpoint", server.reachable, server.errors[0] if server.errors else "answered")


def _ctx_check(plan: RunPlan, server: ServerFacts) -> list[Check]:
    planned_ctx = plan.legs[0].ctx if plan.legs else 0
    if not server.ctx:
        return []
    return [Check("endpoint:ctx", server.ctx >= planned_ctx,
                  f"serving {server.ctx} tokens, the plan expects at least {planned_ctx}")]


def _thinking_check(plan: RunPlan, server: ServerFacts) -> list[Check]:
    """A plan that switches thinking needs an endpoint that honours the flag."""
    if _value(plan, "thinking") not in ("off", "on") or server.thinking_switch is not False:
        return []
    return [Check("endpoint:thinking", False,
                  "the plan switches thinking but this endpoint ignores the flag; every "
                  "thinking-dependent score would be wrong")]


def _slots_check(plan: RunPlan, server: ServerFacts) -> list[Check]:
    """Slot count moves throughput, not scores, so a shortfall is a warning."""
    asks = _standard(plan, "slots")
    if not _value(plan, "slots") or not server.slots:
        return []
    expected = int(asks) if asks.isdigit() else server.slots
    return [Check("endpoint:slots", server.slots >= expected,
                  f"serving {server.slots} slots against the standard's {asks}", hard=False)]


def check_endpoint(plan: RunPlan, server: ServerFacts) -> list[Check]:
    """The serve has to be the one the plan was built against."""
    return ([_reach_check(server)] + _ctx_check(plan, server) + _thinking_check(plan, server)
            + _slots_check(plan, server))


def check_host(plan: RunPlan, host: HostFacts) -> list[Check]:
    """VRAM, RAM and anything else already using them."""
    fit = plan.fit
    checks = [Check("host:fits", fit.fits, fit.reason)]
    if fit.tier == "T2":
        spill = max(0.0, fit.weights_gb - fit.vram_free_gb)
        available = host.ram_available_gb or 0.0
        checks.append(Check("host:ram", available >= spill + RAM_HEADROOM_GB,
                            f"{spill:.1f} GB of weights land in RAM and {available:.1f} GB is "
                            f"free; {RAM_HEADROOM_GB:.0f} GB has to stay spare"))
    if host.compute_apps:
        checks.append(Check("host:idle", True,
                            f"{len(host.compute_apps)} other compute process(es) hold VRAM; "
                            f"the fit above already discounts them", hard=False))
    if host.swap_used_gb and host.swap_used_gb > SWAP_TOLERANCE_GB:
        checks.append(Check("host:swap", True,
                            f"{host.swap_used_gb:.1f} GB in swap: this host is under memory "
                            f"pressure and timing will be noisy", hard=False))
    return checks


def check_pilot(pilot: PilotResult) -> list[Check]:
    """Budgets come from the pilot, so a stalled pilot is a blocking miss."""
    return [Check("pilot", pilot.usable,
                  f"{pilot.tps:.1f} tok/s over {pilot.requests} requests"
                  if pilot.usable else
                  "no rate measured: " + (pilot.errors[0] if pilot.errors else "all requests "
                                          "failed") + ", so every budget is a floor guess")]


def check_out_dir(out_dir: Path | None, resume: bool) -> list[Check]:
    """A run dir is written once. Resuming its own partial run is allowed, a fresh one is not."""
    if out_dir is None:
        return []
    if not out_dir.exists():
        return [Check("out", True, f"{out_dir} does not exist yet, will be created")]
    if not resume:
        return [Check("out", False, f"{out_dir} already exists; pass --resume to continue it "
                                    f"or choose another name")]
    return [Check("out", True, f"resuming {out_dir}")]


def run_preflight(plan: RunPlan, host: HostFacts, server: ServerFacts, pilot: PilotResult,
                  out_dir: Path | None = None, resume: bool = False) -> list[Check]:
    """Every check, in the order the operator should read them."""
    return (check_out_dir(out_dir, resume) + check_endpoint(plan, server) + check_host(plan, host)
            + check_pilot(pilot) + check_data(plan))


def blocked(checks: list[Check]) -> list[Check]:
    return [check for check in checks if check.blocks]


def write_preflight(out_dir: Path, plan: RunPlan, checks: list[Check]) -> Path:
    """Record the gate next to the plan, so the orchestrator can refuse a stale start."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "battery_version": BATTERY_VERSION,
        "standard_version": plan.standard_version,
        "standard_md5": plan.standard_md5,
        "model": plan.model,
        "stamp_line": plan.stamp_line,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": not blocked(checks),
        "checks": [check.to_dict() for check in checks],
    }
    path = out_dir / PREFLIGHT_FILE
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def summarise(checks: list[Check]) -> str:
    """The check table, misses first."""
    lines = []
    for check in blocked(checks) + [item for item in checks if not item.blocks]:
        mark = {True: "ok", False: "MISS"}[check.ok] if check.hard else \
            ("ok" if check.ok else "warn")
        lines.append(f"  {mark:<4} {check.name:<20} {check.detail}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="model profile TOML")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--legs", default="", help="comma-separated subset of leg ids")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--judge-endpoint", default="")
    parser.add_argument("--out", default="", help="run directory to gate")
    parser.add_argument("--resume", action="store_true", help="continue an existing run dir")
    return parser


def _plan_from_args(args, standard: dict, legs):
    """Probe the machine and the endpoint, then plan the run the arguments describe."""
    profile = load_toml(args.profile)
    profile.setdefault("id", args.model or Path(args.profile).stem)
    host = probe_host()
    cfg = ModelConfig(name=args.model or str(profile.get("id")), endpoint=args.endpoint,
                      api_key=args.api_key)
    server = probe_server(cfg)
    pilot = run_pilot(cfg)
    judge = Judge(model=args.judge_model or None, endpoint=args.judge_endpoint or None,
                  under_test=cfg.name)
    selected = tuple(item.strip() for item in args.legs.split(",") if item.strip())
    plan = plan_run(standard, legs, load_variables(), profile, host, server, pilot, judge,
                    selected, STANDARD_FILE, args.profile)
    return plan, host, server, pilot


def _write(out_dir: Path, plan: RunPlan, checks: list[Check]) -> None:
    path = write_preflight(out_dir, plan, checks)
    (out_dir / "run-plan.json").write_text(json.dumps(plan.to_dict(), indent=2) + "\n")
    print(f"wrote {path} and {out_dir / 'run-plan.json'}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    standard = load_toml(STANDARD_FILE)
    plan, host, server, pilot = _plan_from_args(args, standard, load_legs(standard))
    out_dir = Path(args.out).expanduser() if args.out else None
    checks = run_preflight(plan, host, server, pilot, out_dir, args.resume)
    if out_dir is not None:
        _write(out_dir, plan, checks)
    print(summarise(checks))
    misses = blocked(checks)
    if misses:
        print(f"\npreflight failed on {len(misses)} check(s): "
              + ", ".join(check.name for check in misses), file=sys.stderr)
        return 1
    print("\npreflight passed: the run may start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
