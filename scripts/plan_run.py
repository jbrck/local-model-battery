#!/usr/bin/env python3
"""Plan a battery run: probe the host and the endpoint, then print the decision sheet.

    python3 scripts/plan_run.py --profile profiles/arete-27b.toml --endpoint URL \\
        --model served-name [--legs math500,gpqa_diamond] [--json] [--no-probe]

Nothing is started and nothing is written. The sheet is what the setup skill walks the
operator through before a single leg runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.lib.config import load_toml  # noqa: E402
from scripts.lib.env_probe import probe_host  # noqa: E402
from scripts.lib.legs import load_legs  # noqa: E402
from scripts.lib.model import ModelConfig  # noqa: E402
from scripts.lib.pilot import run_pilot  # noqa: E402
from scripts.lib.plan_run import plan_run  # noqa: E402
from scripts.lib.prereqs import Judge  # noqa: E402
from scripts.lib.server_probe import probe_server  # noqa: E402
from scripts.lib.variables import load_variables  # noqa: E402

STANDARD_FILE = "standard.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="model profile TOML")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/v1",
                        help="OpenAI-compatible base URL")
    parser.add_argument("--model", default="", help="served model name")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--legs", default="", help="comma-separated subset of leg ids")
    parser.add_argument("--judge-model", default="", help="scoring model for the writing suite")
    parser.add_argument("--judge-endpoint", default="", help="judge base URL")
    parser.add_argument("--no-probe", action="store_true",
                        help="skip the endpoint probes; fit and prereqs only")
    parser.add_argument("--json", action="store_true", help="emit the plan as JSON")
    return parser


def probe_from_args(args: argparse.Namespace):
    """Plan the run and hand back the raw probes the plan was built from.

    The orchestrator needs the same host, endpoint and pilot the plan saw: the gate re-checks
    them and the manifest records them.
    """
    standard = load_toml(STANDARD_FILE)
    legs = load_legs(standard)
    variables = load_variables()
    profile = load_toml(args.profile)
    profile.setdefault("id", args.model or Path(args.profile).stem)
    host = probe_host()
    cfg = ModelConfig(name=args.model or str(profile.get("id")), endpoint=args.endpoint,
                      api_key=args.api_key)
    server = probe_server(cfg) if not args.no_probe else _unprobed_server(cfg)
    pilot = run_pilot(cfg) if not args.no_probe else _unprobed_pilot()
    judge = Judge(model=args.judge_model or None, endpoint=args.judge_endpoint or None,
                  under_test=cfg.name)
    selected = tuple(item.strip() for item in args.legs.split(",") if item.strip())
    plan = plan_run(standard, legs, variables, profile, host, server, pilot, judge, selected)
    return plan, host, server, pilot


def plan_from_args(args: argparse.Namespace):
    return probe_from_args(args)[0]


def _unprobed_server(cfg: ModelConfig):
    from scripts.lib.server_probe import ServerFacts
    return ServerFacts(reachable=False, errors=("probing was disabled",))


def _unprobed_pilot():
    from scripts.lib.pilot import PilotResult
    return PilotResult(tps=0.0, latency_s=0.0, generated_tokens=0, requests=0, failed=0,
                       stalled=True, errors=("probing was disabled",))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = plan_from_args(args)
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        print(plan.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
