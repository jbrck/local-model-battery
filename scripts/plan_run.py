#!/usr/bin/env python3
"""Plan a battery run: probe the host and the endpoint, then print the decision sheet.

    python3 scripts/plan_run.py --endpoint URL --model served-name [--json] [--no-probe]

The model's shape is read from the model itself: the file the endpoint reports it loaded, or
one named with --model-file. A profile is only needed for a model that neither reader can
reach, and then only for the numbers it holds that the readers could not supply.

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
from scripts.lib.engine_facts import EngineFacts, probe_engine  # noqa: E402
from scripts.lib.env_probe import probe_host, serving_pid_for  # noqa: E402
from scripts.lib.judge import resolve_judge_defaults  # noqa: E402
from scripts.lib.legs import load_legs  # noqa: E402
from scripts.lib.model import ModelConfig  # noqa: E402
from scripts.lib.model_probe import resolve_model_shape  # noqa: E402
from scripts.lib.pilot import run_pilot  # noqa: E402
from scripts.lib.plan_run import plan_run  # noqa: E402
from scripts.lib.prereqs import Judge  # noqa: E402
from scripts.lib.server_probe import probe_server  # noqa: E402
from scripts.lib.variables import load_variables  # noqa: E402

STANDARD_FILE = "standard.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="",
                        help="model profile TOML; only for what the probes cannot read")
    parser.add_argument("--model-file", default="",
                        help="the GGUF or model directory to read the shape from")
    parser.add_argument("--model-dir", default="",
                        help="a HF model directory when the weights are safetensors")
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


def _probes(args: argparse.Namespace, cfg: ModelConfig):
    """The endpoint and pilot the plan is built from, or the honest stubs when probing is off.

    Probing is what separates a sheet built on measurements from one built on guesses, so the
    off switch says so in the facts it hands back rather than pretending they are unknown.
    """
    if args.no_probe:
        return _unprobed_server(cfg), _unprobed_pilot()
    return probe_server(cfg), run_pilot(cfg)


def _judge(args: argparse.Namespace, cfg: ModelConfig, standard: dict) -> Judge:
    """The writing suite's scorer, or a judge that reports the leg as unscored."""
    model, endpoint = resolve_judge_defaults(args.judge_model, args.judge_endpoint,
                                             cfg.endpoint, standard)
    pinned = not args.judge_model and bool((standard.get("judge") or {}).get("model"))
    return Judge(model=model or None, endpoint=endpoint or None, under_test=cfg.name,
                 pinned=pinned)


def probe_from_args(args: argparse.Namespace):
    """Plan the run and hand back the raw probes the plan was built from.

    The orchestrator needs the same host, endpoint and pilot the plan saw: the gate re-checks
    them and the manifest records them.
    """
    standard = load_toml(STANDARD_FILE)
    legs = load_legs(standard)
    variables = load_variables()
    profile = _profile_from_args(args)
    host = probe_host()
    cfg = ModelConfig(name=args.model or str(profile.get("id")), endpoint=args.endpoint,
                      api_key=args.api_key)
    server, pilot = _probes(args, cfg)
    judge = _judge(args, cfg, standard)
    shape = _model_shape(args, server, profile)
    selected = _selected(args.legs)
    pid, engine = _endpoint_state(args, cfg)
    plan = plan_run(standard, legs, variables, profile, host, server, pilot, judge, selected,
                    model_shape=shape, serving_pid=pid, engine=engine)
    return plan, host, server, pilot


def _profile_from_args(args: argparse.Namespace) -> dict:
    """The profile TOML with its id filled in, or an empty one the probes can still name."""
    profile = load_toml(args.profile) if args.profile else {}
    fallback = args.model or (Path(args.profile).stem if args.profile else "model")
    profile.setdefault("id", fallback)
    return profile


def _model_shape(args: argparse.Namespace, server, profile: dict):
    """The shape comes from the model, not from the profile: the file the endpoint says it
    loaded, or the one named on the command line, or the profile as a last resort."""
    return resolve_model_shape(model_file=args.model_file, model_dir=args.model_dir,
                               props=getattr(server, "props", None), profile=profile)


def _selected(legs_arg: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in legs_arg.split(",") if item.strip())


def _endpoint_state(args: argparse.Namespace,
                    cfg: ModelConfig) -> tuple[int | None, EngineFacts | None]:
    """Who is serving the endpoint and which engine it is, unless probing was disabled."""
    if args.no_probe:
        return None, None
    # The process answering the endpoint is holding the model under test. Naming it lets the
    # fit credit those weights to this run instead of counting them as someone else's.
    # The engine is named by its probes, once, here: every stamp below reads from this plan.
    return serving_pid_for(args.endpoint), probe_engine(cfg)


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
