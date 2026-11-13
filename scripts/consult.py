#!/usr/bin/env python3
"""Consult: print the brief an agent uses to set a battery run up with the user.

    python3 scripts/consult.py [--endpoint URL] [--api-key K] [--model-file PATH]
                               [--model-dir DIR] [--engine ID] [--use-case ID]
                               [--judge-model ID] [--out PATH]

The brief is markdown on stdout, and optionally written to --out. It reads the host, the
model and (when something answers) the engine, then states the facts, the decisions and the
exact run command. Nothing is decided on the user's behalf, and no section is omitted for
being unknown: an unknown prints as unknown with what would resolve it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.lib import consult  # noqa: E402
from scripts.lib.consult import BriefInputs, MissingKnowledge  # noqa: E402
from scripts.lib.env_probe import probe_host  # noqa: E402
from scripts.lib.model import ModelConfig  # noqa: E402
from scripts.lib.model_probe import resolve_model_shape  # noqa: E402

DEFAULT_ENDPOINT = "http://127.0.0.1:11434/v1"


def build_parser() -> argparse.ArgumentParser:
    """The consult flags. There is no --model: the endpoint names what it serves."""
    parser = argparse.ArgumentParser(description="Print the consult brief for a battery run.")
    parser.add_argument("--endpoint", default="",
                        help="OpenAI-compatible base URL; empty means nothing is up yet")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--model-file", default="",
                        help="the GGUF to read the shape from")
    parser.add_argument("--model-dir", default="",
                        help="a HF model directory when the weights are safetensors")
    parser.add_argument("--engine", default="",
                        help="engines.toml row id when nothing is up yet")
    parser.add_argument("--use-case", default="",
                        help="use-cases.toml row id the run will answer")
    parser.add_argument("--judge-model", default="",
                        help="the judge you have picked; named in sections 6 and 8")
    parser.add_argument("--out", default="", help="also write the brief to this path")
    return parser


def gather(args: argparse.Namespace) -> BriefInputs:
    """Read the host, the model and the endpoint, and hand the brief its inputs."""
    host = probe_host()
    shape = resolve_model_shape(model_file=args.model_file, model_dir=args.model_dir)
    engine_facts = None
    model_ids: tuple[str, ...] = ()
    model_name = ""
    endpoint = ""
    if args.endpoint:
        cfg = ModelConfig(name="", endpoint=args.endpoint, api_key=args.api_key)
        ids, error = consult.fetch_model_ids(cfg)
        if error:
            print(f"note: the endpoint did not answer its model list ({error}); "
                  "the brief renders the no-endpoint sections", file=sys.stderr)
        else:
            endpoint = args.endpoint
            model_ids = ids
            model_name = ids[0] if ids else ""
            engine_facts = consult.probe_engine_or_none(cfg)
    return BriefInputs(
        host=host,
        shape=shape,
        engine_facts=engine_facts,
        engine_id=args.engine,
        use_case_id=args.use_case,
        judge_pick=args.judge_model,
        endpoint=endpoint,
        model_name=model_name,
        model_file=args.model_file,
        model_dir=args.model_dir,
        model_ids=model_ids,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        brief = consult.render_brief(gather(args))
    except MissingKnowledge as error:
        print(f"consult: {error}", file=sys.stderr)
        return 2
    print(brief)
    if args.out:
        Path(args.out).write_text(brief, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
