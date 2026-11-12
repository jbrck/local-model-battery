#!/usr/bin/env python3
"""Validate the leg and variable catalogs against the frozen standard.

Usage: python3 scripts/check_catalogs.py [--json]
Exits non-zero and names every problem it finds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.lib.config import load_toml  # noqa: E402
from scripts.lib.legs import (  # noqa: E402
    load_legs,
    load_raw_legs,
    resolve_cap,
    validate_legs,
)
from scripts.lib.variables import (  # noqa: E402
    load_raw_variables,
    load_variables,
    validate_variables,
)

CATALOG_PATHS = ("standard.toml", "legs.toml", "variables.toml")


def _caps(legs: list, standard: dict, problems: list[str]) -> dict[str, int]:
    """Every leg's completion cap, naming a leg whose cap_var the standard lacks."""
    caps: dict[str, int] = {}
    for leg in legs:
        try:
            caps[leg.id] = resolve_cap(leg, standard)
        except KeyError as exc:
            problems.append(str(exc.args[0]))
    return caps


def _summary(standard: dict, legs: list, variables: list,
             caps: dict[str, int]) -> dict:
    """The counts a run report quotes back: versions, legs, variables, caps."""
    return {
        "standard_version": standard.get("standard_version"),
        "legs": len(legs),
        "variables": len(variables),
        "implemented": sum(1 for leg in legs if leg.implemented),
        "port_pending": sum(1 for leg in legs if not leg.implemented),
        "caps": caps,
    }


def collect_problems(paths: tuple[str, str, str] = CATALOG_PATHS) -> tuple[list[str], dict]:
    """Run every catalog check and return (problems, summary).

    The paths are injectable so a test can point this at deliberately broken
    catalogs without touching the ones the repo ships.
    """
    standard_path, legs_path, variables_path = paths
    problems: list[str] = []
    standard = load_toml(standard_path)
    legs = load_legs(standard, legs_path)
    variables = load_variables(variables_path)
    problems.extend(validate_legs(legs, load_raw_legs(legs_path)))
    problems.extend(validate_variables(variables, legs, load_raw_variables(variables_path)))
    caps = _caps(legs, standard, problems)
    return problems, _summary(standard, legs, variables, caps)


def _print_json(problems: list[str], summary: dict) -> None:
    print(json.dumps({"problems": problems, **summary}, indent=2))


def _print_text(problems: list[str], summary: dict) -> None:
    print(f"standard {summary['standard_version']}: {summary['legs']} legs "
          f"({summary['implemented']} in repo, {summary['port_pending']} pending), "
          f"{summary['variables']} variables")
    for leg_id, cap in summary["caps"].items():
        print(f"  {leg_id:16s} cap {cap}")
    for problem in problems:
        print(f"  PROBLEM: {problem}", file=sys.stderr)
    if not problems:
        print("catalogs OK")


def main(argv: list[str] | None = None,
         paths: tuple[str, str, str] = CATALOG_PATHS) -> int:
    parser = argparse.ArgumentParser(description="Validate the battery catalogs")
    parser.add_argument("--json", action="store_true", help="Emit the summary as JSON")
    args = parser.parse_args(argv)

    problems, summary = collect_problems(paths)
    if args.json:
        _print_json(problems, summary)
    else:
        _print_text(problems, summary)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
