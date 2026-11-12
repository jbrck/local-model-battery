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


def collect_problems() -> tuple[list[str], dict]:
    """Run every catalog check and return (problems, summary)."""
    problems: list[str] = []
    standard = load_toml("standard.toml")
    legs = load_legs(standard, "legs.toml")
    variables = load_variables("variables.toml")
    problems.extend(validate_legs(legs, load_raw_legs("legs.toml")))
    problems.extend(validate_variables(variables, legs, load_raw_variables("variables.toml")))

    caps: dict[str, int | str] = {}
    for leg in legs:
        try:
            caps[leg.id] = resolve_cap(leg, standard)
        except KeyError as exc:
            problems.append(str(exc))

    summary = {
        "standard_version": standard.get("standard_version"),
        "legs": len(legs),
        "variables": len(variables),
        "implemented": sum(1 for leg in legs if leg.implemented),
        "port_pending": sum(1 for leg in legs if not leg.implemented),
        "caps": caps,
    }
    return problems, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the battery catalogs")
    parser.add_argument("--json", action="store_true", help="Emit the summary as JSON")
    args = parser.parse_args()

    problems, summary = collect_problems()
    if args.json:
        print(json.dumps({"problems": problems, **summary}, indent=2))
    else:
        print(f"standard {summary['standard_version']}: {summary['legs']} legs "
              f"({summary['implemented']} in repo, {summary['port_pending']} pending), "
              f"{summary['variables']} variables")
        for leg_id, cap in summary["caps"].items():
            print(f"  {leg_id:16s} cap {cap}")
        for problem in problems:
            print(f"  PROBLEM: {problem}", file=sys.stderr)
        if not problems:
            print("catalogs OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
