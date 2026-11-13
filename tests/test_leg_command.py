"""The command line one leg runs under: every declared argument reaches the entry.

A shared adapter entry (`scripts/adapters/ifeval_toolbench.py`) serves two legs and cannot
tell them apart from the flags the plan hands every leg, so the selector comes from the
catalog. This asserts the plan carries it and that the resulting argv parses in the entry's
own parser, which is the only authority on what that entry accepts.
"""
from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.adapters.ifeval_toolbench import parse_args as parse_adapter_args  # noqa: E402
from scripts.lib.legs import Leg  # noqa: E402
from scripts.lib.plan_run import PlannedLeg  # noqa: E402
from scripts.run_battery import leg_command  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_leg(**overrides) -> Leg:
    fields = dict(
        id="ifeval", name="IFEval", measures="nothing", scoring="rubric",
        harness="adapter", implementation="in_repo",
        entry="scripts/adapters/ifeval_toolbench.py", cap_var="caps.default",
        ctx_need=8192, thinking="primary_on", thinking_effect="none measured",
        prereqs=(), resume="done_marker", comparability_keys=("cap",),
        known_failures=("it can go wrong",), items=10, budget_floor=60,
    )
    fields.update(overrides)
    return Leg(**fields)  # type: ignore[arg-type]


def make_planned(args: tuple[str, ...]) -> PlannedLeg:
    return PlannedLeg(leg=make_leg(args=args), cap=4096, ctx=131072, thinking="on",
                      budget_s=600, case_timeout_s=600, extra_args=tuple(args))


def make_namespace() -> argparse.Namespace:
    return argparse.Namespace(model="m", endpoint="http://127.0.0.1:1/v1", api_key="k",
                              judge_model="", judge_endpoint="", output_dir="out")


class LegCommandTest(unittest.TestCase):
    def command(self, planned: PlannedLeg) -> list[str]:
        return leg_command(planned, make_namespace(), Path("out"), ROOT, "m")

    def test_a_declared_selector_reaches_the_entry(self):
        command = self.command(make_planned(("--leg", "toolbench")))
        self.assertEqual(command[-2:], ["--leg", "toolbench"])

    def test_a_leg_without_its_own_arguments_adds_none(self):
        command = self.command(make_planned(()))
        self.assertNotIn("--leg", command)

    def test_the_command_line_parses_in_the_entry_it_targets(self):
        command = self.command(make_planned(("--leg", "ifeval")))
        leg_id, leg_args, cli = parse_adapter_args(command[2:])
        self.assertEqual(leg_id, "ifeval")
        self.assertEqual(leg_args.model, "m")
        self.assertEqual(leg_args.cap, 4096)
        self.assertEqual(leg_args.thinking, "on")
        self.assertEqual(cli, "tool-eval-bench")

    def test_a_selector_the_entry_does_not_serve_is_refused_by_its_parser(self):
        command = self.command(make_planned(("--leg", "teleporter")))
        with self.assertRaises(SystemExit):
            parse_adapter_args(command[2:])


if __name__ == "__main__":
    unittest.main()
