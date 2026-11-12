"""Rebuild a plan from run-plan.json.

A run directory has to be readable a day later without the host, the endpoint or the GPUs
that produced it: closing a run, re-checking the gate and writing the notes all read the
plan back off disk instead of re-probing.
"""
from __future__ import annotations

from dataclasses import fields

from scripts.lib.config import load_toml
from scripts.lib.fit import Fit
from scripts.lib.legs import Leg, load_legs
from scripts.lib.plan_run import Decision, Option, PlannedLeg, RunPlan, Skip
from scripts.lib.prereqs import Dependency

STANDARD_FILE = "standard.toml"


def _known(target: type, payload: dict) -> dict:
    """The payload's keys that the dataclass actually declares."""
    names = {item.name for item in fields(target)}
    return {key: value for key, value in payload.items() if key in names}


def _fit(payload: dict) -> Fit:
    return Fit(**_known(Fit, payload))


def _decision(payload: dict) -> Decision:
    """A decision row, translating the sheet's `standard` label back to the field name."""
    names = {item.name for item in fields(Decision)}
    data = {}
    for key, value in payload.items():
        target = "standard_value" if key == "standard" else key
        if target in names:
            data[target] = value
    data["options"] = tuple(Option(**_known(Option, item)) for item in payload.get("options", []))
    return Decision(**data)


def _dependencies(payload: list[dict]) -> tuple[Dependency, ...]:
    return tuple(Dependency(**_known(Dependency, item)) for item in payload)


def _plan_leg(payload: dict, catalog: dict[str, Leg]) -> PlannedLeg:
    leg = catalog.get(payload["id"])
    if leg is None:
        raise KeyError(f"leg {payload['id']!r} is not in the catalog")
    return PlannedLeg(leg=leg, cap=payload["cap"], ctx=payload["ctx"],
                      thinking=payload["thinking"], budget_s=payload["budget_s"],
                      case_timeout_s=int(payload.get("case_timeout_s", 0)),
                      dependencies=_dependencies(payload.get("dependencies", [])),
                      extra_args=tuple(payload.get("extra_args", [])),
                      note=payload.get("note", ""))


def plan_from_dict(payload: dict, legs: list[Leg] | None = None) -> RunPlan:
    """The plan as it was written, with the catalog supplying the leg definitions."""
    catalog = {leg.id: leg for leg in (legs or load_legs(load_toml(STANDARD_FILE)))}
    profile = payload.get("profile") or {}
    return RunPlan(
        standard_version=payload["standard_version"],
        model=payload["model"],
        fit=_fit(payload["fit"]),
        legs=tuple(_plan_leg(item, catalog) for item in payload.get("legs", [])),
        decisions=tuple(_decision(item) for item in payload.get("decisions", [])),
        skips=tuple(Skip(**_known(Skip, item)) for item in payload.get("skips", [])),
        warnings=tuple(payload.get("warnings", [])),
        stamp=payload.get("stamp", {}),
        standard_md5=payload.get("standard_md5", ""),
        profile_path=profile.get("path", ""),
        profile_md5=profile.get("md5", ""),
    )
