"""Variable catalog: every knob that can change a run, its class and its measured effect."""
from __future__ import annotations

from dataclasses import dataclass

from scripts.lib.config import load_toml, missing_fields
from scripts.lib.legs import Leg

VARIABLES_FILE = "variables.toml"
VARIABLE_CLASSES = ("M", "C", "D")
CLASS_NAMES = {
    "M": "measurement",
    "C": "capacity",
    "D": "dependency",
}
ALL_LEGS = "all"

REQUIRED_FIELDS = (
    "id", "class", "applies_to", "stamp_key", "what_it_changes", "effect", "source",
    "decision_rule",
)


@dataclass(frozen=True)
class Variable:
    """One knob, with what changing it costs."""

    id: str
    variable_class: str
    applies_to: tuple[str, ...]
    stamp_key: str
    what_it_changes: str
    effect: str
    source: str
    decision_rule: str

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.variable_class]

    @property
    def is_measurement(self) -> bool:
        return self.variable_class == "M"

    def applies_to_leg(self, leg_id: str) -> bool:
        return ALL_LEGS in self.applies_to or leg_id in self.applies_to

    def to_dict(self) -> dict:
        return {
            "id": self.id, "class": self.variable_class, "class_name": self.class_name,
            "applies_to": list(self.applies_to), "stamp_key": self.stamp_key,
            "what_it_changes": self.what_it_changes, "effect": self.effect,
            "source": self.source, "decision_rule": self.decision_rule,
        }


def load_variables(path: str = VARIABLES_FILE) -> list[Variable]:
    """Load the variable catalog in declaration order."""
    variables = []
    for raw in load_toml(path).get("variable", []):
        applies = raw.get("applies_to", ALL_LEGS)
        if isinstance(applies, str):
            applies = [applies]
        variables.append(Variable(
            id=raw.get("id", ""),
            variable_class=raw.get("class", ""),
            applies_to=tuple(applies),
            stamp_key=raw.get("stamp_key", ""),
            what_it_changes=raw.get("what_it_changes", ""),
            effect=raw.get("effect", ""),
            source=raw.get("source", ""),
            decision_rule=raw.get("decision_rule", ""),
        ))
    return variables


def variables_by_id(variables: list[Variable]) -> dict[str, Variable]:
    return {variable.id: variable for variable in variables}


def variables_for_leg(variables: list[Variable], leg_id: str) -> list[Variable]:
    return [variable for variable in variables if variable.applies_to_leg(leg_id)]


def measurement_variables(variables: list[Variable]) -> list[Variable]:
    return [variable for variable in variables if variable.is_measurement]


def load_raw_variables(path: str = VARIABLES_FILE) -> list[dict]:
    return load_toml(path).get("variable", [])


# The three knobs every leg inherits from the standard. A leg whose cap, ctx or thinking
# mode is not named by some variable has an unexplained knob.
CORE_LEG_VARIABLES = ("cap", "ctx", "thinking")

# Comparability keys that are covered by the core variables rather than a dedicated entry.
CORE_COMPARABILITY_KEYS = (
    "cap", "ctx", "thinking", "seed", "items_sha", "temperature", "kv_type",
)


def _check_variable(variable: Variable, seen: set[str], leg_ids: set[str]) -> list[str]:
    problems = []
    if variable.id in seen:
        problems.append(f"duplicate variable id {variable.id!r}")
    if variable.variable_class not in VARIABLE_CLASSES:
        problems.append(f"{variable.id}: class {variable.variable_class!r} not in {VARIABLE_CLASSES}")
    if not variable.effect:
        problems.append(f"{variable.id}: needs a measured effect or a stated absence of one")
    if not variable.decision_rule:
        problems.append(f"{variable.id}: needs a decision rule for the planner to offer")
    problems.extend(
        f"{variable.id}: applies_to {leg_id!r} is not a known leg"
        for leg_id in variable.applies_to
        if leg_id != ALL_LEGS and leg_id not in leg_ids
    )
    return problems


def _check_leg_knobs(leg: Leg, variables: list[Variable]) -> list[str]:
    """Every core knob and every extra comparability key must have a variable behind it."""
    scoped = variables_for_leg(variables, leg.id)
    named = {variable.id for variable in scoped}
    stamp_keys = {variable.stamp_key for variable in scoped}
    problems = [
        f"leg {leg.id}: knob {knob!r} is not named by any variable"
        for knob in CORE_LEG_VARIABLES
        if knob not in named
    ]
    problems.extend(
        f"leg {leg.id}: comparability key {key!r} has no variable with that stamp_key"
        for key in leg.comparability_keys
        if key not in CORE_COMPARABILITY_KEYS and key not in stamp_keys
    )
    return problems


def validate_variables(variables: list[Variable], legs: list[Leg],
                       raw: list[dict] | None = None) -> list[str]:
    """Return a list of problems. Empty means the catalog is usable."""
    problems: list[str] = []
    seen: set[str] = set()
    leg_ids = {leg.id for leg in legs}
    for variable in variables:
        problems.extend(_check_variable(variable, seen, leg_ids))
        seen.add(variable.id)
    for leg in legs:
        problems.extend(_check_leg_knobs(leg, variables))
    for entry in raw or []:
        problems.extend(
            f"variable {entry.get('id', '?')}: missing field {name!r}"
            for name in missing_fields(entry, REQUIRED_FIELDS)
        )
    return problems
