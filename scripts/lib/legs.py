"""Leg catalog: which benchmarks the battery can run, and what each one needs."""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

from scripts.lib.config import get_dotted, load_toml, missing_fields

LEGS_FILE = "legs.toml"
LEG_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
# An interpreter is named by the environment variable that holds its path, and that name is
# also the key in the standard's `[interpreters]` table. Uppercase keeps them recognisable.
INTERPRETER_ENV_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
IMPORT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
HARNESS_KINDS = ("native", "adapter")
IMPLEMENTATIONS = ("in_repo", "port_pending")
THINKING_MODES = ("primary_on", "primary_off", "required_on", "required_off", "n/a")
RESUME_MODES = ("done_marker", "partial_jsonl", "none")

# Dependency names a leg may declare in its `prereqs` list. The planner maps each one to a
# probe result, so a name that is not here has no way to be checked.
KNOWN_DEPENDENCIES = (
    "docker_sandbox",
    "benchlocal_cli",
    "tool_eval_bench",
    "lcb_source",
    "lcb_data",
    "nltk_data",
    "judge_endpoint",
    "interpreter_imports",
)

REQUIRED_FIELDS = (
    "id", "name", "measures", "scoring", "harness", "implementation", "entry",
    "cap_var", "ctx_need", "thinking", "thinking_effect", "prereqs", "resume",
    "comparability_keys", "known_failures",
)


@dataclass(frozen=True)
class Leg:
    """One benchmark leg, with the frozen counts joined in from the standard."""

    id: str
    name: str
    measures: str
    scoring: str
    harness: str
    implementation: str
    entry: str
    cap_var: str
    ctx_need: int
    thinking: str
    thinking_effect: str
    prereqs: tuple[str, ...]
    resume: str
    comparability_keys: tuple[str, ...]
    known_failures: tuple[str, ...]
    gate: bool = False
    items: int = 0
    budget_floor: int = 0
    data: str = ""
    interpreter: str = ""
    interpreter_env: str = ""
    interpreter_imports: tuple[str, ...] = ()

    @property
    def implemented(self) -> bool:
        """True when the entry script exists in this repo."""
        return self.implementation == "in_repo"

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "measures": self.measures,
            "scoring": self.scoring, "harness": self.harness,
            "implementation": self.implementation, "entry": self.entry,
            "cap_var": self.cap_var, "ctx_need": self.ctx_need,
            "thinking": self.thinking, "thinking_effect": self.thinking_effect,
            "prereqs": list(self.prereqs), "resume": self.resume,
            "comparability_keys": list(self.comparability_keys),
            "known_failures": list(self.known_failures), "gate": self.gate,
            "items": self.items, "budget_floor": self.budget_floor, "data": self.data,
            "interpreter": self.interpreter,
            "interpreter_env": self.interpreter_env,
            "interpreter_imports": list(self.interpreter_imports),
        }


def load_legs(standard: dict, path: str = LEGS_FILE) -> list[Leg]:
    """Load the catalog, in declaration order, with standard values joined in."""
    floors: dict[str, int] = get_dotted(standard, "budget.floors", {}) or {}
    items: dict[str, int] = get_dotted(standard, "budget.items", {}) or {}
    legs = []
    for raw in load_toml(path).get("leg", []):
        leg_id = raw.get("id", "")
        legs.append(Leg(
            id=leg_id,
            name=raw.get("name", ""),
            measures=raw.get("measures", ""),
            scoring=raw.get("scoring", ""),
            harness=raw.get("harness", ""),
            implementation=raw.get("implementation", ""),
            entry=raw.get("entry", ""),
            cap_var=raw.get("cap_var", ""),
            ctx_need=int(raw.get("ctx_need", 0)),
            thinking=raw.get("thinking", ""),
            thinking_effect=raw.get("thinking_effect", ""),
            prereqs=tuple(raw.get("prereqs", ())),
            resume=raw.get("resume", ""),
            comparability_keys=tuple(raw.get("comparability_keys", ())),
            known_failures=tuple(raw.get("known_failures", ())),
            gate=bool(raw.get("gate", False)),
            items=int(items.get(leg_id, 0)),
            budget_floor=int(floors.get(leg_id, 0)),
            data=str(raw.get("data", "")),
            interpreter=str(raw.get("interpreter", "")),
            interpreter_env=str(raw.get("interpreter_env", "")),
            interpreter_imports=tuple(raw.get("interpreter_imports", ())),
        ))
    return legs


def legs_by_id(legs: list[Leg]) -> dict[str, Leg]:
    return {leg.id: leg for leg in legs}


def resolve_cap(leg: Leg, standard: dict) -> int:
    """The completion cap this leg runs at: per-leg override, else the leg's cap_var."""
    override = get_dotted(standard, f"caps.by_leg.{leg.id}")
    if override is not None:
        return int(override)
    value = get_dotted(standard, leg.cap_var)
    if value is None:
        raise KeyError(f"leg {leg.id}: cap_var {leg.cap_var!r} is not in the standard")
    return int(value)


def load_raw_legs(path: str = LEGS_FILE) -> list[dict]:
    """The catalog as written, for validation before any field coercion."""
    return load_toml(path).get("leg", [])


ENUM_FIELDS = (
    ("harness", HARNESS_KINDS),
    ("implementation", IMPLEMENTATIONS),
    ("thinking", THINKING_MODES),
    ("resume", RESUME_MODES),
)


def _check_identity(leg: Leg, seen: set[str]) -> list[str]:
    problems = []
    if not LEG_ID_PATTERN.match(leg.id):
        problems.append(f"leg id {leg.id!r} must be lowercase with underscores")
    if leg.id in seen:
        problems.append(f"duplicate leg id {leg.id!r}")
    return problems


def _check_enums(leg: Leg) -> list[str]:
    return [
        f"{leg.id}: {name} {getattr(leg, name)!r} not in {allowed}"
        for name, allowed in ENUM_FIELDS
        if getattr(leg, name) not in allowed
    ]


def _check_shape(leg: Leg) -> list[str]:
    problems = []
    if leg.ctx_need <= 0:
        problems.append(f"{leg.id}: ctx_need must be positive")
    if leg.items <= 0:
        problems.append(f"{leg.id}: no item count in the standard's budget.items")
    if not leg.entry:
        problems.append(f"{leg.id}: needs an entry script")
    if not leg.comparability_keys:
        problems.append(f"{leg.id}: needs at least one comparability key")
    if not leg.known_failures:
        problems.append(f"{leg.id}: needs at least one known failure mode")
    return problems


def _check_prereqs(leg: Leg) -> list[str]:
    return [
        f"{leg.id}: unknown prerequisite {dependency!r}"
        for dependency in leg.prereqs
        if dependency not in KNOWN_DEPENDENCIES
    ]


def _check_interpreter_path(leg: Leg) -> list[str]:
    """A borrowed interpreter is named by an absolute path, never a relative one.

    The orchestrator runs legs from its own working directory, so a relative interpreter
    would resolve to whatever directory that happens to be.
    """
    if leg.interpreter and not leg.interpreter.startswith("/"):
        return [f"{leg.id}: interpreter {leg.interpreter!r} must be an absolute path"]
    return []


def _check_interpreter_env(leg: Leg) -> list[str]:
    """The variable name has to be a variable name, and imports need one to resolve against."""
    problems = []
    if leg.interpreter_env and not INTERPRETER_ENV_PATTERN.match(leg.interpreter_env):
        problems.append(f"{leg.id}: interpreter_env {leg.interpreter_env!r} must be an "
                        f"environment variable name like LCB_PYTHON")
    if leg.interpreter_imports and not leg.interpreter_env:
        problems.append(f"{leg.id}: interpreter_imports needs an interpreter_env to resolve "
                        f"against")
    return problems


def _check_import_names(leg: Leg) -> list[str]:
    """Every declared import has to be a real dotted module name."""
    return [
        f"{leg.id}: interpreter_imports entry {name!r} is not a module name"
        for name in leg.interpreter_imports
        if not IMPORT_NAME_PATTERN.match(name)
    ]


def _check_interpreter(leg: Leg) -> list[str]:
    """A declared interpreter is how a leg borrows another venv's dependencies."""
    problems = []
    problems.extend(_check_interpreter_path(leg))
    problems.extend(_check_interpreter_env(leg))
    problems.extend(_check_import_names(leg))
    return problems


@dataclass(frozen=True)
class Interpreter:
    """Which python runs a leg, and where that choice came from.

    A leg that imports another project's helper package cannot assume the orchestrator's
    interpreter has it, and it cannot assume the other venv sits where it sat on the machine
    the catalog was written on. So the leg names the environment variable that holds the path,
    the operator's standard supplies the machine-local default, and the fallback is the
    interpreter already running the battery.
    """

    path: str
    source: str
    env_name: str = ""
    imports: tuple[str, ...] = ()

    @property
    def borrowed(self) -> bool:
        """True when the leg runs under an interpreter other than the orchestrator's own."""
        return self.path != sys.executable

    def to_dict(self) -> dict:
        return {"path": self.path, "source": self.source, "env_name": self.env_name,
                "imports": list(self.imports), "borrowed": self.borrowed}


def resolve_interpreter(leg: Leg, standard: dict) -> Interpreter:
    """The interpreter for this leg: the environment variable, then the standard, then this one.

    The environment variable wins because it is the only choice that can be right on a machine
    other than the one the catalog was written on. A variable that is set but wrong is returned
    as-is rather than falling through, because a silently ignored path is harder to find than a
    failed import.
    """
    if leg.interpreter_env:
        from_env = os.environ.get(leg.interpreter_env, "")
        if from_env:
            return Interpreter(path=from_env, source="environment", env_name=leg.interpreter_env,
                               imports=leg.interpreter_imports)
        declared_in_standard = get_dotted(standard, f"interpreters.{leg.interpreter_env}")
        if declared_in_standard:
            return Interpreter(path=str(declared_in_standard), source="standard",
                               env_name=leg.interpreter_env, imports=leg.interpreter_imports)
    if leg.interpreter:
        return Interpreter(path=leg.interpreter, source="declared",
                           imports=leg.interpreter_imports)
    return Interpreter(path=sys.executable, source="this interpreter",
                       env_name=leg.interpreter_env, imports=leg.interpreter_imports)


def interpreter_hint(leg: Leg, choice: Interpreter) -> str:
    """What the operator has to set for this leg's helpers to be importable."""
    if choice.env_name:
        return (f"set {choice.env_name} or add interpreters.{choice.env_name} to the standard")
    return f"declare an interpreter_env on {leg.id}"


def validate_legs(legs: list[Leg], raw: list[dict] | None = None) -> list[str]:
    """Return a list of problems. Empty means the catalog is usable."""
    problems: list[str] = []
    seen: set[str] = set()
    for leg in legs:
        problems.extend(_check_identity(leg, seen))
        seen.add(leg.id)
        problems.extend(_check_enums(leg))
        problems.extend(_check_shape(leg))
        problems.extend(_check_prereqs(leg))
        problems.extend(_check_interpreter(leg))
    for entry in raw or []:
        problems.extend(
            f"leg {entry.get('id', '?')}: missing field {name!r}"
            for name in missing_fields(entry, REQUIRED_FIELDS)
        )
    return problems
