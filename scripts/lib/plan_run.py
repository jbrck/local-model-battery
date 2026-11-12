"""Planning: turn a model, a host and an endpoint into a runnable battery and its decisions.

The plan is the contract between the operator and the run. Everything the standard pins is
either honoured or reported as a deviation, and every variable that a run has to move --
because the weights barely fit, because the window does not reach, because the engine
ignores the thinking flag -- arrives here as an option with its consequence attached.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from scripts.lib.config import get_dotted
from scripts.lib.env_probe import HostFacts
from scripts.lib.fit import (
    BYTES_PER_GB,
    TIER_MANAGED,
    Fit,
    compute_fit,
    kv_bytes_per_token,
    parse_kv_bits,
)
from scripts.lib.legs import Leg, resolve_cap
from scripts.lib.pilot import PilotResult
from scripts.lib.prereqs import (
    Dependency,
    Judge,
    resolve_prereqs,
    thinking_available,
    unavailable,
    verify_bundled_data,
)
from scripts.lib.server_probe import ServerFacts
from scripts.lib.variables import Variable, variables_by_id

DEFAULT_PROMPT_HEADROOM = 16384
KIND_MEASUREMENT = "measurement"
KIND_CAPACITY = "capacity"
KIND_DEPENDENCY = "dependency"
STANDARD = "standard"
PROFILE = "profile"
HOST = "host"
ENGINE = "engine"

# Where each variable's standard value lives. A variable that is neither pinned here nor
# computed by the planner has no defined value, which the plan reports as a warning.
STANDARD_PATHS = {
    "ctx": "serve.ctx",
    "cap": "caps.default",
    "kv_type": "serve.kv_k",
    "slots": "serve.slots",
    "kv_unified": "serve.kv_unified",
    "flash_attn": "serve.flash_attn",
    "seed": "sampling.seed",
    "temperature": "sampling.temperature",
    "batch_size": "limits.lcb_workers",
    "case_timeout": "timeouts.hermesagent20_case_floor",
    "request_timeout": "timeouts.lcb_request_floor",
    "items": "limits",
    "judge_model": None,
    "thinking": None,
    "offload_placement": None,
    "engine_version": None,
    "weight_quant": None,
    "host_vram": None,
    "host_ram": None,
    "power_limit": None,
    "spec_decode": None,
    "draft_model": None,
    "context_limit": None,
    "tool_loop_limit": None,
    "mmlu_sampling": None,
}

THINKING_STANDARD = "on"
THINKING_CHOICES = ("on", "off", "native")


@dataclass(frozen=True)
class Footprint:
    """A model's size on disk and the KV it needs per token, or why that is unknown."""

    weights_bytes: int
    kv_bytes_per_token: int
    known: bool
    missing: tuple[str, ...] = ()
    kv_type: str = ""

    def to_dict(self) -> dict:
        return {
            "weights_bytes": self.weights_bytes,
            "weights_gb": round(self.weights_bytes / BYTES_PER_GB, 2),
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "kv_type": self.kv_type,
            "known": self.known,
            "missing": list(self.missing),
        }


@dataclass(frozen=True)
class Option:
    """One way forward, with what it costs."""

    label: str
    consequence: str
    recommended: bool = False

    def to_dict(self) -> dict:
        return {"label": self.label, "consequence": self.consequence,
                "recommended": self.recommended}


@dataclass(frozen=True)
class Decision:
    """A knob: its value here, the standard's value, and whether the operator must pick."""

    variable: str
    kind: str
    chosen: str
    standard_value: str
    consequence: str
    source: str = STANDARD
    options: tuple[Option, ...] = ()
    needs_operator: bool = False

    @property
    def is_deviation(self) -> bool:
        """Only a measurement variable can be off-standard: capacity is whatever the host is."""
        if self.kind != KIND_MEASUREMENT or self.standard_value in ("", "n/a"):
            return False
        return self.chosen != self.standard_value

    def to_dict(self) -> dict:
        return {
            "variable": self.variable, "kind": self.kind, "chosen": self.chosen,
            "standard": self.standard_value, "deviation": self.is_deviation,
            "source": self.source, "consequence": self.consequence,
            "needs_operator": self.needs_operator,
            "options": [option.to_dict() for option in self.options],
        }


@dataclass(frozen=True)
class Skip:
    """A leg that will not run, and the reason it will not."""

    leg: str
    reason: str

    def to_dict(self) -> dict:
        return {"leg": self.leg, "reason": self.reason}


@dataclass(frozen=True)
class PlannedLeg:
    """One leg as it will actually run on this host."""

    leg: Leg
    cap: int
    ctx: int
    thinking: str
    budget_s: int
    dependencies: tuple[Dependency, ...] = ()
    note: str = ""

    @property
    def id(self) -> str:
        return self.leg.id

    @property
    def is_gate(self) -> bool:
        return self.leg.gate

    def to_dict(self) -> dict:
        return {
            "id": self.leg.id, "name": self.leg.name, "cap": self.cap, "ctx": self.ctx,
            "thinking": self.thinking, "budget_s": self.budget_s, "note": self.note,
            "gate": self.leg.gate, "items": self.leg.items,
            "dependencies": [item.to_dict() for item in self.dependencies],
        }


@dataclass(frozen=True)
class RunPlan:
    """The decision sheet, plus enough machine-readable structure to stamp a manifest."""

    standard_version: str
    model: str
    fit: Fit
    legs: tuple[PlannedLeg, ...]
    decisions: tuple[Decision, ...]
    skips: tuple[Skip, ...] = ()
    warnings: tuple[str, ...] = ()
    stamp: dict = field(default_factory=dict)

    @property
    def needs_operator(self) -> tuple[Decision, ...]:
        return tuple(item for item in self.decisions if item.needs_operator)

    @property
    def deviations(self) -> tuple[Decision, ...]:
        return tuple(item for item in self.decisions if item.is_deviation)

    @property
    def total_budget_s(self) -> int:
        return sum(leg.budget_s for leg in self.legs)

    def leg(self, leg_id: str) -> PlannedLeg | None:
        for planned in self.legs:
            if planned.id == leg_id:
                return planned
        return None

    def to_dict(self) -> dict:
        return {
            "standard_version": self.standard_version,
            "model": self.model,
            "fit": self.fit.to_dict(),
            "legs": [leg.to_dict() for leg in self.legs],
            "skips": [skip.to_dict() for skip in self.skips],
            "warnings": list(self.warnings),
            "decisions": [item.to_dict() for item in self.decisions],
            "needs_operator": [item.variable for item in self.needs_operator],
            "total_budget_s": self.total_budget_s,
            "stamp": self.stamp,
        }

    def render(self) -> str:
        """The sheet the setup skill reads out loud, one decision at a time."""
        lines = [
            f"Battery {self.standard_version} plan -- {self.model}",
            "=" * 60,
            f"fit      {self.fit.tier} {self.fit.verdict}: {self.fit.reason}",
            f"legs     {len(self.legs)} runnable, {len(self.skips)} skipped, "
            f"{_hours(self.total_budget_s)} of worst-case budget",
        ]
        if self.deviations:
            lines.append("off-standard: " + ", ".join(
                f"{item.variable}={item.chosen}" for item in self.deviations))
        if self.skips:
            lines.append("")
            lines.append("Skipped")
            lines.extend(f"  - {skip.leg}: {skip.reason}" for skip in self.skips)
        pending = self.needs_operator
        if pending:
            lines.append("")
            lines.append(f"Decisions that need you ({len(pending)})")
            for index, item in enumerate(pending, start=1):
                lines.append(f"  [{index}] {item.variable} ({item.kind}) -- {_state(item)}")
                lines.append(f"      {item.consequence}")
                for option in item.options:
                    mark = "*" if option.recommended else " "
                    lines.append(f"      {mark} {option.label}: {option.consequence}")
        lines.append("")
        lines.append("Legs")
        lines.extend(
            f"  {planned.id:<16} cap {planned.cap:<7} ctx {planned.ctx:<7} "
            f"think {planned.thinking:<6} {_hours(planned.budget_s)}"
            + (f"  {planned.note}" if planned.note else "")
            for planned in self.legs
        )
        if self.warnings:
            lines.append("")
            lines.append("Warnings")
            lines.extend(f"  ! {warning}" for warning in self.warnings)
        return "\n".join(lines)


def _state(item: Decision) -> str:
    """How a row reads on the sheet: what the standard asks, and what this run gets."""
    if item.kind == KIND_CAPACITY:
        return (f"assumed {item.chosen}" if item.chosen == item.standard_value
                else f"here {item.chosen}")
    return f"standard {item.standard_value}, here {item.chosen}"


def _hours(seconds: int) -> str:
    if seconds <= 0:
        return "no budget"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _number(value: object, default: float) -> float:
    """A config value as a float, or the default when it is absent or the wrong type."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _digit(value: object, default: int) -> int:
    """A config value as an int, for the constants that have to be whole."""
    return int(_number(value, default))


def model_footprint(profile: dict, standard: dict) -> Footprint:
    """Weights in bytes plus KV bytes per token, from the profile's model shape."""
    shape = get_dotted(profile, "model", {}) or {}
    kv_type = str(shape.get("kv_type") or get_dotted(standard, "serve.kv_k", "q8_0"))
    weights_gb = _number(profile.get("weights_gb"), 0.0)
    missing = [name for name in ("n_layers", "n_kv_heads", "head_dim")
               if not isinstance(shape.get(name), int)]
    if weights_gb <= 0:
        missing.insert(0, "weights_gb")
    if missing:
        return Footprint(weights_bytes=0, kv_bytes_per_token=0, known=False,
                         missing=tuple(missing), kv_type=kv_type)
    per_token = kv_bytes_per_token(_digit(shape.get("n_layers"), 0),
                                   _digit(shape.get("n_kv_heads"), 0),
                                   _digit(shape.get("head_dim"), 0), parse_kv_bits(kv_type))
    return Footprint(weights_bytes=int(weights_gb * BYTES_PER_GB),
                     kv_bytes_per_token=per_token, known=True, kv_type=kv_type)


def compute_plan_fit(profile: dict, standard: dict, host: HostFacts) -> tuple[Fit, Footprint]:
    """Fit this model on this host, counting whatever else is already holding VRAM."""
    footprint = model_footprint(profile, standard)
    preferred = _digit(get_dotted(standard, "serve.ctx", 0), 0)
    if not footprint.known:
        # Without a model shape there is no KV number and no honest window, so the placement
        # question goes to the engine instead of being answered with a made-up figure.
        return Fit(tier=TIER_MANAGED, vram_total_gb=host.vram_total_gb,
                   vram_free_gb=max(0.0, host.vram_total_gb - host.vram_used_gb),
                   weights_gb=0.0, vram_needed_gb=0.0, headroom_gb=0.0, max_ctx=preferred,
                   preferred_ctx=preferred, kv_bytes_per_token=0, estimate=True,
                   reason="model shape unknown (" + ", ".join(footprint.missing)
                          + "): placement and window are the engine's call"), footprint
    fit = compute_fit(weights_bytes=footprint.weights_bytes,
                      kv_per_token=footprint.kv_bytes_per_token, gpus=host.gpus,
                      ram_available_gb=host.ram_available_gb, preferred_ctx=preferred,
                      foreign_vram_used_gb=host.vram_used_gb)
    return fit, footprint


def budget_seconds(leg: Leg, cap: int, pilot: PilotResult, standard: dict) -> int:
    """The leg's budget: the larger of its floor and the pilot's extrapolation."""
    headroom = _number(get_dotted(standard, "budget.headroom", 1.25), 1.25)
    estimate = 0.0
    if pilot.usable and pilot.tps > 0:
        estimate = headroom * leg.items * cap / pilot.tps
    return int(max(leg.budget_floor, estimate))


def resolve_thinking(leg: Leg, requested: str, server: ServerFacts) -> tuple[str, str]:
    """(the mode this leg runs in, a note when the endpoint forces it)."""
    honors, why = thinking_available(server)
    if leg.thinking == "n/a":
        return "n/a", ""
    if leg.thinking == "required_off":
        return "off", "the harness needs the channel empty" if requested == "on" else ""
    if leg.thinking == "required_on":
        if honors:
            return "on", ""
        return "on", f"the endpoint ignores the thinking flag ({why}); stamp native-forced"
    if requested == "off":
        if honors:
            return "off", ""
        return "on", f"the endpoint ignores the thinking flag ({why}); stays in the on column"
    return "on", ""


def prompt_headroom(standard: dict) -> int:
    return _digit(get_dotted(standard, "caps.prompt_headroom"), DEFAULT_PROMPT_HEADROOM)


def plan_ctx(standard: dict, fit: Fit, footprint: Footprint,
             largest_cap: int) -> tuple[int, Decision | None]:
    """The window the run will use, and the decision behind it when it is not the standard's."""
    standard_ctx = _digit(get_dotted(standard, "serve.ctx", 0), 0)
    headroom = prompt_headroom(standard)
    needed = largest_cap + headroom
    if fit.estimate and fit.tier != "T4":
        return standard_ctx, Decision(
            variable="ctx", kind=KIND_CAPACITY,
            chosen=str(standard_ctx), standard_value=str(standard_ctx),
            consequence=f"placement is the engine's call here ({fit.reason}), so the window "
                        f"is a guess until /props reports it",
            source=ENGINE, needs_operator=True,
            options=(Option("serve, then read /props", "the run stamps the window the server "
                          "actually reports", recommended=True),))
    if fit.max_ctx >= needed:
        return standard_ctx, None
    options = (
        Option(f"run at {fit.max_ctx} tokens",
               f"every leg that needs {needed} or less still fits; stamp ctx={fit.max_ctx}, "
               f"off-standard", recommended=True),
        Option("keep the standard window and spill to RAM",
               "the window survives, throughput drops badly on a model this size"),
        Option("quantise the KV cache further",
               "buys window at the cost of KV numerics; stamp kv_type, off-standard"),
    )
    return fit.max_ctx, Decision(
        variable="ctx", kind=KIND_MEASUREMENT, chosen=str(fit.max_ctx),
        standard_value=str(standard_ctx),
        consequence=f"the standard asks for {standard_ctx} tokens and this host reaches "
                    f"{fit.max_ctx} with {footprint.kv_type} KV: {fit.reason}",
        source=HOST, options=options, needs_operator=True)


def plan_legs(legs: list[Leg], standard: dict, ctx: int, thinking_request: str,
              server: ServerFacts, host: HostFacts, judge: Judge | None,
              pilot: PilotResult) -> tuple[list[PlannedLeg], list[Skip], list[Decision]]:
    """Every selected leg, with its cap clamped to the window and its prerequisites resolved."""
    headroom = prompt_headroom(standard)
    planned: list[PlannedLeg] = []
    skips: list[Skip] = []
    decisions: list[Decision] = []
    for leg in legs:
        standard_cap = resolve_cap(leg, standard)
        cap = min(standard_cap, max(1, ctx - headroom))
        dependencies = resolve_prereqs(leg, standard, host, judge)
        missing = unavailable(dependencies)
        if missing:
            skips.append(Skip(leg.id, "; ".join(item.detail for item in missing)))
            decisions.append(Decision(
                variable=f"prereq:{leg.id}", kind=KIND_DEPENDENCY, chosen="skip",
                standard_value="run",
                consequence=f"{leg.id} is skipped here: {missing[0].detail}",
                source=HOST, needs_operator=True,
                options=(Option("install it and re-plan", missing[0].detail,
                                recommended=True),
                         Option("accept the skip", "the leg is absent from the manifest "
                                "instead of scoring zero"))))
            continue
        mode, note = resolve_thinking(leg, thinking_request, server)
        planned.append(PlannedLeg(leg=leg, cap=cap, ctx=ctx, thinking=mode,
                                  budget_s=budget_seconds(leg, cap, pilot, standard),
                                  dependencies=dependencies, note=note))
        clamped = cap != standard_cap
        options: tuple[Option, ...] = ()
        if clamped:
            options = (
                Option(f"accept the {cap} ceiling",
                       "runs that would have used more are cut short and score low",
                       recommended=True),
                Option("raise the window", "needs a host or a quant that fits it"),
            )
        decisions.append(Decision(
            variable=f"cap:{leg.id}", kind=KIND_MEASUREMENT, chosen=str(cap),
            standard_value=str(standard_cap),
            consequence=(f"{leg.id} cannot use its {standard_cap}-token cap inside a "
                         f"{ctx}-token window; the cap drops to {cap}" if clamped else
                         f"{leg.id} may generate {cap} tokens an answer, as the standard asks"),
            source=HOST if clamped else STANDARD, options=options, needs_operator=clamped))
        decisions.append(Decision(
            variable=f"harness_version:{leg.id}", kind=KIND_MEASUREMENT,
            chosen=leg.harness, standard_value=leg.harness,
            consequence=f"{leg.id} is scored by {leg.harness} at this commit; a harness bump "
                        f"voids the comparison with earlier runs",
            source=STANDARD))
    return planned, skips, decisions


@dataclass(frozen=True)
class _PlanInputs:
    """Everything the decision builders need, so none of them takes a parameter stampede."""

    standard: dict
    variables: list[Variable]
    profile: dict
    host: HostFacts
    server: ServerFacts
    pilot: PilotResult
    judge: Judge | None
    fit: Fit
    footprint: Footprint
    ctx: int
    ctx_asked: bool
    caps: dict[str, int]
    planned: tuple[PlannedLeg, ...]


def _engine_version(server: ServerFacts) -> str:
    props = server.props if isinstance(server.props, dict) else {}
    for key in ("build_info", "version", "model_path"):
        if props.get(key):
            return str(props[key])
    return "not reported"


def _placement(fit: Fit) -> str:
    if fit.tier == "T1":
        return "gpu_only"
    if fit.tier == "T2":
        return "gpu_plus_ram"
    return "engine_managed"


def capacity_decisions(inputs: _PlanInputs) -> list[Decision]:
    """What the host is made of. Stamped for comparison, never asked about."""
    host, fit = inputs.host, inputs.fit
    names = ", ".join(gpu.name for gpu in host.gpus) or "none reported"
    limits = [gpu.power_limit_w for gpu in host.gpus if gpu.power_limit_w]
    rows = (
        ("host_vram", f"{host.vram_total_gb:g} GB",
         f"{len(host.gpus)} GPU(s): {names}; {host.vram_used_gb:g} GB already in use", HOST),
        ("host_ram", f"{host.ram_total_gb or 0:g} GB",
         f"{host.ram_available_gb or 0:g} GB available to spill into", HOST),
        ("power_limit", f"{min(limits):g} W" if limits else "not reported",
         "the lowest card limit caps throughput for the whole run", HOST),
        ("offload_placement", _placement(fit), f"tier {fit.tier}: {fit.verdict}", HOST),
        ("engine_version", _engine_version(inputs.server),
         f"as reported by {inputs.server.served_model or 'the endpoint'}", ENGINE),
        ("weight_quant", str(inputs.profile.get("weight_quant") or "unknown"),
         "from the profile; two quants of one model are not the same entry", PROFILE),
        ("spec_decode", str(inputs.profile.get("spec_decode") or "off"),
         "draft-model decoding changes tokens per second and sometimes the ranking", PROFILE),
        ("draft_model", str(inputs.profile.get("draft_model") or "none"),
         "only meaningful when spec_decode is on", PROFILE),
    )
    return [
        Decision(variable=name, kind=KIND_CAPACITY, chosen=value, standard_value="n/a",
                 consequence=note, source=source)
        for name, value, note, source in rows
    ]


def _items_summary(standard: dict) -> str:
    """How many item sets this run draws from, bundled and fetched."""
    bundled = get_dotted(standard, "data.bundled", []) or []
    external = get_dotted(standard, "data.external", []) or []
    return f"{len(bundled)} bundled, {len(external)} external item sets"


def _pinned(standard: dict, name: str, fallback: str = "") -> str:
    path = STANDARD_PATHS.get(name)
    if not path:
        return fallback
    value = get_dotted(standard, path)
    if isinstance(value, dict):
        return ", ".join(f"{key} {val}" for key, val in value.items())
    return "n/a" if value is None else str(value)


def measurement_decisions(inputs: _PlanInputs) -> list[Decision]:
    """The knobs that change a score. Each carries the standard's value next to this one."""
    standard = inputs.standard
    pinned_kv = _pinned(standard, "kv_type", "q8_0")
    decisions = [
        _ctx_row(inputs),
        _kv_decision(inputs, pinned_kv),
        _thinking_decision(inputs),
        Decision("seed", KIND_MEASUREMENT, _pinned(standard, "seed"),
                 _pinned(standard, "seed"),
                 "every leg inherits it; a different seed is a different run", STANDARD),
        Decision("temperature", KIND_MEASUREMENT, _pinned(standard, "temperature"),
                 _pinned(standard, "temperature"),
                 "greedy decoding; a nonzero value makes the same item score differently twice",
                 STANDARD),
        Decision("batch_size", KIND_MEASUREMENT, _pinned(standard, "batch_size"),
                 _pinned(standard, "batch_size"),
                 "LiveCodeBench workers; changes wall clock, not the score", STANDARD),
        Decision("request_timeout", KIND_MEASUREMENT, _pinned(standard, "request_timeout"),
                 _pinned(standard, "request_timeout"),
                 "a slow model that trips this looks like a wrong answer", STANDARD),
        Decision("items", KIND_MEASUREMENT, "the standard's limits table",
                 "the standard's limits table",
                 f"mmlu samples {get_dotted(standard, 'limits.mmlu')} of "
                 f"{get_dotted(standard, 'limits.mmlu')}+ items; every leg stamps the hash of "
                 f"the items it actually used", STANDARD),
        Decision("mmlu_sampling", KIND_MEASUREMENT,
                 str(get_dotted(standard, "limits.mmlu_even_per_subject", False)).lower(),
                 str(get_dotted(standard, "limits.mmlu_even_per_subject", False)).lower(),
                 "subject-balanced sampling scores differently from a flat sample", STANDARD),
        Decision("item_set", KIND_MEASUREMENT, _items_summary(standard),
                 _items_summary(standard),
                 "the same items for every model on every host; each leg stamps their hash",
                 STANDARD),
        Decision("flash_attn", KIND_MEASUREMENT, _pinned(standard, "flash_attn"),
                 _pinned(standard, "flash_attn"),
                 "buys throughput and some window; a kernel change can move numerics",
                 STANDARD),
        Decision("slots", KIND_MEASUREMENT, _pinned(standard, "slots"), _pinned(standard, "slots"),
                 f"{_pinned(standard, 'slots')} concurrent slots split the same KV budget, so "
                 f"more slots means a smaller window or a smaller cache", STANDARD),
        _case_timeout_decision(inputs),
        _harness_limit_decision("context_limit", inputs, "the harness stops the case when the "
                                "context fills instead of truncating silently"),
        _harness_limit_decision("tool_loop_limit", inputs, "caps runaway tool loops"),
        _judge_decision(inputs),
    ]
    return [row for row in decisions if row is not None]


def _ctx_row(inputs: _PlanInputs) -> Decision | None:
    """The window, unless plan_ctx already put the question to the operator."""
    if inputs.ctx_asked or inputs.ctx != _pinned_int(inputs.standard, "ctx"):
        return None
    return Decision("ctx", KIND_MEASUREMENT, str(inputs.ctx), _pinned(inputs.standard, "ctx"),
                    f"the window every leg shares; {prompt_headroom(inputs.standard)} tokens of "
                    f"it are reserved for the prompt", STANDARD)


def _pinned_int(standard: dict, name: str) -> int:
    return _digit(get_dotted(standard, STANDARD_PATHS.get(name) or "", 0), 0)


def _kv_decision(inputs: _PlanInputs, pinned: str) -> Decision:
    chosen = inputs.footprint.kv_type or pinned
    options: tuple[Option, ...] = ()
    if chosen != pinned:
        options = (
            Option(f"accept {chosen} KV", "the window fits; the KV cache is less exact, so "
                   "stamp it off-standard", recommended=True),
            Option(f"keep {pinned} KV", "exact KV, smaller window, some legs cannot run"),
        )
    return Decision("kv_type", KIND_MEASUREMENT, chosen, pinned,
                    f"{chosen} KV costs {inputs.footprint.kv_bytes_per_token} bytes per token "
                    f"per sequence" if inputs.footprint.known else "KV type from the profile",
                    PROFILE if chosen != pinned else STANDARD, options=options,
                    needs_operator=bool(options))


def _thinking_decision(inputs: _PlanInputs) -> Decision:
    requested = str(inputs.profile.get("thinking") or THINKING_STANDARD)
    honors, why = thinking_available(inputs.server)
    options: tuple[Option, ...] = ()
    if not honors:
        options = (
            Option("run in the on column", f"the endpoint ignores the switch ({why}); stamp "
                   "thinking=native-forced", recommended=True),
            Option("switch endpoints", "point the run at a server that honours the flag"),
        )
    return Decision("thinking", KIND_MEASUREMENT, requested if honors else "native",
                    THINKING_STANDARD,
                    f"the flag {'works' if honors else 'does nothing'} here ({why}); thinking "
                    f"off costs MATH 7.8 and GPQA 28.3 points",
                    PROFILE if requested != THINKING_STANDARD else STANDARD,
                    options=options, needs_operator=bool(options))


def _case_timeout_decision(inputs: _PlanInputs) -> Decision:
    pinned = _digit(get_dotted(inputs.standard, "timeouts.hermesagent20_case_floor", 600), 600)
    cap = inputs.caps.get("hermesagent20")
    if cap is None:
        return Decision("case_timeout", KIND_MEASUREMENT, str(pinned), str(pinned),
                        "this run has no hermesagent20 leg", STANDARD)
    chosen = inputs.pilot.case_timeout_s(cap)
    options: tuple[Option, ...] = ()
    if chosen != pinned:
        options = (
            Option(f"{chosen}s per case", f"at {inputs.pilot.tps:.1f} tok/s a {cap}-token case "
                   f"needs the longer ceiling; without it the harness kills cases that were "
                   f"still generating", recommended=True),
            Option(f"keep {pinned}s", "slow models lose exactly the cases that need the most "
                   "tokens, which biases the leg against them"),
        )
    return Decision("case_timeout", KIND_MEASUREMENT, str(chosen), str(pinned),
                    f"{inputs.pilot.tps:.1f} tok/s measured, so a {cap}-token case is allowed "
                    f"{chosen}s", source=HOST if chosen != pinned else STANDARD,
                    options=options, needs_operator=bool(options))


def _harness_limit_decision(name: str, inputs: _PlanInputs, why: str) -> Decision:
    chosen = str(inputs.profile.get(name) or "harness default")
    return Decision(name, KIND_MEASUREMENT, chosen, "n/a",
                    f"{why}; only hermesagent20 reads it", PROFILE if chosen != "harness default"
                    else STANDARD)


def _judge_decision(inputs: _PlanInputs) -> Decision:
    judge = inputs.judge
    writing_selected = any(planned.id == "writing" for planned in inputs.planned)
    if judge and judge.configured:
        return Decision("judge_model", KIND_MEASUREMENT, str(judge.model), "n/a",
                        f"scored by {judge.model} at {judge.endpoint}; the judge must never be "
                        f"the model under test", PROFILE)
    return Decision("judge_model", KIND_MEASUREMENT, "none configured", "n/a",
                    "the writing suite can generate but cannot be scored", PROFILE,
                    options=(
                        Option("point the judge at another endpoint",
                               "writing joins the run with a score", recommended=True),
                        Option("run writing unscored", "the prose is kept, the number is absent"),
                    ), needs_operator=writing_selected)


def build_stamp(variables: list[Variable], decisions: list[Decision]) -> dict:
    """The comparability keys as the manifest will record them, per-leg rows included."""
    by_id = variables_by_id(variables)
    stamp: dict = {}
    for decision in decisions:
        base, _, tail = decision.variable.partition(":")
        variable = by_id.get(base)
        if variable is None:
            continue
        if tail:
            bucket = stamp.setdefault(variable.stamp_key, {})
            if isinstance(bucket, dict):
                bucket[tail] = decision.chosen
            continue
        stamp.setdefault(variable.stamp_key, decision.chosen)
    return stamp


def plan_warnings(inputs: _PlanInputs, decisions: list[Decision]) -> list[str]:
    """Everything the operator should know before spending hours on this run."""
    warnings = list(verify_bundled_data(inputs.standard))
    fit, host, pilot = inputs.fit, inputs.host, inputs.pilot
    if not inputs.footprint.known:
        warnings.append("model shape is incomplete ("
                        + ", ".join(inputs.footprint.missing)
                        + "): the window cannot be computed from weights and KV, only reported")
    if fit.window_is_limited:
        warnings.append(f"the standard window does not fit: {fit.reason}")
    elif fit.estimate:
        warnings.append(f"placement is the engine's call: {fit.reason}")
    if not fit.fits:
        warnings.append(f"this model will not run here: {fit.reason}")
    if not pilot.usable:
        warnings.append("the pilot never measured a rate, so every budget falls back to the "
                        "standard floor and the run may overrun")
    if host.compute_apps:
        warnings.append(f"{len(host.compute_apps)} other compute process(es) hold VRAM on this "
                        "host; the fit above already discounts it, but the figure will move "
                        "under load")
    covered = {decision.variable.partition(":")[0] for decision in decisions}
    for variable in inputs.variables:
        if not variable.is_measurement or variable.id in covered:
            continue
        if any(variable.applies_to_leg(planned.id) for planned in inputs.planned):
            warnings.append(f"measurement variable {variable.id} has no value on this plan")
    return warnings


def plan_run(standard: dict, legs: list[Leg], variables: list[Variable], profile: dict,
             host: HostFacts, server: ServerFacts, pilot: PilotResult,
             judge: Judge | None = None, selected: tuple[str, ...] = ()) -> RunPlan:
    """Build the plan: fit, window, per-leg caps, decisions, warnings and the stamp."""
    chosen = [leg for leg in legs if not selected or leg.id in selected]
    fit, footprint = compute_plan_fit(profile, standard, host)
    largest_cap = max((resolve_cap(leg, standard) for leg in chosen), default=0)
    ctx, ctx_decision = plan_ctx(standard, fit, footprint, largest_cap)
    request = str(profile.get("thinking") or THINKING_STANDARD)
    if request not in THINKING_CHOICES:
        request = THINKING_STANDARD
    planned, skips, leg_decisions = plan_legs(chosen, standard, ctx, request, server, host,
                                              judge, pilot)
    inputs = _PlanInputs(standard=standard, variables=variables, profile=profile, host=host,
                         server=server, pilot=pilot, judge=judge, fit=fit,
                         footprint=footprint, ctx=ctx, ctx_asked=ctx_decision is not None,
                         caps={item.id: item.cap for item in planned},
                         planned=tuple(planned))
    decisions = ([ctx_decision] if ctx_decision else []) + leg_decisions
    decisions.extend(capacity_decisions(inputs))
    decisions.extend(measurement_decisions(inputs))
    return RunPlan(
        standard_version=str(standard.get("standard_version", "unknown")),
        model=str(profile.get("id") or profile.get("name") or "unnamed-model"),
        fit=fit, legs=tuple(planned), decisions=tuple(decisions), skips=tuple(skips),
        warnings=tuple(plan_warnings(inputs, decisions)),
        stamp=build_stamp(variables, decisions),
    )
