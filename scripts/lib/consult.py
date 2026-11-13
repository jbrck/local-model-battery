"""The consult brief: what this host, model and engine are, and what to decide.

One command reads the host, the model and (when something is up) the engine, then renders
the brief an agent uses to set a run up with the user. The brief is a document, not a
decision: it names the four use cases, the five variables and the judge candidates, and it
never picks a use case, a cap or a judge on anyone's behalf. Every unknown is printed as
unknown together with what would resolve it, because a section that quietly goes missing
reads as if there was nothing to say.

The three knowledge tables (engines.toml, model-types.toml, use-cases.toml) are read by id,
so a corrected row changes the brief with no code change. A missing table or row fails
loudly instead of printing an empty section.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from scripts.lib.config import get_dotted, load_toml
from scripts.lib.env_probe import HostFacts
from scripts.lib.fit import Fit, compute_fit, parse_kv_bits
from scripts.lib.model import ModelConfig, get_json
from scripts.lib.model_probe import (
    SOURCE_GGUF,
    SOURCE_HF,
    SOURCE_UNKNOWN,
    ModelShape,
)
from scripts.lib.variables import load_variables, variables_by_id

ENGINES_FILE = "engines.toml"
MODEL_TYPES_FILE = "model-types.toml"
USE_CASES_FILE = "use-cases.toml"
STANDARD_FILE = "standard.toml"
VARIABLES_FILE = "variables.toml"

ENGINE_TABLE_KEY = "engine"
MODEL_TYPE_TABLE_KEY = "model_type"
USE_CASE_TABLE_KEY = "use_case"

DEFAULT_PREFERRED_CTX = 131072

# The five decision variables the brief puts to the user, mapped to their rows in
# variables.toml. The drafter is measured there as spec_decode, concurrency as the slot
# count, and the quant as weight_quant; the brief uses the operator's words for them.
DECISION_VARIABLES = (
    ("cap", "cap"),
    ("thinking", "thinking"),
    ("drafter", "spec_decode"),
    ("concurrency", "slots"),
    ("quant", "weight_quant"),
)

# Weight formats the engine rows themselves state. llama.cpp's smoke reads a .gguf
# model_path off /props and ollama manages GGUF models of its own; vllm "loads
# safetensors, not GGUF", mlx "loads MLX weights, not GGUF" from a model directory, and
# exllama serves its own quantization format, also not GGUF. Rows that say nothing about
# format (sglang, strata) stay candidates for either shape. The proxy row is not an
# engine and is never a candidate.
PROXY_ENGINE = "proxy"
GGUF_ENGINES = ("llama.cpp", "ollama")

# A declared window at or above the standard's serve window reads as long context.
LONG_CONTEXT_TOKENS = DEFAULT_PREFERRED_CTX

# Evidence in the probed notes that some layers hold no growing cache, which is what the
# hybrid_recurrent row's tells describe.
HYBRID_MARKERS = ("recurrent", "linear-attention", "state-space")

# A quant tag in the file or directory name, in the shapes the reference collection uses:
# Q4_K_M, Q8_0, IQ3_XXS, F16, BF16, EXL3.
QUANT_PATTERN = re.compile(
    r"(?i)(?:^|[^a-z0-9])(i?q\d(?:_[a-z0-9]+)*|f16|bf16|fp16|f32|fp32|exl\d)(?=$|[^a-z0-9])")

# Model types the shape alone cannot rule in or out, and what would resolve each.
UNRESOLVED_MODEL_TYPES = (
    ("sparse_moe", "expert counts in the model metadata, or a total parameter count far "
                   "above the active one, would resolve it"),
    ("reasoning", "the served chat template, or a live probe of the thinking switch, would "
                  "resolve it"),
)

# What the smoke line means as an actual request, keyed by engine id. The smoke text in
# engines.toml stays the source of truth; this says what to send and what reading it back
# means. {base} is the endpoint with any /v1 suffix stripped, which is where llama.cpp and
# ollama expose their native probes.
SMOKE_CONCRETE = {
    "llama.cpp": ("GET {base}/props",
                  "n_ctx, model_path and chat_template all come back, and n_ctx is the "
                  "window the server will actually enforce"),
    "vllm": ("GET {base}/v1/models and GET {base}/version",
             "max_model_len is present and nonzero, and /version names the build"),
    "sglang": ("GET {base}/get_model_info",
               "the model path and context length come back and match the model that was "
               "meant to be served"),
    "ollama": ("GET {base}/api/show on the native API",
               "model_info, the quantization and parameters.num_ctx come back; run this "
               "even when the runs themselves go through /v1"),
    "proxy": ("GET {base}/models, then one trivial completion through it",
              "the alias list comes back and a completion returns text; the upstream "
              "engine is still unidentified"),
    "openai-compatible": ("GET {base}/models, then one trivial completion",
                          "the list comes back and a completion returns text; neither "
                          "one identifies the engine behind them"),
    "exllama": ("read the server's model list or its startup output",
                "the loaded quant and the launch context are both named"),
    "strata": ("GET the server's health and model endpoints",
               "the loaded model, its context and the drafter it actually loaded are named"),
    "mlx": ("GET {base}/v1/models if the server offers it, and read the model directory's "
            "config.json",
            "depth and head counts come back, and the launch window is stated"),
    "proxy": ("GET {base}/v1/models",
              "the list answers; check whether the router will say which upstream served a "
              "request"),
}

# A smoke check that fails any of these means the setup is broken, not the model.
SMOKE_STOP_CONDITIONS = (
    "the endpoint does not answer the smoke request at all",
    "the field the smoke check names is missing, empty or zero",
    "the model path or id that comes back is not the model that was meant to be served",
)

SECTION_HEADINGS = (
    "What this host is",
    "What the model is",
    "What the engine is",
    "The call",
    "The decisions to put to the user",
    "The judge",
    "The smoke check",
    "The lineup",
)

NO_ENDPOINT_LINE = "Nothing is up yet: no endpoint answered, so nothing has been measured."
LINEUP_WITHHELD = ("Withheld: there is nothing to point at yet. Stand the engine up "
                   "(section 3), then run this command again unchanged -- it renders the "
                   "measured version with no re-entered decisions.")

# Optional peer module: scripts/lib/engine_facts.py is built by the engine-facts pass and
# may not be in the tree yet. The brief still renders without it -- the engine is then
# reported as unknown, with what would resolve it. When the module lands, the CLI picks it
# up with no change here.
try:
    from scripts.lib.engine_facts import probe_engine  # type: ignore
    ENGINE_FACTS_AVAILABLE = True
except ImportError:
    probe_engine = None  # type: ignore[assignment]
    ENGINE_FACTS_AVAILABLE = False


class MissingKnowledge(LookupError):
    """A knowledge table or row the brief needs is absent or does not name this id."""


@dataclass(frozen=True)
class BriefInputs:
    """Everything the brief is rendered from, gathered by the CLI.

    `endpoint` is set only when the endpoint answered; `engine_facts` carries the measured
    engine when the engine-facts module is in the tree. The file fields are overridable so
    a test can point the knowledge tables somewhere else.
    """

    host: HostFacts
    shape: ModelShape
    engine_facts: object | None = None
    engine_id: str = ""
    use_case_id: str = ""
    judge_pick: str = ""
    endpoint: str = ""
    model_name: str = ""
    model_file: str = ""
    model_dir: str = ""
    model_ids: tuple[str, ...] = ()
    engines_file: str = ENGINES_FILE
    model_types_file: str = MODEL_TYPES_FILE
    use_cases_file: str = USE_CASES_FILE
    standard_file: str = STANDARD_FILE
    variables_file: str = VARIABLES_FILE


def load_knowledge(path: str, table_key: str) -> list[dict]:
    """The rows of one knowledge table, in declaration order, or a loud failure."""
    try:
        table = load_toml(path)
    except FileNotFoundError as error:
        raise MissingKnowledge(f"knowledge table {path} is missing: {error}") from error
    rows = table.get(table_key)
    if not isinstance(rows, list) or not rows:
        raise MissingKnowledge(f"knowledge table {path} has no [[{table_key}]] rows")
    return rows


def rows_by_id(rows: list[dict]) -> dict[str, dict]:
    """The rows keyed by id, which is how the brief reads every table."""
    return {str(row.get("id", "")): row for row in rows}


def engine_row(engine_id: str, engines: dict[str, dict]) -> dict:
    """One engine row by id, or a loud failure naming the ids that exist."""
    if engine_id not in engines:
        known = ", ".join(sorted(engines)) or "none"
        raise MissingKnowledge(
            f"engines.toml has no row for {engine_id!r}; known ids: {known}")
    return engines[engine_id]


def engine_candidates(source: str, engine_ids: tuple[str, ...]) -> tuple[str, ...]:
    """The engines that could serve this model file, from what their rows say.

    A GGUF-shaped model goes to the engines whose rows consume GGUF; a HF-shaped one goes
    to the rest. An unknown shape rules nothing out. The operating system is never asked.
    """
    if source == SOURCE_GGUF:
        return tuple(engine_id for engine_id in engine_ids if engine_id in GGUF_ENGINES)
    if source == SOURCE_HF:
        return tuple(engine_id for engine_id in engine_ids
                     if engine_id not in GGUF_ENGINES and engine_id != PROXY_ENGINE)
    return tuple(engine_id for engine_id in engine_ids if engine_id != PROXY_ENGINE)


def detect_model_types(shape: ModelShape) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(matched type ids, permanently-unresolved ids) from what the shape carries.

    Matched ids are looked up in model-types.toml by the renderer, so the table stays the
    source. sparse_moe and reasoning cannot be seen in the shape at all and come back as
    unresolved, which the brief prints as unknown with what would resolve them.
    """
    notes_text = " ".join(shape.notes).lower()
    matched: list[str] = []
    if any(marker in notes_text for marker in HYBRID_MARKERS):
        matched.append("hybrid_recurrent")
    elif shape.source in (SOURCE_GGUF, SOURCE_HF) and shape.n_layers > 0:
        matched.append("dense")
    if QUANT_PATTERN.search(shape.path):
        matched.append("quantized")
    if shape.native_ctx >= LONG_CONTEXT_TOKENS:
        matched.append("long_context")
    return tuple(matched), tuple(unresolved for unresolved, _ in UNRESOLVED_MODEL_TYPES)


def judge_candidates(model_ids: tuple[str, ...], under_test: set[str],
                     ) -> tuple[tuple[str, bool], ...]:
    """(candidate, is-pinned-default) pairs: the endpoint's list minus what is under test.

    The pinned default is not known here; the renderer marks it. A candidate that is the
    model under test is excluded outright -- the judge must never score its own model.
    """
    return tuple((model_id, False) for model_id in model_ids
                 if model_id and model_id not in under_test)


def fetch_model_ids(cfg: ModelConfig) -> tuple[tuple[str, ...], str | None]:
    """The endpoint's model list, or the reason there is none."""
    _, payload, error = get_json(cfg.models_url, cfg.api_key)
    if error:
        return (), error
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return (), "the model list is not a JSON array of entries"
    ids = tuple(str(entry.get("id")) for entry in data
                if isinstance(entry, dict) and entry.get("id"))
    return ids, None if ids else "the model list is empty"


def probe_engine_or_none(cfg: ModelConfig) -> object | None:
    """Measured engine facts when the engine-facts module is in the tree, else None."""
    if not ENGINE_FACTS_AVAILABLE or probe_engine is None:
        return None
    return probe_engine(cfg)


def base_url(endpoint: str) -> str:
    """The endpoint with any /v1 suffix stripped, where the native probes live."""
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def _as_int(value: object, default: int) -> int:
    """An int from a config table, or the default when it is missing or not a number."""
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def brief_fit(shape: ModelShape, host: HostFacts, standard: dict) -> Fit | None:
    """The fit verdict for the call section, or None when the shape cannot be fitted."""
    if not shape.known:
        return None
    kv_type = str(get_dotted(standard, "serve.kv_k", "q8_0"))
    preferred_ctx = _as_int(get_dotted(standard, "serve.ctx"), DEFAULT_PREFERRED_CTX)
    kv_bits = parse_kv_bits(kv_type)
    return compute_fit(
        weights_bytes=shape.weights_bytes,
        kv_per_token=shape.kv_elements_per_token * kv_bits // 8,
        gpus=host.gpus,
        ram_available_gb=host.ram_available_gb,
        preferred_ctx=preferred_ctx,
        foreign_vram_used_gb=host.vram_used_gb,
        kv_fixed_bytes=shape.kv_fixed_bytes(kv_bits),
    )


def _unknown_line(what: str, resolution: str) -> str:
    """One fact the brief could not read, stated as unknown with what would resolve it."""
    return f"- {what}: unknown -- {resolution}"


def render_host(host: HostFacts) -> str:
    """Section 1: the host's facts, each naming its source. Presence, not a verdict."""
    lines: list[str] = []
    if host.gpus:
        for gpu in host.gpus:
            lines.append(
                f"- accelerator {gpu.index}: {gpu.name}, {gpu.vram_total_gb:.1f} GB total, "
                f"{gpu.vram_free_mb / 1024:.1f} GB free (source: nvidia-smi --query-gpu)")
    else:
        reason = host.gpu_error or "no accelerator facts were reported"
        lines.append(_unknown_line(
            "accelerators", f"{reason} (source: nvidia-smi --query-gpu); a working driver "
            "or an installed nvidia-smi would resolve it"))
    if host.ram_total_gb is not None:
        memory = f"- memory: {host.ram_total_gb} GB total"
        if host.ram_available_gb is not None:
            memory += f", {host.ram_available_gb} GB available"
        if host.swap_used_gb:
            memory += f", {host.swap_used_gb} GB swap in use"
        lines.append(f"{memory} (source: /proc/meminfo)")
    else:
        lines.append(_unknown_line("memory", "/proc/meminfo is not readable on this host"))
    if host.cpu_model or host.cpu_threads:
        model = host.cpu_model or "model not reported"
        lines.append(f"- cpu: {model}, {host.cpu_threads} threads (source: /proc/cpuinfo)")
    else:
        lines.append(_unknown_line("cpu", "/proc/cpuinfo is not readable on this host"))
    if host.numa_nodes:
        lines.append(f"- numa nodes: {host.numa_nodes} (source: /sys/devices/system/node)")
    for tool in host.tools:
        if tool.available:
            version = f", reports {tool.version}" if tool.version else ""
            lines.append(f"- tool {tool.name}: present at {tool.path}{version} "
                         "(source: PATH lookup and --version)")
        else:
            lines.append(f"- tool {tool.name}: absent -- any leg that needs it is skipped "
                         "with the reason, never a crash (source: PATH lookup)")
    return "\n".join(lines)


def render_model(shape: ModelShape, model_types: dict[str, dict]) -> str:
    """Section 2: the shape, the source it came from, and the model types it matches."""
    lines: list[str] = []
    if not shape.known:
        lines.append("- shape: unknown -- a model file (--model-file or --model-dir), or a "
                     "running engine that reports the model path it loaded, would resolve it")
        for problem in shape.problems:
            lines.append(f"- problem: {problem}")
    else:
        lines.append(f"- source: {shape.source} ({shape.path or 'no path on record'})")
        lines.append(f"- architecture: {shape.arch or 'not stated'}")
        lines.append(f"- layers that hold a growing cache: {shape.n_layers}")
        lines.append(f"- kv heads: {shape.n_kv_heads}, head_dim: {shape.head_dim}")
        lines.append(f"- weights: {shape.weights_bytes / 1024 ** 3:.1f} GB")
        if shape.native_ctx:
            lines.append(f"- declared window: {shape.native_ctx} tokens")
        else:
            lines.append(_unknown_line("declared window",
                                       "the model metadata does not state one"))
        lines.append(f"- cache cost: {shape.kv_elements_per_token} elements per token of "
                     f"window, {shape.kv_fixed_elements} elements that do not grow")
        for note in shape.notes:
            lines.append(f"- note: {note}")
        for problem in shape.problems:
            lines.append(f"- problem: {problem}")

    matched, _unresolved = detect_model_types(shape)
    if matched:
        lines.append("Model type, matched against model-types.toml:")
        for type_id in matched:
            row = model_types.get(type_id)
            if row is None:
                raise MissingKnowledge(
                    f"model-types.toml has no row for {type_id!r}; known ids: "
                    f"{', '.join(sorted(model_types)) or 'none'}")
            lines.append(f"- `{type_id}`: {row.get('effect', '')}")
            lines.append(f"  records: {row.get('advice', '')}")
    for unresolved_id, resolution in UNRESOLVED_MODEL_TYPES:
        lines.append(f"- model type `{unresolved_id}`: unknown -- {resolution}")
    return "\n".join(lines)


def _engine_fact_lines(facts: object) -> list[str]:
    """The measured engine facts, whatever the engine-facts probe came back with."""
    known = bool(getattr(facts, "known", False))
    lines: list[str] = []
    if not known:
        errors = "; ".join(getattr(facts, "errors", ()) or ("nothing was tried",))
        lines.append(_unknown_line(
            "engine", f"nothing behind the endpoint identified itself ({errors}); a server "
            "that answers /props, /api/show, /version or /models would resolve it"))
        return lines
    lines.append(f"- engine: {getattr(facts, 'engine', 'unknown')} (identified by the "
                 f"{getattr(facts, 'source', '')} probe)")
    version = getattr(facts, "version", "") or "build not reported"
    lines.append(f"- version: {version}")
    window = getattr(facts, "window", None)
    if window:
        lines.append(f"- window: {window} tokens (the number the engine says it serves)")
    else:
        lines.append(_unknown_line("window", "the engine did not report one; the smoke "
                                            "check would read it"))
    slots = getattr(facts, "slots", None)
    if slots:
        lines.append(f"- slots: {slots}")
    kv_type = getattr(facts, "kv_type", None)
    if kv_type:
        lines.append(f"- kv cache type: {kv_type}")
    flash = getattr(facts, "flash_attn", None)
    if flash is not None:
        lines.append(f"- flash attention: {'on' if flash else 'off'}")
    drafter = getattr(facts, "drafter", None)
    if drafter:
        lines.append(f"- drafter: {drafter}")
    errors = tuple(getattr(facts, "errors", ()) or ())
    engine = str(getattr(facts, "engine", "") or "")
    if errors and engine not in ("", "unknown"):
        # Probes this engine does not serve are not failures. Four 404s out of a working
        # router read as a broken endpoint, and this brief gets read once.
        lines.append(f"- {len(errors)} other probes were not served; the engine was "
                     "identified before them")
    else:
        for error in errors:
            lines.append(f"- probe error: {error}")
    return lines


def render_engine(inputs: BriefInputs, engines: dict[str, dict]) -> str:
    """Section 3: measured facts when something is up, serve and smoke when it is not."""
    lines: list[str] = []
    if inputs.engine_facts is not None:
        lines.extend(_engine_fact_lines(inputs.engine_facts))
    elif inputs.endpoint:
        lines.append(_unknown_line(
            "engine", "the endpoint answered but the engine-facts module is not in the "
            "tree; a tree that carries scripts/lib/engine_facts.py would resolve it"))
    else:
        lines.append(NO_ENDPOINT_LINE)
        if inputs.engine_id:
            row = engine_row(inputs.engine_id, engines)
            lines.append(f"How this engine is normally stood up ({inputs.engine_id}):")
            lines.append(f"- serve: {row.get('serve', '')}")
            lines.append(f"- smoke: {row.get('smoke', '')}")
        else:
            candidates = engine_candidates(
                inputs.shape.source, tuple(engines))
            lines.append("Engines that could serve this model file, from what their rows "
                         "say about weight formats:")
            for candidate in candidates:
                lines.append(f"- {candidate}")
            lines.append("Pick one with --engine <id>; the operating system is never asked.")
    return "\n".join(lines)


def _call_verdict(fit: Fit | None, facts: object | None, preferred_ctx: int) -> str:
    """One sentence on whether the combination is worth testing, and why."""
    if fit is None:
        return ("unknown -- no model shape to fit; a model file or a running engine that "
                "reports its model path would resolve it")
    if not fit.fits:
        return f"not worth testing on this host as-is -- {fit.reason}"
    window = getattr(facts, "window", None) if facts is not None else None
    if window and window < preferred_ctx:
        return (f"worth testing at a reduced window -- the engine serves {window} tokens "
                f"against the standard's {preferred_ctx}; rows at that window are stamped "
                "and are not comparable to full-standard rows")
    if window:
        return f"worth testing -- the model fits ({fit.reason}) and the window reaches the standard"
    return ("worth testing on this host once the engine is up -- "
            f"{fit.reason}; the window is unknown until the smoke check reads it")


def render_call(inputs: BriefInputs, fit: Fit | None, engines: dict[str, dict],
                preferred_ctx: int) -> str:
    """Section 4: whether this combination is worth testing, with the reason. Never silent."""
    lines: list[str] = []
    if fit is not None:
        lines.append(f"- fit: {fit.verdict} -- {fit.reason}")
        if fit.estimate:
            lines.append("- the fit is an estimate; placement is the engine's call")
    else:
        lines.append(_unknown_line("fit", "no model shape; --model-file or --model-dir "
                                          "would resolve it"))
    window = getattr(inputs.engine_facts, "window", None) if inputs.engine_facts else None
    if window:
        lines.append(f"- window: the engine serves {window} tokens against the standard's "
                     f"{preferred_ctx}")
    else:
        lines.append(_unknown_line("window", "nothing is serving yet; the smoke check "
                                             "reads it off the server once it is up"))
    row_id = inputs.engine_id
    if not row_id and inputs.engine_facts is not None:
        row_id = getattr(inputs.engine_facts, "engine", "")
    if row_id in engines:
        row = engines[row_id]
        lines.append(f"- what {row_id} does to the numbers: {row.get('effect', '')}")
        lines.append(f"- comparable to: {row.get('compares_to', '')}")
        if inputs.engine_facts is not None and not getattr(inputs.engine_facts,
                                                           "supports_kv_flags", True):
            lines.append("- the KV cache is not a knob on this engine; the plan records "
                         "kv_type as n/a rather than dropping the row")
    else:
        lines.append(_unknown_line(
            "engine behaviour", "no engines.toml row for this engine; --engine <id> or an "
            "engine that identifies itself would resolve it"))
    lines.append(f"- the call: {_call_verdict(fit, inputs.engine_facts, preferred_ctx)}")
    return "\n".join(lines)


def render_decisions(inputs: BriefInputs, use_cases: dict[str, dict],
                     variables: dict) -> str:
    """Section 5: the four use cases and the five variables, with consequences.

    Variable consequences come from what_it_changes and decision_rule, not from `effect`,
    because the effect rows carry measured numbers tied to specific hosts and the brief
    names no host.
    """
    lines: list[str] = []
    for use_case_id, row in use_cases.items():
        marker = " (you named this one)" if use_case_id == inputs.use_case_id else ""
        lines.append(f"### {use_case_id}{marker}")
        lines.append(f"- question: {row.get('question', '')}")
        lines.append(f"- legs: {row.get('legs', '')}")
        lines.append(f"- cannot say: {row.get('cannot_say', '')}")
        lines.append("")
    lines.append("The variables that change the result on this host and model, each with "
                 "its consequence:")
    for label, variable_id in DECISION_VARIABLES:
        variable = variables.get(variable_id)
        if variable is None:
            raise MissingKnowledge(
                f"variables.toml has no row for {variable_id!r}, which the brief's "
                f"{label!r} decision needs")
        lines.append(f"- {label} ({variable.class_name}): {variable.what_it_changes} "
                     f"-- {variable.decision_rule}")
    lines.append("Every one of these is recorded in the run stamp; a value the run did "
                 "not record makes two runs incomparable.")
    return "\n".join(lines)


def render_judge(inputs: BriefInputs, pin: dict) -> str:
    """Section 6: the judge candidates, the pinned default named out loud, the user picks."""
    pinned_model = str(pin.get("model", "") or "none")
    lines = [f"The standard pins {pinned_model} as the default judge (standard.toml "
             f"[judge]); the user picks, and nothing here chooses one."]
    if inputs.model_ids:
        under_test = {name for name in (inputs.model_name,) if name}
        candidates = judge_candidates(inputs.model_ids, under_test)
        excluded = [model_id for model_id in inputs.model_ids
                    if model_id and model_id in under_test]
        if candidates:
            lines.append("Candidates from this endpoint's model list:")
            for candidate, _is_pin in candidates:
                marker = " (the pinned default)" if candidate == pinned_model else ""
                marker = " (you named this one)" if candidate == inputs.judge_pick else marker
                lines.append(f"- {candidate}{marker}")
        else:
            lines.append(_unknown_line("candidates", "every model this endpoint lists is "
                                                     "under test"))
        if excluded:
            lines.append(f"Excluded, because they are under test: {', '.join(excluded)}")
        if pinned_model != "none" and pinned_model not in [c for c, _ in candidates]:
            lines.append(f"The pinned default {pinned_model} is not in this endpoint's "
                         "list; picking it needs an endpoint that serves it.")
    else:
        lines.append(_unknown_line("candidates", "no endpoint is up; an endpoint whose "
                                                 "model list answers would resolve it"))
    if inputs.judge_pick:
        lines.append(f"You named {inputs.judge_pick} as the judge.")
    lines.append("The pin is overridable with --judge-model and --judge-endpoint; inside "
                 "the writing leg WRITING_JUDGE_MODEL and WRITING_JUDGE_ENDPOINT override "
                 "it, and the plan records whether the pin was overridden.")
    lines.append("Calibrate the pick before the run: "
                 "python3 scripts/calibrate_judge.py --judge-model <pick> "
                 "--endpoint <judge endpoint>")
    return "\n".join(lines)


def render_smoke(inputs: BriefInputs, engines: dict[str, dict]) -> str:
    """Section 7: the engine's smoke check made concrete, or the setup checklist."""
    row_id = inputs.engine_id
    if not row_id and inputs.engine_facts is not None:
        row_id = getattr(inputs.engine_facts, "engine", "")
    lines: list[str] = []
    if not inputs.endpoint:
        lines.append("Nothing is up yet, so this is the setup checklist:")
        if row_id in engines:
            row = engines[row_id]
            lines.append(f"1. Stand the engine up: {row.get('serve', '')}")
            lines.append(f"2. Prove it is up and read the real window: {row.get('smoke', '')}")
        else:
            lines.append("1. Pick an engine (--engine <id>); the checklist needs its serve "
                         "and smoke lines.")
            return "\n".join(lines)
        lines.append("3. Stop and fix the setup instead of running when any of these hold:")
        for condition in SMOKE_STOP_CONDITIONS:
            lines.append(f"   - {condition}")
        return "\n".join(lines)

    base = base_url(inputs.endpoint)
    if row_id in engines:
        row = engines[row_id]
        lines.append(f"From engines.toml: {row.get('smoke', '')}")
    else:
        lines.append(_unknown_line("smoke line", "no engines.toml row for this engine"))
    concrete = SMOKE_CONCRETE.get(row_id)
    if concrete is None:
        lines.append(_unknown_line(
            "concrete smoke request", f"no smoke recipe for {row_id!r} in the brief"))
    else:
        request, passes = concrete
        lines.append(f"- what to send: {request.format(base=base)}")
        lines.append(f"- a pass looks like: {passes}")
    lines.append("Stop and fix the setup instead of running when any of these hold:")
    for condition in SMOKE_STOP_CONDITIONS:
        lines.append(f"- {condition}")
    return "\n".join(lines)


def lineup_command(model: str, endpoint: str, model_file: str, model_dir: str,
                   legs: str, judge_model: str, judge_endpoint: str, out_dir: str) -> str:
    """The exact run_battery.py command, plus its --detach and --status lines."""
    parts = ["python3 scripts/run_battery.py"]
    if model:
        parts.append(f"--model {model}")
    parts.append(f"--endpoint {endpoint}")
    if model_file:
        parts.append(f"--model-file {model_file}")
    if model_dir:
        parts.append(f"--model-dir {model_dir}")
    parts.append(f"--legs {legs}")
    if judge_model:
        parts.append(f"--judge-model {judge_model}")
    if judge_endpoint:
        parts.append(f"--judge-endpoint {judge_endpoint}")
    parts.append(f"--out {out_dir}")
    command = " ".join(parts)
    return "\n".join((
        command,
        f"{command} --detach",
        f"python3 scripts/run_battery.py --out {out_dir} --status",
    ))


def render_lineup(inputs: BriefInputs) -> str:
    """Section 8: the exact run command, withheld while nothing is up."""
    if not inputs.endpoint:
        return LINEUP_WITHHELD
    model = inputs.model_name or inputs.shape.path.rsplit("/", 1)[-1] or ""
    legs = (f"the {inputs.use_case_id} leg set (section 5)" if inputs.use_case_id
            else "<the leg set of the use case you choose -- section 5>")
    judge = inputs.judge_pick or "<the judge you choose -- section 6>"
    out_dir = f"runs/{model}" if model else "runs/<name this run>"
    return lineup_command(model, inputs.endpoint, inputs.model_file, inputs.model_dir,
                          legs, judge, "", out_dir)


def render_brief(inputs: BriefInputs) -> str:
    """The whole brief: eight sections, in order, none omitted for being unknown."""
    engines = rows_by_id(load_knowledge(inputs.engines_file, ENGINE_TABLE_KEY))
    model_types = rows_by_id(load_knowledge(inputs.model_types_file, MODEL_TYPE_TABLE_KEY))
    use_cases = rows_by_id(load_knowledge(inputs.use_cases_file, USE_CASE_TABLE_KEY))
    standard = load_toml(inputs.standard_file)
    variables = variables_by_id(load_variables(inputs.variables_file))
    preferred_ctx = _as_int(get_dotted(standard, "serve.ctx"), DEFAULT_PREFERRED_CTX)

    sections = [
        render_host(inputs.host),
        render_model(inputs.shape, model_types),
        render_engine(inputs, engines),
        render_call(inputs, brief_fit(inputs.shape, inputs.host, standard),
                    engines, preferred_ctx),
        render_decisions(inputs, use_cases, variables),
        render_judge(inputs, standard.get("judge") or {}),
        render_smoke(inputs, engines),
        render_lineup(inputs),
    ]
    body = "\n\n".join(f"## {index}. {heading}\n\n{section}"
                       for index, (heading, section) in
                       enumerate(zip(SECTION_HEADINGS, sections), start=1))
    return f"# Consult brief\n\n{body}\n"
