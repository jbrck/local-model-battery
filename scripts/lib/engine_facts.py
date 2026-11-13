"""Identify the serving engine behind an endpoint, and read only what it publishes.

The engine is named by which probe answered, never by guessing from a model name: a tag
that says 128k can be served at a much smaller window, and a file name proves nothing
about the process answering. Each probe yields only the facts its engine actually
publishes, so anything the engine does not say stays None rather than being inferred.
"""
from __future__ import annotations

from dataclasses import dataclass

from scripts.lib.model import ModelConfig, get_json

# The engine ids the knowledge tables use. An endpoint that answers none of the probes
# is identified as none of them.
KNOWN_ENGINES = ("llama.cpp", "vllm", "sglang", "ollama", "exllama", "strata", "mlx",
                 "openai-compatible")

# How each engine's id is written in a sentence that names it.
DISPLAY_NAMES = {"vllm": "vLLM", "sglang": "SGLang", "mlx": "MLX"}


def display_name(engine: str) -> str:
    """The engine's name as it is conventionally written."""
    return DISPLAY_NAMES.get(engine, engine)


@dataclass(frozen=True)
class EngineFacts:
    """What the engine behind an endpoint is, and what it publishes about itself."""

    engine: str = "unknown"          # llama.cpp | vllm | sglang | ollama | exllama | strata | mlx | openai-compatible
    version: str = ""
    window: int | None = None
    slots: int | None = None
    kv_type: str | None = None
    flash_attn: bool | None = None
    model_path: str | None = None
    model_dir: str | None = None
    supports_kv_flags: bool = False
    drafter: str | None = None       # none | mtp | dflash2, only when the engine reports one
    source: str = ""                 # the probe that answered
    errors: tuple[str, ...] = ()

    @property
    def known(self) -> bool:
        """An engine that answered nothing is identified as nothing."""
        return self.engine != "unknown"

    def to_dict(self) -> dict:
        return {
            "engine": self.engine, "version": self.version, "window": self.window,
            "slots": self.slots, "kv_type": self.kv_type, "flash_attn": self.flash_attn,
            "model_path": self.model_path, "model_dir": self.model_dir,
            "supports_kv_flags": self.supports_kv_flags, "drafter": self.drafter,
            "source": self.source, "errors": list(self.errors),
        }


def base_url(cfg: ModelConfig) -> str:
    """The endpoint one level above /v1, where the native probes live.

    Strips a trailing /v1 exactly as server_probe.props_url does, so both readers agree
    on where a server's own endpoints are.
    """
    base = cfg.endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def _int_or_none(value: object) -> int | None:
    """An int as reported, or None for anything the server typed loosely."""
    return value if isinstance(value, int) else None


def _get(url: str, api_key: str, errors: list[str], label: str) -> object:
    """GET one probe endpoint, recording the failure under its label."""
    _, payload, error = get_json(url, api_key)
    if error:
        errors.append(f"{label}: {error}")
    return payload


def _first_entry(models: dict) -> dict:
    """The first entry in an OpenAI-style model list, or an empty dict when there is none."""
    data = models.get("data") or []
    first = data[0] if isinstance(data, list) and data else {}
    return first if isinstance(first, dict) else {}


def _llama_props(raw: dict, errors: list[str]) -> EngineFacts:
    """/props answered, so llama.cpp: its launch flags are the truth about the window.

    The engine is identified by the probe, not by the model path; a .gguf there only
    confirms what the probe already said.
    """
    defaults = raw.get("default_generation_settings")
    defaults = defaults if isinstance(defaults, dict) else {}
    model_path = raw.get("model_path")
    return EngineFacts(
        engine="llama.cpp",
        version=str(raw.get("build_info") or raw.get("version") or ""),
        window=_int_or_none(defaults.get("n_ctx")),
        slots=_int_or_none(raw.get("total_slots")),
        model_path=str(model_path) if model_path else None,
        supports_kv_flags=True,
        source="props",
        errors=tuple(errors),
    )


def _ollama_ctx(show: dict) -> int | None:
    """The window from parameters.num_ctx, or the model_info context length."""
    parameters = show.get("parameters")
    if isinstance(parameters, str):
        for line in parameters.splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] == "num_ctx":
                try:
                    return int(fields[1])
                except ValueError:
                    return None
    info = show.get("model_info")
    if isinstance(info, dict):
        for key, value in info.items():
            if key.endswith(".context_length") and isinstance(value, int):
                return value
    return None


def _ollama(show: dict, errors: list[str]) -> EngineFacts:
    """/api/show answered on the native API, so ollama: the tag is never the window."""
    return EngineFacts(engine="ollama", window=_ollama_ctx(show), source="api/show",
                       errors=tuple(errors))


def _vllm(version: dict, models: dict, errors: list[str]) -> EngineFacts:
    """/version and /models answered, so vllm: max_model_len is the window it serves."""
    reported = version.get("version")
    return EngineFacts(
        engine="vllm",
        version=str(reported) if reported else "",
        window=_int_or_none(_first_entry(models).get("max_model_len")),
        source="version + models",
        errors=tuple(errors),
    )


def _sglang(info: dict, errors: list[str]) -> EngineFacts:
    """/get_model_info answered, so sglang: it reports the loaded path and the context."""
    model_path = info.get("model_path")
    return EngineFacts(
        engine="sglang",
        window=_int_or_none(info.get("context_length")),
        model_path=str(model_path) if model_path else None,
        source="get_model_info",
        errors=tuple(errors),
    )


def _openai_compatible(models: dict, errors: list[str]) -> EngineFacts:
    """Only the model list answers, so a compatible endpoint: the id is all it yields."""
    return EngineFacts(engine="openai-compatible", source="models", errors=tuple(errors))


def probe_engine(cfg: ModelConfig) -> EngineFacts:
    """Name the engine by the probe that answered, in the order the probe table fixes.

    An endpoint that answers nothing is unknown, with the errors naming every probe that
    was tried. An endpoint that answers only the model list is openai-compatible, whatever
    the model name looks like.
    """
    errors: list[str] = []
    key = cfg.api_key
    base = base_url(cfg)
    endpoint = cfg.endpoint.rstrip("/")
    props = _get(f"{base}/props", key, errors, "props")
    if isinstance(props, dict):
        return _llama_props(props, errors)
    show = _get(f"{base}/api/show", key, errors, "api/show")
    if isinstance(show, dict):
        return _ollama(show, errors)
    version = _get(f"{base}/version", key, errors, "version")
    models = _get(f"{endpoint}/models", key, errors, "models")
    if isinstance(version, dict) and isinstance(models, dict):
        return _vllm(version, models, errors)
    info = _get(f"{base}/get_model_info", key, errors, "get_model_info")
    if isinstance(info, dict):
        return _sglang(info, errors)
    if isinstance(models, dict):
        return _openai_compatible(models, errors)
    return EngineFacts(errors=tuple(errors))
