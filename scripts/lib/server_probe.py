"""What the live endpoint will actually accept, measured rather than assumed.

The expensive failures in the reference runs came from assuming: a window smaller than the
cap, a chat template that ignored the thinking switch, and a grammar plus json_schema
combination that a specific engine build rejected. Each one is a probe here.
"""
from __future__ import annotations

from dataclasses import dataclass

from scripts.lib.model import ModelConfig, get_json, post_chat

PROBE_CAP = 256
THINKING_OFF_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
THINKING_ON_BODY = {"chat_template_kwargs": {"enable_thinking": True}}
THINKING_PROMPT = "How many r's are in the word strawberry? Answer with the number."
REASONING_DIFF_RATIO = 0.1
REASONING_DIFF_FLOOR = 8
TOOL_AND_GRAMMAR_BODY = {
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Report the current time",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }],
    "tool_choice": "auto",
    "response_format": {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object",
                                                     "properties": {"answer": {"type": "string"}},
                                                     "required": ["answer"]}},
    },
}
ENGINE_ERROR_MARKERS = ("grammar", "json_schema", "response_format", "400")


@dataclass(frozen=True)
class ServerFacts:
    """Facts about the endpoint, each one measured or explicitly unknown."""

    reachable: bool
    model_id: str | None = None
    served_model: str | None = None
    ctx: int | None = None
    slots: int | None = None
    kv_type: str | None = None
    flash_attn: bool | None = None
    chat_template_mentions_thinking: bool | None = None
    thinking_switch: bool | None = None
    reasoning_channel: bool | None = None
    thinking_on_tokens: int | None = None
    thinking_off_tokens: int | None = None
    tools_with_json_schema_ok: bool | None = None
    tools_error: str | None = None
    props: dict | None = None
    errors: tuple[str, ...] = ()

    @property
    def window_is_known(self) -> bool:
        return self.ctx is not None

    def to_dict(self) -> dict:
        return {
            "reachable": self.reachable, "model_id": self.model_id,
            "served_model": self.served_model, "ctx": self.ctx, "slots": self.slots,
            "kv_type": self.kv_type, "flash_attn": self.flash_attn,
            "chat_template_mentions_thinking": self.chat_template_mentions_thinking,
            "thinking_switch": self.thinking_switch,
            "reasoning_channel": self.reasoning_channel,
            "thinking_on_tokens": self.thinking_on_tokens,
            "thinking_off_tokens": self.thinking_off_tokens,
            "tools_with_json_schema_ok": self.tools_with_json_schema_ok,
            "tools_error": self.tools_error,
            "errors": list(self.errors),
        }


def props_url(cfg: ModelConfig) -> str:
    """llama.cpp exposes server facts at /props, one level above /v1."""
    base = cfg.endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return f"{base}/props"


def _reasoning_lengths(cfg: ModelConfig) -> tuple[int, int, bool, tuple[str, ...]]:
    """(tokens with thinking on, tokens with it off, reasoning channel seen, errors)."""
    errors: list[str] = []
    lengths: list[int] = []
    saw_reasoning = False
    for body in (THINKING_ON_BODY, THINKING_OFF_BODY):
        response = post_chat(cfg, [{"role": "user", "content": THINKING_PROMPT}],
                             max_tokens=PROBE_CAP, temperature=0.0, extra_body=body)
        if not response.ok:
            errors.append(response.error or "thinking probe failed")
            lengths.append(0)
            continue
        if response.reasoning:
            saw_reasoning = True
            lengths.append(response.completion_tokens)
        else:
            # No separate reasoning channel: the model either thinks inline or does not
            # think at all. Token count is then the only signal.
            lengths.append(response.completion_tokens)
    return lengths[0], lengths[1], saw_reasoning, tuple(errors)


def _thinking_switch_moved(on_tokens: int, off_tokens: int) -> bool:
    """A switch that changes nothing is not a switch."""
    if on_tokens == 0 and off_tokens == 0:
        return False
    threshold = max(REASONING_DIFF_FLOOR, REASONING_DIFF_RATIO * max(on_tokens, off_tokens))
    return abs(on_tokens - off_tokens) >= threshold


def _probe_tools_and_grammar(cfg: ModelConfig) -> tuple[bool | None, str | None]:
    """Send tools and a json_schema together, which one engine build rejected outright."""
    response = post_chat(cfg, [{"role": "user", "content": "What time is it?"}],
                         max_tokens=PROBE_CAP, temperature=0.0,
                         extra_body=TOOL_AND_GRAMMAR_BODY)
    if response.ok:
        return True, None
    error = response.error or f"status {response.status}"
    return False, error


@dataclass(frozen=True)
class _Props:
    """What /props (or the models list) told us, before any generation probe."""

    reachable: bool
    served_model: str | None
    ctx: int | None
    slots: int | None
    mentions_thinking: bool | None
    raw: dict | None
    errors: tuple[str, ...]


@dataclass(frozen=True)
class _Thinking:
    """The result of the two-request thinking comparison."""

    on_tokens: int | None
    off_tokens: int | None
    reasoning_seen: bool | None
    switch: bool | None


EMPTY_THINKING = _Thinking(None, None, None, None)


def _first_model_id(models: dict) -> str | None:
    """The id of the first entry in an OpenAI-style model list, or None when there is none."""
    data = models.get("data") or []
    first = data[0] if isinstance(data, list) and data else {}
    if isinstance(first, dict) and first.get("id"):
        return str(first["id"])
    return None


def _probe_models(cfg: ModelConfig) -> tuple[bool, str | None, tuple[str, ...]]:
    """Fallback reachability check for a server that is not llama.cpp and has no /props."""
    _, models, error = get_json(f"{cfg.endpoint.rstrip('/')}/models", cfg.api_key)
    errors = (f"models: {error}",) if error else ()
    if not isinstance(models, dict):
        return False, None, errors
    model_id = _first_model_id(models)
    return (True, model_id, errors) if model_id else (False, None, errors)


def _int_or_none(value: object) -> int | None:
    """An int as reported, or None for anything the server typed loosely."""
    return value if isinstance(value, int) else None


def _props_fields(raw: dict | None) -> tuple[int | None, int | None, bool | None]:
    """(ctx, slots, template mentions thinking) as /props reported them."""
    defaults = raw.get("default_generation_settings") if raw else None
    defaults = defaults if isinstance(defaults, dict) else {}
    template = raw.get("chat_template") if raw else None
    mentions = "enable_thinking" in template if isinstance(template, str) else None
    return (_int_or_none(defaults.get("n_ctx")),
            _int_or_none(raw.get("total_slots") if raw else None),
            mentions)


def _fetch_props(cfg: ModelConfig) -> _Props:
    """Read the server's own description of itself, falling back to the model list."""
    errors: list[str] = []
    status, payload, error = get_json(props_url(cfg), cfg.api_key)
    if error:
        errors.append(f"props: {error}")
    raw = payload if isinstance(payload, dict) else None
    ctx, slots, mentions = _props_fields(raw)
    reachable = raw is not None or status == 200
    served_model = None
    if not reachable:
        reachable, served_model, model_errors = _probe_models(cfg)
        errors.extend(model_errors)
    return _Props(reachable=reachable, served_model=served_model,
                  ctx=ctx, slots=slots,
                  mentions_thinking=mentions, raw=raw, errors=tuple(errors))


def _measure_thinking(cfg: ModelConfig, mentions_thinking: bool | None,
                      errors: list[str]) -> _Thinking:
    """Ask twice with the thinking flag on and off, and see whether anything moved."""
    on_tokens, off_tokens, saw_reasoning, probe_errors = _reasoning_lengths(cfg)
    errors.extend(probe_errors)
    if probe_errors and on_tokens == 0 and off_tokens == 0:
        # The comparison never ran, so the switch is unknown rather than absent.
        return _Thinking(None, None, None, None)
    moved = _thinking_switch_moved(on_tokens, off_tokens)
    return _Thinking(on_tokens, off_tokens, saw_reasoning,
                     bool(mentions_thinking or moved))


def _measure_tools(cfg: ModelConfig, errors: list[str]) -> tuple[bool | None, str | None]:
    """Tools plus a json_schema in one request, which some engine builds reject outright."""
    ok, error = _probe_tools_and_grammar(cfg)
    if error and any(marker in error.lower() for marker in ENGINE_ERROR_MARKERS):
        errors.append(f"tools+json_schema rejected by the engine: {error}")
    return ok, error


def probe_server(cfg: ModelConfig, run_thinking: bool = True,
                 run_tools: bool = True) -> ServerFacts:
    """Measure the endpoint's window, slots, thinking switch and tool support."""
    props = _fetch_props(cfg)
    errors = list(props.errors)
    if not props.reachable:
        return ServerFacts(reachable=False, served_model=props.served_model,
                           chat_template_mentions_thinking=props.mentions_thinking,
                           props=props.raw, errors=tuple(errors))

    thinking = EMPTY_THINKING
    if run_thinking:
        thinking = _measure_thinking(cfg, props.mentions_thinking, errors)
    tools_ok: bool | None = None
    tools_error: str | None = None
    if run_tools:
        tools_ok, tools_error = _measure_tools(cfg, errors)
    return ServerFacts(
        reachable=True,
        served_model=props.served_model or cfg.name,
        ctx=props.ctx,
        slots=props.slots,
        chat_template_mentions_thinking=props.mentions_thinking,
        thinking_switch=thinking.switch,
        reasoning_channel=thinking.reasoning_seen,
        thinking_on_tokens=thinking.on_tokens,
        thinking_off_tokens=thinking.off_tokens,
        tools_with_json_schema_ok=tools_ok,
        tools_error=tools_error,
        props=props.raw,
        errors=tuple(errors),
    )
