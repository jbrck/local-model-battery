"""HTTP client for any OpenAI-compatible chat-completions endpoint."""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_ENDPOINT = "http://localhost:11434/v1"
DEFAULT_TIMEOUT_S = 600


@dataclass
class ModelConfig:
    """Where to send requests, and how the server should sample."""

    name: str
    endpoint: str = DEFAULT_ENDPOINT
    api_key: str = "not-needed"
    token_limit: int = 4096
    temperature: float = 0.2
    timeout_s: int = DEFAULT_TIMEOUT_S
    extra_headers: dict[str, str] = field(default_factory=dict)

    @property
    def chat_url(self) -> str:
        return f"{self.endpoint.rstrip('/')}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.endpoint.rstrip('/')}/models"


REASONING_FIELDS = ("reasoning_content", "reasoning")


@dataclass
class ChatResponse:
    """One chat completion attempt, successful or not, with its timing and token counts."""

    status: int | None = None
    payload: dict | None = None
    elapsed_s: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.payload is not None

    @property
    def message(self) -> dict:
        """The assistant message, or an empty dict when the response is malformed."""
        if not isinstance(self.payload, dict):
            return {}
        choices = self.payload.get("choices") or []
        first = choices[0] if isinstance(choices, list) and choices else {}
        message = first.get("message") if isinstance(first, dict) else None
        return message if isinstance(message, dict) else {}

    @property
    def content(self) -> str:
        return str(self.message.get("content") or "").strip()

    @property
    def reasoning(self) -> str:
        """The reasoning trace under whichever field name the server uses."""
        for field_name in REASONING_FIELDS:
            value = self.message.get(field_name)
            if value:
                return str(value).strip()
        return ""

    @property
    def finish_reason(self) -> str:
        """`length` means the cap was the binding constraint, not the model."""
        if not isinstance(self.payload, dict):
            return ""
        choices = self.payload.get("choices") or []
        first = choices[0] if isinstance(choices, list) and choices else {}
        return str(first.get("finish_reason") or "") if isinstance(first, dict) else ""

    @property
    def usage(self) -> dict:
        if not isinstance(self.payload, dict):
            return {}
        usage = self.payload.get("usage")
        return usage if isinstance(usage, dict) else {}

    @property
    def completion_tokens(self) -> int:
        try:
            return int(self.usage.get("completion_tokens") or 0)
        except (TypeError, ValueError):
            return 0

    def answer(self, allow_reasoning_fallback: bool = True) -> str | None:
        """The scoreable answer, preferring the content channel."""
        if self.content:
            return self.content
        if allow_reasoning_fallback and self.reasoning:
            return self.reasoning
        return None


def build_body(cfg: ModelConfig, messages: list[dict], max_tokens: int,
               temperature: float | None, extra_body: dict | None) -> dict:
    """The request body the server sees, so probes and legs request the same shape."""
    body = {
        "model": cfg.name,
        "messages": messages,
        "max_tokens": min(max_tokens, cfg.token_limit),
        "temperature": temperature if temperature is not None else cfg.temperature,
    }
    if extra_body:
        body.update(extra_body)
    return body


def post_chat(cfg: ModelConfig, messages: list[dict], max_tokens: int = 2048,
              temperature: float | None = None,
              extra_body: dict | None = None) -> ChatResponse:
    """POST one chat request. Failures come back as a ChatResponse with `error` set."""
    request = urllib.request.Request(
        cfg.chat_url,
        data=json.dumps(build_body(cfg, messages, max_tokens, temperature, extra_body)).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
            **cfg.extra_headers,
        },
        method="POST",
    )
    started = time.monotonic()
    status, payload, error = _post_json(request, cfg.timeout_s)
    return ChatResponse(status=status, payload=payload,
                        elapsed_s=time.monotonic() - started, error=error)


def get_json(url: str, api_key: str = "not-needed",
             timeout_s: int = 15) -> tuple[int | None, dict | None, str | None]:
    """GET a JSON endpoint. Returns (status, payload, error) without raising."""
    request = urllib.request.Request(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    return _post_json(request, timeout_s)


def chat_request(
    cfg: ModelConfig,
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float | None = None,
    extra_body: dict | None = None,
    allow_reasoning_fallback: bool = True,
) -> str | None:
    """Send one chat request. Returns the answer text, or None if there is none.

    Transport and protocol failures are reported on stderr and return None so a
    long run can record the failure and continue. Pass
    allow_reasoning_fallback=False to score only the `content` channel.
    """
    response = post_chat(cfg, messages, max_tokens, temperature, extra_body)
    if not response.ok:
        if response.error:
            print(f"  request failed: {response.error}", file=sys.stderr)
        return None
    if not response.message:
        print("  request failed: response has no choices[0].message", file=sys.stderr)
        return None
    return response.answer(allow_reasoning_fallback)


def _post_json(request: urllib.request.Request,
               timeout_s: int) -> tuple[int | None, dict | None, str | None]:
    """Send a request and decode the JSON body. Never raises. Returns (status, payload, error)."""
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.status, json.loads(response.read()), None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        return exc.code, None, f"HTTP {exc.code} {detail}"
    except urllib.error.URLError as exc:
        return None, None, str(exc.reason)
    except json.JSONDecodeError as exc:
        return None, None, f"non-JSON response ({exc})"
    except OSError as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def verify_endpoint(endpoint: str, model: str, api_key: str = "not-needed",
                    timeout_s: int = 15) -> bool:
    """Check that the endpoint answers /models and lists the model."""
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/models",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            models = json.loads(response.read()).get("data", [])
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        print(f"  endpoint check failed: {exc}", file=sys.stderr)
        return False
    return any(model in entry.get("id", "") for entry in models)
