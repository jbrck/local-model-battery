"""HTTP client for any OpenAI-compatible chat-completions endpoint."""
from __future__ import annotations

import json
import sys
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
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
        **cfg.extra_headers,
    }
    body = {
        "model": cfg.name,
        "messages": messages,
        "max_tokens": min(max_tokens, cfg.token_limit),
        "temperature": temperature if temperature is not None else cfg.temperature,
    }
    if extra_body:
        body.update(extra_body)
    request = urllib.request.Request(
        cfg.chat_url,
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    payload = _post_json(request, cfg.timeout_s)
    if payload is None:
        return None

    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        print("  request failed: response has no choices[0].message", file=sys.stderr)
        return None

    content = (message.get("content") or "").strip()
    if content:
        return content
    if allow_reasoning_fallback:
        reasoning = (message.get("reasoning_content") or "").strip()
        if reasoning:
            return reasoning
    return None


def _post_json(request: urllib.request.Request, timeout_s: int) -> dict | None:
    """POST a request and decode the JSON body, reporting failures on stderr."""
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        print(f"  request failed: HTTP {exc.code} {detail}", file=sys.stderr)
    except urllib.error.URLError as exc:
        print(f"  request failed: {exc.reason}", file=sys.stderr)
    except json.JSONDecodeError as exc:
        print(f"  request failed: non-JSON response ({exc})", file=sys.stderr)
    return None


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
