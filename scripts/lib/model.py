"""Shared model endpoint configuration."""
from dataclasses import dataclass, field
from typing import Optional
import json, time, urllib.request, urllib.error


@dataclass
class ModelConfig:
    name: str
    endpoint: str = "http://localhost:11434/v1"
    api_key: str = "not-needed"
    token_limit: int = 4096
    temperature: float = 0.2
    extra_headers: dict[str, str] = field(default_factory=dict)

    @property
    def chat_url(self) -> str:
        return f"{self.rstrip('/')}/chat/completions"


def chat_request(cfg: ModelConfig, messages: list[dict], max_tokens: int = 2048,
                 temperature: float | None = None, extra_body: dict | None = None) -> str | None:
    """Send a chat request to an OpenAI-compatible endpoint. Returns content or None."""
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
    req = urllib.request.Request(
        f"{cfg.endpoint.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        # Handle both content and reasoning_content
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        if not content.strip():
            content = msg.get("reasoning_content", "") or ""
        return content if content.strip() else None
    except Exception as exc:
        # Quiet fail — caller decides what to do with None
        return None


def verify_endpoint(endpoint: str, model: str) -> bool:
    """Check the endpoint is live and model is loaded."""
    try:
        req = urllib.request.Request(
            f"{endpoint.rstrip('/')}/models",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            models = json.loads(resp.read()).get("data", [])
        return any(model in m.get("id", "") for m in models)
    except Exception:
        return False