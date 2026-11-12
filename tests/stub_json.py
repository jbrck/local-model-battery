"""A local HTTP server that replays canned JSON, for tests that need a live endpoint.

Stands in for llama.cpp or any OpenAI-compatible server: fixed routes, recorded
requests, and an optional delay so timing has something to measure.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

Route = tuple[int, object]


def completion(content: str = "42", reasoning: str | None = None, tokens: int = 100,
               finish_reason: str = "stop") -> dict:
    """An OpenAI-shaped completion body."""
    message: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 12, "completion_tokens": tokens, "total_tokens": tokens + 12},
    }


class JsonStub:
    """Serves `routes` keyed by path, or `default` when the path is unknown."""

    def __init__(self, routes: dict[str, Route] | Callable[[dict], Route],
                 default: Route = (404, {"error": "no such route"}), delay_s: float = 0.0):
        self.routes = routes
        self.default = default
        self.delay_s = delay_s
        self.seen: list[dict] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server naming
                stub.respond(self, None)

            def do_POST(self) -> None:  # noqa: N802 - http.server naming
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    body = {"_raw": raw.decode("utf-8", "replace")}
                stub.respond(self, body)

            # The base class names this parameter `format`, which shadows a builtin.
            def log_message(self, fmt: str, *args: object) -> None:  # type: ignore[override]
                """Silence the default request log."""

        return Handler

    def respond(self, handler: BaseHTTPRequestHandler, body: dict | None) -> None:
        """Record the request, then answer from the route table."""
        self.seen.append({"path": handler.path, "body": body})
        if self.delay_s:
            time.sleep(self.delay_s)
        route = self.routes
        if callable(route):
            status, payload = route(body or {})
        else:
            status, payload = route.get(handler.path, self.default)
        data = json.dumps(payload).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    @property
    def endpoint(self) -> str:
        """The OpenAI-style base URL a ModelConfig expects."""
        return f"{self.url}/v1"

    def path(self, path: str) -> str:
        return f"{self.url}{path}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> "JsonStub":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
