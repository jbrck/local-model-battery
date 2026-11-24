"""A local server that streams chat completions as events, with depth-dependent timing.

Prefill delay and per-token delay both grow with the prompt, so a test can assert that the
leg reports decode rate falling with depth and time to first token rising. The copy task is
answered by echoing the end of the reference text, so copy_ratio has something real to read.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CHARS_PER_TOKEN = 4
REFERENCE_MARK = "Reference text:\n\n"
TASK_MARK = "\n\n---\n\n"


def prompt_of(body: dict) -> str:
    """The user turn of a request body."""
    return str(body["messages"][-1]["content"])


def prompt_tokens(prompt: str) -> int:
    return max(1, len(prompt) // CHARS_PER_TOKEN)


def words_for(prompt: str, count: int) -> list[str]:
    """What the stub 'generates': the reference tail for a copy task, counted words otherwise."""
    if "Reproduce the final" in prompt:
        reference = prompt.split(REFERENCE_MARK, 1)[1].rsplit(TASK_MARK, 1)[0].split()
        return reference[-count:]
    return [f"word{index}" for index in range(count)]


class SseStub:
    """Streams `max_tokens` one-word chunks per request, then a usage event and [DONE]."""

    def __init__(self, prefill_s_per_ktok: float = 0.0, token_s: float = 0.0,
                 token_s_per_ktok: float = 0.0, send_usage: bool = True,
                 status: int = 200) -> None:
        self.prefill_s_per_ktok = prefill_s_per_ktok
        self.token_s = token_s
        self.token_s_per_ktok = token_s_per_ktok
        self.send_usage = send_usage
        self.status = status
        self.seen: list[dict] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server naming
                length = int(self.headers.get("Content-Length") or 0)
                stub.stream(self, json.loads(self.rfile.read(length)))

            # The base class names this parameter `format`, which shadows a builtin.
            def log_message(self, fmt: str, *args: object) -> None:  # type: ignore[override]
                """Silence the default request log."""

        return Handler

    def stream(self, handler: BaseHTTPRequestHandler, body: dict) -> None:
        """Record the request, wait out the prefill, then send the events."""
        self.seen.append(body)
        if self.status != 200:
            handler.send_error(self.status, "stub failure")
            return
        prompt = prompt_of(body)
        ktok = prompt_tokens(prompt) / 1000
        time.sleep(self.prefill_s_per_ktok * ktok)
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        count = int(body["max_tokens"])
        pause = self.token_s + self.token_s_per_ktok * ktok
        for word in words_for(prompt, count):
            self._send(handler, {"choices": [{"index": 0, "delta": {"content": word + " "}}]})
            time.sleep(pause)
        self._send(handler, {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
        if self.send_usage:
            self._send(handler, {"choices": [], "timings": {"predicted_per_second": 1.0 / max(pause, 1e-6)},
                                 "usage": {"prompt_tokens": prompt_tokens(prompt),
                                           "completion_tokens": count}})
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()

    @staticmethod
    def _send(handler: BaseHTTPRequestHandler, event: dict) -> None:
        handler.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        handler.wfile.flush()

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def __enter__(self) -> "SseStub":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
