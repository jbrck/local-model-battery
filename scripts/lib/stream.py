"""Streaming chat client: splits one request into time to first token and decode time.

The battery's normal client waits for the whole body, so one number covers prefill and
decode together. At a long prompt the prefill dominates that number and hides the decode
rate, which is the thing a depth test exists to measure. This client reads the event
stream and stamps the first and last token as they arrive.

Decode rate is (completion tokens - 1) / (last token time - first token time). The first
token is the end of prefill, so it opens the decode window instead of counting inside it.
Token counts come from the server's usage block. A server that sends none gets a chunk
count instead and `usage_reported` is False, because speculative decoding packs several
tokens into one chunk and a chunk count would then read low.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterable

from scripts.lib.model import ChatResponse, ModelConfig, build_body

DATA_PREFIX = b"data:"
DONE_MARKER = b"[DONE]"


def as_count(value: object) -> int:
    """A usage figure as a non-negative int. Anything a server sends that is not one is zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return 0
    try:
        return max(int(value), 0)
    except ValueError:
        return 0


@dataclass
class StreamTiming:
    """One streamed request: the response the counters need, plus where the time went."""

    response: ChatResponse
    ttft_s: float = 0.0
    decode_s: float = 0.0
    chunks: int = 0
    usage_reported: bool = False
    server_timings: dict = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        return as_count(self.response.usage.get("prompt_tokens"))

    @property
    def decode_tps(self) -> float:
        """Tokens per second across the decode window, or 0 when there was no window."""
        tokens = self.response.completion_tokens
        if tokens < 2 or self.decode_s <= 0:
            return 0.0
        return (tokens - 1) / self.decode_s


@dataclass
class _Accumulator:
    """What the event loop has seen so far."""

    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    chunks: int = 0
    first_at: float | None = None
    last_at: float | None = None


def _delta_text(event: dict) -> tuple[str, str]:
    """(content, reasoning) carried by one event, either of which may be empty."""
    choices = event.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    raw = choice.get("delta")
    delta = raw if isinstance(raw, dict) else {}
    reasoning = str(delta.get("reasoning_content") or delta.get("reasoning") or "")
    return str(delta.get("content") or ""), reasoning


def _note_tokens(acc: _Accumulator, event: dict, now: float) -> None:
    """Stamp the arrival of any content or reasoning the event carries."""
    content, reasoning = _delta_text(event)
    if not (content or reasoning):
        return
    acc.chunks += 1
    acc.first_at = now if acc.first_at is None else acc.first_at
    acc.last_at = now
    acc.content.append(content)
    acc.reasoning.append(reasoning)


def _note_finish(acc: _Accumulator, event: dict) -> None:
    """Keep the finish reason when the event has one."""
    choices = event.get("choices") or []
    first = choices[0] if choices and isinstance(choices[0], dict) else {}
    acc.finish_reason = str(first.get("finish_reason") or acc.finish_reason)


def _note_meta(acc: _Accumulator, event: dict) -> None:
    """Keep the usage block and the server's own timings when the event has them."""
    if isinstance(event.get("usage"), dict):
        acc.usage = event["usage"]
    if isinstance(event.get("timings"), dict):
        acc.timings = event["timings"]


def _absorb(acc: _Accumulator, event: dict, now: float) -> None:
    """Fold one decoded event into the accumulator."""
    _note_tokens(acc, event, now)
    _note_finish(acc, event)
    _note_meta(acc, event)


def _decode_line(raw: bytes) -> dict | None:
    """The JSON event on one `data:` line, or None for blanks, comments and the end marker."""
    line = raw.strip()
    if not line.startswith(DATA_PREFIX):
        return None
    body = line[len(DATA_PREFIX):].strip()
    if not body or body == DONE_MARKER:
        return None
    try:
        event = json.loads(body)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def _response_from(acc: _Accumulator, started: float, ended: float) -> tuple[ChatResponse, bool]:
    """The ChatResponse the Tally reads, and whether the usage block was the server's own."""
    usage = dict(acc.usage)
    reported = bool(usage.get("completion_tokens"))
    if not reported:
        usage["completion_tokens"] = acc.chunks
    message: dict = {"role": "assistant", "content": "".join(acc.content)}
    if any(acc.reasoning):
        message["reasoning_content"] = "".join(acc.reasoning)
    payload = {"choices": [{"index": 0, "message": message,
                            "finish_reason": acc.finish_reason or "stop"}],
               "usage": usage}
    return ChatResponse(status=200, payload=payload, elapsed_s=ended - started), reported


def parse_stream(lines: Iterable[bytes], clock: Callable[[], float] = time.monotonic,
                 started: float | None = None) -> StreamTiming:
    """Read an event stream to its end and time it. The clock is injectable for tests."""
    begin = clock() if started is None else started
    acc = _Accumulator()
    for raw in lines:
        event = _decode_line(raw)
        if event is not None:
            _absorb(acc, event, clock())
    ended = clock()
    response, reported = _response_from(acc, begin, ended)
    first = acc.first_at if acc.first_at is not None else ended
    last = acc.last_at if acc.last_at is not None else first
    return StreamTiming(response=response, ttft_s=first - begin, decode_s=last - first,
                        chunks=acc.chunks, usage_reported=reported, server_timings=acc.timings)


def stream_chat(cfg: ModelConfig, messages: list[dict], max_tokens: int,
                temperature: float | None = None,
                extra_body: dict | None = None) -> StreamTiming:
    """POST one streamed chat request. Failures come back as a response with `error` set."""
    body = build_body(cfg, messages, max_tokens, temperature, extra_body)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(
        cfg.chat_url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg.api_key}", **cfg.extra_headers})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout_s) as stream:
            return parse_stream(stream, started=started)
    except urllib.error.HTTPError as exc:
        with exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
        return _failed(exc.code, f"HTTP {exc.code} {detail}", started)
    except urllib.error.URLError as exc:
        return _failed(None, str(exc.reason), started)
    except OSError as exc:
        return _failed(None, f"{type(exc).__name__}: {exc}", started)


def _failed(status: int | None, error: str, started: float) -> StreamTiming:
    """A timing for a request that never produced a stream."""
    return StreamTiming(response=ChatResponse(status=status, payload=None, error=error,
                                              elapsed_s=time.monotonic() - started))
