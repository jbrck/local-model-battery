"""HTTP client behaviour, checked against a live stub server."""
from __future__ import annotations

import io
import os
import socket
import sys
import threading
import unittest
from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.lib.model import (  # noqa: E402
    ChatResponse,
    ModelConfig,
    build_body,
    chat_request,
    get_json,
    post_chat,
    verify_endpoint,
)
from tests.stub_json import JsonStub, completion  # noqa: E402


def cfg_for(stub: JsonStub, **overrides) -> ModelConfig:
    fields = {"name": "stub-model", "endpoint": stub.endpoint, "token_limit": 4096,
              "timeout_s": 5}
    fields.update(overrides)
    return ModelConfig(**fields)


class ResponseParsingTest(unittest.TestCase):
    def test_parses_content_reasoning_and_usage(self):
        response = ChatResponse(status=200, payload=completion("the answer", "thinking...", 77))
        self.assertTrue(response.ok)
        self.assertEqual(response.content, "the answer")
        self.assertEqual(response.reasoning, "thinking...")
        self.assertEqual(response.completion_tokens, 77)
        self.assertEqual(response.finish_reason, "stop")

    def test_reasoning_falls_back_to_the_alternate_field_name(self):
        payload = completion("")
        payload["choices"][0]["message"] = {"role": "assistant", "reasoning": "  pondering  "}
        response = ChatResponse(status=200, payload=payload)
        self.assertEqual(response.reasoning, "pondering")

    def test_malformed_payload_is_empty_not_an_exception(self):
        response = ChatResponse(status=200, payload={"nonsense": True})
        self.assertEqual(response.message, {})
        self.assertEqual(response.content, "")
        self.assertEqual(response.completion_tokens, 0)
        self.assertIsNone(response.answer())

    def test_unparseable_token_count_is_zero_not_an_exception(self):
        # Servers in the wild send strings or lists where the usage numbers belong; the
        # count feeds arithmetic in the pilot, so it must degrade to 0.
        payload = completion("x")
        payload["usage"]["completion_tokens"] = "not-a-number"
        self.assertEqual(ChatResponse(status=200, payload=payload).completion_tokens, 0)
        payload["usage"]["completion_tokens"] = [1, 2]
        self.assertEqual(ChatResponse(status=200, payload=payload).completion_tokens, 0)

    def test_failed_response_is_not_ok_and_carries_the_error(self):
        response = ChatResponse(status=500, payload=None, error="HTTP 500 boom")
        self.assertFalse(response.ok)
        self.assertIsNone(response.answer())

    def test_answer_prefers_content_over_reasoning(self):
        response = ChatResponse(status=200, payload=completion("content", "reasoning"))
        self.assertEqual(response.answer(), "content")

    def test_answer_can_be_restricted_to_the_content_channel(self):
        payload = completion("")
        payload["choices"][0]["message"]["reasoning_content"] = "only reasoning here"
        response = ChatResponse(status=200, payload=payload)
        self.assertIsNone(response.answer(allow_reasoning_fallback=False))


class BuildBodyTest(unittest.TestCase):
    def test_body_caps_max_tokens_at_the_config_limit(self):
        cfg = ModelConfig(name="m", token_limit=1024)
        self.assertEqual(build_body(cfg, [], 99999, None, None)["max_tokens"], 1024)

    def test_temperature_comes_from_the_config_when_not_given(self):
        cfg = ModelConfig(name="m", temperature=0.0)
        self.assertEqual(build_body(cfg, [], 10, None, None)["temperature"], 0.0)
        self.assertEqual(build_body(cfg, [], 10, 0.9, None)["temperature"], 0.9)

    def test_extra_body_is_merged_last(self):
        body = build_body(ModelConfig(name="m"), [], 10, None, {"top_k": 5, "model": "over"})
        self.assertEqual(body["top_k"], 5)
        self.assertEqual(body["model"], "over")


class LiveRequestTest(unittest.TestCase):
    def test_successful_request_returns_ok_response(self):
        routes = {"/v1/chat/completions": (200, completion("hello", tokens=40))}
        with JsonStub(routes) as stub:
            response = post_chat(cfg_for(stub), [{"role": "user", "content": "hi"}], max_tokens=64)
        self.assertTrue(response.ok)
        self.assertEqual(response.content, "hello")
        self.assertEqual(response.completion_tokens, 40)
        self.assertGreater(response.elapsed_s, 0)

    def test_request_body_reaches_the_server_as_built(self):
        routes = {"/v1/chat/completions": (200, completion())}
        with JsonStub(routes) as stub:
            cfg = cfg_for(stub, token_limit=2048)
            post_chat(cfg, [{"role": "user", "content": "hi"}], max_tokens=100,
                      temperature=0.0, extra_body={"seed": 42})
            sent = stub.seen[-1]["body"]
        self.assertEqual(sent["model"], "stub-model")
        self.assertEqual(sent["max_tokens"], 100)
        self.assertEqual(sent["temperature"], 0.0)
        self.assertEqual(sent["seed"], 42)

    def test_http_error_comes_back_as_text_not_an_exception(self):
        routes = {"/v1/chat/completions": (400, {"error": "grammar and json_schema"})}
        with JsonStub(routes) as stub:
            response = post_chat(cfg_for(stub), [{"role": "user", "content": "hi"}])
        self.assertFalse(response.ok)
        self.assertEqual(response.status, 400)
        self.assertIn("HTTP 400", response.error or "")
        self.assertIn("grammar", response.error or "")

    def test_dead_endpoint_reports_the_failure(self):
        cfg = ModelConfig(name="m", endpoint="http://127.0.0.1:9/v1", timeout_s=2)
        response = post_chat(cfg, [{"role": "user", "content": "hi"}])
        self.assertFalse(response.ok)
        self.assertTrue(response.error)

    def test_chat_request_returns_none_when_there_is_no_answer(self):
        payload = completion("", None)
        payload["choices"][0]["message"] = {"role": "assistant", "content": ""}
        routes = {"/v1/chat/completions": (200, payload)}
        with JsonStub(routes) as stub:
            self.assertIsNone(chat_request(cfg_for(stub), [{"role": "user", "content": "hi"}]))

    def test_chat_request_falls_back_to_reasoning_by_default(self):
        payload = completion("")
        payload["choices"][0]["message"] = {"role": "assistant", "reasoning_content": "trace"}
        routes = {"/v1/chat/completions": (200, payload)}
        with JsonStub(routes) as stub:
            answer = chat_request(cfg_for(stub), [{"role": "user", "content": "hi"}])
        self.assertEqual(answer, "trace")


class GetJsonTest(unittest.TestCase):
    def test_returns_status_payload_and_no_error(self):
        routes = {"/props": (200, {"total_slots": 4})}
        with JsonStub(routes) as stub:
            status, payload, error = get_json(stub.path("/props"))
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"total_slots": 4})
        self.assertIsNone(error)

    def test_missing_route_reports_the_status(self):
        with JsonStub({}) as stub:
            status, payload, error = get_json(stub.path("/props"))
        self.assertEqual(status, 404)
        self.assertIsNone(payload)
        self.assertIn("HTTP 404", error or "")


def _raw_handler(status: int, body: bytes) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            self.send_response(status)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:  # type: ignore[override]
            """Silence the default request log."""

    return Handler


class RawStub:
    """Answers every request with literal bytes, so tests can send bodies that are not JSON."""

    def __init__(self, status: int, body: bytes):
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _raw_handler(status, body))
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def __enter__(self) -> "RawStub":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def _closing_port() -> int:
    """A listener that accepts one connection and closes it without answering."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def accept_and_drop() -> None:
        conn, _ = srv.accept()
        conn.close()
        srv.close()

    threading.Thread(target=accept_and_drop, daemon=True).start()
    return port


class TransportFailureTest(unittest.TestCase):
    """The failure branches of _post_json, through real sockets: these decide whether a
    long battery run records a clean failure or dies."""

    def test_non_json_body_reports_a_decode_error(self):
        with RawStub(200, b"<html>bad gateway</html>") as raw:
            status, payload, error = get_json(f"{raw.url}/v1/models")
        self.assertIsNone(status)
        self.assertIsNone(payload)
        self.assertIn("non-JSON response", error or "")

    def test_connection_dropped_without_a_response_is_reported_not_raised(self):
        status, payload, error = get_json(f"http://127.0.0.1:{_closing_port()}/v1/models")
        self.assertIsNone(status)
        self.assertIsNone(payload)
        self.assertTrue(error)


class ChatRequestFailureTest(unittest.TestCase):
    """chat_request reports transport and protocol failures on stderr and returns None,
    so a leg can log the miss and keep the run going."""

    def test_transport_failure_is_reported_on_stderr_and_returns_none(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            cfg = ModelConfig(name="m", endpoint="http://127.0.0.1:9/v1", timeout_s=2)
            answer = chat_request(cfg, [{"role": "user", "content": "hi"}])
        self.assertIsNone(answer)
        self.assertIn("request failed", stderr.getvalue())

    def test_http_500_is_reported_on_stderr_and_returns_none(self):
        routes = {"/v1/chat/completions": (500, {"error": "model crashed"})}
        stderr = io.StringIO()
        with JsonStub(routes) as stub, redirect_stderr(stderr):
            answer = chat_request(cfg_for(stub), [{"role": "user", "content": "hi"}])
        self.assertIsNone(answer)
        self.assertIn("HTTP 500", stderr.getvalue())

    def test_response_without_a_message_is_reported_and_returns_none(self):
        routes = {"/v1/chat/completions": (200, {"choices": []})}
        stderr = io.StringIO()
        with JsonStub(routes) as stub, redirect_stderr(stderr):
            answer = chat_request(cfg_for(stub), [{"role": "user", "content": "hi"}])
        self.assertIsNone(answer)
        self.assertIn("no choices[0].message", stderr.getvalue())


class VerifyEndpointTest(unittest.TestCase):
    def test_endpoint_listing_the_model_verifies(self):
        payload = {"data": [{"id": "other-model"}, {"id": "stub-model"}]}
        with JsonStub({"/v1/models": (200, payload)}) as stub:
            self.assertTrue(verify_endpoint(stub.endpoint, "stub-model"))

    def test_model_missing_from_the_list_does_not_verify(self):
        payload = {"data": [{"id": "other-model"}]}
        with JsonStub({"/v1/models": (200, payload)}) as stub:
            self.assertFalse(verify_endpoint(stub.endpoint, "stub-model"))

    def test_dead_endpoint_does_not_verify_and_says_so_on_stderr(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            ok = verify_endpoint("http://127.0.0.1:9/v1", "stub-model", timeout_s=2)
        self.assertFalse(ok)
        self.assertIn("endpoint check failed", stderr.getvalue())

    def test_non_json_models_body_does_not_verify(self):
        stderr = io.StringIO()
        with RawStub(200, b"garbage") as raw, redirect_stderr(stderr):
            ok = verify_endpoint(raw.url, "stub-model")
        self.assertFalse(ok)
        self.assertIn("endpoint check failed", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
