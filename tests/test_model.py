"""HTTP client behaviour, checked against a live stub server."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.lib.model import (  # noqa: E402
    ChatResponse,
    ModelConfig,
    build_body,
    chat_request,
    get_json,
    post_chat,
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


if __name__ == "__main__":
    unittest.main()
