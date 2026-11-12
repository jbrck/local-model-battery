"""Endpoint probes: the window, the slots, the thinking switch and the tool grammar path."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.lib.model import ModelConfig  # noqa: E402
from scripts.lib.server_probe import (  # noqa: E402
    ServerFacts,
    _thinking_switch_moved,
    probe_server,
    props_url,
)
from tests.stub_json import JsonStub, completion  # noqa: E402


def props(ctx: int = 131072, slots: int = 4, template: str = "") -> dict:
    return {
        "default_generation_settings": {"n_ctx": ctx},
        "total_slots": slots,
        "chat_template": template,
    }


class PropsUrlTest(unittest.TestCase):
    def test_v1_suffix_is_stripped_for_props(self):
        cfg = ModelConfig(name="m", endpoint="http://host:11434/v1")
        self.assertEqual(props_url(cfg), "http://host:11434/props")

    def test_endpoint_without_v1_is_left_alone(self):
        cfg = ModelConfig(name="m", endpoint="http://host:11434")
        self.assertEqual(props_url(cfg), "http://host:11434/props")

    def test_trailing_slash_does_not_double_up(self):
        cfg = ModelConfig(name="m", endpoint="http://host:11434/v1/")
        self.assertEqual(props_url(cfg), "http://host:11434/props")


class SwitchMathTest(unittest.TestCase):
    def test_a_measurable_difference_counts_as_a_switch(self):
        self.assertTrue(_thinking_switch_moved(1200, 200))

    def test_identical_counts_are_not_a_switch(self):
        self.assertFalse(_thinking_switch_moved(300, 300))

    def test_no_tokens_at_all_is_not_a_switch(self):
        self.assertFalse(_thinking_switch_moved(0, 0))

    def test_small_differences_do_not_count(self):
        self.assertFalse(_thinking_switch_moved(300, 305))


class ProbeTest(unittest.TestCase):
    def test_props_provide_window_and_slots(self):
        routes = {"/props": (200, props(ctx=65536, slots=2)),
                  "/v1/chat/completions": (200, completion("ok", tokens=120))}
        with JsonStub(routes) as stub:
            cfg = ModelConfig(name="stub", endpoint=stub.endpoint, timeout_s=5)
            facts = probe_server(cfg, run_thinking=False, run_tools=False)
        self.assertTrue(facts.reachable)
        self.assertEqual(facts.ctx, 65536)
        self.assertEqual(facts.slots, 2)
        self.assertTrue(facts.window_is_known)
        self.assertEqual(facts.to_dict()["ctx"], 65536)

    def test_chat_template_hint_marks_the_thinking_switch(self):
        routes = {"/props": (200, props(template="{{ enable_thinking }}")),
                  "/v1/chat/completions": (200, completion("ok", tokens=120))}
        with JsonStub(routes) as stub:
            cfg = ModelConfig(name="stub", endpoint=stub.endpoint, timeout_s=5)
            facts = probe_server(cfg, run_tools=False)
        self.assertTrue(facts.chat_template_mentions_thinking)
        self.assertTrue(facts.thinking_switch)

    def test_a_template_that_ignores_the_flag_is_not_a_switch(self):
        routes = {"/props": (200, props(template="{{ messages }}")),
                  "/v1/chat/completions": (200, completion("ok", tokens=120))}
        with JsonStub(routes) as stub:
            cfg = ModelConfig(name="stub", endpoint=stub.endpoint, timeout_s=5)
            facts = probe_server(cfg, run_tools=False)
        self.assertFalse(facts.thinking_switch)
        self.assertEqual(facts.thinking_on_tokens, facts.thinking_off_tokens)

    def test_the_reasoning_channel_is_detected_and_compared(self):
        def routed(body: dict) -> tuple[int, dict]:
            flags = (body or {}).get("chat_template_kwargs") or {}
            if flags.get("enable_thinking") is False:
                return 200, completion("4", reasoning="", tokens=30)
            return 200, completion("4", reasoning="r" * 400, tokens=1400)

        with JsonStub(routed) as stub:
            cfg = ModelConfig(name="stub", endpoint=stub.endpoint, timeout_s=5)
            facts = probe_server(cfg, run_tools=False)
        self.assertTrue(facts.reasoning_channel)
        self.assertTrue(facts.thinking_switch)
        self.assertGreater(facts.thinking_on_tokens or 0, facts.thinking_off_tokens or 0)

    def test_a_server_without_props_still_reports_reachable_through_models(self):
        routes = {"/v1/models": (200, {"data": [{"id": "some-model"}]}),
                  "/v1/chat/completions": (200, completion("ok", tokens=50))}
        with JsonStub(routes) as stub:
            cfg = ModelConfig(name="stub", endpoint=stub.endpoint, timeout_s=5)
            facts = probe_server(cfg, run_thinking=False, run_tools=False)
        self.assertTrue(facts.reachable)
        self.assertEqual(facts.served_model, "some-model")
        self.assertIsNone(facts.ctx)
        self.assertFalse(facts.window_is_known)

    def test_dead_endpoint_is_reported_as_unreachable_with_a_reason(self):
        cfg = ModelConfig(name="stub", endpoint="http://127.0.0.1:9/v1", timeout_s=2)
        facts = probe_server(cfg)
        self.assertFalse(facts.reachable)
        self.assertTrue(facts.errors)

    def test_tools_with_json_schema_are_reported_when_accepted(self):
        routes = {"/props": (200, props()),
                  "/v1/chat/completions": (200, completion("ok", tokens=60))}
        with JsonStub(routes) as stub:
            cfg = ModelConfig(name="stub", endpoint=stub.endpoint, timeout_s=5)
            facts = probe_server(cfg, run_thinking=False)
        self.assertTrue(facts.tools_with_json_schema_ok)
        self.assertIsNone(facts.tools_error)

    def test_engine_that_rejects_grammar_plus_schema_is_named(self):
        routes = {"/props": (200, props()),
                  "/v1/chat/completions": (400, {"error": "grammar and json_schema are "
                                                          "mutually exclusive"})}
        with JsonStub(routes) as stub:
            cfg = ModelConfig(name="stub", endpoint=stub.endpoint, timeout_s=5)
            facts = probe_server(cfg, run_thinking=False)
        self.assertFalse(facts.tools_with_json_schema_ok)
        self.assertIn("grammar", facts.tools_error or "")
        self.assertTrue(any("engine" in error for error in facts.errors))


class ModelsFallbackTest(unittest.TestCase):
    """The /models fallback, for engines that serve no usable /props."""

    def setUp(self):
        self.routes: dict = {}

    def probe(self) -> ServerFacts:
        with JsonStub(self.routes) as stub:
            cfg = ModelConfig(name="stub", endpoint=stub.endpoint, timeout_s=5)
            return probe_server(cfg, run_thinking=False, run_tools=False)

    def test_an_empty_model_list_is_not_reachable(self):
        self.routes = {"/v1/models": (200, {"data": []})}
        facts = self.probe()
        self.assertFalse(facts.reachable)
        self.assertIsNone(facts.served_model)

    def test_a_model_list_that_is_not_an_object_is_not_reachable(self):
        self.routes = {"/v1/models": (200, ["not", "an", "object"])}
        facts = self.probe()
        self.assertFalse(facts.reachable)

    def test_a_data_field_that_is_not_a_list_is_not_reachable(self):
        self.routes = {"/v1/models": (200, {"data": "unexpected"})}
        facts = self.probe()
        self.assertFalse(facts.reachable)

    def test_an_entry_without_an_id_is_not_reachable(self):
        self.routes = {"/v1/models": (200, {"data": [{"object": "model"}]})}
        facts = self.probe()
        self.assertFalse(facts.reachable)

    def test_a_failing_models_call_is_reported_with_a_reason(self):
        self.routes = {"/v1/models": (500, {"error": "boom"})}
        facts = self.probe()
        self.assertFalse(facts.reachable)
        self.assertTrue(any("models" in error for error in facts.errors))


class PropsShapeTest(unittest.TestCase):
    """/props payloads that answer but do not match the expected shape."""

    def probe(self, props_payload: object) -> ServerFacts:
        routes = {"/props": (200, props_payload)}
        with JsonStub(routes) as stub:
            cfg = ModelConfig(name="stub", endpoint=stub.endpoint, timeout_s=5)
            return probe_server(cfg, run_thinking=False, run_tools=False)

    def test_loosely_typed_fields_read_as_unknown(self):
        facts = self.probe({"default_generation_settings": "bogus",
                            "total_slots": "4", "chat_template": 5})
        self.assertTrue(facts.reachable)
        self.assertIsNone(facts.ctx)
        self.assertIsNone(facts.slots)
        self.assertIsNone(facts.chat_template_mentions_thinking)
        self.assertFalse(facts.window_is_known)

    def test_a_props_payload_that_is_not_an_object_reports_no_window(self):
        facts = self.probe(["unexpected"])
        self.assertTrue(facts.reachable)
        self.assertIsNone(facts.ctx)
        self.assertIsNone(facts.props)


if __name__ == "__main__":
    unittest.main()
