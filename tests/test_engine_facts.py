"""Engine identification: which probe answered decides the engine, never the model name."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.engine_facts import base_url, probe_engine  # noqa: E402
from scripts.lib.model import ModelConfig  # noqa: E402
from tests.stub_json import JsonStub  # noqa: E402

PROPS = {
    "build_info": "b10909-a2878d30d",
    "total_slots": 1,
    "model_path": "/mnt/models/qwen/qwen3-8b-q4_k_m.gguf",
    "chat_template": "{% if enable_thinking %}",
    "default_generation_settings": {"n_ctx": 131072},
}
SHOW = {
    "parameters": "num_keep_alive                      24\nnum_ctx                        8192",
    "model_info": {"qwen3.context_length": 32768, "general.file_type": "Q4_K_M"},
}


def cfg(endpoint: str) -> ModelConfig:
    """A config pointed at the stub, with a model name that proves nothing."""
    return ModelConfig(name="arete-27b", endpoint=endpoint)


class LlamaCppTest(unittest.TestCase):
    def test_props_identify_llama_cpp_and_yield_its_facts(self):
        with JsonStub(routes={"/props": (200, PROPS)}) as stub:
            facts = probe_engine(cfg(stub.endpoint))
        self.assertEqual(facts.engine, "llama.cpp")
        self.assertEqual(facts.version, "b10909-a2878d30d")
        self.assertEqual(facts.window, 131072)
        self.assertEqual(facts.slots, 1)
        self.assertTrue(facts.model_path.endswith(".gguf"))
        self.assertTrue(facts.supports_kv_flags)
        self.assertEqual(facts.source, "props")
        self.assertTrue(facts.known)

    def test_an_empty_props_answer_still_identifies_the_engine(self):
        """Reachable but silent: the probe decides the engine, not the fields in it."""
        with JsonStub(routes={"/props": (200, {})}) as stub:
            facts = probe_engine(cfg(stub.endpoint))
        self.assertEqual(facts.engine, "llama.cpp")
        self.assertEqual(facts.version, "")
        self.assertIsNone(facts.window)
        self.assertTrue(facts.known)
        self.assertEqual(facts.errors, ())


class OllamaTest(unittest.TestCase):
    def test_api_show_identifies_ollama_and_reads_num_ctx(self):
        with JsonStub(routes={"/api/show": (200, SHOW)}) as stub:
            facts = probe_engine(cfg(stub.endpoint))
        self.assertEqual(facts.engine, "ollama")
        # parameters.num_ctx is what the server was told to use; the model_info window is
        # what the model advertises, and the served one is the fact.
        self.assertEqual(facts.window, 8192)
        self.assertEqual(facts.source, "api/show")
        self.assertFalse(facts.supports_kv_flags)
        self.assertTrue(facts.known)


class VllmTest(unittest.TestCase):
    def test_version_and_models_identify_vllm(self):
        routes = {"/version": (200, {"version": "0.6.6"}),
                  "/v1/models": (200, {"data": [{"id": "orca-27b", "max_model_len": 65536}]})}
        with JsonStub(routes=routes) as stub:
            facts = probe_engine(cfg(stub.endpoint))
        self.assertEqual(facts.engine, "vllm")
        self.assertEqual(facts.version, "0.6.6")
        self.assertEqual(facts.window, 65536)
        self.assertEqual(facts.source, "version + models")
        self.assertFalse(facts.supports_kv_flags)
        self.assertTrue(facts.known)


class SglangTest(unittest.TestCase):
    def test_get_model_info_identifies_sglang(self):
        routes = {"/get_model_info": (200, {"model_path": "/models/org-27b",
                                            "context_length": 40960})}
        with JsonStub(routes=routes) as stub:
            facts = probe_engine(cfg(stub.endpoint))
        self.assertEqual(facts.engine, "sglang")
        self.assertEqual(facts.window, 40960)
        self.assertEqual(facts.model_path, "/models/org-27b")
        self.assertEqual(facts.source, "get_model_info")
        self.assertTrue(facts.known)


class OpenAiCompatibleTest(unittest.TestCase):
    def test_models_only_identifies_a_compatible_endpoint(self):
        """The models-only row yields the id alone: no window is claimed from it."""
        routes = {"/v1/models": (200, {"data": [{"id": "alias-27b", "max_model_len": 65536}]})}
        with JsonStub(routes=routes) as stub:
            facts = probe_engine(cfg(stub.endpoint))
        self.assertEqual(facts.engine, "openai-compatible")
        self.assertEqual(facts.version, "")
        self.assertIsNone(facts.window)
        self.assertIsNone(facts.slots)
        self.assertFalse(facts.supports_kv_flags)
        self.assertEqual(facts.source, "models")
        self.assertTrue(facts.known)


class FallbackTest(unittest.TestCase):
    def test_props_404_falls_through_to_the_models_list(self):
        """/props refusing does not end the probe chain: /models still answers."""
        routes = {"/models": (200, {"data": [{"id": "routed-27b"}]})}
        with JsonStub(routes=routes) as stub:
            # An endpoint given without /v1: the base handling has to leave it alone.
            facts = probe_engine(cfg(stub.url))
        self.assertEqual(facts.engine, "openai-compatible")
        self.assertTrue(facts.known)
        self.assertTrue(any("props: HTTP 404" in error for error in facts.errors))


class UnknownTest(unittest.TestCase):
    def test_nothing_answering_is_unknown_with_what_was_tried(self):
        with JsonStub(routes={}) as stub:
            facts = probe_engine(cfg(stub.endpoint))
        self.assertEqual(facts.engine, "unknown")
        self.assertFalse(facts.known)
        self.assertIsNone(facts.window)
        self.assertFalse(facts.supports_kv_flags)
        self.assertEqual(facts.source, "")
        tried = " ".join(facts.errors)
        for probe in ("props", "api/show", "version", "models", "get_model_info"):
            self.assertIn(probe, tried)


class BaseUrlTest(unittest.TestCase):
    def test_a_trailing_v1_is_stripped_like_props_url_strips_it(self):
        self.assertEqual(base_url(cfg("http://h:11434/v1/")), "http://h:11434")
        self.assertEqual(base_url(cfg("http://h:4000")), "http://h:4000")


if __name__ == "__main__":
    unittest.main()
