"""Engine identification: which probe answered decides the engine, never the model name."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.engine_facts import (  # noqa: E402
    EngineFacts,
    _ctx_from_model_info,
    _num_ctx_from_parameters,
    _ollama_ctx,
    base_url,
    display_name,
    probe_engine,
)
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


class DisplayNameTest(unittest.TestCase):
    def test_styled_names_and_the_passthrough(self):
        self.assertEqual(display_name("vllm"), "vLLM")
        self.assertEqual(display_name("sglang"), "SGLang")
        self.assertEqual(display_name("mlx"), "MLX")
        self.assertEqual(display_name("llama.cpp"), "llama.cpp")
        self.assertEqual(display_name("unknown"), "unknown")


class OllamaCtxTest(unittest.TestCase):
    def test_the_first_num_ctx_line_wins(self):
        show = {"parameters": "num_keep_alive 24\nnum_ctx 4096\nnum_ctx 999"}
        self.assertEqual(_ollama_ctx(show), 4096)

    def test_a_parameters_block_that_is_not_text_falls_through(self):
        self.assertIsNone(_ollama_ctx({"parameters": 12, "model_info": {}}))
        self.assertIsNone(_ollama_ctx({}))

    def test_model_info_that_is_not_a_dict_states_nothing(self):
        self.assertIsNone(_ollama_ctx({"model_info": "n/a"}))

    def test_non_integer_context_lengths_are_ignored(self):
        show = {"model_info": {"m.context_length": "big", "other": 5}}
        self.assertIsNone(_ollama_ctx(show))

    def test_a_stated_but_unreadable_num_ctx_does_not_fall_through(self):
        stated, window = _num_ctx_from_parameters("num_ctx bananas")
        self.assertTrue(stated)
        self.assertIsNone(window)
        self.assertEqual(_ollama_ctx({"parameters": "num_ctx bananas",
                                      "model_info": {"a.context_length": 99}}), None)

    def test_no_num_ctx_line_is_reported_as_not_stated(self):
        stated, window = _num_ctx_from_parameters("stop <|eot|>")
        self.assertFalse(stated)
        self.assertIsNone(window)
        self.assertEqual(_num_ctx_from_parameters(None), (False, None))

    def test_a_context_length_under_any_architecture_prefix_is_read(self):
        self.assertEqual(_ctx_from_model_info({"qwen3.context_length": 4096}), 4096)
        self.assertIsNone(_ctx_from_model_info({"context_length": 4096}))
        self.assertIsNone(_ctx_from_model_info({"m.context_length": "big"}))


class SerialisationTest(unittest.TestCase):
    def test_to_dict_carries_every_field_and_lists_the_errors(self):
        facts = EngineFacts(engine="llama.cpp", version="b1", window=8192, slots=1,
                            kv_type="q8_0", flash_attn=True, model_path="m.gguf",
                            model_dir="/models", supports_kv_flags=True, drafter="mtp",
                            source="props", errors=("get_model_info: HTTP 404",))
        payload = facts.to_dict()
        self.assertEqual(payload["engine"], "llama.cpp")
        self.assertEqual(payload["version"], "b1")
        self.assertEqual(payload["window"], 8192)
        self.assertEqual(payload["slots"], 1)
        self.assertEqual(payload["kv_type"], "q8_0")
        self.assertTrue(payload["flash_attn"])
        self.assertEqual(payload["model_path"], "m.gguf")
        self.assertEqual(payload["model_dir"], "/models")
        self.assertTrue(payload["supports_kv_flags"])
        self.assertEqual(payload["drafter"], "mtp")
        self.assertEqual(payload["source"], "props")
        self.assertEqual(payload["errors"], ["get_model_info: HTTP 404"])

    def test_the_default_facts_serialise_with_their_nones(self):
        payload = EngineFacts(errors=("props: refused",)).to_dict()
        self.assertEqual(payload["engine"], "unknown")
        self.assertIsNone(payload["window"])
        self.assertIsNone(payload["slots"])
        self.assertIsNone(payload["kv_type"])
        self.assertIsNone(payload["flash_attn"])
        self.assertIsNone(payload["model_path"])
        self.assertIsNone(payload["model_dir"])
        self.assertFalse(payload["supports_kv_flags"])
        self.assertIsNone(payload["drafter"])
        self.assertEqual(payload["source"], "")
        self.assertEqual(payload["errors"], ["props: refused"])

    def test_an_engine_that_answered_something_is_known(self):
        self.assertTrue(EngineFacts(engine="openai-compatible").known)
        self.assertFalse(EngineFacts().known)


if __name__ == "__main__":
    unittest.main()
