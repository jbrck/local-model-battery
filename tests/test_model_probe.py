"""Tests for the model reader: GGUF header, HF config, and shape resolution order."""
from __future__ import annotations

import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.lib import model_probe
from scripts.lib.model_probe import (
    SOURCE_GGUF,
    SOURCE_HF,
    SOURCE_PROFILE,
    SOURCE_UNKNOWN,
    ModelShape,
    read_gguf_metadata,
    read_gguf_shape,
    read_hf_config,
    resolve_model_shape,
    shape_from_profile,
)
from tests.gguf_fixture import QWEN_METADATA, write_gguf, write_hf_model

# Mirrors Swift-Qwen3.8-27B: 65 blocks, one full-attention layer every fourth block, and an
# explicit key_length that differs from embedding_length / head_count (213.33).
SWIFT_METADATA = {
    "general.architecture": "qwen3next",
    "qwen3next.block_count": 65,
    "qwen3next.embedding_length": 5120,
    "qwen3next.attention.head_count": 24,
    "qwen3next.attention.head_count_kv": 4,
    "qwen3next.attention.key_length": 256,
    "qwen3next.context_length": 262144,
    "qwen3next.attention.recurrent_layers": [i % 4 != 3 for i in range(65)],
}
# Mirrors LFM2.5-24B-A2B: the KV head count arrives as a per-layer array, not a scalar.
LFM_METADATA = {
    "general.architecture": "lfm2moe",
    "lfm2moe.block_count": 40,
    "lfm2moe.embedding_length": 2048,
    "lfm2moe.attention.head_count": 32,
    "lfm2moe.attention.head_count_kv": [8 if i % 4 == 0 else 0 for i in range(40)],
}
QWEN_DIR_CONFIG = {
    "model_type": "qwen3",
    "num_hidden_layers": 62,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "hidden_size": 8192,
    "head_dim": 128,
    "max_position_embeddings": 262144,
}
# Mirrors the GSQ-RCO quant on this box: declares the full-attention interval and the SSM
# geometry, but not the per-layer recurrent flags.
INTERVAL_ONLY_METADATA = {
    "general.architecture": "qwen35",
    "qwen35.block_count": 65,
    "qwen35.embedding_length": 5120,
    "qwen35.attention.head_count": 24,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256,
    "qwen35.context_length": 262144,
    "qwen35.full_attention_interval": 4,
    "qwen35.ssm.conv_kernel": 4,
    "qwen35.ssm.inner_size": 6144,
    "qwen35.ssm.state_size": 128,
    "qwen35.ssm.group_count": 16,
}
# Mirrors Gemma-4-12B: 40 of 48 layers are windowed at 1024 tokens, with a narrower width.
GEMMA_METADATA = {
    "general.architecture": "gemma4",
    "gemma4.block_count": 48,
    "gemma4.embedding_length": 3840,
    "gemma4.attention.head_count": 16,
    "gemma4.attention.head_count_kv": [8] * 48,
    "gemma4.attention.key_length": 512,
    "gemma4.attention.key_length_swa": 256,
    "gemma4.attention.value_length": 512,
    "gemma4.attention.sliding_window": 1024,
    "gemma4.attention.sliding_window_pattern": [True] * 40 + [False] * 8,
    "gemma4.context_length": 131072,
}


class ShapeCase(unittest.TestCase):
    """A temp directory per test, so no fixture outlives its case."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def gguf(self, metadata: dict, name: str = "model.gguf") -> Path:
        return write_gguf(self.tmp / name, metadata)


class GgufHeaderTests(ShapeCase):
    def test_dense_model_shape(self):
        path = self.gguf(QWEN_METADATA)
        shape = read_gguf_shape(path)
        self.assertEqual(shape.arch, "qwen3moe")
        self.assertEqual(shape.n_layers, 48)
        self.assertEqual(shape.n_kv_heads, 4)
        self.assertEqual(shape.head_dim, 128)
        self.assertEqual(shape.native_ctx, 32768)
        self.assertEqual(shape.weights_bytes, path.stat().st_size)
        self.assertEqual(shape.source, SOURCE_GGUF)
        self.assertTrue(shape.known)

    def test_declared_key_length_beats_the_derived_head_dim(self):
        path = self.gguf(SWIFT_METADATA)
        shape = read_gguf_shape(path)
        self.assertEqual(shape.head_dim, 256)
        self.assertNotEqual(shape.head_dim, 5120 // 24)

    def test_hybrid_recurrent_layers_reduce_the_cache_count(self):
        path = self.gguf(SWIFT_METADATA)
        shape = read_gguf_shape(path)
        expected = sum(1 for i in range(65) if i % 4 == 3)
        self.assertEqual(shape.n_layers, expected)
        self.assertLess(shape.n_layers, 65)
        self.assertTrue(any("recurrent" in note for note in shape.notes))

    def test_per_layer_kv_head_array_counts_attention_layers(self):
        path = self.gguf(LFM_METADATA)
        shape = read_gguf_shape(path)
        self.assertEqual(shape.n_layers, 10)
        self.assertEqual(shape.n_kv_heads, 8)

    def test_fractional_head_dim_is_reported_not_guessed(self):
        metadata = dict(QWEN_METADATA)
        metadata["qwen3moe.embedding_length"] = 5120
        metadata["qwen3moe.attention.head_count"] = 24
        metadata.pop("qwen3moe.attention.head_count_kv")
        shape = read_gguf_shape(self.gguf(metadata))
        self.assertEqual(shape.head_dim, 0)
        self.assertFalse(shape.known)
        self.assertTrue(any("head_dim" in problem for problem in shape.problems))

    def test_missing_file_is_reported(self):
        shape = read_gguf_shape(self.tmp / "absent.gguf")
        self.assertFalse(shape.known)
        self.assertTrue(any("not found" in problem for problem in shape.problems))

    def test_non_gguf_file_is_reported(self):
        path = self.tmp / "notes.gguf"
        path.write_text("not a model", encoding="utf-8")
        shape = read_gguf_shape(path)
        self.assertFalse(shape.known)
        self.assertTrue(any("not a GGUF" in problem for problem in shape.problems))

    def test_truncated_header_is_reported(self):
        path = self.tmp / "truncated.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                         + struct.pack("<Q", 40))
        shape = read_gguf_shape(path)
        self.assertFalse(shape.known)
        self.assertTrue(any("truncated" in problem for problem in shape.problems))

    def test_unknown_metadata_type_is_rejected(self):
        reader = model_probe._Reader(io.BytesIO(struct.pack("<I", 99)), 4)
        with self.assertRaises(ValueError):
            model_probe._read_value(reader, 99)

    def test_large_tokenizer_array_is_not_retained(self):
        metadata = dict(QWEN_METADATA)
        metadata["tokenizer.ggml.tokens"] = [f"tok{i}" for i in range(9000)]
        path = self.gguf(metadata)
        raw = read_gguf_metadata(path)
        self.assertIsNone(raw["tokenizer.ggml.tokens"])
        self.assertTrue(read_gguf_shape(path).known)

    def test_value_length_mismatch_is_noted(self):
        metadata = dict(SWIFT_METADATA)
        metadata["qwen3next.attention.value_length"] = 128
        shape = read_gguf_shape(self.gguf(metadata))
        self.assertTrue(any("value_length" in note for note in shape.notes))

    def test_metadata_roundtrip_covers_every_fixture_type(self):
        path = self.gguf({"a": True, "b": 7, "c": 1.5, "d": "x", "e": [1, 2],
                          "f": [True, False]})
        raw = read_gguf_metadata(path)
        self.assertEqual(raw["a"], True)
        self.assertEqual(raw["b"], 7)
        self.assertAlmostEqual(raw["c"], 1.5)
        self.assertEqual(raw["d"], "x")
        self.assertEqual(raw["e"], [1, 2])
        self.assertEqual(raw["f"], [True, False])


class HfConfigTests(ShapeCase):
    def test_dense_config_shape(self):
        directory = write_hf_model(self.tmp / "qwen", QWEN_DIR_CONFIG, shard_bytes=1000,
                                   shards=2)
        shape = read_hf_config(directory)
        self.assertEqual(shape.n_layers, 62)
        self.assertEqual(shape.n_kv_heads, 8)
        self.assertEqual(shape.head_dim, 128)
        self.assertEqual(shape.native_ctx, 262144)
        self.assertEqual(shape.weights_bytes, 2000)
        self.assertEqual(shape.source, SOURCE_HF)
        self.assertTrue(shape.known)

    def test_hybrid_layer_types_reduce_the_cache_count(self):
        config = dict(QWEN_DIR_CONFIG)
        config["layer_types"] = ["full_attention" if i % 4 == 3 else "linear_attention"
                                 for i in range(40)]
        shape = read_hf_config(write_hf_model(self.tmp / "hybrid", config, 500, 1))
        self.assertEqual(shape.n_layers, 10)
        self.assertTrue(any("linear-attention" in note for note in shape.notes))

    def test_head_dim_derives_when_not_declared(self):
        config = dict(QWEN_DIR_CONFIG)
        config.pop("head_dim")
        shape = read_hf_config(write_hf_model(self.tmp / "derived", config, 500, 1))
        self.assertEqual(shape.head_dim, 8192 // 64)

    def test_missing_safetensors_is_reported(self):
        shape = read_hf_config(write_hf_model(self.tmp / "empty", QWEN_DIR_CONFIG))
        self.assertEqual(shape.weights_bytes, 0)
        self.assertFalse(shape.known)
        self.assertTrue(any("safetensors" in problem for problem in shape.problems))

    def test_missing_config_is_reported(self):
        directory = self.tmp / "bare"
        directory.mkdir()
        shape = read_hf_config(directory)
        self.assertFalse(shape.known)
        self.assertTrue(any("config.json" in problem for problem in shape.problems))

    def test_bad_json_is_reported(self):
        directory = self.tmp / "badjson"
        directory.mkdir()
        (directory / "config.json").write_text("{not json", encoding="utf-8")
        shape = read_hf_config(directory)
        self.assertFalse(shape.known)
        self.assertTrue(any("unreadable" in problem for problem in shape.problems))

    def test_non_object_config_is_reported(self):
        directory = self.tmp / "listjson"
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps([1, 2]), encoding="utf-8")
        shape = read_hf_config(directory)
        self.assertFalse(shape.known)
        self.assertTrue(any("not an object" in problem for problem in shape.problems))

    def test_fractional_head_dim_is_reported(self):
        config = dict(QWEN_DIR_CONFIG)
        config.pop("head_dim")
        config["hidden_size"] = 1000
        config["num_attention_heads"] = 3
        shape = read_hf_config(write_hf_model(self.tmp / "fractional", config, 500, 1))
        self.assertEqual(shape.head_dim, 0)
        self.assertTrue(any("head_dim" in problem for problem in shape.problems))


class HybridAndWindowTests(ShapeCase):
    """The cost model for models whose layers do not all behave the same way."""

    def test_full_attention_interval_is_used_when_recurrent_layers_is_absent(self):
        # The GSQ-RCO quant of Qwen3.8-27B declares the interval but not the layer flags.
        shape = read_gguf_shape(self.gguf(INTERVAL_ONLY_METADATA))
        self.assertEqual(shape.n_layers, 17)
        self.assertEqual(shape.n_kv_heads, 4)
        self.assertEqual(shape.head_dim, 256)
        self.assertTrue(any("every 4 blocks" in note for note in shape.notes))

    def test_recurrent_layers_win_over_the_interval(self):
        metadata = dict(SWIFT_METADATA)
        metadata["qwen3next.full_attention_interval"] = 2
        shape = read_gguf_shape(self.gguf(metadata))
        self.assertEqual(shape.n_layers, sum(1 for i in range(65) if i % 4 == 3))

    def test_state_space_state_is_a_fixed_cost(self):
        shape = read_gguf_shape(self.gguf(INTERVAL_ONLY_METADATA))
        self.assertGreater(shape.kv_fixed_elements, 0)
        self.assertTrue(any("state-space" in note for note in shape.notes))

    def test_sliding_window_layers_do_not_scale_with_the_window(self):
        shape = read_gguf_shape(self.gguf(GEMMA_METADATA))
        self.assertEqual(shape.n_layers, 48)
        self.assertEqual(shape.head_dim, 512)
        self.assertEqual(shape.kv_elements_per_token, 8 * 2 * 8 * 512)
        self.assertEqual(shape.kv_fixed_elements, 40 * 2 * 8 * 256 * 1024)
        self.assertTrue(any("sliding-window" in note for note in shape.notes))

    def test_gemma_cache_is_six_times_smaller_than_counting_every_layer(self):
        shape = read_gguf_shape(self.gguf(GEMMA_METADATA))
        as_if_uniform = 48 * 2 * 8 * 512
        self.assertEqual(as_if_uniform, shape.kv_elements_per_token * 6)
        self.assertEqual(shape.kv_bytes_per_token(8), 65536)

    def test_uniform_model_has_no_fixed_cost(self):
        shape = read_gguf_shape(self.gguf(QWEN_METADATA))
        self.assertEqual(shape.kv_fixed_elements, 0)
        self.assertEqual(shape.kv_elements_per_token, 2 * 48 * 4 * 128)
        self.assertEqual(shape.kv_fixed_bytes(8), 0)

    def test_cache_type_scales_the_per_token_cost(self):
        shape = read_gguf_shape(self.gguf(QWEN_METADATA))
        self.assertEqual(shape.kv_bytes_per_token(16), 2 * shape.kv_bytes_per_token(8))

    def test_sliding_window_without_a_pattern_is_ignored(self):
        metadata = dict(GEMMA_METADATA)
        metadata.pop("gemma4.attention.sliding_window_pattern")
        shape = read_gguf_shape(self.gguf(metadata))
        self.assertEqual(shape.kv_fixed_elements, 0)
        self.assertEqual(shape.kv_elements_per_token, 48 * 2 * 8 * 512)

    def test_hf_sliding_window_layers_are_a_fixed_cost(self):
        config = dict(QWEN_DIR_CONFIG)
        config["num_hidden_layers"] = 48
        config["sliding_window"] = 1024
        config["layer_types"] = ["sliding_attention"] * 40 + ["full_attention"] * 8
        shape = read_hf_config(write_hf_model(self.tmp / "swa", config, 1000, 1))
        self.assertEqual(shape.kv_elements_per_token, 8 * 2 * 8 * 128)
        self.assertEqual(shape.kv_fixed_elements, 40 * 2 * 8 * 128 * 1024)


class ProfileShapeTests(unittest.TestCase):
    def test_shape_from_profile(self):
        shape = shape_from_profile({"weights_gb": 16.0,
                                    "model": {"n_layers": 65, "n_kv_heads": 4,
                                              "head_dim": 213}})
        self.assertEqual(shape.source, SOURCE_PROFILE)
        self.assertEqual(shape.n_layers, 65)
        self.assertTrue(shape.known)
        self.assertTrue(any("hand" in note for note in shape.notes))

    def test_profile_without_weights_is_not_known(self):
        shape = shape_from_profile({"model": {"n_layers": 65, "n_kv_heads": 4,
                                             "head_dim": 213}})
        self.assertFalse(shape.known)
        self.assertTrue(any("weights_gb" in problem for problem in shape.problems))

    def test_profile_with_non_dict_model_is_tolerated(self):
        shape = shape_from_profile({"weights_gb": 1.0, "model": "oops"})
        self.assertFalse(shape.known)


class ResolveModelShapeTests(ShapeCase):
    def test_explicit_file_wins(self):
        path = self.gguf(QWEN_METADATA)
        shape = resolve_model_shape(model_file=str(path),
                                    props={"model_path": "/nowhere/other.gguf"})
        self.assertEqual(shape.source, SOURCE_GGUF)
        self.assertEqual(shape.path, str(path))

    def test_argument_beats_env_override(self):
        explicit = self.gguf(QWEN_METADATA, "explicit.gguf")
        with mock.patch.dict(os.environ, {model_probe.ENV_MODEL_FILE: str(
                self.gguf(SWIFT_METADATA, "env.gguf"))}):
            shape = resolve_model_shape(model_file=str(explicit))
        self.assertEqual(shape.path, str(explicit))

    def test_env_override_used_when_no_argument(self):
        path = self.gguf(SWIFT_METADATA, "env.gguf")
        with mock.patch.dict(os.environ, {model_probe.ENV_MODEL_FILE: str(path)}):
            shape = resolve_model_shape()
        self.assertEqual(shape.path, str(path))
        self.assertEqual(shape.n_layers, sum(1 for i in range(65) if i % 4 == 3))

    def test_props_model_path_is_used(self):
        path = self.gguf(QWEN_METADATA, "served.gguf")
        with mock.patch.dict(os.environ, {}, clear=True):
            shape = resolve_model_shape(props={"model_path": str(path)})
        self.assertEqual(shape.path, str(path))
        self.assertEqual(shape.n_layers, 48)

    def test_props_model_dir_uses_the_hf_reader(self):
        directory = write_hf_model(self.tmp / "served", QWEN_DIR_CONFIG, 1000, 2)
        with mock.patch.dict(os.environ, {}, clear=True):
            shape = resolve_model_shape(props={"model_path": str(directory)})
        self.assertEqual(shape.source, SOURCE_HF)
        self.assertEqual(shape.weights_bytes, 2000)

    def test_bad_explicit_path_is_reported_and_falls_through(self):
        path = self.gguf(QWEN_METADATA, "good.gguf")
        with mock.patch.dict(os.environ, {}, clear=True):
            shape = resolve_model_shape(model_file=str(self.tmp / "typo.gguf"),
                                        props={"model_path": str(path)})
        self.assertEqual(shape.path, str(path))
        self.assertTrue(any("typo.gguf" in problem for problem in shape.problems))

    def test_profile_is_the_last_resort(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            shape = resolve_model_shape(profile={"weights_gb": 16.0,
                                                 "model": {"n_layers": 65, "n_kv_heads": 4,
                                                           "head_dim": 213}})
        self.assertEqual(shape.source, SOURCE_PROFILE)

    def test_unknown_when_nothing_is_available(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            shape = resolve_model_shape()
        self.assertEqual(shape.source, SOURCE_UNKNOWN)
        self.assertFalse(shape.known)
        self.assertTrue(any("no model file" in problem for problem in shape.problems))

    def test_props_without_a_model_path_is_tolerated(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            shape = resolve_model_shape(props={"model_path": 7})
        self.assertEqual(shape.source, SOURCE_UNKNOWN)


class ShapeSerialisationTests(unittest.TestCase):
    def test_to_dict_carries_the_source_and_notes(self):
        shape = ModelShape(arch="qwen3moe", n_layers=17, n_kv_heads=4, head_dim=256,
                           weights_bytes=1024 ** 3, native_ctx=32768, source=SOURCE_GGUF,
                           path="/models/x.gguf", notes=("n",), problems=("p",))
        payload = shape.to_dict()
        self.assertEqual(payload["weights_gb"], 1.0)
        self.assertEqual(payload["source"], SOURCE_GGUF)
        self.assertEqual(payload["notes"], ["n"])
        self.assertEqual(payload["problems"], ["p"])

    def test_known_requires_every_kv_number(self):
        def shape(kv_elements_per_token: int, weights_bytes: int = 10) -> ModelShape:
            return ModelShape(arch="a", n_layers=1, n_kv_heads=1, head_dim=1,
                              kv_elements_per_token=kv_elements_per_token,
                              weights_bytes=weights_bytes, source=SOURCE_GGUF)

        self.assertTrue(shape(2).known)
        self.assertFalse(shape(0).known)
        self.assertFalse(shape(2, weights_bytes=0).known)


if __name__ == "__main__":
    unittest.main()
