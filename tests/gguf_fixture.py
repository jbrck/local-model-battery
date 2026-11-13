"""Build minimal valid GGUF headers and HF model directories for tests.

The tests need models with shapes that real files on this box happen to have: a hybrid
model, a per-layer KV-head array, a fractional head width. Writing the header by hand keeps
the tests fast and independent of which models are on disk.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

MAGIC = b"GGUF"
VERSION = 3
TYPE_UINT32 = 4
TYPE_FLOAT32 = 6
TYPE_BOOL = 7
TYPE_STRING = 8
TYPE_ARRAY = 9
TYPE_INT64 = 11


def _write_string(handle, text: str) -> None:
    encoded = text.encode("utf-8")
    handle.write(struct.pack("<Q", len(encoded)))
    handle.write(encoded)


def _scalar_type(value) -> int:
    if isinstance(value, str):
        return TYPE_STRING
    if isinstance(value, bool):
        return TYPE_BOOL
    if isinstance(value, int):
        return TYPE_UINT32 if 0 <= value < 2 ** 32 else TYPE_INT64
    if isinstance(value, float):
        return TYPE_FLOAT32
    raise TypeError(f"no GGUF scalar type for {value!r}")


def _write_typed(handle, kind: int, value) -> None:
    if kind == TYPE_STRING:
        _write_string(handle, value)
    elif kind == TYPE_BOOL:
        handle.write(struct.pack("<?", bool(value)))
    elif kind == TYPE_UINT32:
        handle.write(struct.pack("<I", int(value)))
    elif kind == TYPE_INT64:
        handle.write(struct.pack("<q", int(value)))
    elif kind == TYPE_FLOAT32:
        handle.write(struct.pack("<f", float(value)))
    else:  # pragma: no cover - the fixture builder only emits the types above
        raise TypeError(f"fixture cannot write type {kind}")


def _write_value(handle, value) -> None:
    if isinstance(value, str):
        handle.write(struct.pack("<I", TYPE_STRING))
        _write_string(handle, value)
        return
    if isinstance(value, list):
        element = _scalar_type(value[0]) if value else TYPE_UINT32
        handle.write(struct.pack("<I", TYPE_ARRAY))
        handle.write(struct.pack("<I", element))
        handle.write(struct.pack("<Q", len(value)))
        for item in value:
            _write_typed(handle, element, item)
        return
    kind = _scalar_type(value)
    handle.write(struct.pack("<I", kind))
    _write_typed(handle, kind, value)


def write_gguf(path: str | Path, metadata: dict, pad_bytes: int = 0) -> Path:
    """Write a header-only GGUF. Tensor data does not exist, which readers must not need."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<I", VERSION))
        handle.write(struct.pack("<Q", 0))
        handle.write(struct.pack("<Q", len(metadata)))
        for key, value in metadata.items():
            _write_string(handle, key)
            _write_value(handle, value)
        handle.write(b"\0" * pad_bytes)
    return target


def write_hf_model(directory: str | Path, config: dict, shard_bytes: int = 0,
                   shards: int = 0) -> Path:
    """Write a model directory with a config and optional safetensors shards."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for index in range(shards):
        (target / f"model-{index:05d}-of-{shards:05d}.safetensors").write_bytes(
            b"\0" * shard_bytes)
    return target


QWEN_METADATA = {
    "general.architecture": "qwen3moe",
    "qwen3moe.block_count": 48,
    "qwen3moe.embedding_length": 2048,
    "qwen3moe.attention.head_count": 16,
    "qwen3moe.attention.head_count_kv": 4,
    "qwen3moe.context_length": 32768,
}
