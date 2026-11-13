"""Read a model's shape and size from the model itself, not from a hand-written profile.

The reference runs typed their model shape into a profile by hand. Three of those numbers
were wrong for models already on the box:

  * `head_dim = n_embd / n_head` gave 213.33 where the file declares 256.
  * the layer count included recurrent layers that hold no growing KV cache, which
    overstated the cache by 4x on Arete-27B (64 layers, 16 of them attention).
  * sliding-window layers were counted as full-context layers, which overstated
    Gemma-4-12B by 6x (48 layers, most of them windowed at 1024 tokens).

Together those are the difference between "shrink the window" and "run it at the
standard's window". Everything here reads the file. Nothing guesses. A number that cannot
be read is reported as a problem rather than filled in with something plausible.

KV cost is reported as two element counts rather than one byte count, because the two parts
behave differently: `kv_elements_per_token` scales with the window, `kv_fixed_elements` is
the bounded part (windowed attention and state-space state) that does not. Multiplying by
the standard's cache type is the caller's job, since the cache type is a run decision.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, replace
from pathlib import Path

MAGIC = b"GGUF"
TYPE_STRING = 8
TYPE_ARRAY = 9
SCALAR_FORMATS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
# Arrays larger than this are tokenizer vocabularies. They are consumed and discarded so a
# real file does not cost a hundred megabytes of memory to describe.
MAX_RETAINED_ARRAY = 4096
BYTES_PER_GB = 1024 ** 3
ENV_MODEL_FILE = "BATTERY_MODEL_FILE"
GGUF_SUFFIX = ".gguf"
HF_CONFIG_NAME = "config.json"
SAFETENSORS_SUFFIX = ".safetensors"

SOURCE_GGUF = "gguf"
SOURCE_HF = "hf_config"
SOURCE_PROFILE = "profile"
SOURCE_UNKNOWN = "unknown"

# A KV tensor is a K and a V per layer, so the per-layer cost doubles.
TENSORS_PER_LAYER = 2


@dataclass(frozen=True)
class ModelShape:
    """The numbers the fit math needs, plus where each one came from.

    `n_layers` is every layer that attends. The KV cost is the two element counts, because a
    recurrent or windowed layer costs something different from a full-context one.
    """

    arch: str = ""
    n_layers: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    kv_elements_per_token: int = 0
    kv_fixed_elements: int = 0
    weights_bytes: int = 0
    native_ctx: int = 0
    source: str = SOURCE_UNKNOWN
    path: str = ""
    notes: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()

    @property
    def known(self) -> bool:
        """True when the fit math has every number it needs."""
        return (self.kv_elements_per_token > 0 and self.weights_bytes > 0)

    def kv_bytes_per_token(self, kv_bits: int) -> int:
        """Cache bytes for one token of window, at the run's chosen cache type."""
        return self.kv_elements_per_token * kv_bits // 8

    def kv_fixed_bytes(self, kv_bits: int) -> int:
        """Cache bytes that do not grow with the window: windowed layers and state."""
        return self.kv_fixed_elements * kv_bits // 8

    def to_dict(self) -> dict:
        return {
            "arch": self.arch, "n_layers": self.n_layers, "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim, "weights_gb": round(self.weights_bytes / BYTES_PER_GB, 2),
            "kv_elements_per_token": self.kv_elements_per_token,
            "kv_fixed_elements": self.kv_fixed_elements,
            "native_ctx": self.native_ctx, "source": self.source, "path": self.path,
            "notes": list(self.notes), "problems": list(self.problems),
        }


class _Reader:
    """A cursor over a GGUF header. Reads little-endian, never seeks backwards."""

    def __init__(self, handle, size: int) -> None:
        self.handle = handle
        self.size = size

    def raw(self, count: int) -> bytes:
        data = self.handle.read(count)
        if len(data) != count:
            raise ValueError(f"header truncated: wanted {count} bytes, got {len(data)}")
        return data

    def number(self, fmt: str):
        return struct.unpack(fmt, self.raw(struct.calcsize(fmt)))[0]

    def u32(self) -> int:
        return self.number("<I")

    def u64(self) -> int:
        return self.number("<Q")

    def text(self) -> str:
        return self.raw(self.u64()).decode("utf-8", "replace")


def _read_value(reader: _Reader, value_type: int):
    """One metadata value. Arrays over MAX_RETAINED_ARRAY are consumed but not kept."""
    if value_type == TYPE_STRING:
        return reader.text()
    if value_type == TYPE_ARRAY:
        element_type = reader.u32()
        count = reader.u64()
        if count > MAX_RETAINED_ARRAY:
            for _ in range(count):
                _read_value(reader, element_type)
            return None
        return [_read_value(reader, element_type) for _ in range(count)]
    fmt = SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise ValueError(f"unknown metadata type {value_type}")
    return reader.number(fmt)


def read_gguf_metadata(path: str | Path) -> dict:
    """Every string-keyed metadata pair in a GGUF header. Tensor data is never touched."""
    target = Path(path)
    with target.open("rb") as handle:
        reader = _Reader(handle, target.stat().st_size)
        if reader.raw(4) != MAGIC:
            raise ValueError(f"{target} is not a GGUF file")
        reader.u32()
        reader.u64()
        count = reader.u64()
        metadata = {}
        for _ in range(count):
            key = reader.text()
            metadata[key] = _read_value(reader, reader.u32())
    return metadata


def _arch(metadata: dict) -> str:
    value = metadata.get("general.architecture")
    return str(value) if value else ""


def _meta(metadata: dict, arch: str, suffix: str):
    """A metadata value by its short name, under the model's own architecture prefix."""
    if not arch:
        return None
    return metadata.get(f"{arch}.{suffix}")


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return int(value)


def _flags(raw: object) -> list[bool]:
    """A per-layer boolean list, or an empty list when the value is not one."""
    if not isinstance(raw, list):
        return []
    return [bool(flag) for flag in raw]


def _attention_from_per_layer_heads(metadata: dict, arch: str, notes: list[str]) -> int | None:
    """Attention layers counted from a per-layer KV head array, or None when absent.

    A layer with zero KV heads holds no cache, so a mixed array says exactly which layers
    attend. A uniform array says nothing and is left to the readers below.
    """
    per_layer_heads = _meta(metadata, arch, "attention.head_count_kv")
    if not isinstance(per_layer_heads, list) or not per_layer_heads:
        return None
    active = sum(1 for head in per_layer_heads if _as_int(head) > 0)
    if not 0 < active < len(per_layer_heads):
        return None
    notes.append(f"{active} of {len(per_layer_heads)} layers carry a KV cache "
                 f"(per-layer head counts); the rest have none")
    return active


def _attention_from_recurrent_layers(metadata: dict, arch: str, notes: list[str]) -> int | None:
    """Attention layers counted from a per-layer recurrent flag list, or None when absent."""
    recurrent = _flags(_meta(metadata, arch, "attention.recurrent_layers"))
    if not recurrent or all(recurrent):
        return None
    active = sum(1 for flag in recurrent if not flag)
    notes.append(f"{active} of {len(recurrent)} layers are attention layers; "
                 f"the rest are recurrent and hold no growing cache")
    return active


def _attention_from_interval(metadata: dict, arch: str, block_count: int,
                             notes: list[str]) -> int | None:
    """Attention layers counted from the full-attention interval, or None when absent.

    One full-attention layer every `interval` blocks. Verified against the three qwen35
    files on this box, which report 17, 16 and 17 attention layers for block counts 65, 64
    and 65 at interval 4.
    """
    interval = _as_int(_meta(metadata, arch, "full_attention_interval"))
    if interval <= 1 or block_count <= 0:
        return None
    active = (block_count + interval - 1) // interval
    notes.append(f"one full-attention layer every {interval} blocks gives {active} of "
                 f"{block_count} layers; the rest are recurrent and hold no growing cache")
    return active


def _attention_layer_count(metadata: dict, arch: str, block_count: int) -> tuple[int, list[str]]:
    """How many layers hold a growing, full-context KV cache.

    Hybrid models interleave recurrent or state-space layers with attention layers. Those
    layers carry a fixed-size state, not a cache that grows with the window, so counting
    them inflates the estimate by the ratio between the two. Three different files in the
    reference collection declare this three different ways, so all three are read.
    """
    notes: list[str] = []
    active = _attention_from_per_layer_heads(metadata, arch, notes)
    if active is None:
        active = _attention_from_recurrent_layers(metadata, arch, notes)
    if active is None:
        active = _attention_from_interval(metadata, arch, block_count, notes)
    return (block_count if active is None else active), notes


def _kv_heads(metadata: dict, arch: str) -> int:
    value = _meta(metadata, arch, "attention.head_count_kv")
    if isinstance(value, list):
        return max((_as_int(head) for head in value), default=0)
    return _as_int(value)


def _head_dim(metadata: dict, arch: str) -> tuple[int, list[str]]:
    """The per-head width, taken from the file rather than derived, when the file says it."""
    problems: list[str] = []
    declared = _as_int(_meta(metadata, arch, "attention.key_length"))
    if declared > 0:
        return declared, problems
    embedding = _as_int(_meta(metadata, arch, "embedding_length"))
    heads = _as_int(_meta(metadata, arch, "attention.head_count"))
    if heads > 0 and embedding % heads == 0:
        return embedding // heads, problems
    problems.append("head_dim is not declared and embedding_length does not divide by "
                    f"attention.head_count ({embedding}/{heads}); the KV number would be wrong")
    return 0, problems


@dataclass(frozen=True)
class _KvCost:
    """Cache elements that scale with the window, and those that do not."""

    per_token: int
    fixed: int
    notes: tuple[str, ...]


def _windowed_cost(metadata: dict, arch: str, kv_heads: int, head_dim: int) -> tuple[int, int]:
    """(windowed layers, their fixed element count) for a sliding-window model.

    A windowed layer's cache stops growing at the window, so its total cost is fixed no
    matter how wide the context is. Gemma-4-12B spends 40 of its 48 layers this way.
    """
    pattern = _flags(_meta(metadata, arch, "attention.sliding_window_pattern"))
    window = _as_int(_meta(metadata, arch, "attention.sliding_window"))
    if not window or not pattern:
        return 0, 0
    windowed = sum(1 for flag in pattern if flag)
    if not windowed:
        return 0, 0
    window_dim = _as_int(_meta(metadata, arch, "attention.key_length_swa")) or head_dim
    # Past the window a windowed layer's cache stops growing, so its total is the window.
    per_layer = TENSORS_PER_LAYER * kv_heads * window_dim * window
    return windowed, windowed * per_layer


def _state_cost(metadata: dict, arch: str, recurrent_layers: int) -> tuple[int, list[str]]:
    """State-space layers keep a convolution state and a recurrent state, both fixed size."""
    inner = _as_int(_meta(metadata, arch, "ssm.inner_size"))
    kernel = _as_int(_meta(metadata, arch, "ssm.conv_kernel"))
    state = _as_int(_meta(metadata, arch, "ssm.state_size"))
    groups = _as_int(_meta(metadata, arch, "ssm.group_count"))
    if not (inner and kernel and state and groups) or recurrent_layers <= 0:
        return 0, []
    per_layer = kernel * inner + state * groups
    return recurrent_layers * per_layer, [
        f"{recurrent_layers} state-space layers hold a fixed ~{per_layer / 1024:.0f} KiB of "
        f"state each (an estimate from the declared ssm geometry)"]


def _kv_cost(metadata: dict, arch: str, n_layers: int, block_count: int, kv_heads: int,
             head_dim: int) -> _KvCost:
    """The whole cache cost: what scales with the window, and what does not."""
    notes: list[str] = []
    uniform_per_layer = TENSORS_PER_LAYER * kv_heads * head_dim
    windowed_layers, windowed_fixed = _windowed_cost(metadata, arch, kv_heads, head_dim)
    if windowed_layers:
        notes.append(f"{windowed_layers} of {block_count} layers are sliding-window and hold "
                     f"a bounded cache; only {n_layers} scale with the window")
    scaling_layers = n_layers if not windowed_layers else max(0, n_layers - windowed_layers)
    state_fixed, state_notes = _state_cost(metadata, arch, max(0, block_count - n_layers))
    notes.extend(state_notes)
    return _KvCost(per_token=scaling_layers * uniform_per_layer,
                   fixed=windowed_fixed + state_fixed, notes=tuple(notes))


def read_gguf_shape(path: str | Path) -> ModelShape:
    """Describe a GGUF model from its header: layers, KV cost, weight bytes, window."""
    target = Path(path)
    if not target.is_file():
        return ModelShape(source=SOURCE_GGUF, path=str(target),
                          problems=(f"model file not found: {target}",))
    try:
        metadata = read_gguf_metadata(target)
    except (ValueError, OSError) as error:
        return ModelShape(source=SOURCE_GGUF, path=str(target),
                          problems=(f"unreadable GGUF header: {error}",))

    arch = _arch(metadata)
    block_count = _as_int(_meta(metadata, arch, "block_count"))
    n_layers, notes = _attention_layer_count(metadata, arch, block_count)
    head_dim, problems = _head_dim(metadata, arch)
    kv_heads = _kv_heads(metadata, arch)
    cost = _kv_cost(metadata, arch, n_layers, block_count, kv_heads, head_dim)
    notes.extend(cost.notes)
    value_length = _as_int(_meta(metadata, arch, "attention.value_length"))
    if value_length and head_dim and value_length != head_dim:
        notes.append(f"value_length {value_length} differs from key_length {head_dim}")
    for name, value in (("block_count", block_count), ("head_count_kv", kv_heads),
                        ("head_dim", head_dim)):
        if value <= 0:
            problems.append(f"{name} not found in the GGUF header")

    return ModelShape(
        arch=arch, n_layers=n_layers, n_kv_heads=kv_heads, head_dim=head_dim,
        kv_elements_per_token=cost.per_token, kv_fixed_elements=cost.fixed,
        weights_bytes=target.stat().st_size,
        native_ctx=_as_int(_meta(metadata, arch, "context_length")),
        source=SOURCE_GGUF, path=str(target), notes=tuple(notes), problems=tuple(problems))


def _hf_layer_kinds(config: dict) -> list[str] | None:
    """The config's per-layer type names, lowercased, or None when it declares none."""
    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list) or not layer_types:
        return None
    return [str(kind).lower() for kind in layer_types]


def _hf_layer_notes(kinds: list[str], linear: int, sliding: int) -> list[str]:
    """Notes for the layer kinds that cost less than a full-context attention layer."""
    notes: list[str] = []
    if 0 < linear < len(kinds):
        notes.append(f"{len(kinds) - linear} of {len(kinds)} layers are attention layers; "
                     f"{linear} are linear-attention and hold no growing cache")
    if sliding:
        notes.append(f"{sliding} of {len(kinds)} layers are sliding-window and hold a "
                     f"bounded cache")
    return notes


def _hf_layers_and_windows(config: dict, total: int) -> tuple[int, int, list[str]]:
    """(attention layers, windowed layers, notes) from the config's per-layer type list."""
    kinds = _hf_layer_kinds(config)
    if kinds is None:
        return total, 0, []
    linear = sum(1 for kind in kinds if "linear" in kind)
    sliding = sum(1 for kind in kinds if "sliding" in kind)
    return len(kinds) - linear, sliding, _hf_layer_notes(kinds, linear, sliding)


def _safetensors_bytes(model_dir: Path) -> tuple[int, list[str]]:
    """Weight bytes as the sum of the safetensors shards sitting next to the config."""
    shards = sorted(model_dir.glob(f"*{SAFETENSORS_SUFFIX}"))
    if not shards:
        return 0, [f"no {SAFETENSORS_SUFFIX} shards in {model_dir}; weight size unknown"]
    return sum(shard.stat().st_size for shard in shards), []


def _load_hf_config(directory: Path) -> tuple[dict, ModelShape | None]:
    """The parsed config.json, or an empty dict with the failure shape that stands in for it."""
    config_path = directory / HF_CONFIG_NAME
    if not config_path.is_file():
        return {}, ModelShape(source=SOURCE_HF, path=str(directory),
                              problems=(f"not a model directory: {config_path} missing",))
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {}, ModelShape(source=SOURCE_HF, path=str(directory),
                              problems=(f"unreadable {HF_CONFIG_NAME}: {error}",))
    if not isinstance(config, dict):
        return {}, ModelShape(source=SOURCE_HF, path=str(directory),
                              problems=(f"{HF_CONFIG_NAME} is not an object",))
    return config, None


def _hf_head_dim(config: dict, heads: int, hidden: int) -> int:
    """The declared head width, or the derived one, or zero when neither works."""
    head_dim = _as_int(config.get("head_dim"))
    if head_dim > 0:
        return head_dim
    if heads > 0 and hidden % heads == 0:
        return hidden // heads
    return 0


def _hf_field_problems(head_dim: int, total: int, hidden: int, heads: int) -> list[str]:
    """The config fields the fit math cannot run without, when they are missing."""
    problems: list[str] = []
    if head_dim <= 0:
        problems.append("head_dim is not declared and hidden_size does not divide by "
                        f"num_attention_heads ({hidden}/{heads})")
    if total <= 0:
        problems.append("num_hidden_layers not found in the config")
    return problems


def read_hf_config(model_dir: str | Path) -> ModelShape:
    """Describe a model served from HF weights, which is how vLLM and EXL3 serve.

    A model with no GGUF quant cannot be described by the GGUF reader. OrcaSAQ-2-27B in the
    reference collection is exactly that case.
    """
    directory = Path(model_dir)
    config, problem = _load_hf_config(directory)
    if problem is not None:
        return problem

    problems: list[str] = []
    total = _as_int(config.get("num_hidden_layers"))
    n_layers, windowed, notes = _hf_layers_and_windows(config, total)
    heads = _as_int(config.get("num_attention_heads"))
    kv_heads = _as_int(config.get("num_key_value_heads")) or heads
    hidden = _as_int(config.get("hidden_size"))
    head_dim = _hf_head_dim(config, heads, hidden)
    weights_bytes, weight_problems = _safetensors_bytes(directory)
    problems.extend(weight_problems)
    problems.extend(_hf_field_problems(head_dim, total, hidden, heads))

    cost = _hf_kv_cost(config, n_layers, windowed, kv_heads, head_dim)
    return ModelShape(
        arch=str(config.get("model_type") or ""), n_layers=n_layers, n_kv_heads=kv_heads,
        head_dim=head_dim, kv_elements_per_token=cost.per_token, kv_fixed_elements=cost.fixed,
        weights_bytes=weights_bytes,
        native_ctx=_as_int(config.get("max_position_embeddings")),
        source=SOURCE_HF, path=str(directory), notes=tuple(notes + list(cost.notes)),
        problems=tuple(problems))


def _hf_kv_cost(config: dict, n_layers: int, windowed: int, kv_heads: int,
                head_dim: int) -> _KvCost:
    """The same two-part cost as the GGUF path, read from the config's own keys."""
    window = _as_int(config.get("sliding_window"))
    scaling_layers = max(0, n_layers - windowed)
    per_token = scaling_layers * TENSORS_PER_LAYER * kv_heads * head_dim
    fixed = 0
    if windowed and window:
        fixed = windowed * TENSORS_PER_LAYER * kv_heads * head_dim * window
    return _KvCost(per_token=per_token, fixed=fixed, notes=())


def shape_from_profile(profile: dict) -> ModelShape:
    """The profile's hand-typed numbers, stamped as such so the sheet can show the source."""
    raw_model = profile.get("model")
    model: dict = raw_model if isinstance(raw_model, dict) else {}
    weights_gb = profile.get("weights_gb")
    weights_gb = weights_gb if isinstance(weights_gb, (int, float)) else 0.0
    n_layers = _as_int(model.get("n_layers"))
    kv_heads = _as_int(model.get("n_kv_heads"))
    head_dim = _as_int(model.get("head_dim"))
    return ModelShape(
        arch=str(profile.get("arch") or ""), n_layers=n_layers, n_kv_heads=kv_heads,
        head_dim=head_dim,
        kv_elements_per_token=TENSORS_PER_LAYER * n_layers * kv_heads * head_dim,
        weights_bytes=int(weights_gb * BYTES_PER_GB),
        native_ctx=_as_int(model.get("native_ctx")),
        source=SOURCE_PROFILE,
        notes=("numbers typed into the profile by hand, not read from the model",),
        problems=() if weights_gb > 0 else ("weights_gb missing from the profile",))


def _read_path(path: str) -> ModelShape:
    """Read whichever kind of model artifact the path names."""
    if path.endswith(GGUF_SUFFIX):
        return read_gguf_shape(path)
    if Path(path).is_dir():
        return read_hf_config(path)
    return read_gguf_shape(path)


def _served_path(props: dict | None) -> str:
    """The model file or directory a server reports for what it has loaded, if any."""
    if not isinstance(props, dict):
        return ""
    for key in ("model_path", "model_dir", "model"):
        value = props.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _merge(shape: ModelShape, problems: list[str]) -> ModelShape:
    """One shape carrying the problems every reader hit on the way to it.

    A path that was given and did not work is worth reporting even when something further
    down the chain answered. Otherwise a typo in --model-file looks like a clean run.
    """
    if not problems:
        return shape
    return replace(shape, problems=shape.problems + tuple(problems))


def _candidate(path: str, problems: list[str]) -> ModelShape | None:
    """Read a candidate path, appending its problems when it cannot describe the model."""
    if not path:
        return None
    shape = _read_path(path)
    if shape.known:
        return shape
    problems.append(f"{path}: " + "; ".join(shape.problems or ("shape incomplete",)))
    return None


def _candidate_paths(model_file: str, model_dir: str, props: dict | None) -> list[str]:
    """The paths to try, in priority order: explicit arguments, then what the server reports."""
    override = os.environ.get(ENV_MODEL_FILE, "")
    paths = [model_file, model_dir] if (model_file or model_dir) else [override]
    paths.append(_served_path(props))
    return paths


def _fallback_shape(profile: dict | None) -> ModelShape:
    """The profile's numbers, or an unknown shape when there is no profile either."""
    if profile:
        return shape_from_profile(profile)
    return ModelShape(source=SOURCE_UNKNOWN,
                      problems=("no model file and no profile, so the shape is unknown",))


def resolve_model_shape(model_file: str = "", model_dir: str = "", props: dict | None = None,
                        profile: dict | None = None) -> ModelShape:
    """The best shape available, in priority order, with the source it came from.

    Explicit path, then the path the server reports, then the profile. Every reader that
    failed on the way is reported, so a bad path is visible rather than silently replaced by
    a lower-priority answer.
    """
    problems: list[str] = []
    paths = _candidate_paths(model_file, model_dir, props)
    seen: set[str] = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        shape = _candidate(path, problems)
        if shape is not None:
            return _merge(shape, problems)
    if not any(paths):
        problems.append(f"no model file given, {ENV_MODEL_FILE} unset, and the server "
                        f"reported no model path")
    return _merge(_fallback_shape(profile), problems)
