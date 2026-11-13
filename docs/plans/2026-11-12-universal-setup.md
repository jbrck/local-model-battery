# Universal Battery Setup Implementation Plan

> **For Hermes:** Execute with subagent-driven-development for the bounded tasks; the wiring tasks need the whole tree in view.

**Goal:** The battery sets up a run for any model on any hardware with no hand-written profile and no host-specific paths, by reading the model and the engine itself.

**Architecture:** Two new fact readers (`model_probe`, `engine_facts`) sit beside the existing `env_probe`. Each reports what it measured and names its source per field. `plan_run` consumes the merged facts; the hand-written profile drops from a prerequisite to an override. Anything the engine cannot report is stamped as unknown or not applicable rather than offered as a knob.

**Tech Stack:** Python 3.12 stdlib only (no new dependencies). Existing `scripts/lib` layout, `unittest`, quality gauntlet.

**Evidence this plan was written against (2026-11-12):**

| Fact | Value |
|---|---|
| Engines in the reference collection | llama.cpp, exllamav3, vLLM (OrcaSAQ2-kernel), Ollama, Strata 0.1.57 + DFlash2 |
| Model records | 27 on rtx-3090, 1 on rtx-3080-dual-mod |
| Live probe, llama.cpp :11434 | ctx 131072, slots 1, thinking switch True, build `b10909-a2878d30d` |
| Live probe, LiteLLM :4000 | reachable, every engine fact `None`, `props: HTTP 404` |
| Profile math vs GGUF, Swift-Qwen3.8-27B | 65/4/213 typed vs 17/4/256 real. 108 KiB/token typed vs 34 real |
| Non-GGUF model in the collection | OrcaSAQ-2-27B has no GGUF quant |

---

## Task 1: `ModelShape` and the GGUF header reader

**Objective:** Read a model's numbers from its GGUF header with pure stdlib.

**Files:**
- Create: `scripts/lib/model_probe.py`
- Test: `tests/test_model_probe.py`

**Step 1: Write failing test**

```python
def test_reads_qwen_shape_from_gguf_header(self):
    shape = read_gguf_shape(QWEN_FIXTURE)
    self.assertEqual(shape.arch, "qwen3moe")
    self.assertEqual(shape.n_layers, 17)
    self.assertEqual(shape.head_dim, 256)
    self.assertEqual(shape.n_kv_heads, 4)
```

Build `QWEN_FIXTURE` with a `write_gguf()` test helper that emits a valid minimal header (magic `GGUF`, version 3, metadata KV pairs, no tensors). The helper lives in `tests/gguf_fixture.py`.

**Step 2: Run** `python -m unittest tests.test_model_probe -v` → FAIL, `model_probe` missing.

**Step 3: Implement**

```python
"""Read a model's shape and size from the file itself, not from a hand-written profile."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"GGUF"
TYPE_STRING, TYPE_ARRAY = 8, 9
SCALAR_FORMATS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                  6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
MAX_REASONABLE_ARRAY = 4096
BYTES_PER_GB = 1024 ** 3


@dataclass(frozen=True)
class ModelShape:
    """The numbers the fit math needs, plus where each one came from."""

    arch: str = ""
    n_layers: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    weights_bytes: int = 0
    native_ctx: int = 0
    source: str = "unknown"
    notes: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()
```

Then `read_gguf_shape(path) -> ModelShape` following the reader already proven in this session: parse header, collect `{arch}.*` keys, then map them.

**Step 4: Run** → PASS.

**Step 5: Commit** `feat: read model shape from the gguf header`

---

## Task 2: The hybrid and multi-head cases

**Objective:** Get `n_layers` and `head_dim` right for models whose attention is not uniform. This is the bug that matters.

**Files:**
- Modify: `scripts/lib/model_probe.py`
- Test: `tests/test_model_probe.py`

**Rules, each with a test:**

| GGUF metadata | Rule |
|---|---|
| `attention.key_length` present | `head_dim = key_length`, never `n_embd / n_head` |
| `attention.head_count_kv` is a list | count non-zero entries as attention layers; `n_kv_heads = max` |
| `attention.recurrent_layers` present and not all true | attention layers = entries that are `false` (recurrent layers hold no growing KV) |
| both absent | `n_layers = block_count`, `head_dim = n_embd // n_head` |
| `n_embd % n_head != 0` | take `key_length` if present, else record a problem and refuse the KV number |

```python
def test_hybrid_recurrent_layers_reduce_the_cache_count(self):
    shape = read_gguf_shape(SWIFT_FIXTURE)   # recurrent_layers, full_attention_interval=4
    self.assertEqual(shape.n_layers, 17)
    self.assertEqual(shape.head_dim, 256)

def test_per_layer_kv_head_array_counts_attention_layers(self):
    shape = read_gguf_shape(LFM_FIXTURE)     # head_count_kv = [8,0,0,0,8,...]
    self.assertEqual(shape.n_layers, 10)
```

Use synthetic fixtures built by the helper, not the real GGUFs, so tests stay fast and deterministic.

**Commit** `fix: count only attention layers in the kv math`

---

## Task 3: The HuggingFace `config.json` reader

**Objective:** Describe a model that has no GGUF, which is how vLLM and EXL3 serve.

**Files:**
- Modify: `scripts/lib/model_probe.py`
- Test: `tests/test_model_probe.py`

```python
def test_reads_shape_from_hf_config(self):
    shape = read_hf_config(HF_FIXTURE_DIR)
    self.assertEqual(shape.n_layers, 62)
    self.assertEqual(shape.n_kv_heads, 8)
    self.assertEqual(shape.head_dim, 128)
    self.assertEqual(shape.source, "hf_config")
```

Reads `config.json`: `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `hidden_size`, `head_dim` (fall back to `hidden_size // num_attention_heads`), `max_position_embeddings`, `layer_types` (a `sliding_attention`/`linear_attention` mixture means hybrid, same rule as Task 2), and the `.safetensors` index for weight bytes when the sidecar file is present.

Weights: sum `*.safetensors` sizes when the dir has them; otherwise leave `weights_bytes = 0` and note why. Never guess.

**Commit** `feat: read model shape from a huggingface config`

---

## Task 4: Resolve the shape from what is available, in priority order

**Objective:** One entry point that takes whatever the caller has and returns the best shape it can, naming its source.

**Files:**
- Modify: `scripts/lib/model_probe.py`
- Test: `tests/test_model_probe.py`

```python
def test_explicit_model_file_beats_the_profile(self): ...
def test_props_model_path_is_used_when_no_file_is_given(self): ...
def test_profile_is_the_last_resort_and_is_stamped_as_such(self): ...
def test_unknown_when_nothing_is_available(self): ...

def resolve_model_shape(model_file: str = "", model_dir: str = "",
                        props: dict | None = None,
                        profile: dict | None = None) -> ModelShape:
```

Priority: `model_file`/`model_dir` argument → `props["model_path"]` → `profile`. Env override `BATTERY_MODEL_FILE` wins over everything, matching `path_env` convention in `standard.toml`.

**Commit** `feat: resolve model shape from file, props or profile`

---

## Task 5: `engine_facts` — name the engine and read what it publishes

**Objective:** Identify the engine behind an endpoint and pull the facts each engine actually publishes.

**Files:**
- Create: `scripts/lib/engine_facts.py`
- Test: `tests/test_engine_facts.py`
- Reference: `scripts/lib/server_probe.py` (`props_url`, `_probe_models`, `get_json`)

```python
@dataclass(frozen=True)
class EngineFacts:
    engine: str = "unknown"          # llama.cpp | vllm | ollama | exllama | litellm | openai-compatible
    version: str = ""
    window: int | None = None
    slots: int | None = None
    kv_type: str | None = None
    flash_attn: bool | None = None
    model_path: str | None = None
    model_dir: str | None = None
    supports_kv_flags: bool = False
    source: str = ""
    errors: tuple[str, ...] = ()
```

Endpoint table:

| Probe | Yields | Engine |
|---|---|---|
| `GET {base}/props` → `build_info`, `default_generation_settings.n_ctx`, `total_slots`, `model_path` | all llama.cpp facts, `supports_kv_flags=True` | `llama.cpp` |
| `GET {base}/version` → `{"version": ...}` | version | `vllm` |
| `GET {endpoint}/models` → each entry's `max_model_len` | `window` | `vllm` |
| `GET {base}/api/show` (Ollama native) | `model_info.*`, `parameters.num_ctx` | `ollama` |
| `GET {endpoint}/models` only | model id only | `openai-compatible` |

Identification is by which probe answered, never by guessing from the model name. A `model_path` ending in `.gguf` bumps confidence to llama.cpp.

Use `tests/stub_json.py` (existing) to serve each shape. Tests must prove one case per engine plus one unknown.

**Commit** `feat: identify the serving engine and read its published facts`

---

## Task 6: Wire the engine into the plan and the manifest

**Objective:** Every run names its engine and build, on any engine.

**Files:**
- Modify: `scripts/lib/plan_run.py:550` (`_engine_version` takes `EngineFacts`)
- Modify: `scripts/run_battery.py` (resolve engine facts once, pass into the planner)
- Test: `tests/test_plan_run.py`

`_engine_version` currently reads llama.cpp's props only. It becomes:

```python
def _engine_version(engine: EngineFacts, server: ServerFacts) -> str:
    """The engine build, named on every engine rather than only llama.cpp."""
    if engine.version:
        return f"{engine.engine} {engine.version}"
    props = server.props if isinstance(server.props, dict) else {}
    for key in ("build_info", "version"):
        if props.get(key):
            return f"{engine.engine} {props[key]}"
    return f"{engine.engine} (build not reported)"
```

Add manifest fields `engine`, `engine_version`, `engine_window`, `supports_kv_flags`.

Test: llama.cpp props → `llama.cpp b10909`; vLLM `/version` → `vllm 0.6.x`; nothing → `openai-compatible (build not reported)`.

**Commit** `feat: stamp the serving engine on every run`

---

## Task 7: `plan_run` consumes probed facts; the profile becomes an override

**Objective:** A run sets up with no profile at all.

**Files:**
- Modify: `scripts/lib/plan_run.py:349-385` (`model_footprint`, `compute_plan_fit`)
- Modify: `scripts/plan_run.py:35` and `scripts/preflight.py:195` (`--profile` becomes optional)
- Modify: `scripts/run_battery.py:71` (same, plus `--model-file`/`--model-dir`)
- Test: `tests/test_plan_run.py`

`model_footprint(profile, standard)` keeps its signature and gains an optional first-class path:

```python
def model_footprint(profile: dict, standard: dict,
                    shape: ModelShape | None = None) -> Footprint:
    """Weights in bytes plus KV bytes per token, from the probed shape or the profile."""
```

`Footprint` gains `source: str = ""` and `shape_notes: tuple[str, ...] = ()` so the sheet shows `gguf` vs `profile` for the numbers that decided a run.

When `shape` is provided and known, use it and ignore the profile's numbers, stamping `source`. When it is not, fall back to today's profile path unchanged. When neither, keep today's honest `unknown` branch.

Tests: probed shape wins over a contradictory profile; profile still works when no shape resolves; unknown stays unknown.

**Commit** `feat: plan from the model's own numbers, profile as override`

---

## Task 8: Engine-shaped decisions

**Objective:** Do not offer a knob the engine does not have.

**Files:**
- Modify: `scripts/lib/plan_run.py` (`_kv_decision` ~line 688-710, `flash_attn`/`slots` rows ~640-655)
- Test: `tests/test_plan_run.py`

```python
def test_kv_decision_is_not_offered_when_the_engine_has_no_kv_flags(self): ...
def test_kv_decision_uses_the_standard_on_llama_cpp(self): ...
def test_flash_attn_row_says_not_applicable_on_vllm(self): ...
```

Rule: `supports_kv_flags` false → the row is emitted with `chosen="n/a"` and a consequence naming the engine (`"vLLM manages the KV cache itself (paged attention); this knob does not exist"`). Never silently drop the row: the sheet's job is to say what was and was not controlled.

EXL3's always-on reasoning already behaves correctly in `resolve_thinking`. Add a regression test pinning it.

**Commit** `fix: only offer engine knobs the engine has`

---

## Task 9: Remove the hardcoded interpreter path

**Objective:** The one catalog entry that is not hardware-agnostic.

**Files:**
- Modify: `legs.toml:264-266`
- Modify: `scripts/lib/legs.py:58,76,107,180-187`
- Modify: `scripts/run_battery.py:142`
- Test: `tests/test_legs.py`, `tests/test_run_battery.py`

```toml
# The prompt helpers come from the checkout, so this leg runs under an interpreter that
# has their dependencies. Point BATTERY_LCB_INTERPRETER at one; otherwise the run's own
# interpreter is used and the leg reports `not run` when an import fails.
interpreter_env = "BATTERY_LCB_INTERPRETER"
```

`Leg` gains `interpreter_env: str = ""`. `leg_command` resolves `interpreter` → env var → `sys.executable`. `validate_legs` keeps rejecting a relative `interpreter` and now also rejects setting both `interpreter` and `interpreter_env`.

Tests: env var honored; empty env falls back to `sys.executable`; both set is a validation error; relative still rejected.

**Commit** `fix: resolve the lcb interpreter from the environment`

---

## Task 10: Docs, skill, and the gate

**Objective:** Nothing left claiming a profile is required, and the gate green.

**Files:**
- Modify: `README.md`, `profiles/README.md`, `implementation-notes.md`, `skill/SKILL.md`
- Modify: `/mnt/data/hermes/skills/local-battery-setup/SKILL.md` (via skill_manage)

`profiles/README.md` currently tells the operator to copy numbers off `/props` by hand and derive `head_dim = n_embd / n_head`. That instruction produced the 4x error. Replace it with: profiles are an override for models the probes cannot reach, and the probe is the normal path.

`skill/SKILL.md` gains the setup walkthrough in the operator's terms: what the battery read, what it could not read, and the variables that would change the score given this model on this hardware, with the consequence of each.

Then:

```bash
.venv/bin/python3 -m unittest discover -s tests -t . -q
python3 /mnt/data/hermes/skills/quality-gauntlet/scripts/gauntlet.py --src-dir scripts --test-dir tests --json
git status
```

Both must be green before the final commit. Push.

**Commit** `docs: the battery reads the model, so profiles are an override`

---

## Order and independence

```
Task 1 → 2 → 3 → 4        (model_probe, one file, sequential)
Task 5 → 6                (engine_facts, independent of 1-4)
Task 9                    (independent of all of the above)
Task 7, 8                 (need 4 and 5 both)
Task 10                   (last)
```

## Definition of done

1. `python3 scripts/plan_run.py --endpoint URL` with no `--profile` produces a full sheet with the model's numbers read from the file, on any of the five engines.
2. A vLLM run stamps `engine=vllm` and the window from `max_model_len`.
3. KV and flash-attn rows say what the engine said, or that the engine has no such knob.
4. `grep -rn "/mnt/data" legs.toml standard.toml` returns nothing outside documented env-var examples.
5. Full suite green, gauntlet exit 0, pushed.
