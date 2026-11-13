# Consult, Engine Facts, and the Judge: Implementation Plan

> **For Hermes:** Execute with subagent-driven-development. Tasks 5, 6 and 8 all touch `plan_run.py`, so one implementer owns them. Task 14 onward is new files only and can run beside it.

**Goal:** Finish the battery. The engine layer built as data must reach the plan and the brief, the battery must be able to describe a host before any server is running on it, and the judge must become the user's choice.

**Architecture:** `engine_facts` identifies the serving engine by which probe answered and reports only what that engine publishes. `consult` gathers host, model and engine facts and renders a brief: what was read, what the decisions are, what each one costs, the direction for standing the engine up, the smoke check, and the exact run command. The three knowledge tables written in `6fdf97d` are its source. Nothing decides for the user; the agent running the battery reads the brief and puts the questions.

**Tech Stack:** Python 3.12 stdlib. `unittest`. No new dependencies.

**Related:** `2026-11-12-universal-setup.md` holds Tasks 5, 6, 8 and 10 in their original wording. This document supersedes them where they differ and adds Tasks 14 to 16.

---

## Status

| Task | State |
|---|---|
| 1-4 model_probe | done `53af9ca` |
| 7 plan from the probed shape | done `dbf3152` |
| 9 interpreter | done `f95a842` |
| 11-13 detach, checkpoints, report | done `dde87ab` |
| knowledge tables | done `6fdf97d` |
| 5 engine_facts | this document |
| 6 engine on every run | this document |
| 8 engine-shaped decisions | this document |
| 14 the consult brief | this document |
| 15 the brief before the engine is up | this document |
| 16 the judge is the user's choice | this document |
| 10 docs, skill, gate | last |

---

## Task 5: `engine_facts` identifies the engine and reads what it publishes

**Objective:** Name the engine behind an endpoint from the probe that answered, and report the facts that engine actually publishes.

**Files:**
- Create: `scripts/lib/engine_facts.py`
- Test: `tests/test_engine_facts.py`
- Reference: `scripts/lib/server_probe.py` (`props_url`, `_probe_models`), `scripts/lib/model.py` (`get_json`, `ModelConfig`)
- Reference: `engines.toml` (the `shape_from` field per engine names where the truth lives)

**Interface:**

```python
@dataclass(frozen=True)
class EngineFacts:
    """What the engine behind an endpoint is, and what it publishes about itself."""

    engine: str = "unknown"          # llama.cpp | vllm | sglang | ollama | exllama | strata | mlx | openai-compatible
    version: str = ""
    window: int | None = None
    slots: int | None = None
    kv_type: str | None = None
    flash_attn: bool | None = None
    model_path: str | None = None
    model_dir: str | None = None
    supports_kv_flags: bool = False
    drafter: str | None = None       # none | mtp | dflash2, only when the engine reports one
    source: str = ""                 # the probe that answered
    errors: tuple[str, ...] = ()

    @property
    def known(self) -> bool: ...


def probe_engine(cfg: ModelConfig) -> EngineFacts: ...
```

**Probe table, in the order they are tried:**

| Probe | Identifies | Yields |
|---|---|---|
| `GET {base}/props` | `llama.cpp` (a `model_path` ending `.gguf` raises confidence, it does not decide) | `build_info`, `default_generation_settings.n_ctx`, `total_slots`, `model_path`, `chat_template`; `supports_kv_flags=True` |
| `GET {base}/api/show` | `ollama` | `model_info.*`, `parameters.num_ctx`, quantization |
| `GET {base}/version` + `GET {endpoint}/models` | `vllm` | `version`, `max_model_len` from the first entry |
| `GET {base}/get_model_info` | `sglang` | model path, context length |
| `GET {endpoint}/models` only | `openai-compatible` | model id only; everything else unknown |

**Rules:**
1. Identify by which probe answered. Never from the model name.
2. `window` is the number the engine says it serves, never the model's advertised window.
3. Unknown stays `None`. `supports_kv_flags` is `True` only on `llama.cpp`, because it is the only one of these that exposes the cache as a launch flag.
4. An engine that answers nothing is `unknown`, `known` is `False`, and the errors name what was tried.

**Tests** (`tests/stub_json.py` serves each shape): one case per engine, one unknown, one reachable-but-silent, and one where `/props` answers 404 and `/models` answers (so the fallback path is proven).

**Commit** `feat(engine): identify the serving engine and read what it publishes`

---

## Task 6: Every run names its engine

**Objective:** The manifest stamps the engine on any engine, not only llama.cpp.

**Files:**
- Modify: `scripts/lib/plan_run.py` (`_engine_version` at line 638, `capacity_decisions` at 654)
- Modify: `scripts/run_battery.py` (resolve `probe_engine` once, pass into the planner)
- Test: `tests/test_plan_run.py`

```python
def _engine_version(engine: EngineFacts, server: ServerFacts) -> str:
    """The engine build, named on every engine rather than only llama.cpp."""
```

Fallback order: `engine.version` -> `props["build_info"]` -> `props["version"]` -> `"{engine} (build not reported)"`.

Manifest gains `engine`, `engine_version`, `engine_window`, `supports_kv_flags`, `drafter`. `engine_window` is the engine's reported window and is recorded separately from the plan's `ctx`, because the two differ when the plan caps below the window.

**Tests:** llama.cpp props give `llama.cpp b10909`; vLLM `/version` gives `vllm 0.6.x`; nothing gives `openai-compatible (build not reported)`.

**Commit** `feat(plan): stamp the serving engine on every run`

---

## Task 8: Do not offer a knob the engine does not have

**Objective:** The sheet says what was controlled and what the engine does not expose.

**Files:**
- Modify: `scripts/lib/plan_run.py` (`measurement_decisions` at 700, `_kv_decision`, the `flash_attn` row at 734, the `slots` row at 738)
- Test: `tests/test_plan_run.py`

**Rules:**
1. `supports_kv_flags` false means the `kv_type` row is emitted with `chosen="n/a"` and a consequence naming the engine: `"vLLM manages the KV cache itself (paged attention), so this knob does not exist here"`.
2. The row is never dropped. A sheet that omits a row cannot be read to say whether the setting was controlled or forgotten.
3. A `slots` row above 1 on an engine that batches says so, because slots and batching are different things that both split one KV budget.
4. `resolve_thinking` already returns `on` for a model whose engine runs reasoning unconditionally. Pin it with a regression test so a later change cannot silently drop it.

**Tests:** kv not offered on vllm; kv uses the standard on llama.cpp; flash_attn row reads not applicable on vllm.

**Commit** `fix(plan): only offer engine knobs the engine has`

---

## Task 14: The consult brief

**Objective:** One command that reads the host, the model and the engine, then prints the brief an agent uses to set the run up with the user.

**Files:**
- Create: `scripts/consult.py`, `scripts/lib/consult.py`
- Test: `tests/test_consult.py`
- Read: `engines.toml`, `model-types.toml`, `use-cases.toml`
- Reference: `scripts/lib/env_probe.py`, `scripts/lib/model_probe.py`, `scripts/lib/engine_facts.py`

**Entry:**

```
scripts/consult.py [--endpoint URL] [--api-key K] [--model-file PATH] [--model-dir DIR]
                   [--engine ID] [--use-case ID] [--judge-model ID] [--out PATH]
```

**The brief is markdown, printed to stdout and optionally written with `--out`.** Sections, in order:

1. **What this host is.** Host facts as `env_probe` measured them, each naming its source. Memory and accelerator presence, not a verdict.
2. **What the model is.** The shape from `model_probe`, with the source named (`gguf`, `hf_config`, `profile`, `unknown`). Model type detected from that shape and matched against `model-types.toml`.
3. **What the engine is.** Engine facts when reachable. When not reachable, the engine's `serve` and `smoke` lines from `engines.toml` and a plain statement that nothing is up yet. This is the normal first pass.
4. **The call.** Whether this combination is worth testing at all, with the reason: fits or does not, window available or not, the engine supports the legs or not. Never a silent pass.
5. **The decisions to put to the user.** The four use cases from `use-cases.toml`, each with what it answers and what it cannot say, and the variables that change the result on this host and model: cap, thinking, drafter, concurrency, quant. Every variable carries its consequence for the score.
6. **The judge.** Candidates from the endpoint's model list, minus every model under test, with the pinned default named out loud. The user picks.
7. **The smoke check.** The engine's `smoke` line, made concrete: what to send, what a pass looks like, when to stop and fix the setup instead of running.
8. **The lineup.** The exact `run_battery.py` command with the decisions filled in as flags, plus the `--detach` and `--status` lines.

**Rules:**
- No section is omitted for being unknown. An unknown is printed as unknown with what would resolve it.
- The brief names no data, and no other human, so anyone can run it anywhere.
- Reading the tables is by id, so a corrected row changes the brief without a code change.
- The brief is a document, not a decision. It never chooses a use case, a cap or a judge on the user's behalf.

**Tests:** a brief renders with no endpoint and no model file (the first-pass case); each of the three tables is read by id; the judge list excludes a model under test; a missing table fails loudly rather than printing an empty section.

**Commit** `feat(consult): the brief the agent sets a run up with`

---

## Task 15: The brief before the engine is up

**Objective:** The first thing an operator does on a new machine is stand the engine up. The brief has to be useful before there is anything to probe.

**Files:**
- Modify: `scripts/lib/consult.py`
- Test: `tests/test_consult.py`

**Rules:**
1. With no reachable endpoint, section 3 prints the chosen engine's `serve` and `smoke` from `engines.toml`, section 7 becomes the setup checklist, and section 8 is withheld with the reason: nothing to point at yet.
2. `--engine ID` selects the row. Without it, the brief lists the engines that could serve this model file and asks the user to pick. It does not guess from the operating system.
3. MLX is a normal case of this path, not a special case: Apple silicon is expected to have no endpoint until something starts one. The row carries its own serve and smoke lines like every other engine.
4. After the engine is up, the same command with no changes renders the measured version. Nothing is re-entered.

**Tests:** no endpoint plus `--engine mlx` yields the MLX serve and smoke lines and withholds the lineup; no endpoint and no `--engine` lists candidates; a reachable endpoint yields measured facts instead.

**Commit** `feat(consult): describe the setup before the engine exists`

---

## Task 16: The judge is the user's choice

**Objective:** The scoring model is named, chosen and calibrated rather than inherited.

**Files:**
- Modify: `scripts/lib/consult.py` (section 6)
- Create: `scripts/calibrate_judge.py`
- Test: `tests/test_calibrate_judge.py`

**Already true and not to be rebuilt:** the judge is overridable through `--judge-model`, `--judge-endpoint`, `WRITING_JUDGE_MODEL` and `WRITING_JUDGE_ENDPOINT`, defaults to the pinned alias in `standard.toml`, and `plan_run._judge` records whether the user overrode it.

**Rules:**
1. The candidate list comes from the endpoint, drops everything under test, and marks the pinned default.
2. `calibrate_judge.py` scores a small fixed set of answers whose quality is known by construction: one good, one bad, one truncated, one empty. It prints the score per answer, the spread, and a verdict: `discriminates` or `does not discriminate`.
3. A judge that does not discriminate fails the run setup loudly. A run scored by a judge that cannot tell good from bad is not comparable to anything, including itself.
4. Calibration is one pass, serial, and its result is stamped into the manifest as `judge_calibration`.

**Tests:** the parser on a fixed response; the verdict on a discriminating spread and on a flat one; a judge that errors is reported, not silently passed.

**Commit** `feat(judge): the user picks the judge, and calibration proves it scores`

---

## Task 10: Docs, skill, and the gate

**Objective:** Nothing claims a profile is required, the skill carries the walkthrough, and the gate is green.

**Files:**
- Modify: `README.md`, `profiles/README.md`, `implementation-notes.md`, `docs/plans/2026-11-12-universal-setup.md` (status table)
- Modify: `/mnt/data/hermes/skills/local-battery-setup/SKILL.md` via `skill_manage`
- Add: `skill/SKILL.md` section covering the consult step

`profiles/README.md` tells the operator to copy numbers off `/props` by hand and derive `head_dim = n_embd / n_head`. That instruction produced the 4x error. It becomes: a profile is an override for a model no probe can describe, and the probe is the normal path.

Then, serially, not in parallel:

```bash
.venv/bin/python3 -m unittest discover -s tests -t . -q
.venv/bin/python3 /mnt/data/hermes/skills/quality-gauntlet/scripts/gauntlet.py --src-dir scripts --test-dir tests --json --no-progress
git status
```

The gate carries 24 pre-existing failures (15 CRAP, 9 coverage). `plan_run.py` holds 10 of them and Task 8 rewrites that file, so those are fixed in that pass. The rest are named in the commit message if they are still standing, because a gate that is described as green when it is not is worse than a red one.

**Commit** `docs: the battery reads the model, and the setup is a brief`

---

## Order

```
Task 5 -> 6 -> 8        one implementer, all three touch plan_run.py
Task 14 -> 15 -> 16     one implementer, new files only
Task 10                 last, after both
```

Tasks 5-8 and 14-16 are independent of each other and may run at the same time. Two implementers, no shared file.

**Do not run the full suite in a subagent.** Run the module under test only. The full suite runs once, serially, at integration.

## Definition of done

1. `scripts/consult.py` prints a brief on a host with no endpoint, no model file and no profile, and says what it could not read.
2. The same command on a live endpoint prints measured facts and the exact run command.
3. Every run stamps `engine`, `engine_version` and `engine_window`, on any engine.
4. A vLLM run does not offer a KV knob and says why.
5. The judge candidates exclude every model under test, and the pinned default is named.
6. `calibrate_judge.py` returns a verdict on a discriminating and a flat judge.
7. Full suite green, gate re-run with its remaining failures named, pushed.
