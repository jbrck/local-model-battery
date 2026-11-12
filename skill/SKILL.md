---
name: local-battery-setup
description: Use when setting up or running the Local Model Battery v2 on any OpenAI-compatible endpoint. Walks the operator through every decision the plan surfaces.
---

# Local Model Battery v2 — setup and run

Repo: https://github.com/jbrck/local-model-battery (branch `v2-portable-battery` until merged).

14 legs. One frozen standard, one profile per model, one result block per leg. The battery
answers two questions: what did this model score, and was that number taken under the same
conditions as every other number in the table.

The battery is portable because the standard lives in the repo and the profile describes the
model rather than the box. Anything a host cannot do the same way becomes a decision on the
sheet, and taking that decision stamps the run off-standard instead of quietly changing it.

## What varies, and what decides it

| Variable | Decided by | What it does to the test |
|---|---|---|
| `serve.ctx` (window) | the standard, unless the host cannot serve it | the window is the KV budget. Below the standard's 131072 the run is off-standard even when every leg still fits |
| cap per leg | `standard.toml` caps, clamped to `ctx - prompt_headroom` | at 1024, GPQA put 79 of 108 items on the ceiling; MMLU at 4096 emitted no letter for 77 of 500, and 16384 flipped 49 of them |
| thinking on/off | the leg's requirement, the endpoint's switch | thinking off costs MATH 7.8 points and GPQA 28.3 against the same model with it on |
| case timeout | pilot speed x cap, floor 600 s | a timeout below the cap's wall time is a hidden token cap: HA20 at 600 s was worth about 7200 tokens |
| slots | the standard (4) | slots split one KV budget. More slots means a smaller window or a smaller cache for the same VRAM |
| seed / temperature | the standard (42 / 0.0) | a different seed is a different run; nonzero temperature scores the same item differently twice |
| KV type, flash attention | the standard (q8_0, on) | q8_0 KV costs 131072 bytes per token per sequence; the kernel can move numerics |
| quant | the profile | two quants of one model are two entries, never one row |
| items file | the leg's `data` path, hashed into the manifest | 4096-capped MMLU and 16384-capped MMLU are the same items and different scores |
| judge model | `--judge-model`, or the profile | without a judge the writing leg generates and cannot be scored |

## Step 1: serve the model

Not the battery's job. Serve it however the box does, with the window you intend and a KV type
you can name. `--ctx` is what the engine reports at `/props`, not what you asked for: when it
offloads to RAM the engine picks the split itself and the window is an estimate.

```bash
curl -s http://localhost:11434/v1/models
```

`--model` must match an id the endpoint lists, and the endpoint must serve `chat/completions`.

## Step 2: write the profile

One file per model, `profiles/<model>.toml`. Fields and where each number comes from:
`profiles/README.md`. `weights_gb` plus `[model]`'s `n_layers`, `n_kv_heads` and `head_dim` are
what the plan needs for KV arithmetic. Read them off `GET /props` once the model is serving.

## Step 3: plan, and read the sheet

```bash
python3 scripts/plan_run.py --profile profiles/<model>.toml --model <served-id> \
  --endpoint http://127.0.0.1:11434/v1 --legs math500,mmlu      # --json for a machine copy
```

Four parts. Read them in this order.

**The fit line** says whether the model fits and where. `T1 gpu` is comfortable. `T2
fits_with_offload` means weights spill to RAM, the window is the engine's call, and the same
profile behaves differently on a box with more VRAM. `T3`/`T4` mean slow or will not run.

**Decisions that need you** are the ones this host forces. Each carries its options, what each
option costs, and the measured effect of getting it wrong — the numbers behind that come from
`variables.toml`, not from the planner's opinion:

- `ctx assumed 131072` — the profile has no shape, so the window is the engine's call. Serve,
  read `/props`, re-plan.
- `ctx here N` — the standard window does not fit (weights + KV at 131072 exceeds free VRAM).
  Either accept N and stamp the run off-standard, or move to a smaller quant, or move to a
  bigger card. Do not raise the cap to compensate: that is a different test. When `lcb` is the
  leg that needs the widest window, the row says so: below its 114688-token budget that leg
  stops at the context limit rather than scoring what its cap allows.
- `cap:<leg> N` — the leg's standard cap does not fit the window, so it drops to N. A capped leg
  reports cap hits, and a capped run is a floor rather than a result.
- `thinking` — the endpoint has no switch, the flag did not change the token count, or the
  weights do not fit the card. The last case is the one to think about: when the model spills
  to RAM, every reasoning token costs bandwidth as well as wall clock, so the sheet offers
  on (standard) against off (off-standard, cheaper cases). Thinking off on a leg that needs it
  costs points; thinking on where it is not needed costs wall clock and can still move the score.
- `prereq:<leg>` — an external harness or datum is absent, so the leg is skipped rather than
  scored zero. Skipped is visible in the manifest; zero is a lie.

**Set by the standard here** is every remaining knob, with the conservative reading spelled out.
These are what make the number comparable. Change one and the stamp stops saying ON-STANDARD,
and the manifest keeps the list of what moved.

**Skipped** names the legs that will not run and why.

## Step 4: dry-run the wiring

```bash
python3 scripts/run_battery.py --profile profiles/<model>.toml --model <served-id> --dry-run
```

Same sheet, nothing started. Confirm cap, window and thinking per leg before committing hours.

## Step 5: run it

```bash
python3 scripts/run_battery.py --profile profiles/<model>.toml --model <served-id> \
  --out runs/<model> --judge-model <judge-id> --judge-endpoint <judge-url>
```

Legs run one at a time on purpose. One box serves one model, so overlapping legs fight for the
same GPU and make tps meaningless. The gate (`scripts/preflight.py`, run for you) re-checks the
endpoint, the host, the pilot and the data before the first leg; a blocking miss stops the run
and writes `PREFLIGHT.json` saying what failed.

Each leg gets a wall-clock limit projected from the pilot's measured speed. A leg that overruns
is stopped and recorded as `timeout` with whatever it had already written, never as a pass.
`--no-budget-kill` turns that off. `--resume` skips legs that already passed in the same
directory.

## Step 6: read the result

```
runs/<model>/NOTES.md             the run's own account of itself, stamp on the top line
runs/<model>/MANIFEST.json        standard md5, profile md5, config fingerprint, host, serve
runs/<model>/legs/*.result.json   one block per leg
runs/<model>/logs/*.log           each leg's own stdout
```

Three things decide whether a number belongs in the table:

1. **The stamp.** `ON-STANDARD v2` compares with every other on-standard run. `OFF-STANDARD v2:
   ctx 32768 vs 131072` compares with nothing except runs that moved the same variable the same
   way.
2. **The counters.** `cap_hits > 0` means the leg was truncated and the score is a floor.
   `runaway` means the model looped. `timeout_hits` means slow, not wrong. Any of the three needs
   a note next to the number.
3. **The hashes.** The manifest records the md5 of every items file, every harness and the
   standard itself. If a hash moved, the comparison is between two different tests.

## Polling a running battery

Tests started with `background=true, notify=true` report only on completion.

```
process_manage(action='poll', session_id='proc_XXXX')
```

`output_preview` shows partial output. If Hermes restarted mid-run, the process survives but the
preview is empty — it is still working. Use GPU utilisation as a liveness check.

**0% GPU while a leg should be generating means a stalled server, not a stuck leg.** With `-np 1`
the server queues later requests behind the first client, so one orphaned curl blocks the whole
leg while the leg process still looks alive. Restart the server, then send a fresh probe to
confirm the slot is free. Do not restart the leg.

## Reasoning models (thinking in `reasoning_content`, empty `content`)

Some fine-tunes (`Qwen3.8` with MTP, `Nemotron`, `Muse Glimmer`) put reasoning in a separate
field and produce empty `content` when the budget runs out mid-thought.

- **Detection:** `reasoning_content` present, `content` empty.
- **Fix: raise that leg's cap.** Size the cap from that leg's own measured reasoning: run a few
  dozen items, read `reasoning_len` and `completion_tokens` from the result file, set the cap
  above p100 reasoning plus answer headroom. Verify by counting rows whose `completion_tokens`
  sits exactly on the cap — a nonzero count means the score is still a floor. Reasoning length
  does not transfer between legs: an Instr v2 prompt may spend ~45 tokens where a GPQA item
  spends ~3200 characters.
- **Template-forced thinking.** With `--jinja` plus a template that defaults `enable_thinking` to
  true, no client flag reaches it. The `--no-think` switches and `BENCH_REASONING_EFFORT=none`
  are client-side and the template ignores them.
  - Per request: send `"chat_template_kwargs": {"enable_thinking": false}` in the body.
  - Per server: drop `--chat-template-file` and `--jinja` so the model's built-in template is used.
  - Detect: a 50-token request whose `reasoning_content` opens with meta-commentary and whose
    `content` is empty.
- **Switch-less models.** A template with no thinking variable at all drops the flag silently and
  the model thinks anyway. Flint-abliterated behaves this way: `--thinking off` reads off and the
  model reasons regardless. Budget for it.
- **Budget burn.** Even with no forced thinking, a native template may open with a short reasoning
  preamble. At 10-50 tokens the budget burns before the answer starts. A 50-token answer needs
  100-200 tokens of room. Correct model behaviour, not a hang.
- **Empty content loses pairwise.** A judge counts an empty output as a loss, so a model can beat
  another empty-output model and inflate its ELO. Report head-to-head draws honestly.

## Pointing the battery at this host's data

The frozen standard holds no machine paths. Four legs need data the repo does not redistribute, and each has an env override, so a host that owns the data sets the variable instead of editing the standard.

| Variable | Points at | On Badger-001 |
|---|---|---|
| `LCB_DATA` | `release_v6` lite jsonl, 175 problems | `/mnt/data/lcb-data/test6.jsonl` (134 MB, read in place, never copied into the repo) |
| `LCB_SRC` | LiveCodeBench checkout carrying `lcb_runner` | `/mnt/data/lcb-src/LiveCodeBench` |
| `NLTK_DATA` | punkt/wordnet corpora the IFBench checkers need | `/mnt/data/benchmark-venv/nltk_data` |
| `PATH` | `tool-eval-bench` (ifeval, toolbench) and `benchlocal-cli` (hermesagent20) | prepend `/mnt/data/benchmark-venv/bin` |

The benchmark interpreter carries `nltk`, `ifbench` and the lcb scorer's dependencies: `/mnt/data/benchmark-venv/bin/python3`.

    export LCB_DATA=/mnt/data/lcb-data/test6.jsonl
    export LCB_SRC=/mnt/data/lcb-src/LiveCodeBench
    export NLTK_DATA=/mnt/data/benchmark-venv/nltk_data
    export PATH=/mnt/data/benchmark-venv/bin:$PATH

Ask the planner what a host will actually run before starting a 14-leg day. Every leg that cannot run names its own reason:

    python3 scripts/plan_run.py --profile profiles/<name>.toml --no-probe --json | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['legs']),'run,',len(d['skips']),'skipped'); [print('  skip', s['leg'], '-', s['reason']) for s in d['skips']]"

On Badger-001 that resolves 8 of 14 with nothing exported, and **13 of 14** with the block above. Writing stays out until a judge model is configured, which is the one gap no env var closes.

## A leg that says in_repo is not a ported leg

`legs.toml`'s `implementation` field is a claim. Before trusting a leg or handing the file to someone as a pattern, confirm it is actually on the v2 contract:

```bash
for f in scripts/bench_*.py scripts/adapters/*.py; do
  printf "%-38s harness=%s result_block=%s\n" "$f" \
    "$(grep -c 'lib.harness\|from lib import harness' "$f")" \
    "$(grep -c 'RESULT_SUFFIX\|result.json' "$f")"
done
```

Zero on either line means a v1 script: it imports `scripts.lib.reporting`, parses its own flags and writes no result block. A run reaches it and fails late, which is the opposite of what the gate leg is for. Port it, or flip the flag to `port_pending` so the planner reports it as skipped instead.

Once every leg is on the contract, `scripts/lib/reporting.py` and `scripts/lib/model.chat_request` are dead. Delete them rather than leaving two result conventions in the tree.

## Gate scoping while legs are mid-port

The full gauntlet carries pre-existing complexity failures from the v1 scripts (the worst sits around CC 34). Gate on `--changed`, or grep the report for the files you touched, and leave the legacy failures alone. A port that splits a long if-chain into small functions clears its own file without touching anyone else's.

## Pitfalls

- **One leg at a time.** Never two on the same GPU. Contention corrupts both numbers.
- **Never change the test between models.** Tasks, scoring, judge, output format are fixed. Caps,
  thinking mode and seeds come from the standard; the profile changes nothing about the test.
- **`n_ctx` is not the truncation knob.** The window covers prompt plus generation; the cap limits
  the response. A leg configured to generate more than the window can hold gets cut at the window
  silently, which is why the planner clamps caps to `ctx - prompt_headroom`. LCB's 114688 cap
  needs a `-c` that large to be real.
- **Flash attention.** Some architectures lose 10x speed with `-fa on`. The standard serves it on;
  when a leg is 10x slow for no other reason, that is the knob to report, not to quietly change.
- **VRAM.** A 30B Q4 (~20 GB) fits 24 GB. Two servers or two harnesses at once OOM.
- **Prose scoring is pairwise ELO in four digits.** Never absolute 1-10. `prose_elo.py` is blocked
  (exits with an error by design) — use `prose_elo_pairwise.py`, or merge new outputs with
  `prose_elo_finalize.py`. Never start references at 1500: that erases the established ranking.
- **ELO from merged data.** Head-to-head wins against a weak opponent inflate a number. Check raw
  draws first, and say "N draws out of N tasks", never "N/N".
- **Never kill a running leg because of a remark.** "That looks slow" is not a stop command. If JB
  wants it stopped he will say so; ask when unsure.
- **sympy on a new Python.** Python 3.14 has no sympy wheels: use a 3.10-3.12 venv, or
  `pip install --no-binary sympy sympy`.
- **`PYTHONPATH=` isolation** before benchmarks on Badger-001, to keep the Python 3.14 venv out of
  the harness's imports.

## Checks before calling a run done

```bash
python3 -m unittest discover -s tests -t .
python3 scripts/check_catalogs.py
```

Run `check_catalogs.py` after any edit to `standard.toml`, `legs.toml` or `variables.toml`: it
refuses a catalog that names a variable no leg consumes, or a leg whose cap cannot be resolved.
