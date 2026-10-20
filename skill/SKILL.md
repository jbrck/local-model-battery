---
name: local-battery-setup
description: Use when setting up or running the Local Model Battery on any OpenAI-compatible endpoint. Covers setup, smoke test, full run, writing suite, and how to read results.
---

# Local Model Battery — setup and run

Portable procedure. Works with any AI assistant and any OpenAI-compatible
endpoint. Repo: https://github.com/jbrck/local-model-battery

Tests: MATH-500, HumanEval+, GPQA-Diamond, Instr v2, Writing Suite v1.

## 1. Clone and install

```bash
git clone https://github.com/jbrck/local-model-battery
cd local-model-battery
bash setup.sh
```

If `pip install sympy` fails, the interpreter is too new for the wheels on PyPI —
use a 3.10-3.12 venv, or `pip install --no-binary sympy sympy`.

## 2. Confirm the endpoint

The user needs an OpenAI-compatible endpoint serving the model under test.

```bash
curl -s http://localhost:11434/v1/models
```

`--model` must match an id that endpoint actually lists. Benchmarks request
`chat/completions`; anything that only serves `/completions` will not work.

## 3. Smoke test (~5 min)

```bash
source .venv/bin/activate
python scripts/run_battery.py --model MyModel \
  --endpoint http://localhost:11434/v1 --smoke-only
```

Ten problems per benchmark. If a benchmark warns and continues, fix the endpoint
before starting a full run — a full run wastes hours on a broken pipe.

## 4. Full mechanical battery (2-6 hours)

```bash
python scripts/run_battery.py --model MyModel --endpoint http://localhost:11434/v1
```

Order: MATH-500, HumanEval+, GPQA-Diamond, Instr v2. Roughly 900 requests total;
time scales with tokens per second, not with model size directly.

## 5. Writing suite

Generation is separate from scoring, on purpose. Never score during generation.

```bash
# 1. Generate answers (thin pipe, temperature 0.7, 16k tokens)
python scripts/bench_writing_generate.py --model MyModel \
  --endpoint http://localhost:11434/v1 --answers-dir answers

# 2. Blind the corpus (strips model identity, opaque IDs, seeded shuffle)
python scripts/build_scoring_corpus.py

# 3. Score with a judge that is NOT under test
python scripts/score_writing.py \
  --judge-model JudgeAlias --judge-endpoint http://localhost:8080/v1

# 4. Join and report
python scripts/report_scores.py --scores scoring/scores-<timestamp>.json
```

The judge needs to be a capable instruction-follower and must not be a model in
the comparison set. There is no default judge; pass a model explicitly.

## Pitfalls to tell the user

- **Reasoning models burn the budget before visible output.** The mechanical
  benchmarks use tight token caps, so a model that thinks first can score low on
  Instr v2 while being perfectly capable. That is measured behavior, not a harness
  bug. The writing suite gives 16k tokens for the same reason.
- **Empty content channel scores empty.** The writing leg scores `content` only. A
  model that answers inside `reasoning_content` records as empty. Do not "fix" it
  by raising tokens — it is a real difference in behavior.
- **HumanEval+ runs generated code.** It runs in a subprocess with a 10s timeout in
  a temp directory. That is time-boxing, not a security sandbox. Use a disposable
  machine or a container for models you do not trust.
- **Do not compare scores across suite versions.** A rubric or task change is a
  version bump; prior scores no longer apply.
- **Do not use pairwise ELO for writing.** It was retired: the first-slot answer won
  75% of decided verdicts. Use the writing suite.

## Reading results

| Test | Random baseline | Typical range | Notes |
|---|---|---|---|
| MATH-500 | ~0% | 60-92% | Small models lose on multi-step algebra |
| HumanEval+ | ~0% | 60-95% | Below 60% usually means truncation or format |
| GPQA-Diamond | 25% (4 choices) | 35-60% | Anything near 25% is noise |
| Instr v2 | 0% | 14-20 of 20 | Exact-constraint tasks; few hit 20 |
| Writing Suite v1 | 1.0 (rubric floor) | 4-7 typical | Absolute 1-10 mean, blind judge |

Writing scores are absolute rubric means, not rankings. They are comparable to
other runs on the same suite version and judge only.

Results save to `results/` as timestamped JSON; the orchestrator also writes
`summary-{model}-{timestamp}.json`.
