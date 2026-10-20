# Local Model Battery

Standardized benchmarks for local LLMs. Every test points at any OpenAI-compatible
endpoint — llama.cpp, vLLM, Ollama, LiteLLM, or a hosted API. Two kinds of score:

- **Mechanical** — math, code, science, instruction-following. Graded by script, exact.
- **Writing** — 138 tasks with per-task rubrics, scored 1-10 by a blind judge.

## What it tests

| Test | Size | What it measures | Scoring |
|---|---|---|---|
| MATH-500 | 500 problems | Symbolic math reasoning | sympy answer equivalence |
| HumanEval+ | 164 tasks | Python generation, functional correctness | bundled test suite, subprocess |
| GPQA-Diamond | 198 questions | Graduate-level science | multiple choice, exact letter |
| Instr v2 | 20 tasks | Constraint adherence (one constraint each) | mechanical check |
| Writing Suite v1 | 138 tasks | Prose quality across 8 writing jobs | blind rubric judge, 1-10 |

The writing suite replaced pairwise prose ELO. Pairwise judging proved unusable:
the "Sample A" slot won **75%** of decided verdicts, far from the ~50% an unbiased
judge produces. Absolute rubric scores have no side to be biased about, and a
model added later is comparable without regenerating anything.

## Setup

```bash
git clone https://github.com/jbrck/local-model-battery
cd local-model-battery
bash setup.sh
```

Creates `.venv`, installs `requirements.txt`, and verifies the data files.

## Quick start

```bash
source .venv/bin/activate

# Smoke test: 10 problems per mechanical benchmark (~5 min)
python scripts/run_battery.py --model MyModel \
  --endpoint http://localhost:11434/v1 --smoke-only

# Full mechanical battery (2-6 hours, model speed dependent)
python scripts/run_battery.py --model MyModel \
  --endpoint http://localhost:11434/v1

# Full battery plus writing generation
python scripts/run_battery.py --model MyModel \
  --endpoint http://localhost:11434/v1 --writing
```

`--model` is the alias the endpoint expects, not a display name. `--skip math500`
(repeatable) drops a benchmark.

## Writing suite

Three separated phases. The split exists so generation never sees a score, and
the judge never sees a model name.

**1. Generate.** Thin pipe: one user message, task text verbatim, temperature
0.7, 16k tokens, no system prompt, no cleaning, no retries.

```bash
python scripts/bench_writing_generate.py --model MyModel \
  --endpoint http://localhost:11434/v1 --answers-dir answers
```

Answers land in `answers/<model>/<task_id>.json` with the full request body and
raw response for audit. An empty answer is recorded as empty, not retried.

**2. Build the blind corpus.** Strips model identity, renames every answer to an
opaque ID, shuffles with a seed.

```bash
python scripts/build_scoring_corpus.py --suite data/writing_suite_v1.json
```

Produces `scoring/corpus/` (what the judge reads) and `scoring/key.json` (opaque
ID → model). The key lives outside the corpus on purpose: the scoring step cannot
read it.

**3. Score blind.** Any capable model can judge. Pass it explicitly — there is no
default judge.

```bash
python scripts/score_writing.py \
  --judge-model JudgeAlias --judge-endpoint http://localhost:8080/v1
```

Each answer is scored against its task rubric, 1-10 per criterion, with the
anchors supplied in the prompt. Empty answers take the floor score without a
judge call. Unparseable judge replies are recorded as errors, never as scores.

**4. Join and report.**

```bash
python scripts/report_scores.py --scores scoring/scores-<timestamp>.json
```

Reports overall mean, per-category mean, per-criterion mean, empty rate, and
character/word compliance. Compliance is counted from the answer text by script;
the judge never scores it.

### Rules that make the scores comparable

- One suite version, frozen. Any task or rubric change is a version bump, and a
  version bump invalidates prior scores.
- Same request shape for every model. The only per-model setting is the endpoint.
- Only the `content` channel is scored. Models that put everything in a reasoning
  channel score as empty — that is model behavior, and it is the same rule for
  everyone.
- Judge version pinned in the score file. Changing the judge means re-scoring
  stored answers (cheap; generation never repeats).
- The judge must not be a model under test.

## Individual benchmarks

```bash
python scripts/bench_math500.py        --model MyModel --endpoint http://localhost:11434/v1
python scripts/bench_humaneval_plus.py --model MyModel --endpoint http://localhost:11434/v1
python scripts/bench_gpqa_diamond.py   --model MyModel --endpoint http://localhost:11434/v1
python scripts/bench_instr_v2.py       --model MyModel --endpoint http://localhost:11434/v1
python scripts/validate_suite.py        # check the suite file itself
```

All scripts take `--limit N` for a smoke run and `--output-dir` for a custom path.

## Output

`results/` gets one timestamped JSON per benchmark: per-question pass/fail detail,
aggregate score, and wall-clock timing. The orchestrator also writes
`summary-<model>-<timestamp>.json`.

Writing answers are not scores. Keep `answers/` and `scoring/` out of git — they
are run artifacts, not benchmark data.

## Requirements

- Python 3.10+
- Any OpenAI-compatible chat-completions endpoint
- Python packages: `sympy` only. Everything else is stdlib.

## Data and licensing

Code is MIT (see `LICENSE`). The bundled datasets keep their own licenses:

| File | Source | License |
|---|---|---|
| `data/math500.json` | MATH (Hendrycks et al.) | MIT |
| `data/humaneval_plus.json` | HumanEval (Chen et al.), plus tests | MIT / Apache-2.0 |
| `data/gpqa_diamond.json` | GPQA (Rein et al.) | CC BY 4.0 |
| `data/instr_v2_tasks.json` | Original | MIT |
| `data/writing_suite_v1.json` | Original | MIT |

Check the upstream terms before redistributing a dataset in another product.

## AI-assisted setup

`skill/SKILL.md` is a portable procedure for an AI assistant: setup, smoke test,
full run, and how to read the results. Point your assistant at that file, or paste
it in as instructions. It assumes nothing about which agent or runtime you use.

## Contributing

Results are only comparable within a suite version. If you change a task, a rubric,
or the request shape, bump the version and say so in the PR.
