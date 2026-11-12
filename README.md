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

Two phases. Generation never sees a score, and the judge never sees a model name.

**1. Generate.** Thin pipe: one user message, task text verbatim, temperature
0.7, 16k tokens, no system prompt, no cleaning, no retries.

```bash
python scripts/bench_writing_generate.py --model MyModel \
  --endpoint http://localhost:11434/v1 --answers-dir answers
```

Answers land in `answers/<model>/<task_id>.json` with the full request body and
raw response for audit. An empty answer is recorded as empty, not retried.

**2. Score.** The judge is a model that is not under test. Pass it explicitly,
because there is no default. Pin it: a score is only comparable against another
run on the same judge.

```bash
python scripts/bench_writing_generate.py --model MyModel \
  --endpoint http://localhost:11434/v1 \
  --judge-model nous-glm-5-3-flash --judge-endpoint http://localhost:4000/v1
```

Each answer is scored against its task rubric, 1-10 per criterion, with the
anchors supplied in the prompt. Empty answers take the floor score without a
judge call. Unparseable judge replies are recorded as errors, never as scores.
With no judge the leg still generates everything, reports itself as not scored
and names what is missing. `WRITING_JUDGE_MODEL` and `WRITING_JUDGE_ENDPOINT` do
the same job when `run_battery.py` drives the run, and the judge model lands in
the manifest as a comparability key.

Re-scoring stored answers is cheap, so changing the judge never means repeating
generation.

### Rules that make the scores comparable

- One suite version, frozen. Any task or rubric change is a version bump, and a
  version bump invalidates prior scores.
- Same request shape for every model. The only per-model setting is the endpoint.
- Only the `content` channel is scored. Models that put everything in a reasoning
  channel score as empty — that is model behavior, and it is the same rule for
  everyone.
- Judge model pinned in the result block and the manifest. Changing the judge
  means re-scoring stored answers, which is cheap; generation never repeats.
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

Every change has to pass the quality gate on every method it touches. Three gates,
per method: cyclomatic complexity ≤ 15, line coverage ≥ 80%, CRAP ≤ 8, where
CRAP = complexity² × (1 − coverage)³ + complexity.

```bash
python3 path/to/quality-gauntlet/scripts/gauntlet.py --src-dir scripts --test-dir tests
```

A method over the CRAP line with high coverage is too complex — split it. Under the
coverage line, test the branches it misses. Deleting dead code beats testing it: that
is why the v1 scoring scripts are not here.

```bash
.venv/bin/python3 -m unittest discover -s tests -t .
python3 scripts/check_catalogs.py
```

The second command is the repo's own guard against a lying catalog: every leg's entry
script exists, every bundled data file matches its declared hash, and every leg is
either ported or explicitly marked pending.
