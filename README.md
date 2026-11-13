# Local Model Battery

Standardized benchmarks for local LLMs. Every test points at any OpenAI-compatible
endpoint — llama.cpp, vLLM, Ollama, LiteLLM, or a hosted API. Two kinds of score:

- **Mechanical** — math, code, science, instruction-following. Graded by script, exact.
- **Writing** — 138 tasks with per-task rubrics, scored 1-10 by a blind judge.

## What it tests

| Leg | `--legs` id | Size | What it measures | Scoring |
|---|---|---|---|---|
| Endpoint selftest | `selftest` | 2 items | that the endpoint answers and the harnesses import, before a real leg burns budget | exit code; refuses the battery on failure |
| MATH-500 | `math500` | 500 problems | symbolic math reasoning, boxed final answers | sympy equivalence |
| HumanEval+ | `humaneval_plus` | 164 tasks | Python functional correctness against an extended hidden suite | pass@1, each candidate executed in a subprocess |
| GPQA Diamond | `gpqa_diamond` | 198 questions | graduate-level science reasoning | letter match |
| MMLU (generative) | `mmlu` | 500 of 14042 | broad knowledge, asked as a question rather than a logprob | letter match on a seeded sample |
| Instruction Following v2 | `instr_v2` | 20 tasks | adherence to soft and hard constraints | per-constraint rubric, mechanical |
| IFEval | `ifeval` | prompt set | verifiable instruction following at scale | strict and loose prompt-level and instruction-level accuracy ‡ |
| IFBench | `ifbench` | 300 prompts | out-of-domain instruction following | NLTK verifier, strict and loose ‡ |
| Tool calling | `toolbench` | 69 scenarios | tool selection, argument construction, multi-turn use | scenario pass rate ‡ |
| LiveCodeBench | `lcb` | 175 problems | competitive programming against official test suites | `lcb_runner` codegen metrics, generated code executed ‡ |
| Reasoning | `reasoning` | 8 tasks | whether the model emits a trace at all, and how many tokens it takes | trace presence and trace length |
| Refusal battery | `refusal` | 70 prompts | harmful-request refusal, edgy-prompt discrimination, benign over-refusal | three separate counts, never averaged |
| HermesAgent-20 | `hermesagent20` | 20 scenarios | end-to-end agentic work on real tasks | artifact checks in a Docker sandbox running a pinned agent ‡ |
| Writing Suite v1 | `writing` | 138 tasks | instruction-following and craft on long-form prompts | blind rubric judge, 1-10 per criterion |

‡ needs data or tooling that this repo does not ship: the IFEval and tool-calling legs want
`tool-eval-bench`, IFBench wants the NLTK corpora, LiveCodeBench wants its own checkout and the
release_v6 lite jsonl, and HermesAgent-20 wants Docker plus a pinned agent CLI. The setup skill names
the environment variable for each, and the planner reports a missing prerequisite as a leg that does
not run rather than a leg that scores zero.

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

Then run the consult brief:

```bash
python3 scripts/consult.py --endpoint http://localhost:11434/v1
```

It reads the host and the model, probes the endpoint, and prints a brief: what
it measured and where each fact came from, the call and fit verdict, the use
cases and the variables you can still set, judge candidates, the smoke check,
and the exact `run_battery.py` command with the `--detach` and `--status`
lines. It decides nothing. The legs and the judge stay as placeholders in the
printed lineup until you name them.

With no endpoint running, name the engine you intend to serve with instead:

```bash
python3 scripts/consult.py --engine llama.cpp
```

The brief prints that engine's serve and smoke instructions rather than
measured facts. The path is the same for every engine, so `--engine mlx` on a
Mac reads the same way. The OS is never consulted.

The brief's source of truth is the knowledge tables in the repo root
(`engines.toml`, `model-types.toml`, `use-cases.toml`, `variables.toml`). It
reads them by id, so correcting a row corrects the brief with no code change.
A missing table or row exits 2 naming the ids that exist. No section renders
silently empty.

## Quick start

The steps in order: the consult brief, your answers to it, the decision sheet
from `plan_run.py`, the run detached, `--status` to check on it, and the report
when it is done.

```bash
source .venv/bin/activate

# The brief: what the host and the model are, and what to decide
python scripts/consult.py --endpoint http://localhost:11434/v1

# The decision sheet: reads the model's shape from the model itself
python scripts/plan_run.py --model MyModel --endpoint http://localhost:11434/v1

# Start the run and walk away
python scripts/run_battery.py --model MyModel \
  --endpoint http://localhost:11434/v1 --detach

# Check on it
python scripts/run_battery.py --status

# The report, when the run is done
python scripts/report.py results/<run>
```

`--model` is the alias the endpoint expects, not a display name. `--legs` takes
a comma-separated subset of leg ids.

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

Which judge is your call. `scripts/consult.py` lists the endpoint's models,
excludes every model under test, and names the pinned default out loud even
when the endpoint does not serve it. Before a run, prove the one you picked can
score: `scripts/calibrate_judge.py` sends four answers of known quality (good,
bad, truncated, empty) through the real judge path, serially, and prints the
per-answer scores, the spread, and a verdict of `discriminates` or
`does not discriminate`.

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
| `data/gpqa_diamond.json` | GPQA (Rein et al.), diamond subset | CC BY 4.0 |
| `data/mmlu_test.json` | `cais/mmlu` test split, seeded 500-row sample | MIT |
| `data/instr_v2_tasks.json` | Original | MIT |
| `data/ifbench_prompts.json` | `allenai/IFBench` test data | ODC-BY-1.0, Ai2 responsible-use terms |
| `data/reasoning_tasks.json` | Original | MIT |
| `data/refusal_prompts.json` | Original | MIT |
| `data/writing_suite_v1.json` | Original | MIT |

The data the four outside legs read is not redistributed here. Check the upstream terms before
redistributing a dataset in another product.

## AI-assisted setup

`scripts/consult.py` prints the setup brief an assistant works from: what it
read on the host and the model, what it measured on the engine, the decisions
to put to you, and the exact run command. `skill/SKILL.md` is a portable
procedure for an AI assistant that walks through it: setup, smoke test, full
run, and how to read the results. Point your assistant at those files, or paste
them in as instructions. It assumes nothing about which agent or runtime you use.

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
