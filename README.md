# Local Model Battery

Standardized local LLM benchmarks. Tests math, coding, science knowledge, instruction-following, and prose quality — all runnable against any OpenAI-compatible endpoint.

## What it tests

| Benchmark | Problems | What it measures |
|---|---|---|
| MATH-500 | 500 math word problems | Symbolic math reasoning, step-by-step accuracy |
| HumanEval+ | 164 Python coding tasks | Code generation, functional correctness (evalplus harness) |
| GPQA-Diamond | 198 graduate-level MC questions | Scientific knowledge, multiple-choice accuracy |
| Instr v2 | 20 instruction-following tasks | Constraint adherence (exact word, line count, format) |
| Prose ELO | 10 writing tasks × pairwise comparison | Writing quality via LLM judge, ELO-rated |

## Setup

```bash
git clone https://github.com/jbrck/local-model-battery
cd local-model-battery
bash setup.sh
```

This creates a virtualenv, installs requirements, and verifies everything.

## AI-assisted setup

This repo bundles a companion skill (`skill/SKILL.md`) designed to be loaded by any AI assistant. It tells the AI how to guide you through endpoint setup, smoke tests, full benchmarks, and interpreting results.

To use it with an AI agent:
- **Hermes Agent:** `hermes skill install ./skill/`
- **Any other AI:** Share the file directly — or let the AI read `skill/SKILL.md` as a system instruction

## Usage

You need a running OpenAI-compatible inference server (llama.cpp, vLLM, LiteLLM, Ollama, etc.):

```bash
source .venv/bin/activate

# Quick smoke test (10 problems per benchmark, ~5 min)
python scripts/run_battery.py --model MyModel \
  --endpoint http://localhost:11434/v1 --smoke-only

# Full battery (takes 2-6 hours depending on model speed)
python scripts/run_battery.py --model MyModel \
  --endpoint http://localhost:11434/v1
```

### Prose ELO

Prose ELO needs a separate judge model (a capable instruction-following LLM):

```bash
python scripts/run_battery.py --model MyModel \
  --endpoint http://localhost:11434/v1 \
  --judge-model hermes-4-70b --judge-endpoint http://judge-server:11434/v1
```

Or skip prose entirely:

```bash
python scripts/run_battery.py --model MyModel \
  --endpoint http://localhost:11434/v1 --skip-prose
```

### Individual benchmarks

```bash
python scripts/bench_math500.py --model MyModel --endpoint http://localhost:11434/v1
python scripts/bench_gpqa_diamond.py --model MyModel --endpoint http://localhost:11434/v1
python scripts/bench_humaneval_plus.py --model MyModel --endpoint http://localhost:11434/v1
python scripts/bench_instr_v2.py --model MyModel --endpoint http://localhost:11434/v1
python scripts/bench_prose_elo.py --model MyModel --endpoint http://localhost:11434/v1 \
  --judge-model hermes-4-70b
```

All scripts support `--limit N` for quick smoke tests and `--output-dir` for custom output paths.

## Output

Results land in `results/` as timestamped JSON files with:
- Per-question pass/fail breakdown
- Aggregated scores (accuracy %, pass@1, ELO)
- Wall-clock timing

## Data

All datasets are included in `data/` as JSON:
- `math500.json` — 500 problems from the MATH test set
- `humaneval_plus.json` — 164 coding tasks with test cases
- `gpqa_diamond.json` — 198 graduate-level science questions
- `instr_v2_tasks.json` — 20 instruction-following tasks
- `prose_tasks.json` — 10 writing prompts

No API keys or HuggingFace tokens needed to run the benchmarks.

## Requirements

- Python 3.10+
- An OpenAI-compatible inference endpoint (local or remote)
- For HumanEval+: evalplus test harness (installed via pip)
- For Prose ELO: a capable judge model (70B-class recommended)

## License

MIT — use freely, share results, contribute improvements.