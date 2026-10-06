---
name: local-battery-setup
description: Use when setting up local model battery. Guide setup.
---

# Local Model Battery Setup

Guides through setting up and running the [Local Model Battery](https://github.com/jbrck/local-model-battery). Tests: MATH-500, HumanEval+, GPQA-Diamond, Instr v2, Prose ELO.

## Clone & install

```bash
git clone https://github.com/jbrck/local-model-battery
cd local-model-battery
bash setup.sh
```

Pitfall: Python 3.14+ lacks sympy wheels — `pip install sympy --no-binary sympy`.

## Start model server

llama.cpp, vLLM, Ollama — any OpenAI-compatible endpoint.

```bash
curl http://localhost:11434/v1/chat/completions -d '{"model":"x","messages":[{"role":"user","content":"hi"}]}'
```

## Smoke test (~5 min)

```bash
source .venv/bin/activate
python scripts/run_battery.py --model MyModel --endpoint http://localhost:11434/v1 --smoke-only
```

## Full battery (2-6 hours)

```bash
python scripts/run_battery.py --model MyModel --endpoint http://localhost:11434/v1
```

Order: MATH-500 → HumanEval+ → GPQA-Diamond → Instr v2 → Prose ELO.

## Pitfalls

- **Reasoning models**: Thinking models burn budget before visible output. Depresses Instr v2 and Prose. Try `--no-think` or raise budget.
- **Flash-attention**: Some architectures lose 10x speed with `-fa on`. Try `-fa off`.
- **VRAM**: 30B Q4 (~20 GB) + draft (~3 GB) fits 24 GB with tight headroom.
- **ELO formula**: Draw-excluded. Never include draws in denominator.

## Interpreting results

| Test | 7B | 27B | 70B |
|---|---|---|---|
| MATH-500 | 60-75% | 80-90% | 85-92% |
| HE+ | 60-75% | 85-95% | 85-92% |
| GPQA-D | 35-45% | 45-55% | 50-60% |
| Instr v2 | 14-17/20 | 16-19/20 | 17-20/20 |
| Prose ELO | 1000-1300 | 1400-1700 | 1600-1900 |

## Prose ELO (needs judge)

```bash
python scripts/run_battery.py --model MyModel --endpoint ... \
  --judge-model hermes-4-70b --judge-endpoint http://...
```

70B+ judge recommended. `--skip-prose` if none.

## Installing as a Hermes skill

Anyone using Hermes Agent can install this skill:

```bash
skill_manage(action='create',
  name='local-battery-setup',
  content=$(cat skill/SKILL.md))
```

Or from the directory directly:

```bash
hermes skill install ./skill/
```