---
name: local-battery-setup
description: Use when setting up local model battery. Guide setup.
---

# Local Model Battery Setup

This skill instructs an AI assistant on how to set up and run the [Local Model Battery](https://github.com/jbrck/local-model-battery). Tests: MATH-500, HumanEval+, GPQA-Diamond, Instr v2, Prose ELO.

## Steps

### 1. Clone & install

```bash
git clone https://github.com/jbrck/local-model-battery
cd local-model-battery
bash setup.sh
```

Pitfall: Python 3.14+ lacks sympy wheels — `pip install sympy --no-binary sympy`.

### 2. Start model server

The user needs an OpenAI-compatible endpoint running — llama.cpp, vLLM, Ollama, or LiteLLM.

```bash
curl http://localhost:11434/v1/chat/completions -d '{"model":"x","messages":[{"role":"user","content":"hi"}]}'
```

### 3. Smoke test (~5 min)

```bash
source .venv/bin/activate
python scripts/run_battery.py --model MyModel --endpoint http://localhost:11434/v1 --smoke-only
```

### 4. Full battery (2-6 hours)

```bash
python scripts/run_battery.py --model MyModel --endpoint http://localhost:11434/v1
```

Order: MATH-500 → HumanEval+ → GPQA-Diamond → Instr v2 → Prose ELO.

### 5. Prose ELO (optional, needs judge)

```bash
python scripts/run_battery.py --model MyModel --endpoint http://localhost:11434/v1 \
  --judge-model hermes-4-70b --judge-endpoint http://judge:11434/v1
```

70B+ judge recommended. Use `--skip-prose` if none available.

## Pitfalls to tell the user

- **Reasoning models**: Models that output `reasoning_content` burn token budget before visible output. Depresses Instr v2 and Prose scores. Try `--no-think` if the model supports it, or increase token budget.
- **Flash-attention**: Some architectures lose 10x speed with `-fa on`. Try `-fa off` if the model is unexpectedly slow.
- **VRAM**: A 30B Q4 model (~20 GB) plus draft model (~3 GB) barely fits 24 GB. Reduce context (`-c 4096`) if the server OOMs.
- **ELO formula**: The prose ELO uses draw-excluded win rate. Never include draws in the denominator.
- **HumanEval+ sandbox**: The test harness executes generated code. Sandboxed via subprocess timeout; container recommended for multi-user systems.

## Interpreting results

| Test | 7B | 27B | 70B | Notes |
|---|---|---|---|---|
| MATH-500 | 60-75% | 80-90% | 85-92% | Qwen variants dominate |
| HE+ | 60-75% | 85-95% | 85-92% | Coding harder for small models |
| GPQA-D | 35-45% | 45-55% | 50-60% | 34% is random baseline |
| Instr v2 | 14-17/20 | 16-19/20 | 17-20/20 | Few hit 20/20 |
| Prose ELO | 1000-1300 | 1400-1700 | 1600-1900 | Qwen3.8 variants top local |

Results save to `results/` as timestamped JSON. The orchestrator also creates `summary-{model}-{ts}.json`.