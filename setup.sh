#!/usr/bin/env bash
# Local Model Battery — setup script
# Run: bash setup.sh
set -euo pipefail

echo "==> Local Model Battery setup"
echo "==> Python: $(python3 --version 2>&1 || echo 'not found')"

if [ ! -d .venv ]; then
    echo "==> Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "==> Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "==> Verifying imports..."
python3 -c "import sympy, json, subprocess, urllib.request; print('All imports OK')"

echo "==> Checking data files..."
missing=0
for f in data/math500.json data/humaneval_plus.json data/gpqa_diamond.json \
         data/instr_v2_tasks.json data/writing_suite_v1.json; do
    if [ -f "$f" ]; then
        echo "  ok  $f"
    else
        echo "  MISSING  $f"
        missing=1
    fi
done

if [ "$missing" -eq 1 ]; then
    echo "ERROR: data files missing; re-clone the repo." >&2
    exit 1
fi

echo "==> Validating the writing suite..."
python3 scripts/validate_suite.py

echo "==> Running tests..."
python3 -m unittest discover -s tests -q

echo ""
echo "==> Setup complete."
echo ""
echo "Next steps:"
echo "  1. Start your model server (llama.cpp, vLLM, Ollama, LiteLLM)."
echo "  2. Smoke test:"
echo "     source .venv/bin/activate"
echo "     python scripts/run_battery.py --model MyModel \\"
echo "       --endpoint http://localhost:11434/v1 --smoke-only"
echo "  3. Full battery:"
echo "     python scripts/run_battery.py --model MyModel \\"
echo "       --endpoint http://localhost:11434/v1"
echo ""
echo "Writing scores need a judge model that is not under test:"
echo "     python scripts/bench_writing_generate.py --model MyModel"
echo "     python scripts/build_scoring_corpus.py"
echo "     python scripts/score_writing.py --judge-model JudgeAlias \\"
echo "       --judge-endpoint http://localhost:8080/v1"
