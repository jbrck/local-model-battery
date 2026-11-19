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

echo "==> Checking the catalog..."
python3 scripts/check_catalogs.py

echo "==> Running tests..."
python3 -m unittest discover -s tests -t . -q

echo ""
echo "==> Agent setup"
echo "This repo ships an agent-facing run procedure at skill/SKILL.md"
echo "(frontmatter name: local-battery-setup). An agent driving this battery"
echo "should read that file before planning or starting a run."
if [ -d "$HOME/.hermes/skills" ]; then
    echo ""
    echo "Detected a Hermes skills directory at \$HOME/.hermes/skills. Install or"
    echo "refresh the skill there with:"
    echo "  mkdir -p \"\$HOME/.hermes/skills/local-battery-setup\""
    echo "  cp skill/SKILL.md \"\$HOME/.hermes/skills/local-battery-setup/SKILL.md\""
    echo "(this will not run automatically — a live copy there may already carry"
    echo "local additions worth keeping; diff before overwriting)"
else
    echo ""
    echo "No Hermes skills directory found at \$HOME/.hermes/skills. If your agent"
    echo "uses a different skills/procedures directory, copy skill/SKILL.md there,"
    echo "or point the agent at this path directly."
fi

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
echo "     python scripts/bench_writing_generate.py --model MyModel \\"
echo "       --judge-model nous-glm-5-3-flash \\"
echo "       --judge-endpoint http://localhost:4000/v1"
echo "     (the judge reuses the run's --api-key; pass --judge-api-key when they differ)"
