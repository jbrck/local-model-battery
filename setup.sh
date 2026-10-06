#!/usr/bin/env bash
# Local Model Battery — setup script
# Run: bash setup.sh
set -euo pipefail

echo "==> Local Model Battery setup"
echo "==> Python: $(python3 --version 2>&1 || echo 'not found')"

# Create virtual environment
if [ ! -d .venv ]; then
    echo "==> Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate and install
source .venv/bin/activate
echo "==> Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Verify key imports
echo "==> Verifying imports..."
python3 -c "import requests, sympy, json, urllib.request; print('All imports OK')"

# Check data files
echo "==> Checking data files..."
for f in data/math500.json data/humaneval_plus.json data/gpqa_diamond.json \
         data/instr_v2_tasks.json data/prose_tasks.json; do
    if [ -f "$f" ]; then
        echo "  ✓ $f ($(wc -l < "$f") lines)"
    else
        echo "  ✗ $f MISSING"
    fi
done

echo ""
echo "==> Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Start your model server (llama.cpp, vLLM, etc.)"
echo "  2. Run a smoke test:"
echo "     source .venv/bin/activate"
echo "     python scripts/run_battery.py --model MyModel \\"
echo "       --endpoint http://localhost:11434/v1 --smoke-only"
echo "  3. Run the full battery:"
echo "     python scripts/run_battery.py --model MyModel \\"
echo "       --endpoint http://localhost:11434/v1"
echo ""
echo "For prose ELO, you need a judge model:"
echo "     python scripts/run_battery.py --model MyModel \\"
echo "       --endpoint http://localhost:11434/v1 \\"
echo "       --judge-model hermes-4-70b --judge-endpoint http://..."