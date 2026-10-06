#!/usr/bin/env python3
"""MATH-500: mathematical reasoning benchmark.

Reads problems from data/math500.json, sends each to the model endpoint,
and grades using sympy-based symbolic math verification.
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.lib.model import ModelConfig, chat_request
from scripts.lib.reporting import start_timer, elapsed, save_result, report_stdout

# ---------- grading ----------

def _extract_boxed(text: str) -> str | None:
    """Extract the content inside the LAST \\boxed{}."""
    import re
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    brace = text.find("{", idx)
    if brace == -1:
        return None
    depth = 1
    i = brace + 1
    while depth > 0 and i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[brace + 1 : i - 1] if depth == 0 else None


def _extract_answer(text: str) -> str:
    """Try to extract a boxed answer, else return the cleaned text."""
    boxed = _extract_boxed(text)
    if boxed:
        return boxed.strip()
    text = text.strip()
    # Last line heuristics
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        for prefix in ["answer is", "answer:", "result:", "= "]:
            if prefix in last.lower():
                return last.split(prefix)[-1].strip()
        return last
    return text


def _grade_answer(gold: str, pred: str) -> bool:
    """Compare gold and predicted answers using sympy simplification."""
    from sympy import simplify, sympify, SympifyError

    gold = _extract_answer(gold.strip())
    pred = _extract_answer(pred.strip())

    if not pred or not gold:
        return False
    if pred.lower() == gold.lower():
        return True
    try:
        return bool(simplify(gold) == simplify(pred))
    except SympifyError:
        return gold.strip().lower() == pred.strip().lower()


# ---------- main ----------

def run_math500(cfg: ModelConfig, data_path: str, limit: int | None = None) -> dict:
    """Run MATH-500. Returns {n_correct, n_total, accuracy}."""
    with open(data_path) as f:
        problems = json.load(f)
    if limit:
        problems = problems[:limit]

    system = "Answer directly and concisely. No preamble. No meta-commentary. No sign-off. Stay on topic; stop when the answer is complete."

    correct = 0
    total = len(problems)
    for i, item in enumerate(problems):
        start_t = time.time()
        response = chat_request(cfg, [
            {"role": "system", "content": system},
            {"role": "user", "content": item["problem"]},
        ], max_tokens=2048, temperature=0.0)
        elapsed_t = time.time() - start_t

        if response is None:
            print(f"  [{i+1}/{total}] SKIP (no response)")
            continue

        gold = item["answer"]
        passed = _grade_answer(gold, response)
        if passed:
            correct += 1
        print(f"  [{i+1}/{total}] {'✓' if passed else '✗'} ({elapsed_t:.1f}s)")
        sys.stdout.flush()

    accuracy = correct / total if total > 0 else 0.0
    return {"n_correct": correct, "n_total": total, "accuracy": accuracy,
            "accuracy_pct": round(accuracy * 100, 1)}


def main():
    parser = argparse.ArgumentParser(description="MATH-500 benchmark")
    parser.add_argument("--model", required=True, help="Model name for logging")
    parser.add_argument("--endpoint", default="http://localhost:11434/v1", help="OpenAI-compatible API endpoint")
    parser.add_argument("--api-key", default="not-needed", help="API key if required")
    parser.add_argument("--data", default="data/math500.json", help="Path to math500.json")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit problems (for smoke tests)")
    args = parser.parse_args()

    cfg = ModelConfig(name=args.model, endpoint=args.endpoint, api_key=args.api_key)
    timer = start_timer()
    print(f"MATH-500 [{args.model}]", flush=True)
    results = run_math500(cfg, args.data, args.limit)
    results["_elapsed"] = int(time.time() - timer)
    results["elapsed"] = elapsed(timer)
    path = save_result("math500", args.model, results, args.output_dir)
    report_stdout(results, f"MATH-500 [{args.model}] done")
    print(f"Results saved to {path}")


if __name__ == "__main__":
    main()