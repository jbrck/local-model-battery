#!/usr/bin/env python3
"""GPQA-Diamond: graduate-level science QA.

Reads questions from data/gpqa_diamond.json (hendrydong mirror), sends each
to the model endpoint with answer options, and checks for the correct letter.
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.lib.model import ModelConfig, chat_request
from scripts.lib.reporting import start_timer, elapsed, save_result, report_stdout

SYSTEM = "Answer the multiple-choice question. Reply with ONLY the letter of the correct answer (A, B, C, or D). No explanation."


def run_gpqa(cfg: ModelConfig, data_path: str, limit: int | None = None) -> dict:
    with open(data_path) as f:
        problems = json.load(f)
    if limit:
        problems = problems[:limit]

    correct = 0
    total = len(problems)
    details = []

    for i, item in enumerate(problems):
        prompt = item["question"]
        gold = item["answer"].strip().upper()
        start_t = time.time()
        response = chat_request(cfg, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ], max_tokens=512, temperature=0.0)
        elapsed_t = time.time() - start_t

        if response is None:
            print(f"  [{i+1}/{total}] SKIP (no response)")
            continue

        # Extract the letter answer
        pred = response.strip().upper().replace(".", "")
        # Find first A/B/C/D that appears
        match = re.search(r'\b([ABCD])\b', pred)
        got = match.group(1) if match else "?"
        passed = got == gold
        if passed:
            correct += 1
        details.append({"id": i, "gold": gold, "predicted": got, "passed": passed})
        print(f"  [{i+1}/{total}] {'✓' if passed else '✗'} (gold={gold}, got={got}) ({elapsed_t:.1f}s)")
        sys.stdout.flush()

    accuracy = correct / total if total > 0 else 0.0
    return {"n_correct": correct, "n_total": total, "accuracy": accuracy,
            "accuracy_pct": round(accuracy * 100, 1)}


def main():
    parser = argparse.ArgumentParser(description="GPQA-Diamond benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--data", default="data/gpqa_diamond.json")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cfg = ModelConfig(name=args.model, endpoint=args.endpoint, api_key=args.api_key)
    timer = start_timer()
    print(f"GPQA-Diamond [{args.model}]", flush=True)
    results = run_gpqa(cfg, args.data, args.limit)
    results["_elapsed"] = int(time.time() - timer)
    results["elapsed"] = elapsed(timer)
    path = save_result("gpqa_diamond", args.model, results, args.output_dir)
    report_stdout(results, f"GPQA-Diamond [{args.model}] done")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()