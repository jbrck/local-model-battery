#!/usr/bin/env python3
"""Instruction-following battery v2 — ONE constraint per test, mechanically checked.

20 tests with objective pass/fail. Reads tasks from data/instr_v2_tasks.json.
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.lib.model import ModelConfig, chat_request
from scripts.lib.reporting import start_timer, elapsed, save_result, report_stdout


def check(test_id: str, checker: str, response: str) -> bool:
    """Check a single constraint. Returns True if the response passes."""
    text = response.strip()

    if checker == "exact:banana":
        return text == "BANANA"
    elif checker == "one_word":
        return len(text.split()) == 1
    elif checker == "two_words":
        return len(text.split()) == 2
    elif checker == "three_sentences":
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        return len(sentences) == 3
    elif checker == "five_bullets":
        bullets = [l for l in text.split("\n") if l.strip().startswith("-")]
        return len(bullets) == 5
    elif checker == "seven_items":
        items = [x.strip() for x in text.split(",")]
        return len(items) == 7
    elif checker == "all_caps":
        return text == text.upper()
    elif checker == "starts_with":
        return text.lower().startswith("although")
    elif checker == "ends_with":
        return text.lower().rstrip(".").endswith("silence")
    elif checker == "lowercase":
        return text == text.lower() and "the quick brown fox" in text.lower()
    elif checker == "no_e":
        return "e" not in text.lower()
    elif checker == "no_commas":
        return "," not in text
    elif checker == "max20":
        return len(text.split()) <= 20
    elif checker == "min15":
        return len(text.split()) >= 15
    elif checker == "json":
        try:
            obj = json.loads(text)
            return all(k in obj for k in ["city", "temp_c", "conditions"])
        except (json.JSONDecodeError, TypeError):
            return False
    elif checker == "three_lines":
        return len([l for l in text.split("\n") if l.strip()]) == 3
    elif checker == "no_digits":
        return not any(c.isdigit() for c in text)
    elif checker == "exact_phrase":
        return text.rstrip(".") == "The eagle lands at dawn"
    elif checker == "ends_period":
        return text.rstrip().endswith(".")
    elif checker == "one_line_three":
        words = text.split()
        return len(words) == 3 and not text.startswith("-") and "\n" not in text

    return False


def run_instr(cfg: ModelConfig, data_path: str) -> dict:
    with open(data_path) as f:
        tasks = json.load(f)

    passed = 0
    total = len(tasks)
    results = []
    for item in tasks:
        tid = item["id"]
        prompt = item["prompt"]
        checker = item["checker"]

        response = chat_request(cfg, [
            {"role": "user", "content": prompt},
        ], max_tokens=400, temperature=0.0)

        if response is None:
            response = ""
        p = check(tid, checker, response)
        if p:
            passed += 1
        results.append({"id": tid, "passed": p, "response": response[:80]})
        print(f"  {tid:20s} {'✓' if p else '✗'}")
        sys.stdout.flush()

    return {"n_passed": passed, "n_total": total,
            "score_pct": round(passed / total * 100, 1) if total > 0 else 0.0,
            "score": f"{passed}/{total}"}


def main():
    parser = argparse.ArgumentParser(description="Instruction-following v2 benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--data", default="data/instr_v2_tasks.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    cfg = ModelConfig(name=args.model, endpoint=args.endpoint, api_key=args.api_key)
    timer = start_timer()
    print(f"Instr v2 [{args.model}]", flush=True)
    results = run_instr(cfg, args.data)
    results["_elapsed"] = int(time.time() - timer)
    path = save_result("instr_v2", args.model, results, args.output_dir)
    report_stdout(results, f"Instr v2 [{args.model}] done")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()