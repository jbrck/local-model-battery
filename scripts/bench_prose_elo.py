#!/usr/bin/env python3
"""Prose ELO — pairwise comparison of model outputs.

Runs 10 writing tasks through the model, then compares every pair using
a judge LLM. Reports ELO ratings and head-to-head results.
"""
import argparse, itertools, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.lib.model import ModelConfig, chat_request
from scripts.lib.judge import setup_judge, judge_pair
from scripts.lib.reporting import start_timer, elapsed, save_result, report_stdout


def run_prose_elo(cfg: ModelConfig, judge_cfg: ModelConfig,
                  data_path: str, other_model_cfgs: list[ModelConfig] | None = None) -> dict:
    """Run prose ELO: generate outputs for this model, then compare against baselines.

    other_model_cfgs: models to compare against. If empty, uses a single-model
    reference baseline (absolute scores only).
    """
    with open(data_path) as f:
        tasks = json.load(f)

    model_name = cfg.name
    print(f"Generating prose for {model_name} ({len(tasks)} tasks)...", flush=True)
    outputs = {}
    for item in tasks:
        tid = item["id"]
        response = chat_request(cfg, [
            {"role": "user", "content": item["prompt"]},
        ], max_tokens=1500, temperature=0.7)
        outputs[tid] = response or "[empty]"
        print(f"  {tid}: {'ok' if response else 'EMPTY'}", flush=True)

    # If no reference models, just return absolute scores
    if not other_model_cfgs:
        print("No reference models — outputting absolute scores only.", flush=True)
        return {
            "model": model_name,
            "tasks_generated": len(outputs),
            "empty_tasks": sum(1 for v in outputs.values() if v == "[empty]"),
            "mode": "absolute_only",
        }

    # Pairwise ELO against reference models
    from scripts.lib.judge import JUDGE_SYSTEM
    k_factor = 32
    elo = {model_name: 1500}
    for ref in other_model_cfgs:
        elo[ref.name] = 1500
        elo.setdefault(model_name, 1500)

    matches = []
    for ref in other_model_cfgs:
        print(f"  Judging {model_name} vs {ref.name}...", flush=True)
        # Generate reference outputs if not already cached
        pass  # TODO: cache reference outputs per session

        for item in tasks:
            a_out = outputs[item["id"]]
            # Need ref outputs here — this requires a cache or pre-generation
            # Simpler: generate ref outputs on demand
            ref_out = chat_request(ref, [
                {"role": "user", "content": item["prompt"]},
            ], max_tokens=1500, temperature=0.7)
            b_out = ref_out or "[empty]"

            verdict = judge_pair(judge_cfg, item["prompt"], a_out, b_out)
            # Store match data for ELO
            matches.append({
                "task": item["id"],
                "a": model_name, "b": ref.name,
                "a_wins": 1 if verdict == "A" else 0,
                "b_wins": 1 if verdict == "B" else 0,
                "draws": 1 if verdict == "TIE" else 0,
            })

    # Compute ELO (draw-excluded formula)
    # Aggregate matches per pair
    pair_matches: dict = {}
    for m in matches:
        key = tuple(sorted([m["a"], m["b"]]))
        if key not in pair_matches:
            pair_matches[key] = {"a": m["a"], "b": m["b"], "a_wins": 0, "b_wins": 0, "draws": 0}
        if (m["a"], m["b"]) == key:
            pair_matches[key]["a_wins"] += m["a_wins"]
            pair_matches[key]["b_wins"] += m["b_wins"]
        else:
            pair_matches[key]["a_wins"] += m["b_wins"]
            pair_matches[key]["b_wins"] += m["a_wins"]
        pair_matches[key]["draws"] += m["draws"]

    for _ in range(100):
        for pair, pm in pair_matches.items():
            a, b = pm["a"], pm["b"]
            total_decided = pm["a_wins"] + pm["b_wins"]
            if total_decided == 0:
                continue
            actual_a = pm["a_wins"] / total_decided
            actual_b = pm["b_wins"] / total_decided
            expected_a = 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400.0))
            expected_b = 1.0 - expected_a
            elo[a] += k_factor * (actual_a - expected_a)
            elo[b] += k_factor * (actual_b - expected_b)

    ranked = sorted(elo.items(), key=lambda x: -x[1])
    e = model_name
    return {
        "model": model_name,
        "elo": round(elo[model_name]),
        "all_ratings": {k: round(v) for k, v in elo.items()},
        "ranked": [{"model": k, "elo": round(v)} for k, v in ranked],
        "empty_tasks": sum(1 for v in outputs.values() if v == "[empty]"),
        "matches": matches,
    }


def main():
    parser = argparse.ArgumentParser(description="Prose ELO benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--judge-endpoint", default="http://localhost:11434/v1")
    parser.add_argument("--judge-model", default="hermes-4-70b")
    parser.add_argument("--ref-models", nargs="*", default=[],
                        help="Reference models to compare against (<name>:<endpoint> format)")
    parser.add_argument("--data", default="data/prose_tasks.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    cfg = ModelConfig(name=args.model, endpoint=args.endpoint, api_key=args.api_key)
    judge_cfg = setup_judge(args.judge_endpoint, args.judge_model)

    ref_cfgs = []
    for ref in args.ref_models:
        if ":" in ref:
            name, ep = ref.split(":", 1)
            ref_cfgs.append(ModelConfig(name=name, endpoint=ep))
        else:
            ref_cfgs.append(ModelConfig(name=ref, endpoint=args.endpoint))

    timer = start_timer()
    print(f"Prose ELO [{args.model}]", flush=True)
    results = run_prose_elo(cfg, judge_cfg, args.data, ref_cfgs)
    results["_elapsed"] = int(time.time() - timer)
    path = save_result("prose_elo", args.model, results, args.output_dir)
    report_stdout(results, f"Prose ELO [{args.model}] done")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()