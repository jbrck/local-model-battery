"""Standardized result output."""
import json, os, time, sys
from pathlib import Path


def start_timer() -> float:
    return time.time()


def elapsed(start: float) -> str:
    s = int(time.time() - start)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def save_result(name: str, model: str, data: dict, outdir: str = "results") -> str:
    """Save a benchmark result JSON to outdir. Returns the file path."""
    os.makedirs(outdir, exist_ok=True)
    ts = time.strftime("%Y-%m-%d-%H%M%S")
    safe_name = model.replace("/", "--").replace(" ", "-")
    path = os.path.join(outdir, f"{name}-{safe_name}-{ts}.json")
    record = {
        "benchmark": name,
        "model": model,
        "timestamp": ts,
        "unix": int(time.time()),
        **data,
        "_wall_elapsed_s": data.get("_elapsed"),
    }
    # Move _elapsed inside data where it belongs
    if "_elapsed" in data:
        record["elapsed_s"] = record.pop("_elapsed")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def report_stdout(result: dict, headline: str = ""):
    """Print a clean, parseable summary to stdout."""
    if headline:
        print(f"\n{'=' * 50}")
        print(headline)
        print(f"{'=' * 50}")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print()
    sys.stdout.flush()