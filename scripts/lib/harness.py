"""What every leg script shares: one CLI, one result block, one place the counters live.

A leg is any script that reads items, asks the endpoint one question per item, grades what
comes back and writes a result block the orchestrator can read. The block is the contract:
`status`, `items`, `correct`, `cap_hits`, `runaway`, `timeout_hits`, `tps`. Those five numbers
are what the standard is built to compare, so they have to mean the same thing in every leg.

    --cap        the token ceiling for one answer: the standard's value for that leg
    --ctx        the window the server was started with, recorded, not enforced here
    --thinking   on/off: off sends the server's thinking switch, nothing else changes
    --seed       sampling seed, so a rerun sees the same questions the same way
    --timeout    seconds one case may take before it is abandoned and counted
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

from scripts.lib.model import ChatResponse, ModelConfig, post_chat

RESULT_SUFFIX = ".result.json"
DEFAULT_CAP = 8192
DEFAULT_TIMEOUT_S = 1800
DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.0
THINKING_OFF_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

# What a leg points at when no --endpoint is given. The test package sets this to a dead
# port, so a leg test that forgets the flag fails at once instead of quietly calling
# whatever happens to be listening on the operator's machine.
DEFAULT_ENDPOINT = os.environ.get("BATTERY_DEFAULT_ENDPOINT", "http://localhost:11434/v1")

# A runaway is the model failing to stop: the same stretch of text over and over. Twelve
# tokens repeated three times is long enough that prose will not trip it by accident.
RUNAWAY_GRAM = 12
RUNAWAY_REPEATS = 3
# Below this many tokens a `length` finish reason is the cap doing its job on a short leg,
# not a signal that the cap is cutting real work.
CAP_HIT_FLOOR_TOKENS = 32


@dataclass
class LegArgs:
    """The arguments every leg takes, in the order they matter to an operator."""

    model: str
    endpoint: str = DEFAULT_ENDPOINT
    api_key: str = "not-needed"
    data: str = ""
    output_dir: str = "results"
    cap: int = DEFAULT_CAP
    ctx: int = 0
    thinking: str = "on"
    seed: int = DEFAULT_SEED
    temperature: float = DEFAULT_TEMPERATURE
    timeout_s: int = DEFAULT_TIMEOUT_S
    limit: int = 0
    tag: str = ""


def add_leg_args(parser: argparse.ArgumentParser, default_data: str = "") -> None:
    """Register the shared leg flags on a parser. Legs with their own flags add them after."""
    parser.add_argument("--model", required=True, help="model name the endpoint serves")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--data", default=default_data, help="items file for this leg")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP,
                        help="token ceiling for one answer (the standard's cap for this leg)")
    parser.add_argument("--ctx", type=int, default=0,
                        help="window the server was started with, for the record")
    parser.add_argument("--thinking", choices=("on", "off"), default="on")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S,
                        dest="timeout_s", help="seconds one case may take")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N items")
    parser.add_argument("--tag", default="", help="free label for this run")


def parse_leg_args(description: str, default_data: str = "",
                   argv: list[str] | None = None) -> LegArgs:
    """Parse the shared leg CLI for a leg that needs no flags of its own."""
    parser = argparse.ArgumentParser(description=description)
    add_leg_args(parser, default_data)
    return LegArgs(**vars(parser.parse_args(argv)))


def leg_config(args: LegArgs) -> ModelConfig:
    """The client for this leg: the cap is the per-request limit, the timeout is per case."""
    return ModelConfig(name=args.model, endpoint=args.endpoint, api_key=args.api_key,
                       token_limit=max(1, args.cap), temperature=args.temperature,
                       timeout_s=args.timeout_s)


def thinking_body(mode: str) -> dict | None:
    """The body fragment that switches thinking off, or None to leave the server alone."""
    return dict(THINKING_OFF_BODY) if mode == "off" else None


def ask(cfg: ModelConfig, messages: list[dict], args: LegArgs,
        cap: int | None = None) -> ChatResponse:
    """One request with this leg's cap, temperature and thinking setting applied."""
    return post_chat(cfg, messages, max_tokens=cap or args.cap, temperature=args.temperature,
                     extra_body=thinking_body(args.thinking))


def repeated_gram(text: str, gram: int = RUNAWAY_GRAM,
                  repeats: int = RUNAWAY_REPEATS) -> bool:
    """True when the tail of the text is the same window of words over and over."""
    words = text.split()
    if len(words) < gram * repeats:
        return False
    tail = words[-gram:]
    runs = 0
    index = len(words) - gram
    while index >= 0 and words[index:index + gram] == tail:
        runs += 1
        index -= gram
    return runs >= repeats


def is_runaway(response: ChatResponse) -> bool:
    """A long answer that is going round in circles rather than finishing."""
    text = response.content or response.reasoning
    return bool(text) and repeated_gram(text)


def is_cap_hit(response: ChatResponse) -> bool:
    """The cap bound the answer: the server stopped for length with real tokens behind it."""
    return response.finish_reason == "length" and response.completion_tokens >= CAP_HIT_FLOOR_TOKENS


def is_timeout(response: ChatResponse) -> bool:
    error = (response.error or "").lower()
    return "timeout" in error or "timed out" in error


@dataclass
class Tally:
    """The counters the result block is built from. One per leg."""

    items: int = 0
    scored: int = 0
    correct: int = 0
    cap_hits: int = 0
    runaway: int = 0
    timeout_hits: int = 0
    failed: int = 0
    tokens: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def tps(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.scored if self.scored else 0.0

    def record(self, response: ChatResponse, correct: bool | None = None) -> bool:
        """Count one case. Returns whether the answer was scoreable at all."""
        self.items += 1
        self.seconds += response.elapsed_s
        self.tokens += response.completion_tokens
        if not self._note_failure(response):
            return False
        if is_cap_hit(response):
            self.cap_hits += 1
        if is_runaway(response):
            self.runaway += 1
        if correct is None:
            return True
        self.scored += 1
        self.correct += int(correct)
        return True

    def _note_failure(self, response: ChatResponse) -> bool:
        """Split a timeout from a transport failure. False means there was nothing to score."""
        if is_timeout(response):
            self.timeout_hits += 1
            self.errors.append(response.error or "timeout")
            return False
        if not response.ok:
            self.failed += 1
            self.errors.append(response.error or f"HTTP {response.status}")
            return False
        return True

    def status(self) -> str:
        """A one-word verdict the manifest can print without knowing the leg's internals."""
        if not self.items:
            return "not run"
        if self.timeout_hits:
            return "timeout"
        if self.failed > self.scored:
            return "fail"
        if self.failed or self.runaway:
            return "partial"
        return "pass"

    def to_result(self, leg_id: str, args: LegArgs, extra: dict | None = None) -> dict:
        """The result block the orchestrator reads."""
        block = {
            "leg": leg_id,
            "model": args.model,
            "status": self.status(),
            "items": self.items,
            "scored": self.scored,
            "correct": self.correct,
            "score_pct": round(self.accuracy * 100, 1),
            "tps": round(self.tps, 1),
            "cap_hits": self.cap_hits,
            "runaway": self.runaway,
            "timeout_hits": self.timeout_hits,
            "failed": self.failed,
            "tokens": self.tokens,
            "elapsed_s": round(self.seconds),
            "cap": args.cap,
            "ctx": args.ctx,
            "thinking": args.thinking,
            "seed": args.seed,
            "temperature": args.temperature,
        }
        if args.tag:
            block["tag"] = args.tag
        if self.notes:
            block["notes"] = self.notes
        if self.errors:
            block["errors"] = self.errors[:5]
        if extra:
            block.update(extra)
        return block


def result_path(leg_id: str, args: LegArgs) -> str:
    return os.path.join(args.output_dir, f"{leg_id}{RESULT_SUFFIX}")


# ---------- per-case checkpoints ----------
#
# A killed run must lose at most the case in flight, not the leg. Every leg writes one
# checkpoint file per case the moment the case lands, and on start replays the files
# already on disk into its Tally, so a resumed run's counters come out identical to an
# uninterrupted one's. Legs share one --output-dir, so the files are namespaced by leg.

CHECKPOINT_DIRNAME = "cases"


def case_path(args: LegArgs, leg_id: str, case_id: str) -> str:
    """Where one case's checkpoint lives: <output-dir>/cases/<leg>/<case>.json."""
    return os.path.join(args.output_dir, CHECKPOINT_DIRNAME, leg_id, f"{case_id}.json")


def write_case(args: LegArgs, leg_id: str, case_id: str, record: dict) -> None:
    """Checkpoint one case as soon as it lands, staged then renamed so a kill cannot
    leave a half-written file behind for the resume to trip over."""
    path = case_path(args, leg_id, case_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    staging = f"{path}.partial"
    with open(staging, "w") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    os.replace(staging, path)


def load_case(args: LegArgs, leg_id: str, case_id: str) -> dict | None:
    """The checkpoint already on disk for one case, or None when it has not run."""
    path = case_path(args, leg_id, case_id)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        record = json.load(handle)
    return record if isinstance(record, dict) else None


def replay_payload(response: ChatResponse) -> dict | None:
    """The slice of a response the counters need, small enough to keep per case."""
    if not response.ok:
        return None
    return {
        "choices": [{"finish_reason": response.finish_reason,
                     "message": response.message}],
        "usage": {"completion_tokens": response.completion_tokens},
    }


def case_record(case_id: str, response: ChatResponse,
                correct: bool | None = None, **extra) -> dict:
    """The checkpoint for one case: everything a resumed run needs to count it again."""
    record = {
        "case": case_id,
        "status": response.status,
        "elapsed_s": response.elapsed_s,
        "error": response.error,
        "payload": replay_payload(response),
        "correct": correct,
    }
    record.update(extra)
    return record


def replay_response(record: dict) -> ChatResponse:
    """Rebuild the response a checkpoint saved, so a replayed Tally matches the original."""
    return ChatResponse(status=record.get("status"), payload=record.get("payload"),
                        elapsed_s=float(record.get("elapsed_s") or 0.0),
                        error=record.get("error"))


def finish(leg_id: str, args: LegArgs, tally: Tally, extra: dict | None = None) -> dict:
    """Write the result block and print it, so the orchestrator and a human both see it."""
    block = tally.to_result(leg_id, args, extra)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(result_path(leg_id, args), "w") as handle:
        json.dump(block, handle, indent=2)
        handle.write("\n")
    print("\n" + "=" * 50)
    print(f"{leg_id} [{args.model}] {block['status']}: {block['correct']}/{block['scored']}"
          f" ({block['score_pct']}%)  {block['tps']} tok/s  cap hits {block['cap_hits']}"
          f"  runaway {block['runaway']}  timeouts {block['timeout_hits']}")
    print("=" * 50, flush=True)
    print("@@RESULT " + json.dumps(block), flush=True)
    return block


def load_items(path: str, limit: int = 0) -> list:
    """Read the leg's items file: a JSON list, or a dict with an `items` list."""
    with open(path) as handle:
        payload = json.load(handle)
    items = payload.get("items", []) if isinstance(payload, dict) else payload
    if limit:
        items = items[:limit]
    if not items:
        print(f"no items in {path}", file=sys.stderr)
    return list(items)
