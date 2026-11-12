#!/usr/bin/env python3
"""IFBench leg: 300 out-of-domain instruction-following prompts scored by the
ifbench package's own verifiers (strict and loose).

Ifbench is IFEval's harder sibling: 25 classic constraint types plus 58 new
out-of-distribution ones. The verifier lives in the `ifbench` package and in
the NLTK corpora it tokenizes with. This leg calls the package's OWN checkers
(ifbench.instructions_registry) rather than reimplementing them, so it cannot
drift from the upstream scorer the way a hand-rolled parser could.

Every check runs inside the interpreter named by --ifbench-python (default:
this interpreter), as a worker subprocess that imports ifbench and answers one
JSON exchange per item on stdin/stdout, so another interpreter's packages can
never corrupt this process. NLTK_DATA is set before the import like the legacy
runner did. When the package or its NLTK corpora are missing, the leg reports
itself as not run with a readable reason and exits 2 instead of crashing or
silently scoring nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass

sys.path.insert(0,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.lib.harness import LegArgs, Tally, add_leg_args, ask, finish  # noqa: E402,F401,E501 pylint: disable=wrong-import-position,C0413
from scripts.lib.harness import leg_config, load_items  # noqa: E402,F401,E501 pylint: disable=wrong-import-position,C0413
from scripts.lib.model import ChatResponse, ModelConfig  # noqa: E402,F401,E501 pylint: disable=wrong-import-position,C0413

DEFAULT_DATA = "data/ifbench_prompts.json"
DEFAULT_NLTK_DATA = "/mnt/data/benchmark-venv/nltk_data"
LEG_ID = "ifbench"
EXIT_NOT_RUN = 2

PROBE_SRC = (
    "import json\n"
    "def check():\n"
    "    try:\n"
    "        from nltk.corpus import wordnet\n"
    "        wordnet.synsets('dog')\n"
    "        import nltk\n"
    "        nltk.sent_tokenize('One sentence. Two.')\n"
    "        from ifbench import instructions_registry\n"
    "        registry = instructions_registry.INSTRUCTION_DICT\n"
    "        if not registry:\n"
    "            return {'ok': False, 'error': 'instructions_registry is empty'}\n"
    "        return {'ok': True, 'registry': len(registry)}\n"
    "    except Exception as exc:\n"
    "        return {'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)}\n"
    "print(json.dumps(check()))\n"
)

WORKER_SRC = (
    "import json,sys\n"
    "def build(registry, instruction_id, kwargs, prompt):\n"
    "    cls = registry[instruction_id]\n"
    "    instruction = cls(instruction_id)\n"
    "    clean = {k: v for k, v in kwargs.items() if v is not None}\n"
    "    instruction.build_description(**clean)\n"
    "    args = instruction.get_instruction_args()\n"
    "    if args and 'prompt' in args:\n"
    "        instruction.build_description(prompt=prompt)\n"
    "    return instruction\n"
    "def check_mode(registry, prompt, ids, kwargs_list, variants):\n"
    "    if not variants or not (variants[0] or '').strip():\n"
    "        return [False] * len(ids)\n"
    "    out = []\n"
    "    for iid, kw in zip(ids, kwargs_list):\n"
    "        try:\n"
    "            instruction = build(registry, iid, dict(kw), prompt)\n"
    "            passed = any(v.strip() and instruction.check_following(v)\n"
    "                        for v in variants)\n"
    "        except Exception:\n"
    "            passed = False\n"
    "        out.append(bool(passed))\n"
    "    return out\n"
    "def main():\n"
    "    from ifbench import instructions_registry\n"
    "    registry = instructions_registry.INSTRUCTION_DICT\n"
    "    print(json.dumps({'id': 'ready', 'registry': len(registry)}), flush=True)\n"
    "    for line in sys.stdin:\n"
    "        req = json.loads(line)\n"
    "        reply = {'id': req['id']}\n"
    "        try:\n"
    "            reply['strict'] = check_mode(\n"
    "                registry, req['prompt'], req['ids'], req['kwargs'],\n"
    "                req['strict_variants'])\n"
    "            reply['loose'] = check_mode(\n"
    "                registry, req['prompt'], req['ids'], req['kwargs'],\n"
    "                req['loose_variants'])\n"
    "        except Exception as exc:\n"
    "            reply['error'] = type(exc).__name__ + ': ' + str(exc)\n"
    "        print(json.dumps(reply), flush=True)\n"
    "main()\n"
)

@dataclass(frozen=True)
class IfbenchArgs:
    """The harness's shared flags plus this leg's verifier knobs."""

    leg: LegArgs
    ifbench_python: str
    nltk_data: str


class Checker:
    """Scores one response against one prompt's instruction list."""

    def check(self, prompt, instruction_ids, kwargs_list,
              response) -> tuple:
        raise NotImplementedError


class WorkerChecker(Checker):
    """Delegates every check to the --ifbench-python process."""

    def __init__(self, python_exe: str):
        process = subprocess.Popen(
            [python_exe, "-u", "-c", WORKER_SRC],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True)
        assert process.stdin is not None and process.stdout is not None
        self._proc = process
        self._stdin = process.stdin
        self._stdout = process.stdout
        self._ready = self._read_first_line()

    def _read_first_line(self) -> dict:
        """Parse the worker's startup handshake or report why it failed."""
        first = self._stdout.readline()
        try:
            return json.loads(first)
        except json.JSONDecodeError:
            return {"error": f"worker did not start: {first!r}"}

    @property
    def ready(self) -> bool:
        return self._ready.get("id") == "ready"

    def check(self, prompt, instruction_ids, kwargs_list,
              response) -> tuple:
        request = {"id": 0, "prompt": prompt,
                   "ids": instruction_ids,
                   "kwargs": kwargs_list,
                   "strict_variants": [response],
                   "loose_variants": loose_variants(response)}
        data = self._exchange(request)
        return data["strict"], data["loose"]

    def _exchange(self, request: dict) -> dict:
        """One request over stdin/stdout; raises on a dead or erroring worker."""
        self._stdin.write(json.dumps(request) + "\n")
        self._stdin.flush()
        reply_line = self._stdout.readline()
        if not reply_line:
            raise RuntimeError("ifbench worker died mid-run")
        data = json.loads(reply_line)
        if data.get("error"):
            raise RuntimeError(data["error"])
        return data

    def close(self) -> None:
        try:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - the worker already left or hangs; nothing to do
            pass

def loose_variants(response: str) -> list[str]:
    """The 8-way loose transform from allenai's evaluation_lib, order preserved.

    Strip the first line, the last line, or both, drop asterisks, and combine.
    The order matters for nothing but parity with the reference runner, which
    applied exactly this list.
    """
    lines = response.split("\n")
    variants = [
        response,
        response.replace("*", ""),
        "\n".join(lines[1:]).strip(),
        "\n".join(lines[:-1]).strip(),
        "\n".join(lines[1:-1]).strip(),
    ]
    variants += [variant.replace("*", "") for variant in variants[2:]]
    return variants


@dataclass
class RunSummary:
    """The numbers the result block is built from, plus the strict/loose split."""

    tally: Tally
    strict_prompts: int = 0
    loose_prompts: int = 0
    strict_instr_pass: int = 0
    strict_instr_total: int = 0
    loose_instr_pass: int = 0
    loose_instr_total: int = 0

def run_items(items, cfg: ModelConfig, args: LegArgs,
              checker: Checker) -> RunSummary:
    """Ask the model one question per item and score each answer, both ways."""
    tally = Tally()
    strict_prompts = 0
    loose_prompts = 0
    strict_pass = 0
    strict_total = 0
    loose_pass = 0
    loose_total = 0
    total = len(items)

    for index, item in enumerate(items, start=1):
        prompt = item["prompt"]
        ids = item["instruction_id_list"]
        kwargs = item["kwargs"]
        response: ChatResponse = ask(cfg,
                                     [{"role": "user", "content": prompt}],
                                     args)
        content = response.answer() or ""
        try:
            strict_list, loose_list = checker.check(prompt, ids, kwargs,
                                                    content)
        except Exception as exc:  # noqa: BLE001 - one bad check must not kill the run
            strict_list = [False] * len(ids)
            loose_list = [False] * len(ids)
            tally.notes.append(f"key {item['key']}: checker error "
                               f"{type(exc).__name__}: {exc}")

        strict_total += len(strict_list)
        strict_pass += sum(strict_list)
        loose_total += len(loose_list)
        loose_pass += sum(loose_list)

        tally.record(response, correct=None)
        if content.strip():
            tally.scored += 1
            if all(strict_list):
                tally.correct += 1
                strict_prompts += 1
            if all(loose_list):
                loose_prompts += 1

        print(f"  [{index}/{total}] key={item['key']} "
              f"strict={all(strict_list)} loose={all(loose_list)} "
              f"({response.elapsed_s:.1f}s)", flush=True)

    return RunSummary(tally=tally, strict_prompts=strict_prompts,
                      loose_prompts=loose_prompts,
                      strict_instr_pass=strict_pass,
                      strict_instr_total=strict_total,
                      loose_instr_pass=loose_pass,
                      loose_instr_total=loose_total)


def parse_args(argv=None) -> IfbenchArgs:
    parser = argparse.ArgumentParser(description="IFBench instruction-following leg")
    add_leg_args(parser, default_data=DEFAULT_DATA)
    parser.add_argument("--ifbench-python", default="",
                        help="python interpreter with the ifbench package; "
                             "defaults to this interpreter")
    parser.add_argument("--nltk-data",
                        default=os.environ.get("NLTK_DATA", DEFAULT_NLTK_DATA),
                        help="NLTK corpus dir for the verifier")
    namespace = parser.parse_args(argv)
    values = vars(namespace)
    python_exe = values.pop("ifbench_python")
    nltk_data = values.pop("nltk_data")
    return IfbenchArgs(leg=LegArgs(**values), ifbench_python=python_exe,
                       nltk_data=nltk_data)


def resolve_checker(python_exe: str, nltk_dir: str):
    """Return a ready WorkerChecker, or (None, readable reason) when unusable.

    The probe runs under the target interpreter, which is the only one that
    can tell us whether ifbench and its corpora actually work there.
    """
    if not python_exe:
        python_exe = sys.executable
    os.environ.setdefault("NLTK_DATA", nltk_dir)

    stdout, err = run_probe(python_exe)
    if err:
        return None, err

    assert stdout is not None
    info = parse_probe(stdout)
    if isinstance(info, str):
        return None, f"{python_exe!r} {info}"

    if not info.get("ok"):
        return None, classify_probe_failure(info, python_exe, nltk_dir)

    checker = WorkerChecker(python_exe)
    if not checker.ready:
        return None, (f"{python_exe!r} ifbench worker failed to start "
                      f"({checker._ready})")
    return checker, None


def run_probe(python_exe: str, timeout: float = 120):
    """Run the verifier probe; returns (stdout_text | None, reason | None)."""
    try:
        result = subprocess.run([python_exe, "-c", PROBE_SRC],
                                capture_output=True, text=True,
                                timeout=timeout)
    except OSError as exc:
        return None, f"--ifbench-python {python_exe!r} does not exist ({exc})"
    except subprocess.TimeoutExpired:
        return None, f"--ifbench-python {python_exe!r} timed out probing"

    if result.returncode != 0:
        return (None,
                f"{python_exe!r} cannot run the verifier probe: "
                f"{result.stderr.strip()[:160]}")
    return result.stdout, None


def parse_probe(stdout: str):
    """Extract the probe's JSON; returns a dict or a failure reason."""
    try:
        return json.loads(stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return "returned an unreadable probe result"


def classify_probe_failure(info: dict, python_exe: str,
                           nltk_dir: str) -> str:
    """Turn a failed probe into a readable reason for the not-run block."""
    message = str(info.get("error", "verifier unavailable"))
    corpus_hint = ("wordnet" in message or "punkt" in message
                   or "sent_tokenize" in message)
    if corpus_hint:
        return (f"NLTK corpus missing under NLTK_DATA={nltk_dir!r} "
                f"({message})")
    if "instructions_registry" in message:
        return f"ifbench instructions_registry is empty ({message})"
    return f"{python_exe!r} has no usable 'ifbench' package ({message})"


def render_extra(summary: RunSummary) -> dict:
    """The leg-specific numbers that ride alongside the harness block."""
    scored = summary.tally.scored
    return {
        "strict_prompt_correct": summary.strict_prompts,
        "strict_prompt_scored": scored,
        "loose_prompt_correct": summary.loose_prompts,
        "loose_prompt_scored": scored,
        "strict_prompt_accuracy": round(summary.strict_prompts / scored * 100, 1)
        if scored else 0.0,
        "loose_prompt_accuracy": round(summary.loose_prompts / scored * 100, 1)
        if scored else 0.0,
        "strict_instruction_correct": summary.strict_instr_pass,
        "strict_instruction_total": summary.strict_instr_total,
        "loose_instruction_correct": summary.loose_instr_pass,
        "loose_instruction_total": summary.loose_instr_total,
    }


def main(argv=None) -> int:
    parsed = parse_args(argv)
    args = parsed.leg

    checker, reason = resolve_checker(parsed.ifbench_python, parsed.nltk_data)
    if checker is None:
        reason = reason or "verifier unavailable"
        tally = Tally()
        tally.notes.append(reason)
        tally.errors.append(reason)
        print(f"ifbench not run: {reason}", file=sys.stderr)
        finish(LEG_ID, args, tally)
        return EXIT_NOT_RUN

    try:
        items = load_items(args.data, limit=args.limit)
        cfg = leg_config(args)
        print(f"IFBench: {len(items)} items | thinking={args.thinking} "
              f"| verifier={parsed.ifbench_python or sys.executable}",
              flush=True)
        summary = run_items(items, cfg, args, checker)
        block = finish(LEG_ID, args, summary.tally,
                       extra=render_extra(summary))
    finally:
        checker.close()
    print(f"IFBENCH DONE: loose {block['loose_prompt_correct']}/"
          f"{block['loose_prompt_scored']} "
          f"strict {block['strict_prompt_correct']}/"
          f"{block['strict_prompt_scored']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
