# Changelog

## Unreleased

New leg: `ctxspeed`, decode speed against context depth.

**Added**
- `scripts/ctxspeed_bench.py` measures decode tokens per second as the prompt grows (512 up to
  1048576), on free prose and on a verbatim copy of the context. It does not trust the served
  window: it climbs until a rung is refused, clipped, timed out or crashes the server, then
  bisects to within 4096 tokens. The result's `ceiling` block names the deepest depth that
  worked, the first that failed and why, so the leg reports what the host takes. Before it
  climbs it reads the server's own window from `/props` and does not send rungs that window can
  only refuse, so no hour goes on a prefill that is certain to fail; the top rung lands at the
  window's usable edge. The result says when the window, not the hardware, ended the climb.
  `--ignore-window` tries every rung anyway. A streaming client
  (`scripts/lib/stream.py`) splits time to first token from decode time, so prefill no longer
  hides inside a whole-body timing. Filler comes from the bundled datasets, shuffled by seed. A prompt past one corpus pass
  (about a million characters) takes more passes with the words reshuffled inside each piece,
  so no run of words repeats and a speculative drafter cannot win on repetition.
- The result block carries `per_depth` (median rate, retention against the shallowest depth,
  prefill rate, copy ratio) and `per_trial`. Drafter on versus off is two runs on two server
  launches, compared by `per_depth`.

**Comparability**
- Keyed on `cap`, `ctx`, `thinking`, `seed` and `items_sha` (hash of the filler corpus).

---

MATH-500 grader reads inline-math final answers.

**Changed**
- `scripts/bench_math500.py` now takes a final answer wrapped in `\(...\)` as well as
  `\boxed{}`, and rewrites common LaTeX (`\dfrac`, `\cdot`, `\pi`, spacing) on both gold and
  prediction before the sympy compare. The prompt never asks for `\boxed{}`, so a model that
  answers in prose or inline math scored near zero for format, not for the math.
- A lone `$` is not trusted to open a math span (it collides with currency).

**Comparability**
- Scores from the old grader are lower for any model that does not box. The first run that
  showed it: inkling-2-mini on Project Maya, 214/500 (42.8%) under the old grader, 267/500
  (53.4%) under this one, 478/500 (95.6%) under a looser offline rescore that is not in the repo.
  Regrade older runs from their saved outputs before comparing them with a run on this grader.
- The prompt is unchanged.

## 2.0.0 — 2026-10-20

Writing benchmark replaced. Pairwise prose ELO is gone.

**Removed**
- `scripts/bench_prose_elo.py` and `data/prose_tasks.json`. Pairwise judging showed
  the first-slot answer winning 75% of decided verdicts, so the ratings were not
  measuring prose quality. No replacement for pairwise ELO: use the suite below.

**Added**
- `data/writing_suite_v1.json` — 138 tasks across 8 categories (xpost 36, xthread 12,
  long_form 12, revision 24, summary 12, craft 12, persuasion 12, copy 18). Per-task
  rubrics with 2/5/8 anchors, scored 1-10 per criterion.
- `scripts/bench_writing_generate.py` — thin generation pipe. One user message, task
  verbatim, temperature 0.7, 16k tokens, no system prompt, no cleaning, no retries.
- `scripts/build_scoring_corpus.py` — strips model identity, assigns opaque IDs,
  seeds the shuffle, writes the key outside the corpus.
- `scripts/score_writing.py` — blind rubric scoring against any endpoint. Records
  judge model and timestamp in the score file; unparseable replies become errors,
  never scores.
- `scripts/report_scores.py` — joins scores to models, reports mean/per-category/
  per-criterion, plus mechanical word and character compliance.
- `scripts/validate_suite.py` — suite schema and rubric validation.
- `scripts/lib/code_exec.py` — subprocess Python test runner.
- `tests/test_writing_suite.py` — 14 tests: suite validation, corpus blinding,
  seeded shuffle, compliance checks, judge parsing.

**Fixed**
- `scripts/lib/model.py`: `chat_url` called `self.rstrip('/')` on a ModelConfig
  instead of the endpoint string, so every request would have crashed.
- `bench_humaneval_plus.py`: called `evalplus.evaluate()` with a signature that does
  not exist, so the benchmark could not run. Replaced with a self-contained
  subprocess test runner. Also stripped the completion's leading indentation, which
  broke every indented HumanEval body.
- `bench_math500.py`: inline imports moved to module level.

**Changed**
- Dependencies down to `sympy` alone (`requests`, `urllib3`, `evalplus`, `astor`
  were unused or replaced).
- README and `skill/SKILL.md` rewritten. Installation and setup are generic: no
  machine names, no internal paths, no private model aliases.
- Requests carry explicit `max_tokens` and `temperature` instead of relying on
  endpoint defaults, so runs are reproducible across providers.
