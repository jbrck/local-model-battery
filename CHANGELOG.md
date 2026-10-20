# Changelog

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
