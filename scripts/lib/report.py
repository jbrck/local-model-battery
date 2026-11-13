"""Rebuild the run's document from the run directory's own records.

Every number in the report comes off disk -- MANIFEST.json, run-plan.json, legs/*.result.json
-- so the report can be re-run at any time and always says the same thing. Where a section has
no data it says so in one line rather than going quiet: the point of the document is to say
what was and was not controlled.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_FILE = "MANIFEST.json"
PLAN_FILE = "run-plan.json"
RUN_STATE_FILE = "run.json"
REPORT_MD_FILE = "report.md"
REPORT_JSON_FILE = "report.json"
LEGS_DIRNAME = "legs"
RESULT_SUFFIX = ".result.json"
NOT_RUN_STATUS = "not run"
NO_RESULT_STATUS = "no result"
NO_RUN_STATUSES = (NOT_RUN_STATUS, NO_RESULT_STATUS)
COMPARABLE_KEYS = ("cap", "ctx", "thinking")
SEED_VARIABLE = "seed"
ENGINE_VARIABLE = "engine_version"
HARNESS_PREFIX = "harness_version:"


class ReportError(Exception):
    """A run directory that cannot be reported on; the message is the whole explanation."""


def _load_json(path: Path) -> tuple[dict | None, str]:
    """The payload, or (None, a one-line reason) when the file is absent or unreadable."""
    if not path.is_file():
        return None, f"{path.name} is missing"
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        return None, f"{path.name} is unreadable ({error})"
    if not isinstance(payload, dict):
        return None, f"{path.name} is not a JSON object"
    return payload, ""


def _leg_result_path(run_dir: Path, leg_id: str) -> Path:
    return run_dir / LEGS_DIRNAME / f"{leg_id}{RESULT_SUFFIX}"


def _result_leg_ids(run_dir: Path) -> list[str]:
    legs_dir = run_dir / LEGS_DIRNAME
    if not legs_dir.is_dir():
        return []
    names = []
    for path in sorted(legs_dir.glob(f"*{RESULT_SUFFIX}")):
        if path.is_file():
            names.append(path.name[:-len(RESULT_SUFFIX)])
    return names


def _decision_index(plan: dict) -> dict[str, dict]:
    return {row.get("variable", ""): row for row in plan.get("decisions", [])}


def _planned_index(plan: dict) -> dict[str, dict]:
    return {row.get("id", ""): row for row in plan.get("legs", [])}


def _leg_ids(plan: dict | None, run_dir: Path, run_state: dict | None) -> list[str]:
    ids = set(_result_leg_ids(run_dir))
    if plan:
        ids.update(_planned_index(plan))
    if run_state and isinstance(run_state.get("results"), dict):
        ids.update(run_state["results"])
    return sorted(ids)


def _note_error(leg_id: str, error: str | None, broken: dict[str, str]) -> None:
    """Record a leg result file that could not be read. A missing file is not an error."""
    if error and "is missing" not in error:
        broken[leg_id] = error


def _state_fallback(run_state: dict | None, leg_id: str) -> dict | None:
    """The leg's block as run.json recorded it, when no result file supplied one."""
    if not run_state or not isinstance(run_state.get("results"), dict):
        return None
    fallback = run_state["results"].get(leg_id)
    return fallback if isinstance(fallback, dict) else None


def _block_for(leg_id: str, run_dir: Path, run_state: dict | None,
               broken: dict[str, str]) -> dict | None:
    """The leg's result block, preferring its own file and falling back to the run state."""
    block, error = _load_json(_leg_result_path(run_dir, leg_id))
    if block is not None:
        return block
    _note_error(leg_id, error, broken)
    return _state_fallback(run_state, leg_id)


def _shape_line(plan: dict | None, manifest: dict) -> dict:
    """Where the model's numbers came from: the plan's footprint, or the manifest's fit."""
    if plan and isinstance(plan.get("footprint"), dict):
        return dict(plan["footprint"])
    fit = manifest.get("fit")
    if isinstance(fit, dict):
        return {"known": not fit.get("estimate"), "source": "manifest fit",
                "weights_gb": fit.get("weights_gb", 0),
                "kv_bytes_per_token": fit.get("kv_bytes_per_token", 0),
                "path": "", "notes": [], "missing": ["the plan's shape record is absent"]}
    return {"known": False, "source": "", "weights_gb": 0, "kv_bytes_per_token": 0,
            "path": "", "notes": [], "missing": ["neither the plan nor the manifest describes "
                                                 "the shape"]}


def _first_text(*values: Any) -> str:
    """The first value that is truthy, as text. Empty string when there is none."""
    for value in values:
        if value:
            return str(value)
    return ""


def _engine_info(manifest: dict, decisions: dict[str, dict]) -> dict:
    props_raw = manifest.get("props")
    props: dict = props_raw if isinstance(props_raw, dict) else {}
    build = _first_text(manifest.get("engine_version"), props.get("build_info"),
                        props.get("version"))
    return {"name": _first_text(manifest.get("engine")), "build": build,
            "served_from": _first_text(props.get("model_path"), manifest.get("serve_cmdline")),
            "recorded": bool(build),
            "decision": decisions.get(ENGINE_VARIABLE, {}).get("chosen", "")}


def _checkpoint(weights: dict) -> str:
    """The weights file the run tested, or the honest placeholder when nothing recorded it."""
    return str(weights.get("path") or "")


def _format_elapsed(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        return "-"
    total = int(seconds)
    hours, seconds = divmod(total, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


def _leg_row(leg_id: str, planned: dict | None, block: dict | None,
             broken: dict[str, str]) -> dict:
    row: dict[str, Any] = {
        "leg": leg_id,
        "status": NOT_RUN_STATUS,
        "reason": "",
        "correct": None, "scored": None, "score_pct": None,
        "items": None, "cases_scored": None, "cases_not_scored": None,
        "failed": None, "timeout_hits": None, "cap_hits": None, "runaway": None,
        "tps": None, "elapsed_s": None, "elapsed": "-",
        "cap": None, "ctx": None, "thinking": "", "seed": None,
        "items_sha": "", "harness_version": "",
    }
    if planned:
        row["planned_cap"] = planned.get("cap")
        row["planned_ctx"] = planned.get("ctx")
        row["planned_thinking"] = planned.get("thinking")
    if block is None:
        row["status"] = NO_RESULT_STATUS
        row["reason"] = broken.get(leg_id) or (
            f"legs/{leg_id}{RESULT_SUFFIX} is missing and no run state carries it")
        return row
    row["status"] = str(block.get("status", "unknown"))
    row["reason"] = str(block.get("reason", ""))
    for key, name in (("correct", "correct"), ("scored", "scored"), ("items", "items"),
                      ("failed", "failed"), ("timeout_hits", "timeout_hits"),
                      ("cap_hits", "cap_hits"), ("runaway", "runaway"), ("cap", "cap"),
                      ("ctx", "ctx"), ("seed", "seed")):
        row[name] = block.get(key)
    row["score_pct"] = block.get("score_pct")
    row["tps"] = block.get("tps")
    row["thinking"] = str(block.get("thinking", ""))
    row["items_sha"] = str(block.get("items_sha", ""))
    row["harness_version"] = str(block.get("harness_version", ""))
    row["elapsed_s"] = block.get("elapsed_s")
    row["elapsed"] = _format_elapsed(block.get("elapsed_s"))
    items = row["items"] if isinstance(row["items"], int) else 0
    scored = row["scored"] if isinstance(row["scored"], int) else 0
    row["cases_scored"] = scored
    row["cases_not_scored"] = max(0, items - scored)
    return row


def _deviation_findings(decisions: dict) -> list[dict]:
    """Every decision the plan moved off the standard, as a finding."""
    findings: list[dict] = []
    for row in decisions.values():
        if not row.get("deviation"):
            continue
        variable = row.get("variable", "")
        leg = variable.split(":", 1)[1] if ":" in variable else ""
        findings.append({"leg": leg, "key": variable, "recorded": row.get("chosen"),
                         "baseline": row.get("standard"),
                         "detail": f"the plan moved {variable} to {row.get('chosen')} "
                                   f"(standard {row.get('standard')})"})
    return findings


def _harness_finding(row: dict, decisions: dict, leg_id: str) -> tuple[list[dict], list[str]]:
    """The harness version this leg recorded, against the one the plan pinned."""
    recorded = row["harness_version"]
    baseline = decisions.get(f"{HARNESS_PREFIX}{leg_id}", {}).get("chosen")
    if recorded == baseline:
        return [], []
    if not recorded:
        return [], [f"{leg_id} records no harness version"]
    if not baseline:
        return [], [f"{leg_id} records harness {recorded}; the plan does not name "
                    "a baseline for it"]
    return [{"leg": leg_id, "key": "harness_version", "recorded": recorded,
             "baseline": baseline,
             "detail": f"harness bump: {leg_id} scored by {recorded}; "
                       f"the plan pinned {baseline}"}], []


def _seed_finding(leg_id: str, row: dict, seed_baseline) -> list[dict]:
    """The seed this leg ran with, against the one the plan pinned."""
    seed = row.get("seed")
    if seed is None or seed_baseline is None or str(seed) == str(seed_baseline):
        return []
    return [{"leg": leg_id, "key": SEED_VARIABLE, "recorded": seed, "baseline": seed_baseline,
             "detail": f"{leg_id} ran with seed {seed}; the plan pinned {seed_baseline}"}]


def _leg_findings(row: dict, baseline_leg: dict, seed_baseline) -> list[dict]:
    """One leg's recorded values against its planned baseline, as findings."""
    leg_id = row["leg"]
    findings: list[dict] = []
    for key in COMPARABLE_KEYS:
        recorded, baseline = row.get(key), baseline_leg.get(key)
        if recorded is None or baseline is None or str(recorded) == str(baseline):
            continue
        findings.append({"leg": leg_id, "key": key, "recorded": recorded, "baseline": baseline,
                         "detail": f"{leg_id} ran at {key} {recorded}; the plan set {baseline}"})
    return findings + _seed_finding(leg_id, row, seed_baseline)


def _no_baseline_note(row: dict) -> str:
    """One leg's recorded values, stated plainly when there is no baseline to check them."""
    note = f"{row['leg']} recorded cap {row['cap']}, thinking {row['thinking'] or '?'}"
    return note + (f", items hash {row['items_sha']}" if row["items_sha"] else "")


def _items_note(row: dict) -> str:
    """The items hash another run has to match, or the fact that none was recorded."""
    if row["items_sha"]:
        return (f"{row['leg']} items hash {row['items_sha']}: another run compares only "
                "at the same hash")
    return f"{row['leg']} records no items hash"


def _engine_finding(engine: dict) -> tuple[list[dict], list[str]]:
    """The endpoint's engine build, against the one the plan pinned."""
    if not engine["recorded"]:
        return [], ["the manifest records no engine build"]
    if engine["decision"] and engine["build"] != engine["decision"]:
        return [{"leg": "", "key": ENGINE_VARIABLE, "recorded": engine["build"],
                 "baseline": engine["decision"],
                 "detail": f"engine change: the endpoint reports {engine['build']}; "
                           f"the plan pinned {engine['decision']}"}], []
    return [], []


def _leg_comparability(row: dict, planned: dict, decisions: dict, seed_baseline,
                       baseline_recorded: bool) -> tuple[list[dict], list[str]]:
    """One leg's findings and notes, against the baseline the plan recorded for it."""
    if not baseline_recorded:
        return [], [_no_baseline_note(row)]
    findings: list[dict] = []
    notes: list[str] = []
    baseline_leg = planned.get(row["leg"])
    if baseline_leg is None:
        notes.append(f"{row['leg']} has a result but no planned baseline in run-plan.json")
    else:
        findings += _leg_findings(row, baseline_leg, seed_baseline)
    harness_findings, harness_notes = _harness_finding(row, decisions, row["leg"])
    return findings + harness_findings, notes + harness_notes + [_items_note(row)]


def _comparability(plan: dict | None, manifest: dict, rows: list[dict],
                   plan_error: str) -> dict:
    """The baseline the run names for itself, and every recorded value that misses it."""
    decisions = _decision_index(plan) if plan else {}
    baseline_recorded = plan is not None
    stamp_line = str((plan or manifest).get("stamp_line", ""))
    notes = [] if baseline_recorded else [
        f"the run records no baseline: {plan_error}, so the recorded values "
        "have nothing on file to be compared against"]
    findings = _deviation_findings(decisions)
    seed_baseline = decisions.get(SEED_VARIABLE, {}).get("chosen")
    planned = _planned_index(plan or {})
    for row in rows:
        if row["status"] in NO_RUN_STATUSES:
            continue
        row_findings, row_notes = _leg_comparability(
            row, planned, decisions, seed_baseline, baseline_recorded)
        findings += row_findings
        notes += row_notes
    engine_findings, engine_notes = _engine_finding(_engine_info(manifest, decisions))
    return {"baseline_recorded": baseline_recorded, "stamp_line": stamp_line,
            "findings": findings + engine_findings, "notes": notes + engine_notes}


def _sources(plan_error: str, run_state: dict | None, run_state_error: str,
             broken: dict[str, str]) -> dict:
    """Where each number came from, and every file the report could not read."""
    if run_state_error and "is missing" not in run_state_error:
        note = run_state_error
    elif run_state and not isinstance(run_state.get("results"), dict):
        note = f"{RUN_STATE_FILE} is present but carries no results mapping"
    else:
        note = "read" if run_state else "not present"
    return {"manifest": MANIFEST_FILE, "plan": plan_error or "read",
            "run_state": note, "result_errors": dict(broken)}


def build_report(run_dir: Path) -> dict:
    """Everything the report says, straight from the run directory. Raises ReportError
    when there is no manifest to report on."""
    run_dir = Path(run_dir)
    manifest, manifest_error = _load_json(run_dir / MANIFEST_FILE)
    if manifest is None:
        raise ReportError(
            f"cannot report on {run_dir}: {manifest_error} -- a run directory holds "
            f"{MANIFEST_FILE}, {PLAN_FILE} and {LEGS_DIRNAME}/*{RESULT_SUFFIX}")
    plan, plan_error = _load_json(run_dir / PLAN_FILE)
    run_state, run_state_error = _load_json(run_dir / RUN_STATE_FILE)
    decisions = _decision_index(plan) if plan else {}
    broken: dict[str, str] = {}
    planned = _planned_index(plan) if plan else {}
    rows = [_leg_row(leg_id, planned.get(leg_id), _block_for(leg_id, run_dir, run_state, broken),
                     broken) for leg_id in _leg_ids(plan, run_dir, run_state)]
    weights = _shape_line(plan, manifest)
    return {
        "run_dir": str(run_dir),
        "model": manifest.get("model", ""),
        "battery_version": manifest.get("battery_version", ""),
        "standard_version": manifest.get("standard_version", ""),
        "stamp_line": manifest.get("stamp_line", ""),
        "shape": weights,
        "engine": _engine_info(manifest, decisions),
        "checkpoint": _checkpoint(weights),
        "decisions": list(plan.get("decisions", [])) if plan else [],
        "legs": rows,
        "comparability": _comparability(plan, manifest, rows, plan_error),
        "sources": _sources(plan_error, run_state, run_state_error, broken),
    }


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        cells = [str(value).replace("|", "\\|") if value is not None else "-" for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _missing_text(weights: dict) -> str:
    """The reasons no reader described the shape, as one line."""
    return ", ".join(weights.get("missing") or []) or "no reader described it"


def _shape_detail(weights: dict) -> str:
    """The one-line footprint: weights, KV bytes per token, and where it was read from."""
    path = f" ({weights['path']})" if weights.get("path") else ""
    return (f"- shape: {weights.get('weights_gb', 0)} GB of weights, "
            f"{weights.get('kv_bytes_per_token', 0)} B/token of "
            f"{weights.get('kv_type', '?')} KV, read from {weights.get('source') or '?'}{path}")


def _shape_lines(weights: dict) -> list[str]:
    """The footprint the run knew, or the plain statement that nobody described it."""
    if not weights.get("known"):
        return [f"- shape: unknown ({_missing_text(weights)})"]
    lines = [_shape_detail(weights)]
    for note in weights.get("notes") or []:
        lines.append(f"  - note: {note}")
    return lines


def _engine_line(engine: dict) -> str:
    """The engine build the manifest recorded, or the fact that none was."""
    if not engine.get("recorded"):
        return "- engine: the manifest records no engine build"
    name = f"{engine['name']} " if engine.get("name") else ""
    return f"- engine: {name}{engine['build']}"


def _render_what_was_tested(report: dict) -> list[str]:
    lines = ["## What was tested", ""]
    lines.append(f"- model: {report['model']} (battery {report['battery_version']}, standard "
                 f"{report['standard_version']})")
    lines += _shape_lines(report["shape"])
    lines.append(_engine_line(report["engine"]))
    checkpoint = report["checkpoint"]
    lines.append(f"- checkpoint: {checkpoint}" if checkpoint
                 else "- checkpoint: not recorded; neither the manifest nor the plan names "
                      "the weights file")
    return lines


def _render_decisions(report: dict) -> list[str]:
    lines = ["## How it was set up", ""]
    if not report["decisions"]:
        lines.append("The run records no decision rows; run-plan.json is missing or empty.")
        return lines
    lines.append("Chosen against the standard's value, with the source of each number:")
    lines.append("")
    lines += _table(
        ["variable", "chosen", "standard", "source", "consequence"],
        [[row.get("variable"), row.get("chosen"), row.get("standard"), row.get("source"),
          row.get("consequence")] for row in report["decisions"]])
    return lines


def _render_legs(report: dict) -> list[str]:
    lines = ["## The legs", ""]
    if not report["legs"]:
        lines.append("No planned legs and no leg result files in legs/.")
        return lines
    lines += _table(
        ["leg", "status", "score", "cases scored", "cases not scored", "reason"],
        [[row["leg"], row["status"],
          f"{row['correct']}/{row['scored']} ({row['score_pct']}%)"
          if row["scored"] is not None else "-",
          row["cases_scored"] if row["cases_scored"] else "-",
          row["cases_not_scored"] if row["cases_not_scored"] else "-",
          row["reason"]] for row in report["legs"]])
    return lines


def _render_speed(report: dict) -> list[str]:
    lines = ["## Throughput and wall time", ""]
    timed = [row for row in report["legs"]
             if row["tps"] is not None or row["elapsed_s"] is not None]
    if not timed:
        lines.append("No leg recorded a throughput or a wall time.")
        return lines
    lines += _table(["leg", "tok/s", "wall time"],
                    [[row["leg"], row["tps"] if row["tps"] is not None else "-",
                      row["elapsed"]] for row in timed])
    return lines


def _render_comparability(report: dict) -> list[str]:
    comp = report["comparability"]
    lines = ["## What this run cannot be compared to", ""]
    if comp["stamp_line"]:
        lines.append(f"The run names this baseline for itself: {comp['stamp_line']}.")
    else:
        lines.append("The run names no stamp; nothing here declares what it compares with.")
    lines.append("")
    if comp["findings"]:
        for finding in comp["findings"]:
            lines.append(f"- not comparable: {finding['detail']}.")
    else:
        lines.append("- Nothing recorded differs from the baseline the run names: these "
                     "numbers compare with other runs at the same stamp.")
    if comp["notes"]:
        lines.append("")
        for note in comp["notes"]:
            lines.append(f"- {note}")
    return lines


def _render_sources(report: dict) -> list[str]:
    sources = report["sources"]
    lines = ["## Sources", "",
             f"- {MANIFEST_FILE}: read.",
             f"- {PLAN_FILE}: {sources['plan']}.",
             f"- {RUN_STATE_FILE}: {sources['run_state']}."]
    if sources["result_errors"]:
        lines += [f"- {leg_id}: {error}." for leg_id, error in sorted(sources["result_errors"].items())]
    else:
        lines.append("- Every leg result file read cleanly.")
    return lines


def render_markdown(report: dict) -> str:
    """The document: every section present, an empty section saying so in one line."""
    lines = [f"# {report['model']} on the portable battery {report['standard_version']}", ""]
    if report["stamp_line"]:
        lines.append(f"Stamp: {report['stamp_line']}")
        lines.append("")
    for section in (_render_what_was_tested, _render_decisions, _render_legs, _render_speed,
                    _render_comparability, _render_sources):
        lines += section(report)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def write_report(run_dir: Path) -> tuple[Path, Path, str]:
    """Build the report, write report.md and report.json, and hand back the markdown."""
    run_dir = Path(run_dir)
    report = build_report(run_dir)
    markdown = render_markdown(report)
    md_path = run_dir / REPORT_MD_FILE
    json_path = run_dir / REPORT_JSON_FILE
    md_path.write_text(markdown)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    return md_path, json_path, markdown
