#!/usr/bin/env python3
"""Report on a finished battery run: rebuild the document from the run directory's own records.

    python3 scripts/report.py results/<run>

Writes report.md and report.json into the run directory and prints the markdown. It reads
only what the run itself recorded -- MANIFEST.json, run-plan.json and legs/*.result.json --
so it can be re-run at any time and writes nothing else.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.lib.report import REPORT_JSON_FILE, REPORT_MD_FILE  # noqa: E402
from scripts.lib.report import ReportError, write_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="the run directory to report on (e.g. results/<run>)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        md_path, json_path, markdown = write_report(Path(args.run_dir).expanduser())
    except ReportError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(markdown, end="" if markdown.endswith("\n") else "\n")
    print(f"wrote {md_path} and {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
