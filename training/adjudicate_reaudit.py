#!/usr/bin/env python3
"""Re-audit adjudication driver: judge the 23-arc re-audit wave flags."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adjudicate_flags as af

ARCS = ("arc_0010,arc_0012,arc_0016,arc_0021,arc_0022,arc_0052,arc_0063,"
        "arc_0068,arc_0079,arc_0085,arc_0141,arc_0157,arc_0162,arc_0213,"
        "arc_0214,arc_0226,arc_0230,arc_0274,arc_0300,arc_0310,arc_0355,"
        "pilot_03,pilot_05")

out = af.OUT_DIR / "reaudit23"
af.OUT_DIR = out
af.JUDGED_PATH = out / "judged_flags.jsonl"
af.REPORT_PATH = out / "report.md"
af.SUMMARY_PATH = out / "summary.json"

sys.argv = ["adjudicate_flags.py", "--arcs", ARCS, "--last", "23"]
raise SystemExit(af.main())
