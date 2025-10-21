"""
Safety Monitoring CLI

Reads JSONL with {"text": "...", "source": "..."} or raw strings in "text" key,
produces per-line safety events and summary counters. Model-free.

Usage:
  uv run python ai/pixel/training/safety_monitoring_cli.py --input logs.jsonl --events out_events.jsonl --summary summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai.pixel.training.safety_monitoring import SafetyMonitor


def main() -> None:
    parser = argparse.ArgumentParser(description="Safety Monitoring CLI")
    parser.add_argument("--input", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    monitor = SafetyMonitor()

    inp = Path(args.input)
    ev_out = Path(args.events)
    sum_out = Path(args.summary)
    ev_out.parent.mkdir(parents=True, exist_ok=True)
    sum_out.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    total_events = 0

    with inp.open("r", encoding="utf-8") as fin, ev_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text") or obj.get("response") or obj.get("content") or ""
            source = obj.get("source", "unknown")
            events = monitor.record_from_text(text, source=source)
            for ev in events:
                fout.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
            total_events += len(events)
            total_lines += 1

    summary = {"lines": total_lines, "events": total_events, "counters": monitor.counters}
    sum_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "lines": total_lines, "events": total_events, "summary": str(sum_out)}))


if __name__ == "__main__":
    main()
