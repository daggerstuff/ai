#!/usrv/bin/env python3
"""Run deslop + anti-sycophancy on deduped JSONL using streaming I/O."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add deslop to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "deslop"))

from deslop.engine import CleanOptions, clean_record
from deslop.rules.core import load_rule_set

INPUT = Path("/home/vivi/pixelated/ai/data/raw/deduped/all_deduped.jsonl")
OUTPUT = Path("/home/vivi/pixelated/ai/data/raw/deduped/all_desloped.jsonl")
REPORT = Path("/home/vivi/pixelated/ai/data/raw/deduped/deslop_report.txt")

PACKS = ["generic-ai", "sycophancy", "chatbot-assistant", "fabrication-signal", "synthetic-evals"]

# Fields to scan: message content fields
FIELDS = ("messages.*.content",)


def main() -> None:
    rules = load_rule_set(None, PACKS)
    options = CleanOptions(rules=rules, fields=FIELDS)

    total = 0
    rewritten = 0
    fields_changed = 0

    start = time.time()
    last_report = start

    with INPUT.open("r", encoding="utf-8") as fin, OUTPUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_id = record.get("id", record.get("source", f"row_{total}"))
            cleaned, changed = clean_record(record, str(record_id), options)

            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            total += 1
            if changed > 0:
                rewritten += 1
                fields_changed += changed

            if total % 50000 == 0:
                now = time.time()
                elapsed = now - start
                rate = total / elapsed
                eta = (849760 - total) / rate if rate > 0 else 0
                print(
                    f"  {total:>8d} records | {rewritten:>6d} rewritten | {rate:.0f} rec/s | ETA {eta:.0f}s",
                    file=sys.stderr,
                )
                last_report = now

    elapsed = time.time() - start
    density = (rewritten / total * 100) if total > 0 else 0

    lines = [
        f"Deslop Report",
        f"=============",
        f"Input: {INPUT}",
        f"Output: {OUTPUT}",
        f"Packs: {', '.join(PACKS)}",
        f"Fields: {', '.join(FIELDS)}",
        f"",
        f"Records processed: {total}",
        f"Records rewritten: {rewritten}",
        f"Fields rewritten: {fields_changed}",
        f"Slop density: {density:.2f}%",
        f"Time: {elapsed:.1f}s ({total / elapsed:.0f} rec/s)",
    ]

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nDone: {total} processed, {rewritten} rewritten ({density:.2f}% density)")
    print(f"Report: {REPORT}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
