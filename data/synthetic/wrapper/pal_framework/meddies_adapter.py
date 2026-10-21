"""Adapt real ``Meddies/meddies-persona-vie`` records to PAL fixture schema.

The real Meddies dataset ships Vietnamese field values and has slightly
different key names than the simplified fixture schema used by
``meddies_to_pal.format_persona`` (e.g. ``demographics.province`` not
``location``, ``healthcare_behavior.health_literacy_level`` not
``health_literacy``, ``healthcare_seeking_pattern`` not ``preference``).

This adapter performs an explicit, deterministic mapping so the rest of the
PAL pipeline (Phase 1 NL formatter, Phase 2.1/2.2/3.1 generators) can operate
on canonical fixtures while consuming real records.

The adapter is intentionally loss-less at the fixture-shape layer; the full
upstream record is preserved as ``_raw`` for downstream consumers that need it
(e.g. the synthesizer that builds preference-pair prompts).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Vietnamese -> English translation tables for the small closed enum of values
# the PAL pipeline reads. We keep these explicit so persona strings remain
# stable and English-only (PAL paper specifies English NL persona strings).
# --------------------------------------------------------------------------- #

_GENDER_VI_TO_EN: dict[str, str] = {
    "Nam": "male",
    "Nữ": "female",
    "Khác": "non-binary",
}

_HEALTH_LITERACY_VI_TO_EN: dict[str, str] = {
    "Thấp": "low",
    "Trung bình": "average",
    "Cao": "high",
}

_HEALTH_SEEKING_VI_TO_EN: dict[str, str] = {
    "Ưu tiên Đông y": "traditional medicine",
    "Ưu tiên Tây y": "modern medicine",
    "Kết hợp": "integrated medicine",
    "Kết hợp Đông/Tây y": "integrated medicine",
    "Ngay lập tức": "immediate care",
    "Tự điều trị": "self-treatment",
    "Chưa khám bệnh": "no prior care",
}

# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #

def _translate(value: Any, table: dict[str, str], default: str) -> str:
    """Translate a Vietnamese enum value to English, falling through to default."""
    if value is None:
        return default
    if isinstance(value, str) and value in table:
        return table[value]
    if isinstance(value, str) and value.strip():
        # Unknown Vietnamese string — leave as-is rather than drop info.
        return value
    return default


def adapt_record(meddies_record: dict[str, Any]) -> dict[str, Any]:
    """Convert a real Meddies record to the canonical PAL fixture shape.

    Returns a dict with the structure expected by ``format_persona``:

        {
            "demographics": {"age": int, "gender": str, "location": str},
            "healthcare_behavior": {"health_literacy": str, "preference": str},
            "_raw": <full upstream record>
        }
    """
    demo = meddies_record.get("demographics", {}) or {}
    health = meddies_record.get("healthcare_behavior", {}) or {}

    age = demo.get("age")
    if not isinstance(age, int):
        # Fall back to date_of_birth if present, else unknown-age sentinel.
        age = "unknown age"

    gender_en = _translate(demo.get("gender"), _GENDER_VI_TO_EN, "person")
    # Real Meddies has ``province`` (e.g. "Hà Nội") — map to fixture "location".
    location = demo.get("province") or demo.get("location") or "Vietnam"

    health_literacy_en = _translate(
        health.get("health_literacy_level"),
        _HEALTH_LITERACY_VI_TO_EN,
        "average",
    )
    preference_en = _translate(
        health.get("healthcare_seeking_pattern"),
        _HEALTH_SEEKING_VI_TO_EN,
        "standard medicine",
    )

    return {
        "demographics": {
            "age": age,
            "gender": gender_en,
            "location": location,
        },
        "healthcare_behavior": {
            "health_literacy": health_literacy_en,
            "preference": preference_en,
        },
        "_raw": meddies_record,
    }


def adapt_records(records: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Lazily adapt an iterable of Meddies records."""
    for r in records:
        yield adapt_record(r)


def main(argv: list[str] | None = None) -> int:
    """CLI: read raw Meddies JSONL → write adapted PAL fixture JSONL.

    Usage:
        python meddies_adapter.py <input.jsonl> <output.jsonl>
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Adapt raw Meddies/meddies-persona-vie records to PAL fixture schema.",
    )
    parser.add_argument("input", type=Path, help="Raw Meddies JSONL (one record per line).")
    parser.add_argument("output", type=Path, help="Output adapted fixture JSONL.")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 1

    n = 0
    with args.input.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            adapted = adapt_record(record)
            fout.write(json.dumps(adapted, ensure_ascii=False) + "\n")
            n += 1
    print(f"adapted {n} records -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
