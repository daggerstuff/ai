from __future__ import annotations

import json
from pathlib import Path

import meddies_to_pal as m


def test_format_persona_basic():
    record = {
        "demographics": {"age": 45, "gender": "female", "location": "Hanoi"},
        "healthcare_behavior": {"health_literacy": "low", "preference": "traditional medicine"},
    }
    assert (
        m.format_persona(record)
        == "This patient is a 45-year-old female from Hanoi with low health literacy who prefers traditional medicine."
    )


def test_format_persona_missing_fields():
    record = {"demographics": {"age": 30}}
    assert (
        m.format_persona(record)
        == "This patient is a 30-year-old person from Vietnam with average health literacy who prefers standard medicine."
    )


def test_format_persona_empty_record():
    assert (
        m.format_persona({})
        == "This patient is a unknown age-year-old person from Vietnam with average health literacy who prefers standard medicine."
    )


def test_no_json_leakage():
    record = {
        "demographics": {"age": 50, "gender": "male", "location": "HCMC"},
        "healthcare_behavior": {"health_literacy": "high", "preference": "standard medicine"},
    }
    out = m.format_persona(record)
    for ch in ("{", "}", "'", '"'):
        assert ch not in out


def test_process_file_roundtrip(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    records = [
        {
            "demographics": {"age": 45, "gender": "female", "location": "Hanoi"},
            "healthcare_behavior": {"health_literacy": "low", "preference": "traditional medicine"},
        },
        {
            "demographics": {"age": 30, "gender": "male", "location": "HCMC"},
            "healthcare_behavior": {"health_literacy": "high", "preference": "standard medicine"},
        },
    ]
    inp.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    m.process_file(str(inp), str(out))
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert "persona_string" in rec
        assert rec["persona_string"] == m.format_persona(records[i])


def test_process_file_skips_blank_lines(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    record = {"demographics": {"age": 22, "gender": "male", "location": "Hanoi"}}
    inp.write_text(
        json.dumps(record) + "\n\n   \n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )
    m.process_file(str(inp), str(out))
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_process_file_preserves_unicode(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    record = {"demographics": {"age": 60, "gender": "female", "location": "Đà Nẵng"}}
    inp.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    m.process_file(str(inp), str(out))
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "Đà Nẵng" in rec["persona_string"]
