from __future__ import annotations

from pathlib import Path

from lint_dpo_dataset import lint_file, validate_record


def _valid_record() -> dict:
    return {
        "prompt": "sys",
        "chosen": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "good response"},
        ],
        "rejected": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "bad response"},
        ],
    }


def test_valid_record_passes():
    assert validate_record(_valid_record(), 1) == []


def test_missing_required_key():
    issues = validate_record({"prompt": "x", "chosen": []}, 2)
    assert any(i.field == "rejected" and "missing" in i.message for i in issues)


def test_prompt_not_str():
    rec = _valid_record()
    rec["prompt"] = 1
    issues = validate_record(rec, 1)
    assert any(i.field == "prompt" for i in issues)


def test_invalid_role():
    rec = _valid_record()
    rec["chosen"] = [{"role": "bot", "content": "a"}]
    issues = validate_record(rec, 1)
    assert any("invalid role" in i.message for i in issues)


def test_message_content_not_str():
    rec = _valid_record()
    rec["chosen"] = [{"role": "assistant", "content": 123}]
    issues = validate_record(rec, 1)
    assert any(i.field == "chosen[0]" for i in issues)


def test_prefix_diverges_before_final():
    rec = _valid_record()
    rec["rejected"] = [
        {"role": "user", "content": "DIFFERENT"},
        {"role": "assistant", "content": "bad"},
    ]
    issues = validate_record(rec, 1)
    assert any(i.field == "prefix" for i in issues)


def test_prefix_identical_final():
    rec = _valid_record()
    rec["rejected"] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "good response"},
    ]
    issues = validate_record(rec, 1)
    assert any("must differ" in i.message for i in issues)


def test_unequal_length():
    rec = _valid_record()
    rec["rejected"] = [{"role": "user", "content": "hi"}]
    issues = validate_record(rec, 1)
    assert any("equal length" in i.message for i in issues)


def test_non_assistant_divergent_turn():
    rec = _valid_record()
    rec["rejected"] = [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "bad"},
    ]
    issues = validate_record(rec, 1)
    assert any("assistant message" in i.message for i in issues)


def test_lint_file_ok(tmp_path: Path):
    p = tmp_path / "d.jsonl"
    p.write_text(__import__("json").dumps(_valid_record()) + "\n", encoding="utf-8")
    report = lint_file(p)
    assert report.ok and report.total == 1


def test_lint_file_bad_json(tmp_path: Path):
    p = tmp_path / "d.jsonl"
    p.write_text("{not json}\n", encoding="utf-8")
    report = lint_file(p)
    assert not report.ok and any(i.field == "json" for i in report.issues)


def test_lint_file_mixed(tmp_path: Path):
    import json

    p = tmp_path / "d.jsonl"
    lines = [json.dumps(_valid_record()), "{bad}", json.dumps(_valid_record())]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = lint_file(p)
    assert not report.ok
    assert report.total == 3
    assert any(i.field == "json" for i in report.issues)
