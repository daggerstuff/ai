"""Fail-closed preparation tests; no network, model weights, or patient data."""

import json
from pathlib import Path

import pytest

from ai.qa.validation.therapy_bench.prepare_glimmer_eval import (
    audit_lineage,
    prepare,
    read_jsonl,
    render_prompt,
    sha256,
)


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    manifest = []
    for split in ("train", "val", "test"):
        row = {
            "messages": [
                {"role": "user", "content": f"{split} question"},
                {"role": "assistant", "content": f"{split} secret reference"},
            ]
        }
        (pilot / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        manifest.append(
            {
                "arc_id": split,
                "session_n": 1,
                "split": split,
                "source_family": "synthetic",
                "source_id": split,
                "difficulty": "moderate",
            }
        )
    (pilot / "manifest_internal.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest),
        encoding="utf-8",
    )
    refresh_identity(pilot)
    (tmp_path / "seed_map.jsonl").write_text(
        "".join(
            json.dumps({"arc_id": split, "key": split, "status": "written"}) + "\n"
            for split in ("train", "val", "test")
        ),
        encoding="utf-8",
    )
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "bos": "BOS",
                "turns": {role: f"<{role}>{{content}}</{role}>" for role in ("system", "user", "assistant")},
                "system_prompt": "Date: {date}",
                "system_prompt_date_seen_during_training": "2026-10-10",
                "generation_prefix": "<assistant>",
                "stop_token": "</assistant>",
            }
        ),
        encoding="utf-8",
    )
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            [
                {
                    "id": "control",
                    "category": "General",
                    "prompt": "control question",
                    "expected_behavior": "hidden scoring instructions",
                }
            ]
        ),
        encoding="utf-8",
    )
    return pilot, template, questions


def refresh_identity(pilot: Path) -> None:
    files = {p.name: {"sha256": sha256(p), "rows": len(read_jsonl(p))} for p in pilot.glob("*.jsonl")}
    (pilot / "identity.json").write_text(json.dumps({"files": files}), encoding="utf-8")


def test_prepare_excludes_answers_and_pins_date(inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    result = prepare(*inputs, output, tmp_path / "seed_map.jsonl")
    cases = read_jsonl(output / "cases.jsonl")
    assert result["suites"] == {"heldout_final_turn": 1, "golden_controls": 1}
    assert cases[0]["reference"] == "test secret reference"
    for case in cases:
        assert "secret reference" not in case["prompt"]
        assert "hidden scoring instructions" not in case["prompt"]
        assert "2026-10-10" in case["prompt"]
    assert result["cases_sha256"] == sha256(output / "cases.jsonl")
    with pytest.raises(FileExistsError):
        prepare(*inputs, output, tmp_path / "seed_map.jsonl")


def test_rejects_changed_pilot(inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    pilot, _, _ = inputs
    with (pilot / "test.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare(*inputs, tmp_path / "evaluation", tmp_path / "seed_map.jsonl")
    assert not (tmp_path / "evaluation").exists()


def test_rejects_seed_leakage(inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    pilot, _, _ = inputs
    path = pilot / "manifest_internal.jsonl"
    manifest = read_jsonl(path)
    manifest[-1]["source_id"] = "train"
    path.write_text("".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8")
    refresh_identity(pilot)
    with pytest.raises(ValueError, match="crosses splits"):
        prepare(*inputs, tmp_path / "evaluation", tmp_path / "seed_map.jsonl")


def test_rejects_assistant_terminated_prompt(inputs: tuple[Path, Path, Path]) -> None:
    _, template_path, _ = inputs
    template = json.loads(template_path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="end with a user"):
        render_prompt([{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}], template)


def test_repeat_preparation_is_deterministic(inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    first = prepare(*inputs, tmp_path / "first", tmp_path / "seed_map.jsonl")
    second = prepare(*inputs, tmp_path / "second", tmp_path / "seed_map.jsonl")
    assert first == second


def test_reused_seed_hidden_by_arc_fallback_is_excluded(inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    seed_map = tmp_path / "seed_map.jsonl"
    rows = read_jsonl(seed_map)
    rows[-1]["key"] = "train:v2"
    seed_map.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="No seed-isolated"):
        prepare(*inputs, tmp_path / "evaluation", seed_map)


def test_ambiguous_seed_is_excluded(inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    pilot, _, _ = inputs
    seed_map = tmp_path / "seed_map.jsonl"
    with seed_map.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"arc_id": "test", "key": "another_seed", "status": "written"}) + "\n")
    provenance = {
        split: [row for row in read_jsonl(pilot / "manifest_internal.jsonl") if row["split"] == split]
        for split in ("train", "val", "test")
    }
    lineage = audit_lineage(provenance, seed_map)
    assert lineage["excluded_test_arcs"] == {"test": "unresolved_or_ambiguous_seed"}
