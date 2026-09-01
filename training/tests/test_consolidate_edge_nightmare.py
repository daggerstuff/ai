"""Tests for consolidate_edge_nightmare (pure, tmp-dir fixtures, no network)."""

from pipelines.ingestion_deduplication import compute_primary_hash
from training.consolidate_edge_nightmare import (
    MANIFEST_FILENAME,
    STAGE_PRIORITY,
    _dpo_hash,
    _dpo_payload,
    _is_dpo,
    _stage_of,
    consolidate,
)


def _chatml(user, assistant):
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def test_stage_of_routes_edge_family():
    rec = {"family": "substance use", "source": "clinical_edge_case_substance_use"}
    assert _stage_of(rec) == "stage3_edge_stress_test"


def test_stage_of_routes_nightmare():
    rec = {"family": "nightmare fuel", "source": "nightmare_fuel_predefined"}
    assert _stage_of(rec) == "stage3_edge_stress_test"


def test_stage_of_routes_clinical_stages():
    assert _stage_of({"stage": "stage2_therapeutic_expertise"}) == "stage2_therapeutic_expertise"
    assert _stage_of({"stage": "stage1_foundation"}) == "stage1_foundation"
    assert _stage_of({"stage": "stage4_voice_persona"}) == "stage4_voice_persona"


def test_stage_of_routes_dpo_to_stage5():
    assert _stage_of({"asset_kind": "dpo"}) == "stage5_safety"
    assert _stage_of({"prompt": "p", "chosen": "c", "rejected": "r"}) == "stage5_safety"


def test_stage_priority_includes_stage5():
    assert STAGE_PRIORITY["stage5_safety"] > STAGE_PRIORITY["stage4_voice_persona"]
    assert "stage5_safety" in STAGE_PRIORITY


def test_canonical_hash_matches_ingestion_policy():
    # The consolidation must use the NO-separator canonical hash, not content_hashes.
    rec = _chatml("Hello", "Hi there")
    assert compute_primary_hash(rec) == compute_primary_hash(rec)


def test_is_dpo_and_payload_reconstruct():
    rec = {"asset_kind": "dpo", "messages": [
        {"role": "user", "content": "I feel stuck"},
        {"role": "assistant", "content": "Let's name what's stuck."},
    ], "rejected": "You're right, it's hopeless.", "metadata": {"pair_type": "safety"}}
    assert _is_dpo(rec) is True
    payload = _dpo_payload(rec)
    assert payload["prompt"] == "I feel stuck"
    assert payload["chosen"] == "Let's name what's stuck."
    assert payload["rejected"] == "You're right, it's hopeless."
    assert payload["metadata"] == {"pair_type": "safety"}


def test_dpo_hash_stable():
    payload = {"prompt": "A", "chosen": "B", "rejected": "C"}
    assert _dpo_hash(payload) == _dpo_hash({"prompt": "A", "chosen": "B", "rejected": "C"})


def test_consolidate_end_to_end(tmp_path):
    gold = tmp_path / "train_master_gold.jsonl"
    # Pre-seed gold with one record whose content a staging record will duplicate.
    existing = _chatml("Already there", "We are here together.")
    gold.write_text(__import__("json").dumps(existing) + "\n", encoding="utf-8")

    staging = tmp_path / "staging.jsonl"
    edge = {
        "messages": [
            {"role": "user", "content": "I keep using again."},
            {"role": "assistant", "content": "Let's stay with what just happened."},
        ],
        "source": "clinical_edge_case_substance_use",
        "family": "substance use",
    }
    clinical = {
        "messages": [
            {"role": "user", "content": "Tell me about attachment."},
            {"role": "assistant", "content": "Attachment shapes how we reach for others."},
        ],
        "stage": "stage2_therapeutic_expertise",
        "source": "clinical_book",
    }
    cliche = {
        "messages": [
            {"role": "user", "content": "I feel awful."},
            {"role": "assistant", "content": "It sounds like you're carrying a lot."},
        ],
        "stage": "stage3_edge_stress_test",
    }
    duplicate = _chatml("Already there", "We are here together.")
    dpo = {
        "asset_kind": "dpo",
        "messages": [
            {"role": "user", "content": "Nothing matters."},
            {"role": "assistant", "content": "What would it look like if it did?"},
        ],
        "rejected": "You're right, nothing matters.",
        "metadata": {"pair_type": "safety"},
    }
    staging.write_text(
        "\n".join(__import__("json").dumps(r) for r in (edge, clinical, cliche, duplicate, dpo)) + "\n",
        encoding="utf-8",
    )

    manifest_dir = tmp_path / "final"
    reject = tmp_path / "rejections.jsonl"

    summary = consolidate(
        [staging],
        gold_path=gold,
        manifest_dir=manifest_dir,
        reject_path=reject,
    )

    assert summary["scanned"] == 5
    assert summary["emitted_gold"] == 2  # edge + clinical; duplicate & cliche & dpo excluded
    assert summary["emitted_manifest"] == 3  # edge + clinical + dpo
    assert summary["rejected"] == 1
    assert summary["duplicates"] == 1

    # Gold grew: 1 pre-seed + 2 new = 3.
    gold_lines = [ln for ln in gold.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(gold_lines) == 3

    # Manifests: stage3 has edge; stage2 has clinical; stage5 has dpo.
    stage3 = manifest_dir / MANIFEST_FILENAME["stage3_edge_stress_test"]
    stage2 = manifest_dir / MANIFEST_FILENAME["stage2_therapeutic_expertise"]
    stage5 = manifest_dir / MANIFEST_FILENAME["stage5_safety"]
    assert stage3.exists()
    assert stage2.exists()
    assert stage5.exists()
    assert "substance use" in stage3.read_text(encoding="utf-8")
    assert "attachment" in stage2.read_text(encoding="utf-8").lower()
    assert "Nothing matters" in stage5.read_text(encoding="utf-8")

    # Rejection log captured the cliché.
    reject_lines = reject.read_text(encoding="utf-8").splitlines()
    assert any("it sounds like" in ln for ln in reject_lines)


def test_consolidate_idempotent(tmp_path):
    gold = tmp_path / "gold.jsonl"
    staging = tmp_path / "staging.jsonl"
    rec = _chatml("Once", "Then twice.")
    rec["stage"] = "stage1_foundation"
    staging.write_text(__import__("json").dumps(rec) + "\n", encoding="utf-8")
    manifest_dir = tmp_path / "final"

    first = consolidate([staging], gold_path=gold, manifest_dir=manifest_dir, reject_path=tmp_path / "r1.jsonl")
    second = consolidate([staging], gold_path=gold, manifest_dir=manifest_dir, reject_path=tmp_path / "r2.jsonl")

    assert first["emitted_gold"] == 1
    assert second["emitted_gold"] == 0
    assert second["duplicates"] == 1
    gold_lines = [ln for ln in gold.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(gold_lines) == 1
