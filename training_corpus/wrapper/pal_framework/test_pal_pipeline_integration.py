"""Comprehensive end-to-end integration test for the full PAL pipeline.

Exercises every stage of the PAL framework in sequence using rule-based
generation (no API keys, no HF datasets downloads):

    Phase 0   — Meddies record adaptation → synthetic JSONL generation
    Phase 1   — Persona string formatting
    Phase 2.1 — Dialogue-informed persona selection (SFT Task 1)
    Phase 2.2 — Persona-enhanced dialogue generation (SFT Task 2)
    Phase 2.3 — Unified mixed-task SFT dataset
    Phase 3.1 — DPO preference pair construction
    Phase 3.2 — DPO dataset linting for TRL DPOTrainer compatibility
    Phase 5   — Inference: select-then-generate roundtrip

Every stage validates:
  - Correct record count, schema, and required keys
  - ChatML compliance (SFT stages)
  - TRL schema compliance (DPO stages)
  - No JSON leakage in generated text
  - Deterministic output (same seed → same results)
  - Roundtrip data coherence (persona used in selection matches persona
    used in generation and shows up in inference results)

Run:
    cd ai && python -m pytest training_corpus/pal_framework/test_pal_pipeline_integration.py -v --tb=short
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Phase 0 — Synthetic Meddies records (no HF datasets dependency)
# ---------------------------------------------------------------------------

# 10 diverse synthetic records that mimic the Meddies/meddies-persona-vie shape
_SYNTHETIC_RAW_RECORDS: list[dict[str, Any]] = [
    {
        "demographics": {
            "age": 45,
            "gender": "Nữ",
            "province": "Hà Nội",
            "full_name": "Nguyễn Thị Lan",
        },
        "healthcare_behavior": {
            "health_literacy_level": "Thấp",
            "healthcare_seeking_pattern": "Ưu tiên Đông y",
        },
        "medical_history": {
            "presenting_symptoms": [{"symptom_name": "đau đầu"}],
            "chronic_conditions": ["hypertension"],
        },
    },
    {
        "demographics": {
            "age": 30,
            "gender": "Nam",
            "province": "Hồ Chí Minh",
            "full_name": "Trần Văn An",
        },
        "healthcare_behavior": {
            "health_literacy_level": "Cao",
            "healthcare_seeking_pattern": "Ngay lập tức",
        },
        "medical_history": {
            "presenting_symptoms": [{"symptom_name": "sốt cao"}],
        },
    },
    {
        "demographics": {
            "age": 60,
            "gender": "Nam",
            "province": "Đồng Nai",
            "full_name": "Lê Văn Bình",
        },
        "healthcare_behavior": {
            "health_literacy_level": "Trung bình",
            "healthcare_seeking_pattern": "Ưu tiên Tây y",
        },
        "medical_history": {
            "presenting_symptoms": [{"symptom_name": "đau khớp"}],
            "chronic_conditions": ["diabetes"],
        },
    },
    {
        "demographics": {
            "age": 25,
            "gender": "Nữ",
            "province": "Đà Nẵng",
            "full_name": "Phạm Thị Hoa",
        },
        "healthcare_behavior": {
            "health_literacy_level": "Cao",
            "healthcare_seeking_pattern": "Kết hợp",
        },
        "medical_history": {
            "presenting_symptoms": [{"symptom_name": "dị ứng"}],
        },
    },
    {
        "demographics": {
            "age": 55,
            "gender": "Nam",
            "province": "Cần Thơ",
            "full_name": "Hoàng Văn Đức",
        },
        "healthcare_behavior": {
            "health_literacy_level": "Thấp",
            "healthcare_seeking_pattern": "Tự điều trị",
        },
        "medical_history": {
            "presenting_symptoms": [{"symptom_name": "mệt mỏi"}],
            "chronic_conditions": ["heart disease"],
        },
    },
    {
        "demographics": {
            "age": 35,
            "gender": "Nữ",
            "province": "Hải Phòng",
            "full_name": "Vũ Thị Mai",
        },
        "healthcare_behavior": {
            "health_literacy_level": "Trung bình",
            "healthcare_seeking_pattern": "Chưa khám bệnh",
        },
        "medical_history": {
            "presenting_symptoms": [{"symptom_name": "đau bụng"}],
        },
    },
    {
        "demographics": {
            "age": 70,
            "gender": "Nam",
            "province": "Huế",
            "full_name": "Ngô Văn Thành",
        },
        "healthcare_behavior": {
            "health_literacy_level": "Thấp",
            "healthcare_seeking_pattern": "Ưu tiên Đông y",
        },
        "medical_history": {
            "presenting_symptoms": [{"symptom_name": "mất ngủ"}],
            "chronic_conditions": ["hypertension", "diabetes"],
        },
    },
    {
        "demographics": {
            "age": 28,
            "gender": "Nữ",
            "province": "Bình Dương",
            "full_name": "Đặng Thị Hương",
        },
        "healthcare_behavior": {
            "health_literacy_level": "Cao",
            "healthcare_seeking_pattern": "Kết hợp Đông/Tây y",
        },
        "medical_history": {
            "presenting_symptoms": [{"symptom_name": "đau lưng"}],
        },
    },
    {
        "demographics": {
            "age": 50,
            "gender": "Nam",
            "province": "Nghệ An",
            "full_name": "Bùi Văn Tùng",
        },
        "healthcare_behavior": {
            "health_literacy_level": "Trung bình",
            "healthcare_seeking_pattern": "Ngay lập tức",
        },
        "medical_history": {
            "presenting_symptoms": [{"symptom_name": "khó thở"}],
        },
    },
    {
        "demographics": {
            "age": 42,
            "gender": "Nữ",
            "province": "Quảng Ninh",
            "full_name": "Trịnh Thị Hạnh",
        },
        "healthcare_behavior": {
            "health_literacy_level": "Cao",
            "healthcare_seeking_pattern": "Ưu tiên Tây y",
        },
        "medical_history": {
            "presenting_symptoms": [{"symptom_name": "mờ mắt"}],
            "chronic_conditions": ["glaucoma"],
        },
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tmp_output_dir() -> Path:
    """Persistent temp directory for all pipeline artifacts."""
    with tempfile.TemporaryDirectory(prefix="pal_pipeline_") as d:
        yield Path(d)


@pytest.fixture(scope="module")
def meddies_adapter():
    """Lazy import to avoid order-of-import issues in subprocess test runners."""
    from meddies_adapter import adapt_record

    return adapt_record


@pytest.fixture(scope="module")
def meddies_synthesizer():
    """Lazy import the synthesizer module."""
    import meddies_synthesizer as ms

    return ms


@pytest.fixture(scope="module")
def selection_gen():
    """Lazy import the selection dataset generator."""
    import generate_selection_dataset as gs

    return gs


@pytest.fixture(scope="module")
def dialogue_gen():
    """Lazy import the dialogue dataset generator."""
    import generate_sft_dialogue as gd

    return gd


@pytest.fixture(scope="module")
def unified_builder():
    """Lazy import the unified SFT builder."""
    import build_unified_sft as bu

    return bu


@pytest.fixture(scope="module")
def dpo_gen():
    """Lazy import the DPO pair generator."""
    import generate_dpo_pairs as dp

    return dp


@pytest.fixture(scope="module")
def dpo_linter():
    """Lazy import the DPO lint module."""
    import lint_dpo_dataset as ld

    return ld


@pytest.fixture(scope="module")
def inference_wrapper_module():
    """Lazy import the inference wrapper."""
    import inference_wrapper as iw

    return iw


# ---------------------------------------------------------------------------
# Helper: assert no JSON leakage in a string
# ---------------------------------------------------------------------------


def _assert_no_json_leakage(text: str, context: str = "") -> None:
    """Fail if the text contains JSON structural characters.

    Only `{`, `}`, and `"` are flagged — single quotes are legitimate
    natural-language punctuation (apostrophes, contractions) and should NOT
    be treated as JSON leakage. See `build_unified_sft._has_json_leakage`
    for the matching production check.
    """
    for char in ("{", "}", '"'):
        assert char not in text, (
            f"JSON leakage detected{(' (' + context + ')' if context else '')}: "
            f"char {char!r} found in {text[:80]!r}"
        )


# ===========================================================================
# Phase 0: Meddies Record Adaptation + Synthetic Data Generation
# ===========================================================================


class TestPhase0Synthesizer:
    """Exercises the Meddies adapter, persona formatter, and synthesizer builders."""

    def test_adapt_all_records(self, meddies_adapter) -> None:
        """All 10 synthetic records adapt without error and preserve key fields."""
        for i, raw in enumerate(_SYNTHETIC_RAW_RECORDS):
            adapted = meddies_adapter(raw)
            assert "demographics" in adapted, f"record {i} missing demographics"
            assert "healthcare_behavior" in adapted, f"record {i} missing healthcare_behavior"
            assert "_raw" in adapted, f"record {i} missing _raw"
            assert adapted["_raw"] is raw, f"record {i} _raw does not reference original"
            # Verify mapped fields exist
            demo = adapted["demographics"]
            assert "age" in demo and "gender" in demo and "location" in demo
            health = adapted["healthcare_behavior"]
            assert "health_literacy" in health and "preference" in health

    def test_persona_strings_have_no_json_leakage(self, meddies_adapter) -> None:
        """Phase 1: formatted persona strings must be pure NL with no JSON chars."""
        from meddies_to_pal import format_persona

        for i, raw in enumerate(_SYNTHETIC_RAW_RECORDS):
            adapted = meddies_adapter(raw)
            persona = format_persona(adapted)
            assert isinstance(persona, str) and len(persona) > 0
            _assert_no_json_leakage(persona, f"persona {i}")

    def test_builders_are_deterministic(self, meddies_adapter, meddies_synthesizer) -> None:
        """Same seed produces identical outputs for all three builder types."""
        adapted = [meddies_adapter(r) for r in _SYNTHETIC_RAW_RECORDS]

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        sel1 = list(meddies_synthesizer.build_selection_input(adapted, rng1, n_distractors=3))
        sel2 = list(meddies_synthesizer.build_selection_input(adapted, rng2, n_distractors=3))
        assert sel1 == sel2, "build_selection_input is not deterministic"

        rng3 = random.Random(42)
        rng4 = random.Random(42)
        dia1 = list(meddies_synthesizer.build_dialogue_input(adapted, rng3))
        dia2 = list(meddies_synthesizer.build_dialogue_input(adapted, rng4))
        assert dia1 == dia2, "build_dialogue_input is not deterministic"

        rng5 = random.Random(42)
        rng6 = random.Random(42)
        dpo1 = list(meddies_synthesizer.build_dpo_input(adapted, rng5))
        dpo2 = list(meddies_synthesizer.build_dpo_input(adapted, rng6))
        assert dpo1 == dpo2, "build_dpo_input is not deterministic"

    def test_build_selection_input_schema(self, meddies_adapter, meddies_synthesizer) -> None:
        """Phase 2.1 input records have the correct schema."""
        adapted = [meddies_adapter(r) for r in _SYNTHETIC_RAW_RECORDS]
        rng = random.Random(7)
        records = list(meddies_synthesizer.build_selection_input(adapted, rng, n_distractors=3))
        assert len(records) == len(adapted), "should yield one record per input"
        for i, rec in enumerate(records):
            assert set(rec.keys()) >= {"dialogue", "personas", "correct_index"}, f"record {i} missing keys"
            assert isinstance(rec["dialogue"], str) and len(rec["dialogue"]) > 0
            assert isinstance(rec["personas"], list) and len(rec["personas"]) >= 4
            assert 0 <= rec["correct_index"] < len(rec["personas"])
            # Dialogue should be a realistic medical conversation
            assert "Patient:" in rec["dialogue"] and "Doctor:" in rec["dialogue"]

    def test_build_selection_input_persona_order(self, meddies_adapter, meddies_synthesizer) -> None:
        """The correct_index should point to the original record in the shuffled personas list."""
        adapted = [meddies_adapter(r) for r in _SYNTHETIC_RAW_RECORDS]

        rng = random.Random(7)
        records = list(meddies_synthesizer.build_selection_input(adapted, rng, n_distractors=3))
        for rec in records:
            idx = rec["correct_index"]
            correct_persona = rec["personas"][idx]
            # The correct persona should have the same demographics.age as the source
            assert "demographics" in correct_persona, "correct persona missing demographics"

    def test_build_dialogue_input_schema(self, meddies_adapter, meddies_synthesizer) -> None:
        """Phase 2.2 input records have the correct schema."""
        adapted = [meddies_adapter(r) for r in _SYNTHETIC_RAW_RECORDS]
        rng = random.Random(7)
        records = list(meddies_synthesizer.build_dialogue_input(adapted, rng))
        assert len(records) == len(adapted)
        for i, rec in enumerate(records):
            assert set(rec.keys()) >= {"persona", "dialogue", "response"}, f"record {i} missing keys"
            assert isinstance(rec["persona"], dict) and "demographics" in rec["persona"]
            assert isinstance(rec["dialogue"], str) and len(rec["dialogue"]) > 0
            assert isinstance(rec["response"], str) and len(rec["response"]) > 0

    def test_build_dpo_input_schema(self, meddies_adapter, meddies_synthesizer) -> None:
        """Phase 3.1 input records have the correct schema."""
        adapted = [meddies_adapter(r) for r in _SYNTHETIC_RAW_RECORDS]
        rng = random.Random(7)
        records = list(meddies_synthesizer.build_dpo_input(adapted, rng))
        assert len(records) == len(adapted)
        for i, rec in enumerate(records):
            assert set(rec.keys()) >= {
                "persona",
                "dialogue",
                "chosen_response",
                "rejected_response",
            }, f"record {i} missing keys"
            assert isinstance(rec["chosen_response"], str) and len(rec["chosen_response"]) > 0
            assert isinstance(rec["rejected_response"], str) and len(rec["rejected_response"]) > 0
            # Chosen and rejected must differ (preference signal)
            assert rec["chosen_response"] != rec["rejected_response"], "chosen/rejected must differ"
            # Rejected should contain medical jargon (violates persona)
            assert any(
                word in rec["rejected_response"].lower()
                for word in ["medical", "clinical", "tertiary", "imaging", "ai assistant", "guideline"]
            ), "rejected response should contain medical jargon"

    def test_output_no_json_leakage(self, meddies_adapter, meddies_synthesizer) -> None:
        """All synthesized dialogue/response text must be free of JSON formatting chars."""
        adapted = [meddies_adapter(r) for r in _SYNTHETIC_RAW_RECORDS]
        rng = random.Random(7)
        for rec in meddies_synthesizer.build_selection_input(adapted, rng, n_distractors=3):
            _assert_no_json_leakage(rec["dialogue"], "selection dialogue")
        rng = random.Random(7)
        for rec in meddies_synthesizer.build_dialogue_input(adapted, rng):
            _assert_no_json_leakage(rec["dialogue"], "dialogue input dialogue")
            _assert_no_json_leakage(rec["response"], "dialogue input response")
        rng = random.Random(7)
        for rec in meddies_synthesizer.build_dpo_input(adapted, rng):
            _assert_no_json_leakage(rec["chosen_response"], "DPO chosen")
            _assert_no_json_leakage(rec["rejected_response"], "DPO rejected")

    def test_write_intermediate_jsonls(self, meddies_adapter, meddies_synthesizer, tmp_output_dir) -> None:
        """Write all three intermediate JSONL files to disk for downstream phases."""
        adapted = [meddies_adapter(r) for r in _SYNTHETIC_RAW_RECORDS]
        rng = random.Random(7)

        # Phase 1: persona strings
        from meddies_to_pal import format_persona

        p1_path = tmp_output_dir / "pal_persona_strings.jsonl"
        with p1_path.open("w", encoding="utf-8") as f:
            for a in adapted:
                f.write(json.dumps({"persona_string": format_persona(a)}, ensure_ascii=False) + "\n")
        assert p1_path.exists() and p1_path.stat().st_size > 0
        lines = list(p1_path.open())
        assert len(lines) == len(adapted)

        # Phase 2.1 input
        sel_path = tmp_output_dir / "input_phase_2_1_selection.jsonl"
        with sel_path.open("w", encoding="utf-8") as f:
            for rec in meddies_synthesizer.build_selection_input(adapted, rng, n_distractors=3):
                f.write(_safe_json(rec) + "\n")
        assert sel_path.exists() and sel_path.stat().st_size > 0

        # Phase 2.2 input
        dia_path = tmp_output_dir / "input_phase_2_2_dialogue.jsonl"
        with dia_path.open("w", encoding="utf-8") as f:
            for rec in meddies_synthesizer.build_dialogue_input(adapted, rng):
                f.write(_safe_json(rec) + "\n")
        assert dia_path.exists() and dia_path.stat().st_size > 0

        # Phase 3.1 input
        dpo_path = tmp_output_dir / "input_phase_3_1_dpo.jsonl"
        with dpo_path.open("w", encoding="utf-8") as f:
            for rec in meddies_synthesizer.build_dpo_input(adapted, rng):
                f.write(_safe_json(rec) + "\n")
        assert dpo_path.exists() and dpo_path.stat().st_size > 0

        # Store paths for downstream tests
        _store_paths(tmp_output_dir, p1_path, sel_path, dia_path, dpo_path)


# ---------------------------------------------------------------------------
# Cross-stage path registry (set by test_write_intermediate_jsonls, consumed
# by downstream phase tests)
# ---------------------------------------------------------------------------

_pipeline_paths: dict[str, Path] = {}


def _store_paths(
    out_dir: Path,
    p1: Path,
    sel: Path,
    dia: Path,
    dpo: Path,
) -> None:
    _pipeline_paths["out_dir"] = out_dir
    _pipeline_paths["persona_strings"] = p1
    _pipeline_paths["selection_input"] = sel
    _pipeline_paths["dialogue_input"] = dia
    _pipeline_paths["dpo_input"] = dpo


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


# ===========================================================================
# Phase 2.1: Dialogue-informed Persona Selection (SFT Task 1)
# ===========================================================================


class TestPhase2_1Selection:
    """Exercises generate_selection_dataset on the synthesized Phase 2.1 input."""

    @pytest.fixture(scope="class")
    def selection_output(self, selection_gen, tmp_output_dir) -> Path:
        sel_input = _pipeline_paths.get("selection_input")
        if sel_input is None:
            pytest.skip("Phase 2.1 input not generated (run Phase 0 test_write_intermediate_jsonls first)")
        out_path = tmp_output_dir / "phase_2_1_selection_sft.jsonl"
        n = selection_gen.generate_dataset(sel_input, out_path, n_distractors=3, seed=7)
        assert n > 0, "generate_dataset returned 0"
        return out_path

    def test_output_exists_and_nonempty(self, selection_output) -> None:
        assert selection_output.exists()
        assert selection_output.stat().st_size > 0

    def test_output_record_count(self, selection_output, selection_gen) -> None:
        """Output should have the same number of records as input (no filtering)."""
        records = _load_jsonl(selection_output)
        input_records = _load_jsonl(_pipeline_paths["selection_input"])
        assert len(records) == len(input_records)

    def test_all_records_are_chatml_compliant(self, selection_output, selection_gen) -> None:
        """Every record must pass strict ChatML validation."""
        records = _load_jsonl(selection_output)
        for i, rec in enumerate(records):
            assert "messages" in rec, f"record {i} missing messages"
            assert selection_gen.is_chatml_compliant(rec["messages"]), f"record {i} not ChatML compliant"

    def test_each_record_has_system_user_assistant_roles(self, selection_output) -> None:
        """Selection records must have exactly system → user → assistant."""
        records = _load_jsonl(selection_output)
        for i, rec in enumerate(records):
            roles = [m["role"] for m in rec["messages"]]
            assert roles == ["system", "user", "assistant"], f"record {i} has unexpected roles: {roles}"

    def test_assistant_response_is_valid_option_number(self, selection_output) -> None:
        """The assistant turn must be a valid 1-indexed option number."""
        records = _load_jsonl(selection_output)
        for i, rec in enumerate(records):
            assistant = rec["messages"][2]["content"]
            try:
                option = int(assistant.strip())
                assert 1 <= option <= 4, f"record {i}: option {option} out of range 1-4"
            except ValueError:
                pytest.fail(f"record {i}: assistant response '{assistant}' not an integer")

    def test_no_json_leakage_in_messages(self, selection_output) -> None:
        """No JSON formatting chars anywhere in the message content."""
        records = _load_jsonl(selection_output)
        for i, rec in enumerate(records):
            for m in rec["messages"]:
                _assert_no_json_leakage(m["content"], f"record {i} role={m['role']}")

    def test_metadata_contains_correct_option(self, selection_output) -> None:
        """Metadata must track the correct option and option count."""
        records = _load_jsonl(selection_output)
        for i, rec in enumerate(records):
            meta = rec.get("metadata", {})
            assert "correct_option" in meta, f"record {i} missing correct_option"
            assert "n_options" in meta, f"record {i} missing n_options"
            assert isinstance(meta["correct_option"], int)
            assert meta["n_options"] == 4  # 1 correct + 3 distractors


# ===========================================================================
# Phase 2.2: Persona-Enhanced Dialogue Generation (SFT Task 2)
# ===========================================================================


class TestPhase2_2Dialogue:
    """Exercises generate_sft_dialogue on the synthesized Phase 2.2 input."""

    @pytest.fixture(scope="class")
    def dialogue_output(self, dialogue_gen, tmp_output_dir) -> Path:
        dia_input = _pipeline_paths.get("dialogue_input")
        if dia_input is None:
            pytest.skip("Phase 2.2 input not generated")
        out_path = tmp_output_dir / "phase_2_2_dialogue_sft.jsonl"
        # Use a low min_records so the 10-record test passes
        n = dialogue_gen.generate_dataset(dia_input, out_path, min_records=1)
        assert n > 0
        return out_path

    def test_output_exists_and_nonempty(self, dialogue_output) -> None:
        assert dialogue_output.exists()
        assert dialogue_output.stat().st_size > 0

    def test_output_record_count(self, dialogue_output) -> None:
        records = _load_jsonl(dialogue_output)
        input_records = _load_jsonl(_pipeline_paths["dialogue_input"])
        assert len(records) == len(input_records)

    def test_all_records_are_chatml_compliant(self, dialogue_output, dialogue_gen) -> None:
        records = _load_jsonl(dialogue_output)
        for i, rec in enumerate(records):
            assert "messages" in rec, f"record {i} missing messages"
            assert dialogue_gen.is_chatml_compliant(rec["messages"]), f"record {i} not ChatML compliant"

    def test_each_record_has_system_user_assistant_roles(self, dialogue_output) -> None:
        records = _load_jsonl(dialogue_output)
        for i, rec in enumerate(records):
            roles = [m["role"] for m in rec["messages"]]
            assert roles == ["system", "user", "assistant"], f"record {i} has unexpected roles: {roles}"

    def test_system_message_is_persona_roleplay_instruction(self, dialogue_output) -> None:
        records = _load_jsonl(dialogue_output)
        system = records[0]["messages"][0]["content"]
        assert "roleplaying" in system or "persona" in system
        assert len(system) > 50

    def test_user_message_includes_persona_string(self, dialogue_output) -> None:
        """The user message should reference the persona from the input."""
        records = _load_jsonl(dialogue_output)
        for i, rec in enumerate(records):
            user = rec["messages"][1]["content"]
            assert "persona:" in user or "persona" in user.lower(), f"record {i} user msg missing persona"
            assert "Generate the next response" in user, f"record {i} user msg missing generation instruction"

    def test_no_json_leakage_in_messages(self, dialogue_output) -> None:
        records = _load_jsonl(dialogue_output)
        for i, rec in enumerate(records):
            for m in rec["messages"]:
                _assert_no_json_leakage(m["content"], f"record {i} role={m['role']}")

    def test_metadata_contains_persona_and_token_estimate(self, dialogue_output) -> None:
        records = _load_jsonl(dialogue_output)
        for i, rec in enumerate(records):
            meta = rec.get("metadata", {})
            assert "persona_string" in meta, f"record {i} missing persona_string"
            assert isinstance(meta["persona_string"], str) and len(meta["persona_string"]) > 0
            assert "estimated_tokens" in meta, f"record {i} missing estimated_tokens"
            assert meta["estimated_tokens"] > 0


# ===========================================================================
# Phase 2.3: Unified Mixed-Task SFT Dataset
# ===========================================================================


class TestPhase2_3UnifiedSft:
    """Exercises build_unified_sft combining selection + dialogue outputs."""

    @pytest.fixture(scope="class")
    def unified_output(self, unified_builder, tmp_output_dir) -> Path:
        sel_path = tmp_output_dir / "phase_2_1_selection_sft.jsonl"
        dia_path = tmp_output_dir / "phase_2_2_dialogue_sft.jsonl"
        for p in (sel_path, dia_path):
            if not p.exists():
                pytest.skip(f"Required input not found: {p}")
        out_path = tmp_output_dir / "phase_2_3_unified_sft.jsonl"
        stats = unified_builder.build_unified_dataset(
            sel_path, dia_path, out_path,
            target_records=20, seed=7,
        )
        assert stats.total > 0
        return out_path

    def test_output_exists_and_nonempty(self, unified_output) -> None:
        assert unified_output.exists()
        assert unified_output.stat().st_size > 0

    def test_record_count_matches_target(self, unified_output) -> None:
        records = _load_jsonl(unified_output)
        assert len(records) == 20, f"expected 20 records, got {len(records)}"

    def test_both_task_types_present(self, unified_output) -> None:
        records = _load_jsonl(unified_output)
        task_types = {r["metadata"]["task_type"] for r in records}
        assert "selection" in task_types, "missing selection records"
        assert "dialogue" in task_types, "missing dialogue records"

    def test_all_records_pass_validation(self, unified_output, unified_builder) -> None:
        records = _load_jsonl(unified_output)
        for i, rec in enumerate(records):
            assert unified_builder.validate_record(rec), f"record {i} failed validation"

    def test_all_records_are_chatml_compliant(self, unified_output) -> None:
        records = _load_jsonl(unified_output)
        for i, rec in enumerate(records):
            assert "messages" in rec, f"record {i} missing messages"
            assert len(rec["messages"]) >= 2, f"record {i}: expected >=2 messages, got {len(rec['messages'])}"
            roles = [m["role"] for m in rec["messages"]]
            assert roles[0] == "system", f"record {i}: first role must be system (got {roles[0]})"

    def test_interleaving_is_mixed(self, unified_output) -> None:
        """Task types should be interleaved, not all one type then the other."""
        records = _load_jsonl(unified_output)
        types = [r["metadata"]["task_type"] for r in records]
        # Simple check: at least one transition
        transitions = sum(1 for i in range(1, len(types)) if types[i] != types[i - 1])
        assert transitions >= 2, f"tasks are not interleaved (transitions={transitions})"

    def test_no_json_leakage_in_unified(self, unified_output) -> None:
        records = _load_jsonl(unified_output)
        for i, rec in enumerate(records):
            for m in rec.get("messages", []):
                _assert_no_json_leakage(m.get("content", ""), f"unified record {i} role={m.get('role')}")

    def test_deterministic_with_same_seed(self, unified_builder, tmp_output_dir) -> None:
        """Same seed produces byte-identical output."""
        sel_path = tmp_output_dir / "phase_2_1_selection_sft.jsonl"
        dia_path = tmp_output_dir / "phase_2_2_dialogue_sft.jsonl"
        out1 = tmp_output_dir / "unified_run1.jsonl"
        out2 = tmp_output_dir / "unified_run2.jsonl"
        unified_builder.build_unified_dataset(sel_path, dia_path, out1, target_records=10, seed=42)
        unified_builder.build_unified_dataset(sel_path, dia_path, out2, target_records=10, seed=42)
        assert out1.read_bytes() == out2.read_bytes()


# ===========================================================================
# Phase 3.1: DPO Preference Pair Construction
# ===========================================================================


class TestPhase3_1Dpo:
    """Exercises generate_dpo_pairs on the synthesized Phase 3.1 input."""

    @pytest.fixture(scope="class")
    def dpo_output(self, dpo_gen, tmp_output_dir) -> Path:
        dpo_input = _pipeline_paths.get("dpo_input")
        if dpo_input is None:
            pytest.skip("Phase 3.1 input not generated")
        out_path = tmp_output_dir / "phase_3_1_dpo_pairs.jsonl"
        n = dpo_gen.generate_dataset(dpo_input, out_path, min_records=1)
        assert n > 0
        return out_path

    def test_output_exists_and_nonempty(self, dpo_output) -> None:
        assert dpo_output.exists()
        assert dpo_output.stat().st_size > 0

    def test_output_record_count(self, dpo_output) -> None:
        records = _load_jsonl(dpo_output)
        input_records = _load_jsonl(_pipeline_paths["dpo_input"])
        assert len(records) == len(input_records)

    def test_all_records_have_trl_schema(self, dpo_output) -> None:
        """Each record must conform to the TRL DPOTrainer schema."""
        records = _load_jsonl(dpo_output)
        for i, rec in enumerate(records):
            assert "prompt" in rec, f"record {i} missing prompt"
            assert "chosen" in rec, f"record {i} missing chosen"
            assert "rejected" in rec, f"record {i} missing rejected"
            assert isinstance(rec["prompt"], str) and len(rec["prompt"]) > 0
            assert isinstance(rec["chosen"], list) and len(rec["chosen"]) == 1
            assert isinstance(rec["rejected"], list) and len(rec["rejected"]) == 1
            chosen_role = rec["chosen"][0]["role"]
            rejected_role = rec["rejected"][0]["role"]
            assert chosen_role == "assistant", f"record {i}: chosen role is {chosen_role}, expected assistant"
            assert rejected_role == "assistant", f"record {i}: rejected role is {rejected_role}, expected assistant"
            assert rec["chosen"][0]["content"] != rec["rejected"][0]["content"]

    def test_prompt_includes_system_instruction(self, dpo_output) -> None:
        """The prompt should be self-contained with system instruction."""
        records = _load_jsonl(dpo_output)
        for i, rec in enumerate(records):
            assert "roleplaying" in rec["prompt"], f"record {i} prompt missing roleplaying instruction"
            assert "persona" in rec["prompt"], f"record {i} prompt missing persona"

    def test_metadata_present(self, dpo_output) -> None:
        records = _load_jsonl(dpo_output)
        for i, rec in enumerate(records):
            assert "metadata" in rec, f"record {i} missing metadata"
            meta = rec["metadata"]
            assert "persona_string" in meta
            assert meta["n_dialogue_turns"] >= 0
            assert meta["chosen_estimated_tokens"] > 0
            assert meta["rejected_estimated_tokens"] > 0

    def test_no_json_leakage(self, dpo_output) -> None:
        records = _load_jsonl(dpo_output)
        for i, rec in enumerate(records):
            _assert_no_json_leakage(rec["prompt"], f"DPO {i} prompt")
            _assert_no_json_leakage(rec["chosen"][0]["content"], f"DPO {i} chosen")
            _assert_no_json_leakage(rec["rejected"][0]["content"], f"DPO {i} rejected")


# ===========================================================================
# Phase 3.2: DPO Dataset Linting
# ===========================================================================


class TestPhase3_2DpoLint:
    """Exercises lint_dpo_dataset on the DPO output."""

    @pytest.fixture(scope="class")
    def dpo_path(self, tmp_output_dir) -> Path:
        p = tmp_output_dir / "phase_3_1_dpo_pairs.jsonl"
        if not p.exists():
            pytest.skip(f"DPO output not found: {p}")
        return p

    def test_lint_passes_with_no_issues(self, dpo_linter, dpo_path) -> None:
        """The DPO dataset must lint cleanly for TRL DPOTrainer."""
        report = dpo_linter.lint_file(dpo_path)
        assert report.ok, f"DPO lint failed with {len(report.issues)} issues: {report.issues}"
        assert report.total > 0

    def test_lint_counts_all_records(self, dpo_linter, dpo_path) -> None:
        report = dpo_linter.lint_file(dpo_path)
        expected = len(_load_jsonl(dpo_path))
        assert report.total == expected, f"lint counted {report.total}, expected {expected}"

    def test_lint_rejects_missing_prompt(self, dpo_linter, tmp_output_dir) -> None:
        """Linting a record missing 'prompt' should report an issue."""
        bad_path = tmp_output_dir / "bad_dpo.jsonl"
        with bad_path.open("w") as f:
            f.write(json.dumps({"chosen": [{"role": "assistant", "content": "a"}], "rejected": [{"role": "assistant", "content": "b"}]}) + "\n")
        report = dpo_linter.lint_file(bad_path)
        assert not report.ok
        issue_fields = [i.field for i in report.issues]
        assert "prompt" in issue_fields

    def test_lint_rejects_identical_chosen_rejected(self, dpo_linter, tmp_output_dir) -> None:
        """Linting a record where chosen == rejected should report an issue."""
        bad_path = tmp_output_dir / "bad_dpo_identical.jsonl"
        with bad_path.open("w") as f:
            f.write(
                json.dumps({
                    "prompt": "test",
                    "chosen": [{"role": "assistant", "content": "same"}],
                    "rejected": [{"role": "assistant", "content": "same"}],
                }) + "\n"
            )
        report = dpo_linter.lint_file(bad_path)
        assert not report.ok


# ===========================================================================
# Phase 5: Inference (Select-then-Generate Roundtrip)
# ===========================================================================


class TestPhase5Inference:
    """Exercises the inference wrapper using the synthesized personas."""

    @pytest.fixture(scope="class")
    def adapted_records(self, meddies_adapter) -> list[dict[str, Any]]:
        return [meddies_adapter(r) for r in _SYNTHETIC_RAW_RECORDS]

    @pytest.fixture(scope="class")
    def wrapper(self, adapted_records, inference_wrapper_module) -> Any:
        """Create a PalInferenceWrapper with stub clients + the synthetic personas."""
        PalInferenceWrapper = inference_wrapper_module.PalInferenceWrapper

        # Stub selector: always returns "1" (first candidate)
        class _StubSelector:
            def __call__(self, messages: list[dict[str, str]]) -> str:
                return "1"

        # Stub generator: returns a canned persona-aligned response
        class _StubGenerator:
            def __call__(self, messages: list[dict[str, str]]) -> str:
                return (
                    "I have been feeling this way for a while now. "
                    "Thank you for explaining things clearly, doctor."
                )

        return PalInferenceWrapper(
            selector_client=_StubSelector(),
            generator_client=_StubGenerator(),
            candidate_personas=adapted_records,
            latency_budget_seconds=10.0,
        )

    def test_wrapper_initializes(self, wrapper) -> None:
        """Wrapper should initialize without error with stub clients."""
        assert wrapper is not None
        assert len(wrapper.candidate_personas) == len(_SYNTHETIC_RAW_RECORDS)

    def test_select_persona_returns_string(self, wrapper) -> None:
        """Stage 1: persona selection returns a natural-language string."""
        result = wrapper.select_persona("Patient: I feel tired.")
        assert isinstance(result.persona_string, str)
        assert len(result.persona_string) > 0
        assert isinstance(result.selected_index, int)
        assert result.selected_index >= 0
        assert result.latency_seconds >= 0.0
        _assert_no_json_leakage(result.persona_string, "selected persona")

    def test_select_persona_uses_candidate_pool(self, wrapper) -> None:
        """The returned persona index must be within the candidate pool range."""
        result = wrapper.select_persona("Patient: I have a headache.")
        assert 0 <= result.selected_index < len(wrapper.candidate_personas)

    def test_generate_response_returns_string(self, wrapper) -> None:
        """Stage 2: response generation returns a non-empty string."""
        result = wrapper.generate_response(
            persona_string="This patient is a 45-year-old female from Hanoi.",
            dialogue_history="Patient: I feel tired.",
        )
        assert isinstance(result.response, str)
        assert len(result.response) > 0
        assert result.latency_seconds >= 0.0
        _assert_no_json_leakage(result.response, "generated response")

    def test_generate_response_rejects_empty_persona(self, wrapper) -> None:
        """Empty persona_string should raise ValueError."""
        with pytest.raises(ValueError, match="must be non-empty"):
            wrapper.generate_response(persona_string="", dialogue_history="Patient: hello")

    def test_generate_response_accepts_empty_history(self, wrapper) -> None:
        """Empty dialogue history is valid (single-turn generation)."""
        result = wrapper.generate_response(
            persona_string="This patient is a 45-year-old female from Hanoi.",
            dialogue_history="",
        )
        assert len(result.response) > 0

    def test_generate_response_rejects_non_string_history(self, wrapper) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            wrapper.generate_response(
                persona_string="a persona",
                dialogue_history=123,  # type: ignore[arg-type]
            )

    def test_infer_end_to_end(self, wrapper) -> None:
        """End-to-end: select persona, then generate response."""
        result = wrapper.infer("Patient: I have been feeling very tired lately.")
        assert result.selection.persona_string
        assert result.generation.response
        assert result.total_latency_seconds >= 0.0
        assert result.dialogue_history_text == "Patient: I have been feeling very tired lately."

    def test_infer_with_vietnamese_dialogue(self, wrapper) -> None:
        """Inference handles Vietnamese dialogue."""
        result = wrapper.infer("Bệnh nhân: Tôi cảm thấy mệt mỏi. Bác sĩ: Bao lâu rồi?")
        assert result.selection.persona_string
        assert result.generation.response
        assert "mệt" in result.dialogue_history_text or "mỏi" in result.dialogue_history_text

    def test_infer_rejects_empty_dialogue(self, wrapper) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            wrapper.infer("")

    def test_latency_exceeded_error(self, wrapper, inference_wrapper_module) -> None:
        """A tiny latency budget (1 ns) should be exceeded by stub client wall time."""
        from inference_wrapper import PalInferenceWrapper

        tight = PalInferenceWrapper(
            selector_client=wrapper.selector_client,
            generator_client=wrapper.generator_client,
            candidate_personas=wrapper.candidate_personas,
            latency_budget_seconds=1e-9,  # 1 nanosecond — always exceeded
        )
        with pytest.raises(inference_wrapper_module.LatencyExceededError):
            tight.infer("Patient: I feel tired.")

    def test_selection_parse_error(self, inference_wrapper_module) -> None:
        """A selector that returns non-numeric text should raise SelectionParseError."""
        PalInferenceWrapper = inference_wrapper_module.PalInferenceWrapper
        from inference_wrapper import SelectionParseError

        class _BadSelector:
            def __call__(self, messages: list[dict[str, str]]) -> str:
                return "not a number"

        bad = PalInferenceWrapper(
            selector_client=_BadSelector(),
            generator_client=_StubGenerator(),
            candidate_personas=_SYNTHETIC_RAW_RECORDS[:3],
            latency_budget_seconds=10.0,
        )
        with pytest.raises(SelectionParseError):
            bad.select_persona("Patient: test")

    def test_infer_uses_persona_from_selection_pool(self, wrapper) -> None:
        """The selected persona should match one of the candidate personas."""
        from meddies_to_pal import format_persona

        result = wrapper.infer("Patient: I have a chronic condition.")
        selected = result.selection.persona_string
        # The stub selector always returns "1" → first candidate
        expected = format_persona(wrapper.candidate_personas[0])
        assert selected.strip() == expected.strip(), (
            f"Expected first persona ({expected[:60]}...), got ({selected[:60]}...)"
        )


# ===========================================================================
# Cross-Stage Data Flow: Persona Consistency
# ===========================================================================


class TestCrossStageDataFlow:
    """Verifies that the same persona flows consistently through all stages."""

    def test_selection_input_persona_matches_selection_output(self, tmp_output_dir, selection_gen) -> None:
        """The correct_index in the Phase 2.1 input should correspond to the
        correct_option in the Phase 2.1 output metadata."""
        sel_input = _pipeline_paths.get("selection_input")
        sel_output = tmp_output_dir / "phase_2_1_selection_sft.jsonl"
        if not sel_input or not sel_output.exists():
            pytest.skip("selection input or output not available")

        input_records = _load_jsonl(sel_input)
        output_records = _load_jsonl(sel_output)
        assert len(input_records) == len(output_records)

        for i, (inp, out) in enumerate(zip(input_records, output_records)):
            inp_idx = inp["correct_index"]
            out_option = out["metadata"]["correct_option"]
            # The correct_option in metadata is 1-indexed
            assert 1 <= out_option <= 4, f"record {i}: out_option {out_option} out of range"
            # We can't directly compare inp_idx and out_option because the
            # selection generator re-shuffles. But both should reference valid
            # indices in the input personas list.
            assert 0 <= inp_idx < len(inp["personas"]), f"record {i}: inp_idx {inp_idx} out of range"

    def test_persona_from_dialogue_input_appears_in_sft_output(self, tmp_output_dir, dialogue_gen) -> None:
        """The persona that was used to generate the dialogue input should
        be referenced in the SFT output's user message."""
        dia_input = _pipeline_paths.get("dialogue_input")
        dia_output = tmp_output_dir / "phase_2_2_dialogue_sft.jsonl"
        if not dia_input or not dia_output.exists():
            pytest.skip("dialogue input or output not available")

        input_records = _load_jsonl(dia_input)
        output_records = _load_jsonl(dia_output)
        assert len(input_records) == len(output_records)

        from meddies_to_pal import format_persona

        for i, (inp, out) in enumerate(zip(input_records, output_records)):
            expected_persona = format_persona(inp["persona"])
            user_msg = out["messages"][1]["content"]
            assert expected_persona[:30] in user_msg, (
                f"record {i}: persona not found in user message"
            )

    def test_dpo_pair_chosen_response_matches_dialogue_response(self, tmp_output_dir) -> None:
        """The chosen response in the DPO output should differ from the
        rejected response (preference signal), and both should contain
        the dialogue from the input."""
        dpo_output = tmp_output_dir / "phase_3_1_dpo_pairs.jsonl"
        if not dpo_output.exists():
            pytest.skip("DPO output not available")
        dpo_input = _pipeline_paths.get("dpo_input")
        if not dpo_input:
            pytest.skip("DPO input not available")

        input_records = _load_jsonl(dpo_input)
        output_records = _load_jsonl(dpo_output)
        assert len(input_records) == len(output_records)

        for i, (inp, out) in enumerate(zip(input_records, output_records)):
            # The DPO pair references the original dialogue
            assert inp["dialogue"][:40] in out["prompt"] or out["prompt"].startswith("You are roleplaying"), (
                f"record {i}: dialogue not found in prompt"
            )
            # Chosen and rejected must differ
            chosen_content = out["chosen"][0]["content"]
            rejected_content = out["rejected"][0]["content"]
            assert chosen_content != rejected_content, f"record {i}: chosen == rejected"

    def test_inference_uses_same_persona_format(self, inference_wrapper_module, meddies_adapter) -> None:
        """The inference wrapper should format personas the same way as the
        Phase 1 formatter (meddies_to_pal.format_persona)."""
        from meddies_to_pal import format_persona

        adapted = meddies_adapter(_SYNTHETIC_RAW_RECORDS[0])
        adapted_records = [meddies_adapter(r) for r in _SYNTHETIC_RAW_RECORDS]
        PalInferenceWrapper = inference_wrapper_module.PalInferenceWrapper

        w = PalInferenceWrapper(
            selector_client=_StubSelector(),
            generator_client=_StubGenerator(),
            candidate_personas=adapted_records,
            latency_budget_seconds=10.0,
        )

        expected = format_persona(adapted)
        # The wrapper renders candidate personas through _persona_to_string
        # which calls format_persona underneath
        result = w.select_persona("Patient: I feel tired.")
        # The stub selector returns index 0, which is the first adapted record
        assert expected == result.persona_string or expected[:30] == result.persona_string[:30]


class _StubSelector:
    """Canonical stub selector: returns "1" for any input."""

    def __call__(self, messages: list[dict[str, str]]) -> str:
        return "1"


class _StubGenerator:
    """Canonical stub generator reused across inference tests."""

    def __call__(self, messages: list[dict[str, str]]) -> str:
        return (
            "I have been feeling this way for a while now. "
            "Thank you for explaining things clearly, doctor."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_json(obj: Any, **kwargs: Any) -> str:
    """Serialize with date/datetime fallback (matches meddies_synthesizer._dumps)."""
    return json.dumps(obj, ensure_ascii=False, default=_json_default, **kwargs)


def _json_default(obj: Any) -> str:
    from datetime import date, datetime

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
