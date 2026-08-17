"""Comprehensive IAA tests — PIX-4344 blueprint §B.4.

Covers all 10 requirements:
1. Fleiss kappa computation (≥0.75 fair, ≥0.85 T1_GOLD)
2. Cohen's kappa pairwise
3. Label Studio JSONL export → IAA data conversion
4. Quality bucketing (5 bands)
5. Rubric XML generation
6. Landis-Koch threshold evaluation
7. Reviewer override tracking (annotation_stage="v3_adjudicated")
8. compute_iaa_from_labels integration
9. curate_pipeline classify_tier with IAA module present (T1_GOLD upgrade)
10. CLI entry point
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from training.annotation.iaa import (
    AnnotationStage,
    AnnotatorLabel,
    FLEISS_KAPPA_GOLD,
    FLEISS_KAPPA_MINIMUM,
    LANDIS_KOCH_THRESHOLDS,
    bucket_quality,
    cohen_kappa_simple,
    compute_iaa_from_labels,
    evaluate_agreement,
    fleiss_kappa,
    generate_label_studio_rubric,
    label_studio_export_to_iaa,
)
from training.curate_pipeline import classify_tier


# ---------------------------------------------------------------------------
# 1. Fleiss kappa
# ---------------------------------------------------------------------------


class TestFleissKappa:
    def test_perfect_agreement(self) -> None:
        """All annotators agree perfectly across multiple categories → kappa = 1.0."""
        n = 3
        N = 3
        k = 3
        # Sample 1: all pick cat 0; sample 2: all pick cat 1; sample 3: all pick cat 2
        n_i = [3, 0, 0, 0, 3, 0, 0, 0, 3]
        result = fleiss_kappa(n, N, k, n_i)
        assert result == pytest.approx(1.0, abs=0.001)

    def test_zero_agreement(self) -> None:
        """Complete disagreement → kappa near 0."""
        n = 3
        N = 3
        k = 3
        # Each sample: 1 annotator per category — no agreement beyond chance
        n_i = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        result = fleiss_kappa(n, N, k, n_i)
        assert result <= 0.01  # kappa ≤ 0 for random

    def test_invalid_inputs(self) -> None:
        assert fleiss_kappa(0, 3, 2, []) == 0.0
        assert fleiss_kappa(3, 0, 2, []) == 0.0

    def test_mismatched_n_i_length(self) -> None:
        with pytest.raises(ValueError, match="n_i length"):
            fleiss_kappa(3, 3, 2, [1, 0])  # n*k=6 but len=2

    def test_fair_threshold(self) -> None:
        """A moderately-agreeing batch should land at or above 0.75."""
        n = 5
        N = 4
        k = 3
        # 4 perfect samples split across categories; 1 sample: 3-1 split
        # Balanced categories keep P_e low so kappa stays high (~0.83).
        n_i = [4, 0, 0, 0, 4, 0, 0, 0, 4, 4, 0, 0, 3, 1, 0]
        result = fleiss_kappa(n, N, k, n_i)
        assert result >= FLEISS_KAPPA_MINIMUM

    def test_gold_threshold(self) -> None:
        """Near-perfect agreement should land at or above 0.85."""
        n = 10
        N = 3
        k = 3
        # 9 perfect samples balanced across 3 categories; 1 sample: 2-1 split.
        # Balanced categories keep P_e low so kappa ≈ 0.91.
        n_i = [3, 0, 0, 0, 3, 0, 0, 0, 3, 3, 0, 0, 0, 3, 0, 0, 0, 3, 3, 0, 0, 0, 3, 0, 0, 0, 3, 2, 1, 0]
        result = fleiss_kappa(n, N, k, n_i)
        assert result >= FLEISS_KAPPA_GOLD


# ---------------------------------------------------------------------------
# 2. Cohen's kappa
# ---------------------------------------------------------------------------


class TestCohenKappa:
    def test_perfect_agreement(self) -> None:
        a = ["yes", "no", "yes", "no", "yes"]
        result = cohen_kappa_simple(a, a[:])
        assert result == pytest.approx(1.0, abs=0.001)

    def test_no_agreement(self) -> None:
        a = ["yes", "no", "yes", "no"]
        b = ["no", "yes", "no", "yes"]
        result = cohen_kappa_simple(a, b)
        assert result <= 0.0

    def test_empty_raters(self) -> None:
        assert cohen_kappa_simple([], []) == 0.0

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same number"):
            cohen_kappa_simple(["a"], ["a", "b"])

    def test_partial_agreement(self) -> None:
        a = ["yes", "yes", "no", "no", "yes"]
        b = ["yes", "no", "no", "no", "yes"]
        result = cohen_kappa_simple(a, b)
        assert 0.0 < result < 1.0


# ---------------------------------------------------------------------------
# 3. Label Studio JSONL export → IAA
# ---------------------------------------------------------------------------


class TestLabelStudioExport:
    def _write_ls_jsonl(self, tmp_path: Path, records: list[dict]) -> Path:
        p = tmp_path / "ls_export.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return p

    def test_basic_conversion(self, tmp_path: Path) -> None:
        records = [
            {
                "id": "sample-1",
                "data": {"text": "I feel sad", "category": "depression"},
                "annotations": [
                    {"annotator_id": "a1", "value": {"quality_score": 0.9, "category": "depression"}},
                    {"annotator_id": "a2", "value": {"quality_score": 0.8, "category": "depression"}},
                    {"annotator_id": "a3", "value": {"quality_score": 0.85, "category": "depression"}},
                ],
            },
            {
                "id": "sample-2",
                "data": {"text": "I feel anxious", "category": "anxiety"},
                "annotations": [
                    {"annotator_id": "a1", "value": {"quality_score": 0.6, "category": "anxiety"}},
                    {"annotator_id": "a2", "value": {"quality_score": 0.5, "category": "anxiety"}},
                ],
            },
        ]
        path = self._write_ls_jsonl(tmp_path, records)
        all_labels, labels_by_sample = label_studio_export_to_iaa(
            str(path), annotator_ids=["a1", "a2", "a3"]
        )
        assert len(all_labels) == 5
        assert len(labels_by_sample) == 2
        assert len(labels_by_sample["sample-1"]) == 3
        assert labels_by_sample["sample-1"][0].quality_score == 0.9

    def test_missing_quality_score_defaults(self, tmp_path: Path) -> None:
        records = [
            {
                "id": "s1",
                "data": {"text": "text"},
                "annotations": [
                    {"annotator_id": "a1", "value": {}},
                ],
            },
        ]
        path = self._write_ls_jsonl(tmp_path, records)
        all_labels, _ = label_studio_export_to_iaa(str(path), annotator_ids=["a1"])
        assert all_labels[0].quality_score == 0.5  # default

    def test_reject_reason_captured(self, tmp_path: Path) -> None:
        records = [
            {
                "id": "s1",
                "data": {"text": "text"},
                "annotations": [
                    {"annotator_id": "a1", "value": {"reject_reason": "off_topic"}},
                ],
            },
        ]
        path = self._write_ls_jsonl(tmp_path, records)
        all_labels, _ = label_studio_export_to_iaa(str(path), annotator_ids=["a1"])
        assert all_labels[0].reject_reason == "off_topic"


# ---------------------------------------------------------------------------
# 4. Quality bucketing (5 bands)
# ---------------------------------------------------------------------------


class TestBucketQuality:
    def test_all_five_bands(self) -> None:
        assert bucket_quality(0.9) == "excellent"
        assert bucket_quality(0.8) == "excellent"
        assert bucket_quality(0.65) == "good"
        assert bucket_quality(0.6) == "good"
        assert bucket_quality(0.5) == "acceptable"
        assert bucket_quality(0.4) == "acceptable"
        assert bucket_quality(0.3) == "marginal"
        assert bucket_quality(0.2) == "marginal"
        assert bucket_quality(0.1) == "poor"
        assert bucket_quality(0.0) == "poor"

    def test_boundary_values(self) -> None:
        # Exact boundaries — lower bound is inclusive
        assert bucket_quality(0.8) == "excellent"
        assert bucket_quality(0.79) == "good"
        assert bucket_quality(0.6) == "good"
        assert bucket_quality(0.59) == "acceptable"
        assert bucket_quality(0.4) == "acceptable"
        assert bucket_quality(0.39) == "marginal"
        assert bucket_quality(0.2) == "marginal"
        assert bucket_quality(0.19) == "poor"


# ---------------------------------------------------------------------------
# 5. Rubric XML generation
# ---------------------------------------------------------------------------


class TestRubricXML:
    def test_basic_rubric(self) -> None:
        xml = generate_label_studio_rubric(
            categories=["depression", "anxiety", "bipolar"],
        )
        assert "<?xml" in xml
        assert "<View>" in xml
        assert "depression" in xml
        assert "anxiety" in xml
        assert "bipolar" in xml
        assert "quality_score" in xml
        assert "domain" in xml
        assert "difficulty" in xml

    def test_custom_description(self) -> None:
        xml = generate_label_studio_rubric(
            categories=["test"],
            description="Custom rubric description",
        )
        assert "Custom rubric description" in xml

    def test_quality_scale(self) -> None:
        xml = generate_label_studio_rubric(
            categories=["test"],
            quality_scale=(0.0, 10.0),
        )
        assert 'min="0.0"' in xml
        assert 'max="10.0"' in xml

    def test_xml_well_formed(self) -> None:
        xml = generate_label_studio_rubric(categories=["a", "b"])
        # Check basic XML structure balance
        assert xml.count("<View>") == xml.count("</View>")
        assert xml.count("<Dropdown") == xml.count("</Dropdown>")
        assert xml.count("<Choices") == xml.count("</Choices>")


# ---------------------------------------------------------------------------
# 6. Landis-Koch threshold evaluation
# ---------------------------------------------------------------------------


class TestEvaluateAgreement:
    def test_gold_standard(self) -> None:
        result = evaluate_agreement(fleiss_kappa=0.90)
        assert result["fleiss_classification"] == "T1_GOLD final release"
        assert "gold standard" in result["fleiss_recommendation"].lower()

    def test_fair_release(self) -> None:
        result = evaluate_agreement(fleiss_kappa=0.78)
        assert result["fleiss_classification"] == "fair quality release"

    def test_retraining_zone(self) -> None:
        result = evaluate_agreement(fleiss_kappa=0.50)
        assert "retraining" in result["fleiss_recommendation"].lower()

    def test_quarantined(self) -> None:
        result = evaluate_agreement(fleiss_kappa=0.20)
        assert "quarant" in result["fleiss_recommendation"].lower()

    def test_with_cohen_kappa(self) -> None:
        result = evaluate_agreement(fleiss_kappa=0.85, cohen_kappa=0.90)
        assert "cohen_classification" in result
        assert result["cohen_classification"] == "T1_GOLD gold standard"

    def test_cohen_retraining_zone(self) -> None:
        result = evaluate_agreement(fleiss_kappa=0.80, cohen_kappa=0.50)
        assert result["cohen_classification"] == "retraining zone"

    def test_cohen_poor(self) -> None:
        result = evaluate_agreement(fleiss_kappa=0.80, cohen_kappa=0.20)
        assert result["cohen_classification"] == "poor"

    def test_landis_koch_thresholds_defined(self) -> None:
        assert "poor" in LANDIS_KOCH_THRESHOLDS
        assert "substantial" in LANDIS_KOCH_THRESHOLDS
        assert "perfect" in LANDIS_KOCH_THRESHOLDS


# ---------------------------------------------------------------------------
# 7. Reviewer override tracking
# ---------------------------------------------------------------------------


class TestReviewerOverrides:
    def test_adjudicated_counted(self) -> None:
        labels = [
            AnnotatorLabel(
                annotator_id="a1",
                sample_id="s1",
                quality_score=0.9,
                metadata={"annotation_stage": AnnotationStage.ADJUDICATED.value},
            ),
            AnnotatorLabel(
                annotator_id="a2",
                sample_id="s1",
                quality_score=0.85,
                metadata={"annotation_stage": AnnotationStage.FINAL.value},
            ),
            AnnotatorLabel(
                annotator_id="a3",
                sample_id="s1",
                quality_score=0.88,
                metadata={"annotation_stage": AnnotationStage.ADJUDICATED.value},
            ),
        ]
        result = compute_iaa_from_labels(labels, num_annotators=3)
        assert result.reviewer_overrides == 2

    def test_no_overrides(self) -> None:
        labels = [
            AnnotatorLabel(
                annotator_id="a1",
                sample_id="s1",
                quality_score=0.9,
                metadata={"annotation_stage": AnnotationStage.INITIAL.value},
            ),
            AnnotatorLabel(
                annotator_id="a2",
                sample_id="s1",
                quality_score=0.85,
                metadata={"annotation_stage": AnnotationStage.INITIAL.value},
            ),
            AnnotatorLabel(
                annotator_id="a3",
                sample_id="s1",
                quality_score=0.88,
                metadata={"annotation_stage": AnnotationStage.INITIAL.value},
            ),
        ]
        result = compute_iaa_from_labels(labels, num_annotators=3)
        assert result.reviewer_overrides == 0

    def test_annotation_stage_enum_values(self) -> None:
        assert AnnotationStage.ADJUDICATED.value == "v3_adjudicated"
        assert AnnotationStage.FINAL.value == "v3_final"
        assert AnnotationStage.INITIAL.value == "v1_initial"
        assert AnnotationStage.SECONDARY.value == "v2_secondary"


# ---------------------------------------------------------------------------
# 8. compute_iaa_from_labels integration
# ---------------------------------------------------------------------------


class TestComputeIAA:
    def _make_labels(
        self, n_samples: int = 5, scores: list[float] | None = None
    ) -> list[AnnotatorLabel]:
        labels = []
        for i in range(n_samples):
            s = scores[i] if scores and i < len(scores) else 0.8
            for a in range(3):
                labels.append(
                    AnnotatorLabel(
                        annotator_id=f"annotator_{a}",
                        sample_id=f"sample_{i}",
                        quality_score=s,
                    )
                )
        return labels

    def test_high_agreement_gold(self) -> None:
        labels = self._make_labels(n_samples=10, scores=[0.9] * 10)
        result = compute_iaa_from_labels(labels, num_annotators=3)
        assert result.num_samples == 10
        assert result.num_annotators == 3
        assert result.fleiss_kappa >= FLEISS_KAPPA_GOLD

    def test_mixed_quality(self) -> None:
        labels = self._make_labels(
            n_samples=6,
            scores=[0.9, 0.1, 0.8, 0.3, 0.85, 0.5],
        )
        result = compute_iaa_from_labels(labels, num_annotators=3)
        # With mixed scores, some samples should be quarantined (kappa < 0.40)
        assert len(result.quarantine_samples) + len(result.retraining_samples) + \
               len(result.gold_standard_samples) == 6

    def test_empty_labels(self) -> None:
        result = compute_iaa_from_labels([], num_annotators=3)
        assert result.num_samples == 0
        assert result.fleiss_kappa == 0.0

    def test_per_sample_kappas(self) -> None:
        labels = self._make_labels(n_samples=3, scores=[0.9, 0.9, 0.9])
        result = compute_iaa_from_labels(labels, num_annotators=3)
        assert len(result.per_sample_kappas) == 3
        # All samples perfect agreement → per-sample kappa should be high
        for v in result.per_sample_kappas.values():
            assert v >= 0.85

    def test_quality_scores_averaged(self) -> None:
        labels = [
            AnnotatorLabel("a1", "s1", quality_score=0.8),
            AnnotatorLabel("a2", "s1", quality_score=0.6),
            AnnotatorLabel("a3", "s1", quality_score=0.7),
        ]
        result = compute_iaa_from_labels(labels, num_annotators=3)
        assert result.quality_scores["s1"] == pytest.approx(0.7, abs=0.01)

    def test_stages_distribution(self) -> None:
        labels = [
            AnnotatorLabel("a1", "s1", quality_score=0.9,
                           metadata={"annotation_stage": "v1_initial"}),
            AnnotatorLabel("a2", "s1", quality_score=0.9,
                           metadata={"annotation_stage": "v2_secondary"}),
            AnnotatorLabel("a3", "s1", quality_score=0.9,
                           metadata={"annotation_stage": "v3_final"}),
        ]
        result = compute_iaa_from_labels(labels, num_annotators=3)
        assert result.stages_distribution["v1_initial"] == 1
        assert result.stages_distribution["v2_secondary"] == 1
        assert result.stages_distribution["v3_final"] == 1

    def test_reject_reasons_tallied(self) -> None:
        labels = [
            AnnotatorLabel("a1", "s1", quality_score=0.1, reject_reason="off_topic"),
            AnnotatorLabel("a2", "s1", quality_score=0.1, reject_reason="off_topic"),
            AnnotatorLabel("a3", "s1", quality_score=0.1, reject_reason="low_quality"),
        ]
        result = compute_iaa_from_labels(labels, num_annotators=3)
        assert result.reject_reasons.get("off_topic") == 2
        assert result.reject_reasons.get("low_quality") == 1


# ---------------------------------------------------------------------------
# 9. curate_pipeline classify_tier with IAA module present
# ---------------------------------------------------------------------------


class TestClassifyTierWithIAA:
    """When the IAA module is importable, classify_tier should upgrade
    adjudicated records with strong Fleiss kappa to T1_GOLD."""

    def test_iaa_upgrade_to_gold(self) -> None:
        """Adjudicated record with fleiss_kappa >= 0.85 → T1_GOLD."""
        record: dict[str, Any] = {
            "task_type": "therapy_response_generation",
            "source": "annomi",
            "messages": [{"role": "user", "content": str(i)} for i in range(5)],
            "annotation_stage": AnnotationStage.ADJUDICATED.value,
            "fleiss_kappa": 0.90,
        }
        assert classify_tier(record) == "T1_GOLD"

    def test_iaa_no_upgrade_below_threshold(self) -> None:
        """Adjudicated but kappa < 0.85 → NOT upgraded, falls to T2_SILVER."""
        record: dict[str, Any] = {
            "task_type": "therapy_response_generation",
            "source": "annomi",
            "messages": [{"role": "user", "content": str(i)} for i in range(5)],
            "annotation_stage": AnnotationStage.ADJUDICATED.value,
            "fleiss_kappa": 0.70,
        }
        assert classify_tier(record) == "T2_SILVER"

    def test_iaa_no_upgrade_wrong_stage(self) -> None:
        """Non-adjudicated record with high kappa → NOT upgraded."""
        record: dict[str, Any] = {
            "task_type": "therapy_response_generation",
            "source": "annomi",
            "messages": [{"role": "user", "content": str(i)} for i in range(5)],
            "annotation_stage": AnnotationStage.FINAL.value,
            "fleiss_kappa": 0.95,
        }
        assert classify_tier(record) == "T2_SILVER"

    def test_iaa_no_upgrade_missing_kappa(self) -> None:
        """Adjudicated but no fleiss_kappa field → NOT upgraded."""
        record: dict[str, Any] = {
            "task_type": "therapy_response_generation",
            "source": "annomi",
            "messages": [{"role": "user", "content": str(i)} for i in range(5)],
            "annotation_stage": AnnotationStage.ADJUDICATED.value,
        }
        assert classify_tier(record) == "T2_SILVER"

    def test_iaa_safety_takes_priority(self) -> None:
        """Adversarial safety records are always T4_SAFETY regardless of IAA."""
        record: dict[str, Any] = {
            "task_type": "adversarial_safety",
            "source": "",
            "messages": [],
            "annotation_stage": AnnotationStage.ADJUDICATED.value,
            "fleiss_kappa": 0.95,
        }
        assert classify_tier(record) == "T4_SAFETY"

    def test_iaa_clinical_reviewed_still_gold(self) -> None:
        """Clinical_reviewed=True still takes priority over everything."""
        record: dict[str, Any] = {
            "task_type": "",
            "source": "",
            "messages": [],
            "clinical_reviewed": True,
            "annotation_stage": AnnotationStage.ADJUDICATED.value,
            "fleiss_kappa": 0.90,
        }
        assert classify_tier(record) == "T1_GOLD"

    def test_iaa_exact_threshold(self) -> None:
        """kappa exactly 0.85 should upgrade (>= comparison)."""
        record: dict[str, Any] = {
            "task_type": "",
            "source": "annomi",
            "messages": [{"role": "user", "content": str(i)} for i in range(5)],
            "annotation_stage": AnnotationStage.ADJUDICATED.value,
            "fleiss_kappa": 0.85,
        }
        assert classify_tier(record) == "T1_GOLD"


# ---------------------------------------------------------------------------
# 10. CLI entry point
# ---------------------------------------------------------------------------


class TestCLI:
    def _write_ls_jsonl(self, tmp_path: Path, records: list[dict]) -> Path:
        p = tmp_path / "ls_export.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return p

    def test_cli_basic_run(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        records = [
            {
                "id": f"s{i}",
                "data": {"text": "text", "category": "dep"},
                "annotations": [
                    {"annotator_id": "a1", "value": {"quality_score": 0.9, "category": "dep"}},
                    {"annotator_id": "a2", "value": {"quality_score": 0.85, "category": "dep"}},
                    {"annotator_id": "a3", "value": {"quality_score": 0.88, "category": "dep"}},
                ],
            }
            for i in range(5)
        ]
        ls_path = self._write_ls_jsonl(tmp_path, records)
        output_path = tmp_path / "iaa_result.json"

        from training.annotation.iaa import main

        import sys
        old_argv = sys.argv
        sys.argv = [
            "iaa",
            "--ls-jsonl", str(ls_path),
            "--output", str(output_path),
            "--num-annotators", "3",
        ]
        try:
            ret = main()
        finally:
            sys.argv = old_argv

        assert ret == 0
        assert output_path.exists()
        result = json.loads(output_path.read_text())
        assert "fleiss_kappa" in result
        assert "classification" in result
        assert "num_samples" in result
        captured = capsys.readouterr()
        assert "Fleiss kappa:" in captured.out

    def test_cli_with_rubric(self, tmp_path: Path) -> None:
        records = [
            {
                "id": "s1",
                "data": {"text": "text", "category": "dep"},
                "annotations": [
                    {"annotator_id": "a1", "value": {"quality_score": 0.9}},
                    {"annotator_id": "a2", "value": {"quality_score": 0.85}},
                    {"annotator_id": "a3", "value": {"quality_score": 0.88}},
                ],
            },
        ]
        ls_path = self._write_ls_jsonl(tmp_path, records)
        output_path = tmp_path / "iaa_result.json"
        rubric_path = tmp_path / "rubric.xml"

        from training.annotation.iaa import main

        import sys
        old_argv = sys.argv
        sys.argv = [
            "iaa",
            "--ls-jsonl", str(ls_path),
            "--output", str(output_path),
            "--num-annotators", "3",
            "--categories", "depression", "anxiety", "bipolar",
            "--rubric-xml", str(rubric_path),
        ]
        try:
            ret = main()
        finally:
            sys.argv = old_argv

        assert ret == 0
        assert rubric_path.exists()
        rubric_content = rubric_path.read_text()
        assert "depression" in rubric_content
        assert "anxiety" in rubric_content
