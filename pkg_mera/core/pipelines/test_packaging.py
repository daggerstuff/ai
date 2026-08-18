"""Tests for the training-ready packaging module (PIX-303).

These tests verify that:
1. Package bundles are created with correct structure
2. Manifest reflects readiness validation results
3. Promotion tokens are only created when gates pass
4. Stage thresholds are correctly applied
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from ai.pkg_mera.core.pipelines.packaging import (
    DatasetManifest,
    DatasetPackager,
    PackageBundle,
    create_training_package,
)
from ai.pkg_mera.core.pipelines.training_readiness_gates import STAGE_QUALITY_THRESHOLDS


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for packages."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_records():
    """Sample records for testing."""
    return [
        {
            "source": "test_source",
            "stage": "stage1_foundation",
            "created_at": "2026-05-13T00:00:00Z",
            "content": "Test content 1",
            "empathy_score": 0.75,
            "clinical_score": 0.35,
            "safety_score": 1.0,
        },
        {
            "source": "test_source",
            "stage": "stage1_foundation",
            "created_at": "2026-05-13T00:00:00Z",
            "content": "Test content 2",
            "empathy_score": 0.72,
            "clinical_score": 0.38,
            "safety_score": 1.0,
        },
    ]


@pytest.fixture
def gate_audit():
    """Sample gate audit from privacy/content gates."""
    return {
        "gate0_content_classification": "pass",
        "gate1_pii_treatment": "pass",
        "gate2_content_safety": "pass",
        "gate3_license_consent": "pass",
        "gate4_human_review": "pass",
    }


class TestDatasetManifest:
    """Test manifest creation and serialization."""

    def test_manifest_from_readiness_result(self, sample_records):
        """Manifest correctly reflects readiness validation."""
        packager = DatasetPackager()
        readiness_result = packager.readiness_gates.validate_package(
            package_id="test-pkg",
            stage_id="stage1_foundation",
            records=sample_records,
            metrics={"empathy": 0.75, "clinical": 0.35, "safety": 1.0},
        )

        manifest = DatasetManifest.from_readiness_result(
            readiness_result=readiness_result,
            stage_thresholds=STAGE_QUALITY_THRESHOLDS["stage1_foundation"],
        )

        assert manifest.name == "stage1_foundation-slice-v1"
        assert manifest.stage == "stage1_foundation"
        assert manifest.record_count == 2
        assert manifest.promotion_status in ["READY", "BLOCKED", "NOT_READY"]

    def test_manifest_serialization(self, sample_records):
        """Manifest serializes to dictionary correctly."""
        packager = DatasetPackager()
        readiness_result = packager.readiness_gates.validate_package(
            package_id="test-pkg",
            stage_id="stage1_foundation",
            records=sample_records,
        )

        manifest = DatasetManifest.from_readiness_result(
            readiness_result=readiness_result,
            stage_thresholds=STAGE_QUALITY_THRESHOLDS["stage1_foundation"],
        )

        manifest_dict = manifest.to_dict()

        assert "name" in manifest_dict
        assert "stage" in manifest_dict
        assert "validation_gates" in manifest_dict
        assert "promotion_status" in manifest_dict


class TestDatasetPackager:
    """Test package creation workflow."""

    def test_create_package_structure(self, temp_output_dir, sample_records, gate_audit):
        """Package creates correct file structure."""
        packager = DatasetPackager(output_dir=temp_output_dir)
        bundle = packager.create_package(
            stage_id="stage1_foundation",
            records=sample_records,
            gate_audit=gate_audit,
        )

        # Verify file structure
        assert bundle.data_path.exists()
        assert bundle.metrics_path.exists()
        assert bundle.readiness_path.exists()
        assert (
            Path(temp_output_dir)
            / "stage1_foundation"
            / bundle.manifest.package_id
            / "manifest.json"
        ).exists()

    def test_promotion_token_only_when_ready(self, temp_output_dir, sample_records, gate_audit):
        """Promotion token only created when all gates pass."""
        packager = DatasetPackager(output_dir=temp_output_dir)
        bundle = packager.create_package(
            stage_id="stage1_foundation",
            records=sample_records,
            gate_audit=gate_audit,
        )

        # Check promotion status matches token existence
        if bundle.manifest.promotion_status == "READY":
            assert bundle.promotion_token_path is not None
            assert bundle.promotion_token_path.exists()
        else:
            # If not ready, token should be None
            assert bundle.promotion_token_path is None

    def test_data_hash_computed(self, temp_output_dir, sample_records):
        """Data hash is computed and stored in manifest."""
        packager = DatasetPackager(output_dir=temp_output_dir)
        bundle = packager.create_package(
            stage_id="stage1_foundation",
            records=sample_records,
        )

        assert bundle.manifest.data_hash != ""
        assert len(bundle.manifest.data_hash) == 64  # SHA256 hex length

    def test_stage_thresholds_applied(self, temp_output_dir, sample_records):
        """Stage-specific thresholds are correctly applied."""
        packager = DatasetPackager(output_dir=temp_output_dir)

        for stage_id, expected_thresholds in STAGE_QUALITY_THRESHOLDS.items():
            bundle = packager.create_package(
                stage_id=stage_id,
                records=sample_records,
            )

            assert bundle.manifest.stage_thresholds == expected_thresholds

    def test_validation_gates_recorded(self, temp_output_dir, sample_records):
        """All validation gates are recorded in manifest."""
        packager = DatasetPackager(output_dir=temp_output_dir)
        bundle = packager.create_package(
            stage_id="stage1_foundation",
            records=sample_records,
        )

        expected_gates = [
            "completeness",
            "quality_floors",
            "dedup_retention",
            "privacy_compliance",
            "slice_boundaries",
        ]

        for gate in expected_gates:
            assert gate in bundle.manifest.validation_gates
            assert bundle.manifest.validation_gates[gate] in ["PASS", "FAIL"]


class TestCreateTrainingPackage:
    """Test convenience function."""

    def test_convenience_function(self, temp_output_dir, sample_records):
        """Convenience function creates package correctly."""
        bundle = create_training_package(
            stage_id="stage1_foundation",
            records=sample_records,
            output_dir=temp_output_dir,
        )

        assert isinstance(bundle, PackageBundle)
        assert bundle.manifest.stage == "stage1_foundation"


class TestPackageIntegration:
    """Integration tests with other pipeline components."""

    def test_integration_with_readiness_gates(self, temp_output_dir, sample_records):
        """Package correctly integrates with PIX-506 readiness gates."""
        packager = DatasetPackager(output_dir=temp_output_dir)
        bundle = packager.create_package(
            stage_id="stage1_foundation",
            records=sample_records,
        )

        # Verify readiness result is embedded in manifest
        assert bundle.manifest.readiness_result is not None
        assert "gate_results" in bundle.manifest.readiness_result
        assert "can_promote" in bundle.manifest.readiness_result

    def test_metrics_persistence(self, temp_output_dir, sample_records):
        """Metrics are correctly persisted and reloadable."""
        packager = DatasetPackager(output_dir=temp_output_dir)
        bundle = packager.create_package(
            stage_id="stage1_foundation",
            records=sample_records,
        )

        # Reload metrics from disk
        with open(bundle.metrics_path) as f:
            reloaded_metrics = json.load(f)

        # Should match manifest metrics
        assert reloaded_metrics == bundle.manifest.actual_metrics


class TestEdgeCases:
    """Edge case handling."""

    def test_stale_promotion_token_removed(self, temp_output_dir, gate_audit):
        """Stale promotion token from a previous run is removed when package is not promotable."""
        stage_dir = Path(temp_output_dir) / "stage1_foundation"
        # Simulate a previous promotable run by pre-creating a token
        stage_dir.mkdir(parents=True, exist_ok=True)
        old_token_dir = stage_dir / "pkg-oldtoken"
        old_token_dir.mkdir(parents=True, exist_ok=True)
        stale_token = old_token_dir / "promotion_token.json"
        stale_token.write_text('{"status": "STALE"}')

        packager = DatasetPackager(output_dir=temp_output_dir)
        # Create a package with empty records (will be BLOCKED, not promotable)
        bundle = packager.create_package(
            stage_id="stage1_foundation",
            records=[],
            gate_audit=gate_audit,
            package_id="pkg-oldtoken",
        )

        # Stale token should be removed since this run is not promotable
        assert not stale_token.exists()
        assert bundle.promotion_token_path is None

    def test_directory_collision_avoided(self, temp_output_dir, sample_records, gate_audit):
        """Multiple packages for same stage get distinct directories."""
        packager = DatasetPackager(output_dir=temp_output_dir)

        bundle1 = packager.create_package(
            stage_id="stage1_foundation",
            records=sample_records,
            gate_audit=gate_audit,
        )
        bundle2 = packager.create_package(
            stage_id="stage1_foundation",
            records=sample_records,
            gate_audit=gate_audit,
        )

        # Different package IDs → different directories
        assert bundle1.manifest.package_id != bundle2.manifest.package_id
        assert bundle1.data_path != bundle2.data_path
        assert bundle1.data_path.exists()
        assert bundle2.data_path.exists()

    def test_empty_records(self, temp_output_dir):
        """Empty record list handled gracefully."""
        packager = DatasetPackager(output_dir=temp_output_dir)
        bundle = packager.create_package(
            stage_id="stage1_foundation",
            records=[],
        )

        assert bundle.manifest.record_count == 0
        assert bundle.manifest.promotion_status == "BLOCKED"

    def test_unknown_stage_defaults_to_supplementary(self, temp_output_dir, sample_records):
        """Unknown stage IDs use supplementary thresholds."""
        packager = DatasetPackager(output_dir=temp_output_dir)
        bundle = packager.create_package(
            stage_id="unknown_stage",
            records=sample_records,
        )

        # Should use supplementary thresholds
        assert bundle.manifest.stage_thresholds == STAGE_QUALITY_THRESHOLDS["supplementary"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
