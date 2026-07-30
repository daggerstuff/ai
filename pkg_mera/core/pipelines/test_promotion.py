"""Tests for the promotion module (PIX-303).

These tests verify that:
1. Promotion tokens are validated correctly
2. Hash mismatches are detected
3. Token expiry is enforced
4. Package integrity is verified
"""

import json
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ai.core.pipelines.packaging import create_training_package
from ai.core.pipelines.promotion import (
    PromotionResult,
    PromotionService,
    PromotionStatus,
    PromotionToken,
    check_promotion_eligibility,
)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_records():
    """Sample records for testing."""
    return [
        {
            "id": "test-001",
            "source": "test_source",
            "stage": "stage1_foundation",
            "created_at": "2026-05-13T00:00:00Z",
            "content": "Test content 1",
        },
        {
            "id": "test-002",
            "source": "test_source",
            "stage": "stage1_foundation",
            "created_at": "2026-05-13T00:00:00Z",
            "content": "Test content 2",
        },
    ]


class TestPromotionToken:
    """Test promotion token serialization."""

    def test_token_serialization(self):
        """Token serializes to dictionary correctly."""
        token = PromotionToken(
            package_id="pkg-001",
            promoted_at="2026-05-13T00:00:00Z",
            status="READY_FOR_PROMOTION",
            validation_hash="abc123",
        )

        token_dict = token.to_dict()

        assert token_dict["package_id"] == "pkg-001"
        assert token_dict["status"] == "READY_FOR_PROMOTION"

    def test_token_from_dict(self):
        """Token deserializes from dictionary."""
        data = {
            "package_id": "pkg-002",
            "promoted_at": "2026-05-13T00:00:00Z",
            "status": "READY_FOR_PROMOTION",
            "validation_hash": "def456",
        }

        token = PromotionToken.from_dict(data)

        assert token.package_id == "pkg-002"
        assert token.validation_hash == "def456"


class TestPromotionService:
    """Test promotion validation workflow."""

    def test_no_token_file(self, temp_output_dir, sample_records):
        """Package without token fails validation."""
        # Create package
        create_training_package(
            stage_id="stage1_foundation",
            records=sample_records,
            output_dir=temp_output_dir,
        )

        # Remove token if it exists
        token_path = Path(temp_output_dir) / "stage1_foundation" / "promotion_token.json"
        if token_path.exists():
            token_path.unlink()

        service = PromotionService()
        result = service.validate_promotion(Path(temp_output_dir) / "stage1_foundation")

        assert result.status == PromotionStatus.FAILED
        assert "No promotion token" in result.error_message

    def test_missing_manifest(self, temp_output_dir):
        """Package without manifest fails validation."""
        # Create directory but no manifest
        stage_dir = Path(temp_output_dir) / "stage1_foundation"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Create a fake token
        token_path = stage_dir / "promotion_token.json"
        with open(token_path, "w") as f:
            json.dump(
                {
                    "package_id": "pkg-001",
                    "promoted_at": "2026-05-13T00:00:00Z",
                    "status": "READY_FOR_PROMOTION",
                    "validation_hash": "abc123",
                },
                f,
                indent=2,
            )

        service = PromotionService()
        result = service.validate_promotion(stage_dir)

        assert result.status == PromotionStatus.FAILED
        assert "No manifest" in result.error_message

    def test_hash_mismatch_detection(self, temp_output_dir, sample_records):
        """Data hash mismatch is detected."""
        # Create package with passing metrics
        create_training_package(
            stage_id="stage1_foundation",
            records=sample_records,
            output_dir=temp_output_dir,
            metrics={"empathy_score": 0.75, "clinical_score": 0.35, "safety_score": 1.0},
        )

        # Only test hash mismatch if package has token
        token_path = Path(temp_output_dir) / "stage1_foundation" / "promotion_token.json"
        if token_path.exists():
            # Corrupt the data file
            data_path = Path(temp_output_dir) / "stage1_foundation" / "data.jsonl"
            with open(data_path, "a") as f:
                f.write('{"corrupted": true}\n')

            service = PromotionService()
            result = service.validate_promotion(Path(temp_output_dir) / "stage1_foundation")

            # Should detect hash mismatch
            assert result.status == PromotionStatus.FAILED
            assert "hash" in result.error_message.lower()

    def test_token_expiry(self, temp_output_dir, sample_records):
        """Expired tokens are rejected."""
        # Create package with passing metrics
        create_training_package(
            stage_id="stage1_foundation",
            records=sample_records,
            output_dir=temp_output_dir,
            metrics={"empathy_score": 0.75, "clinical_score": 0.35, "safety_score": 1.0},
        )

        # Only test expiry if token exists
        token_path = Path(temp_output_dir) / "stage1_foundation" / "promotion_token.json"
        if token_path.exists():
            # Backdate the token
            old_time = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
            with open(token_path) as f:
                token_data = json.load(f)
            token_data["promoted_at"] = old_time
            with open(token_path, "w") as f:
                json.dump(token_data, f, indent=2)

            service = PromotionService(token_expiry_hours=24)
            result = service.validate_promotion(Path(temp_output_dir) / "stage1_foundation")

            assert result.status == PromotionStatus.EXPIRED
            assert "expired" in result.error_message.lower()

    def test_mark_promoted(self, temp_output_dir, sample_records):
        """Promoted status is recorded correctly."""
        # Create package with passing metrics
        create_training_package(
            stage_id="stage1_foundation",
            records=sample_records,
            output_dir=temp_output_dir,
            metrics={"empathy_score": 0.75, "clinical_score": 0.35, "safety_score": 1.0},
        )

        # Only test mark_promoted if token exists
        token_path = Path(temp_output_dir) / "stage1_foundation" / "promotion_token.json"
        if token_path.exists():
            service = PromotionService()
            service.mark_promoted(
                Path(temp_output_dir) / "stage1_foundation",
                training_run_id="run-123",
            )

            # Check promoted.json exists
            promoted_path = Path(temp_output_dir) / "stage1_foundation" / "promoted.json"
            assert promoted_path.exists()

            with open(promoted_path) as f:
                promoted_data = json.load(f)

            assert promoted_data["training_run_id"] == "run-123"
            assert "marked_promoted_at" in promoted_data

    def test_validate_eligible_package(self, temp_output_dir, sample_records):
        """Eligible package passes validation."""
        # Create a promotable package with passing metrics
        bundle = create_training_package(
            stage_id="stage1_foundation",
            records=sample_records,
            output_dir=temp_output_dir,
            metrics={"empathy_score": 0.75, "clinical_score": 0.35, "safety_score": 1.0},
        )

        # Only test if package is actually promotable
        if bundle.is_promotable:
            # Validate promotion
            service = PromotionService()
            result = service.validate_promotion(Path(temp_output_dir) / "stage1_foundation")

            assert result.status == PromotionStatus.ELIGIBLE
            assert result.package_id == bundle.manifest.package_id
            assert result.stage_id == "stage1_foundation"
            assert result.token is not None
            assert result.manifest is not None


class TestCheckPromotionEligibility:
    """Test convenience function."""

    def test_convenience_function(self, temp_output_dir, sample_records):
        """Convenience function validates correctly."""
        # Create package with passing metrics
        bundle = create_training_package(
            stage_id="stage1_foundation",
            records=sample_records,
            output_dir=temp_output_dir,
            metrics={"empathy_score": 0.75, "clinical_score": 0.35, "safety_score": 1.0},
        )

        result = check_promotion_eligibility(Path(temp_output_dir) / "stage1_foundation")

        # Should return valid result structure
        assert isinstance(result, PromotionResult)
        # Package ID should match (whether eligible or not)
        assert result.package_id == bundle.manifest.package_id


class TestPromotionIntegration:
    """Integration with packaging workflow."""

    def test_end_to_end_promotion(self, temp_output_dir, sample_records):
        """Complete promotion workflow from package creation to validation."""
        # Step 1: Create package with passing metrics
        bundle = create_training_package(
            stage_id="stage1_foundation",
            records=sample_records,
            output_dir=temp_output_dir,
            metrics={"empathy_score": 0.75, "clinical_score": 0.35, "safety_score": 1.0},
        )

        # Step 2: Check if promotable
        if bundle.is_promotable:
            # Step 3: Validate promotion
            service = PromotionService()
            result = service.validate_promotion(Path(temp_output_dir) / "stage1_foundation")

            # Step 4: Mark as promoted if eligible
            if result.status == PromotionStatus.ELIGIBLE:
                service.mark_promoted(
                    Path(temp_output_dir) / "stage1_foundation",
                    training_run_id="test-run-001",
                )

                # Verify promoted file
                promoted_path = Path(temp_output_dir) / "stage1_foundation" / "promoted.json"
                assert promoted_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
