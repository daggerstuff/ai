"""End-to-end test for VAL-CROSS-002: SDG borderline routing to annotation queue."""

from __future__ import annotations

from unittest.mock import Mock, patch

from training.sdg_pipeline import _validate_clinical_validity


def test_val_cross_002_borderline_routing_to_annotation_queue():
    """Test that borderline samples from SDG pipeline appear in annotation queue.

    This test validates VAL-CROSS-002 assertion.
    """
    # Mock the annotation queue API response
    mock_response = Mock()
    mock_response.status_code = 201  # HTTP_OK + 1
    mock_response.json.return_value = {"id": 123, "status": "pending"}

    with patch("requests.post", return_value=mock_response) as mock_post:
        # Create a sample with borderline score
        sample = {
            "instruction": "Test instruction about CBT techniques",
            "output": "Let's explore your thought patterns using cognitive restructuring."
        }

        # Mock the ClinicalValidityJudge.score to return borderline score (0.55)
        with patch("training.sdg_pipeline.ClinicalValidityJudge.score", return_value=0.55):
            # Mock the scorer to return annotation_needed classification
            with patch("training.sdg_pipeline.ClinicalValidityScorer.classify_score",
                      return_value="annotation_needed"):
                # Call _validate_clinical_validity with min_clinical_validity > 0
                result = _validate_clinical_validity(
                    sample=sample,
                    output=sample["output"],
                    min_clinical_validity=0.1,
                    nemo_config=object()  # truthy sentinel routes to (mocked) LLM judge
                )

        # Verify sample has expected fields
        assert sample["clinical_validity_score"] == 0.55
        assert sample["clinical_validity_classification"] == "annotation_needed"
        assert "clinical_validity_reason" in sample
        assert "borderline range" in sample["clinical_validity_reason"]

        # Verify API was called
        mock_post.assert_called_once()

        # Extract call arguments
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:3102/queue"

        # Verify payload structure
        payload = call_args[1]["json"]
        assert payload["sample_text"] == sample["output"]
        assert payload["original_score"] == 0.55
        assert "per_dimension_scores" in payload
        assert "routing_reason" in payload

        # Validation should succeed (returns None)
        assert result is None


def test_val_cross_002_non_borderline_samples_not_routed():
    """Test that non-borderline samples don't get routed to annotation queue."""
    test_cases = [
        (0.75, "accepted"),  # Accepted score (>=0.6)
        (0.30, "excluded"),  # Rejected score (<0.4)
    ]

    for score, expected_classification in test_cases:
        with patch("requests.post") as mock_post:
            sample = {
                "instruction": "Test instruction",
                "output": "Test output"
            }

            with patch("training.sdg_pipeline.ClinicalValidityJudge.score", return_value=score):
                with patch("training.sdg_pipeline.ClinicalValidityScorer.classify_score",
                          return_value=expected_classification):
                    _validate_clinical_validity(
                        sample=sample,
                        output=sample["output"],
                        min_clinical_validity=0.1,
                        nemo_config=object()  # truthy sentinel routes to (mocked) LLM judge
                    )

            # API should not be called for non-borderline samples
            mock_post.assert_not_called()

            # Sample should still have clinical validity score
            assert sample["clinical_validity_score"] == score

            # For "excluded" classification, field should be present
            # For "accepted" classification, field is not added (by design)
            if expected_classification == "excluded":
                assert sample.get("clinical_validity_classification") == expected_classification
                assert "clinical_validity_reason" in sample


def test_val_cross_002_api_failure_graceful_handling():
    """Test that SDG pipeline continues even if annotation queue API fails."""
    # Mock API to return error
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch("requests.post", return_value=mock_response):
        sample = {
            "instruction": "Test instruction",
            "output": "Test output with therapeutic content"
        }

        with patch("training.sdg_pipeline.ClinicalValidityJudge.score", return_value=0.55):
            with patch("training.sdg_pipeline.ClinicalValidityScorer.classify_score",
                      return_value="annotation_needed"):
                result = _validate_clinical_validity(
                    sample=sample,
                    output=sample["output"],
                    min_clinical_validity=0.1,
                    nemo_config=object()  # truthy sentinel routes to (mocked) LLM judge
                )

        # Sample should still be marked as borderline even if API fails
        assert sample["clinical_validity_score"] == 0.55
        assert sample["clinical_validity_classification"] == "annotation_needed"
        assert result is None  # Validation should succeed
