"""Tests for Stage 5 DPO Dataset Ingestion Pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipelines.dpo_ingestion import (
    compute_hash,
    ingest_dataset,
    normalize_schema,
    validate_record,
)


class TestNormalizeSchema:
    """Test schema normalization for different dataset formats."""

    def test_standard_schema(self):
        """Test standard prompt/chosen/rejected format."""
        record = {
            "prompt": "What is machine learning?",
            "chosen": "Machine learning is a subset of AI...",
            "rejected": "ML is when computers learn stuff...",
        }
        result = normalize_schema(record)
        assert result == record

    def test_alternative_field_names(self):
        """Test alternative field names (question/response_a/response_b)."""
        record = {
            "question": "Explain quantum computing",
            "response_a": "Quantum computing uses qubits...",
            "response_b": "It's like regular computing but faster",
        }
        result = normalize_schema(record)
        assert result["prompt"] == "Explain quantum computing"
        assert result["chosen"] == "Quantum computing uses qubits..."
        assert result["rejected"] == "It's like regular computing but faster"

    def test_messages_format(self):
        """Test nested messages format returns None when rejected is missing."""
        record = {
            "messages": [
                {"role": "user", "content": "How do neural networks work?"},
                {"role": "assistant", "content": "Neural networks are..."},
            ]
        }
        result = normalize_schema(record)
        assert result is None

    def test_missing_fields(self):
        """Test record with missing required fields."""
        record = {"prompt": "Some question"}
        result = normalize_schema(record)
        assert result is None

    def test_empty_fields(self):
        """Test record with empty string fields."""
        record = {"prompt": "", "chosen": "", "rejected": ""}
        result = normalize_schema(record)
        assert result is None


class TestValidateRecord:
    """Test record validation."""

    def test_valid_record(self):
        """Test valid record passes validation."""
        record = {
            "prompt": "What is AI?",
            "chosen": "AI is artificial intelligence...",
            "rejected": "AI is when computers think",
        }
        assert validate_record(record) is True

    def test_empty_prompt(self):
        """Test record with empty prompt fails."""
        record = {
            "prompt": "",
            "chosen": "Some response",
            "rejected": "Another response",
        }
        assert validate_record(record) is False

    def test_whitespace_only(self):
        """Test record with whitespace-only fields fails."""
        record = {
            "prompt": "   ",
            "chosen": "Some response",
            "rejected": "Another response",
        }
        assert validate_record(record) is False

    def test_missing_field(self):
        """Test record with missing field fails."""
        record = {"prompt": "Question", "chosen": "Response"}
        assert validate_record(record) is False


class TestComputeHash:
    """Test hash computation for deduplication."""

    def test_deterministic_hash(self):
        """Test same record produces same hash."""
        record = {"prompt": "Q", "chosen": "A", "rejected": "B"}
        hash1 = compute_hash(record)
        hash2 = compute_hash(record)
        assert hash1 == hash2

    def test_different_records_different_hash(self):
        """Test different records produce different hashes."""
        record1 = {"prompt": "Q1", "chosen": "A1", "rejected": "B1"}
        record2 = {"prompt": "Q2", "chosen": "A2", "rejected": "B2"}
        assert compute_hash(record1) != compute_hash(record2)

    def test_hash_is_sha256(self):
        """Test hash is valid SHA256 (64 hex chars)."""
        record = {"prompt": "Q", "chosen": "A", "rejected": "B"}
        hash_value = compute_hash(record)
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)


class TestIngestDataset:
    """Test dataset ingestion with mocking."""

    @patch("pipelines.dpo_ingestion.load_dataset")
    def test_successful_ingestion(self, mock_load_dataset):
        """Test successful dataset ingestion."""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "prompt": "Question 1",
                        "chosen": "Good answer",
                        "rejected": "Bad answer",
                    },
                    {
                        "prompt": "Question 2",
                        "chosen": "Better answer",
                        "rejected": "Worse answer",
                    },
                ]
            )
        )
        mock_dataset.__len__ = MagicMock(return_value=2)
        mock_load_dataset.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test.jsonl"
            seen_hashes = set()
            stats = {
                "total_records": 0,
                "total_invalid": 0,
                "total_duplicate": 0,
                "failed_datasets": 0,
            }

            ingest_dataset("test/dataset", seen_hashes, output_file, stats)

            # Verify output
            assert output_file.exists()
            lines = output_file.read_text().strip().split("\n")
            assert len(lines) == 2

            # Verify stats
            assert stats["total_records"] == 2
            assert stats["total_invalid"] == 0
            assert stats["total_duplicate"] == 0

            # Verify deduplication
            assert len(seen_hashes) == 2

    @patch("pipelines.dpo_ingestion.load_dataset")
    def test_deduplication(self, mock_load_dataset):
        """Test duplicate records are skipped."""
        # Mock dataset with duplicate
        duplicate_record = {
            "prompt": "Same question",
            "chosen": "Same answer",
            "rejected": "Same rejection",
        }
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter([duplicate_record, duplicate_record]))
        mock_dataset.__len__ = MagicMock(return_value=2)
        mock_load_dataset.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test.jsonl"
            seen_hashes = set()
            stats = {
                "total_records": 0,
                "total_invalid": 0,
                "total_duplicate": 0,
                "failed_datasets": 0,
            }

            ingest_dataset("test/dataset", seen_hashes, output_file, stats)

            # Only one record should be written
            lines = output_file.read_text().strip().split("\n")
            assert len(lines) == 1
            assert stats["total_records"] == 1
            assert stats["total_duplicate"] == 1

    @patch("pipelines.dpo_ingestion.load_dataset")
    def test_invalid_records_skipped(self, mock_load_dataset):
        """Test invalid records are skipped."""
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(
            return_value=iter(
                [
                    {"prompt": "Valid"},  # Missing chosen/rejected
                    {
                        "prompt": "Question",
                        "chosen": "Answer",
                        "rejected": "Rejection",
                    },
                ]
            )
        )
        mock_dataset.__len__ = MagicMock(return_value=2)
        mock_load_dataset.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test.jsonl"
            seen_hashes = set()
            stats = {
                "total_records": 0,
                "total_invalid": 0,
                "total_duplicate": 0,
                "failed_datasets": 0,
            }

            ingest_dataset("test/dataset", seen_hashes, output_file, stats)

            # Only valid record should be written
            lines = output_file.read_text().strip().split("\n")
            assert len(lines) == 1
            assert stats["total_records"] == 1
            assert stats["total_invalid"] == 1

    @patch("pipelines.dpo_ingestion.load_dataset")
    def test_dataset_load_failure(self, mock_load_dataset):
        """Test graceful handling of dataset load failure."""
        mock_load_dataset.side_effect = Exception("Network error")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test.jsonl"
            seen_hashes = set()
            stats = {
                "total_records": 0,
                "total_invalid": 0,
                "total_duplicate": 0,
                "failed_datasets": 0,
            }

            ingest_dataset("invalid/dataset", seen_hashes, output_file, stats)

            # No output should be written
            assert not output_file.exists() or output_file.stat().st_size == 0
            assert stats["failed_datasets"] == 1
