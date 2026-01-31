"""Unit tests for quarantine.py module.

Tests QuarantineRecord model, QuarantineStore operations, status updates,
reprocessing workflow. Uses pytest with mocks for MongoDB to avoid live DB.
"""

from unittest.mock import MagicMock, patch

import pytest

from ai.pipelines.orchestrator.quarantine import (
    QuarantineRecord,
    QuarantineStatus,
    QuarantineStore,
)


@pytest.fixture
def sample_quarantine_record():
    return QuarantineRecord(
        quarantine_id="test_qid",
        original_record_id="test_rec_id",
        raw_payload={"test": "data"},
        validation_errors=["test error"],
        metadata={"test": "meta"},
        status=QuarantineStatus.PENDING_REVIEW,
    )


def test_quarantine_record_model(sample_quarantine_record):
    assert sample_quarantine_record.quarantine_id == "test_qid"
    assert sample_quarantine_record.status == QuarantineStatus.PENDING_REVIEW

    # Test status update
    sample_quarantine_record.update_status(QuarantineStatus.APPROVED, "Looks good")
    assert sample_quarantine_record.status == QuarantineStatus.APPROVED
    assert sample_quarantine_record.review_notes == "Looks good"
    assert sample_quarantine_record.updated_at is not None


def test_quarantine_store_connection(mocker):
    mocker.patch("ai.pipelines.orchestrator.quarantine.MongoClient")
    mocker.patch("ai.pipelines.orchestrator.quarantine.MongoClient.admin.command")
    store = QuarantineStore(mongo_uri="mock://localhost")
    assert store.collection is not None  # Connection succeeds


@patch("ai.pipelines.orchestrator.quarantine.MongoClient")
def test_quarantine_store_quarantine(mock_client, sample_quarantine_record):
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_collection.insert_one.return_value = MagicMock(inserted_id="mock_id")
    mock_client.return_value = MagicMock()
    mock_client.return_value.__getitem__.return_value = mock_db
    mock_db.__getitem__.return_value = mock_collection

    store = QuarantineStore("mock_uri")
    qid = store.quarantine_record(sample_quarantine_record, ["error1"])
    assert qid == sample_quarantine_record.quarantine_id
    mock_collection.insert_one.assert_called_once()


def test_quarantine_store_update_status(mocker):
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = {
        "quarantine_id": "test_qid",
        "status": QuarantineStatus.PENDING_REVIEW.value,
    }
    mock_collection.replace_one.return_value = MagicMock(modified_count=1)

    store = MagicMock()
    store.collection = mock_collection

    QuarantineRecord(
        quarantine_id="test_qid",
        original_record_id="test_id",
        raw_payload={},
        validation_errors=[],
        metadata={},
    )
    success = store.update_status("test_qid", QuarantineStatus.APPROVED, "Approved")
    assert success is True
    mock_collection.replace_one.assert_called_once()


@patch("ai.pipelines.orchestrator.quarantine.MongoClient")
def test_quarantine_store_reprocess_success(mock_client, mocker):
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = {
        "quarantine_id": "test_qid",
        "reprocess_attempts": 0,
        "raw_payload": {"test": "data"},
        "original_record_id": "rec_id",
        "metadata": {},
    }
    mock_collection.replace_one.return_value = MagicMock(modified_count=1)
    mock_client.return_value = MagicMock()
    mock_client.return_value.__getitem__.return_value = mock_db
    mock_db.__getitem__.return_value = mock_collection

    mocker.patch("ai.pipelines.orchestrator.quarantine.IngestRecord")
    mocker.patch("ai.pipelines.orchestrator.quarantine.datetime.utcnow")
    mocker.patch("ai.pipelines.orchestrator.quarantine.validate_record", return_value=MagicMock())

    store = QuarantineStore("mock_uri")
    rec = store.reprocess_record("test_qid")
    assert rec is not None
    assert mock_collection.replace_one.call_count == 2  # One for update, one for reprocess


@patch("ai.pipelines.orchestrator.quarantine.MongoClient")
def test_quarantine_store_reprocess_fail(mock_client, mocker):
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = {
        "quarantine_id": "test_qid",
        "reprocess_attempts": 0,
        "raw_payload": {"invalid": "data"},
        "original_record_id": "rec_id",
        "metadata": {},
    }
    mock_collection.replace_one.return_value = MagicMock(modified_count=1)
    mock_client.return_value = MagicMock()
    mock_client.return_value.__getitem__.return_value = mock_db
    mock_db.__getitem__.return_value = mock_collection

    mocker.patch("ai.pipelines.orchestrator.quarantine.IngestRecord")
    mocker.patch("ai.pipelines.orchestrator.quarantine.validate_record", side_effect=ValueError("Invalid"))

    store = QuarantineStore("mock_uri")
    rec = store.reprocess_record("test_qid")
    assert rec is None
    assert mock_collection.replace_one.call_count == 1


def test_operator_workflow():
    store = MagicMock()
    store.update_status.return_value = True
    store.delete_record.return_value = True

    # Test approve
    assert approve_record("test_qid", "Good") is True

    # Test reject (updates status then deletes)
    assert reject_record("test_qid", "Bad") is True
