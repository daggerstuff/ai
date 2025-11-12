"""
Test fixtures and mocks for journal dataset research system.

This module provides shared fixtures, mock objects, and test utilities
for all test modules.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from unittest.mock import Mock, MagicMock
from uuid import uuid4

import pandas as pd
import pytest

from ai.journal_dataset_research.models.dataset_models import (
    AccessRequest,
    AcquiredDataset,
    DatasetEvaluation,
    DatasetSource,
    IntegrationPlan,
    ResearchSession,
)
from ai.journal_dataset_research.orchestrator.research_orchestrator import (
    ResearchOrchestrator,
)
from ai.journal_dataset_research.orchestrator.types import OrchestratorConfig


# ============================================================================
# Dataset Source Fixtures
# ============================================================================

@pytest.fixture
def sample_dataset_source() -> DatasetSource:
    """Create a sample dataset source."""
    return DatasetSource(
        source_id="test-001",
        title="Therapeutic Conversation Dataset for Mental Health",
        authors=["Smith, J.", "Doe, A."],
        publication_date=datetime(2024, 1, 1),
        source_type="journal",
        url="https://example.com/dataset",
        doi="10.1000/test",
        abstract="A comprehensive dataset of therapy session transcripts for training AI models in mental health counseling.",
        keywords=["therapy", "counseling", "mental health", "transcripts"],
        open_access=True,
        data_availability="available",
        discovery_date=datetime.now(),
        discovery_method="pubmed_search",
    )


@pytest.fixture
def high_quality_source() -> DatasetSource:
    """Create a high-quality dataset source."""
    return DatasetSource(
        source_id="high-quality-001",
        title="Evidence-Based CBT Therapy Transcripts Dataset",
        authors=["Researcher, A.", "Clinician, B."],
        publication_date=datetime(2024, 1, 1),
        source_type="repository",
        url="https://example.com/high-quality",
        doi="10.1000/high-quality",
        abstract="A large dataset of Cognitive Behavioral Therapy session transcripts with comprehensive metadata and anonymized patient data. Includes outcome measures and treatment protocols.",
        keywords=["cbt", "cognitive behavioral therapy", "transcripts", "evidence-based"],
        open_access=True,
        data_availability="available",
        discovery_date=datetime.now(),
        discovery_method="repository_api",
    )


@pytest.fixture
def low_quality_source() -> DatasetSource:
    """Create a low-quality dataset source."""
    return DatasetSource(
        source_id="low-quality-001",
        title="Some Dataset",
        authors=[],
        publication_date=datetime(2024, 1, 1),
        source_type="training_material",
        url="",
        doi=None,
        abstract="Some abstract",
        keywords=[],
        open_access=False,
        data_availability="unknown",
        discovery_date=datetime.now(),
        discovery_method="citation",
    )


@pytest.fixture
def multiple_sources(sample_dataset_source, high_quality_source, low_quality_source) -> List[DatasetSource]:
    """Create multiple dataset sources."""
    return [sample_dataset_source, high_quality_source, low_quality_source]


# ============================================================================
# Evaluation Fixtures
# ============================================================================

@pytest.fixture
def sample_evaluation(sample_dataset_source) -> DatasetEvaluation:
    """Create a sample dataset evaluation."""
    return DatasetEvaluation(
        source_id=sample_dataset_source.source_id,
        therapeutic_relevance=8,
        therapeutic_relevance_notes="High therapeutic value with evidence-based practices",
        data_structure_quality=7,
        data_structure_notes="Well structured with comprehensive metadata",
        training_integration=8,
        integration_notes="Compatible with training pipeline format",
        ethical_accessibility=9,
        ethical_notes="Open access and fully anonymized",
        overall_score=8.0,
        priority_tier="high",
        evaluation_date=datetime.now(),
        evaluator="system",
        competitive_advantages=["Contains therapy transcripts", "Evidence-based"],
        compliance_checked=True,
        compliance_status="compliant",
        compliance_score=0.95,
        license_compatible=True,
        privacy_compliant=True,
        hipaa_compliant=True,
    )


@pytest.fixture
def high_score_evaluation(high_quality_source) -> DatasetEvaluation:
    """Create a high-score evaluation."""
    return DatasetEvaluation(
        source_id=high_quality_source.source_id,
        therapeutic_relevance=9,
        data_structure_quality=9,
        training_integration=9,
        ethical_accessibility=9,
        overall_score=9.0,
        priority_tier="high",
        evaluation_date=datetime.now(),
        evaluator="system",
    )


@pytest.fixture
def low_score_evaluation(low_quality_source) -> DatasetEvaluation:
    """Create a low-score evaluation."""
    return DatasetEvaluation(
        source_id=low_quality_source.source_id,
        therapeutic_relevance=3,
        data_structure_quality=2,
        training_integration=4,
        ethical_accessibility=5,
        overall_score=3.5,
        priority_tier="low",
        evaluation_date=datetime.now(),
        evaluator="system",
    )


# ============================================================================
# Access Request Fixtures
# ============================================================================

@pytest.fixture
def sample_access_request(sample_dataset_source) -> AccessRequest:
    """Create a sample access request."""
    return AccessRequest(
        source_id=sample_dataset_source.source_id,
        access_method="direct",
        request_date=datetime.now(),
        status="pending",
        access_url=sample_dataset_source.url,
        credentials_required=False,
        institutional_affiliation_required=False,
        estimated_access_date=datetime.now() + timedelta(hours=1),
        notes="Direct download available",
    )


@pytest.fixture
def approved_access_request(sample_dataset_source) -> AccessRequest:
    """Create an approved access request."""
    return AccessRequest(
        source_id=sample_dataset_source.source_id,
        access_method="direct",
        request_date=datetime.now() - timedelta(days=1),
        status="approved",
        access_url=sample_dataset_source.url,
    )


# ============================================================================
# Acquired Dataset Fixtures
# ============================================================================

@pytest.fixture
def sample_acquired_dataset(sample_dataset_source, tmp_path) -> AcquiredDataset:
    """Create a sample acquired dataset."""
    # Create a test file
    test_file = tmp_path / "test_dataset.zip"
    test_file.write_bytes(b"test data content")

    return AcquiredDataset(
        source_id=sample_dataset_source.source_id,
        acquisition_date=datetime.now(),
        storage_path=str(test_file),
        file_format="zip",
        file_size_mb=0.001,
        license="CC-BY",
        usage_restrictions=[],
        attribution_required=False,
        checksum="abc123def456",
        encrypted=False,
        compliance_status="compliant",
        compliance_score=0.95,
        hipaa_compliant=True,
        privacy_assessed=True,
    )


# ============================================================================
# Integration Plan Fixtures
# ============================================================================

@pytest.fixture
def sample_integration_plan(sample_dataset_source) -> IntegrationPlan:
    """Create a sample integration plan."""
    return IntegrationPlan(
        source_id=sample_dataset_source.source_id,
        dataset_format="csv",
        schema_mapping={
            "message": "messages",
            "role": "role",
            "timestamp": "timestamp",
        },
        required_transformations=["format_conversion", "field_mapping"],
        preprocessing_steps=[
            "Load CSV file",
            "Map fields to pipeline schema",
            "Validate data types",
            "Convert to JSONL format",
        ],
        complexity="medium",
        estimated_effort_hours=4,
        dependencies=[],
        integration_priority=1,
        created_date=datetime.now(),
    )


# ============================================================================
# Research Session Fixtures
# ============================================================================

@pytest.fixture
def sample_research_session() -> ResearchSession:
    """Create a sample research session."""
    return ResearchSession(
        session_id=str(uuid4()),
        start_date=datetime.now(),
        target_sources=["pubmed", "zenodo", "dryad"],
        search_keywords={
            "therapy": ["cbt", "dbt", "act"],
            "mental_health": ["depression", "anxiety"],
        },
        weekly_targets={
            "sources_identified": 10,
            "datasets_evaluated": 5,
            "datasets_acquired": 2,
        },
        current_phase="discovery",
        progress_metrics={},
    )


# ============================================================================
# Test Dataset Fixtures
# ============================================================================

@pytest.fixture
def sample_csv_dataset(tmp_path) -> Path:
    """Create a sample CSV dataset file."""
    csv_path = tmp_path / "test_dataset.csv"
    df = pd.DataFrame({
        "id": [f"id_{i}" for i in range(10)],
        "message": [f"Message {i}" for i in range(10)],
        "role": ["user", "assistant"] * 5,
        "timestamp": [datetime.now().isoformat()] * 10,
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_jsonl_dataset(tmp_path) -> Path:
    """Create a sample JSONL dataset file."""
    jsonl_path = tmp_path / "test_dataset.jsonl"
    records = [
        {
            "conversation_id": f"conv_{i}",
            "messages": [
                {"role": "user", "content": f"User message {i}"},
                {"role": "assistant", "content": f"Assistant response {i}"},
            ],
        }
        for i in range(10)
    ]

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return jsonl_path


@pytest.fixture
def sample_json_dataset(tmp_path) -> Path:
    """Create a sample JSON dataset file."""
    json_path = tmp_path / "test_dataset.json"
    data = {
        "conversations": [
            {
                "id": f"conv_{i}",
                "turns": [
                    {"speaker": "user", "text": f"User message {i}"},
                    {"speaker": "assistant", "text": f"Assistant response {i}"},
                ],
            }
            for i in range(10)
        ]
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return json_path


# ============================================================================
# Mock API Responses
# ============================================================================

@pytest.fixture
def mock_pubmed_response():
    """Mock PubMed API response."""
    return {
        "esearchresult": {
            "count": "2",
            "idlist": ["12345678", "87654321"],
        }
    }


@pytest.fixture
def mock_pubmed_xml_response():
    """Mock PubMed XML response."""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345678</PMID>
                <Article>
                    <ArticleTitle>Therapeutic Conversation Dataset</ArticleTitle>
                    <Abstract>
                        <AbstractText>A dataset of therapy sessions.</AbstractText>
                    </Abstract>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    </PubmedArticleSet>"""


@pytest.fixture
def mock_zenodo_response():
    """Mock Zenodo API response."""
    return {
        "hits": {
            "hits": [
                {
                    "id": 12345,
                    "metadata": {
                        "title": "Therapy Dataset",
                        "doi": "10.5281/zenodo.12345",
                        "creators": [{"name": "Researcher, A."}],
                        "description": "A dataset of therapy sessions",
                        "keywords": ["therapy", "mental health"],
                        "access_right": "open",
                    },
                }
            ]
        }
    }


@pytest.fixture
def mock_dryad_response():
    """Mock Dryad API response."""
    return {
        "results": [
            {
                "id": "doi:10.5061/dryad.test",
                "title": "Therapy Dataset",
                "authors": ["Smith, J."],
                "abstract": "A dataset of therapy sessions",
                "keywords": ["therapy"],
                "is_available": True,
            }
        ]
    }


@pytest.fixture
def mock_clinical_trials_response():
    """Mock ClinicalTrials.gov API response."""
    return {
        "StudyFieldsResponse": {
            "NStudiesFound": 1,
            "StudyFields": [
                {
                    "NCTId": ["NCT12345"],
                    "BriefTitle": ["Therapy Study"],
                    "Condition": ["Depression"],
                    "OverallStatus": ["Completed"],
                }
            ],
        }
    }


# ============================================================================
# Mock Services
# ============================================================================

@pytest.fixture
def mock_discovery_service():
    """Create a mock discovery service."""
    service = Mock()
    service.discover_sources = Mock(return_value=[])
    return service


@pytest.fixture
def mock_evaluation_engine():
    """Create a mock evaluation engine."""
    engine = Mock()
    engine.evaluate_dataset = Mock(return_value=DatasetEvaluation(
        source_id="test",
        therapeutic_relevance=8,
        data_structure_quality=7,
        training_integration=8,
        ethical_accessibility=9,
        overall_score=8.0,
        priority_tier="high",
    ))
    return engine


@pytest.fixture
def mock_acquisition_manager():
    """Create a mock acquisition manager."""
    manager = Mock()
    manager.submit_access_request = Mock(return_value=AccessRequest(
        source_id="test",
        access_method="direct",
        status="pending",
    ))
    manager.download_dataset = Mock(return_value=AcquiredDataset(
        source_id="test",
        storage_path="/tmp/test.zip",
    ))
    return manager


@pytest.fixture
def mock_integration_engine():
    """Create a mock integration engine."""
    engine = Mock()
    engine.create_integration_plan = Mock(return_value=IntegrationPlan(
        source_id="test",
        dataset_format="csv",
        complexity="medium",
    ))
    return engine


@pytest.fixture
def orchestrator(
    temp_dir,
    mock_discovery_service,
    mock_evaluation_engine,
    mock_acquisition_manager,
    mock_integration_engine,
):
    """Provide a fully wired orchestrator for integration tests."""
    config = OrchestratorConfig(
        session_storage_path=temp_dir,
        max_retries=3,
        retry_delay_seconds=0.0,
        fallback_on_failure=True,
        progress_history_limit=100,
    )
    return ResearchOrchestrator(
        discovery_service=mock_discovery_service,
        evaluation_engine=mock_evaluation_engine,
        acquisition_manager=mock_acquisition_manager,
        integration_engine=mock_integration_engine,
        config=config,
    )


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_log_dir(temp_dir):
    """Create a temporary log directory."""
    log_dir = temp_dir / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def temp_report_dir(temp_dir):
    """Create a temporary report directory."""
    report_dir = temp_dir / "reports"
    report_dir.mkdir()
    return report_dir


@pytest.fixture
def temp_storage_dir(temp_dir):
    """Create a temporary storage directory."""
    storage_dir = temp_dir / "storage"
    storage_dir.mkdir()
    return storage_dir


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration dictionary."""
    return {
        "storage_base_path": str(temp_dir / "storage"),
        "log_directory": str(temp_dir / "logs"),
        "report_directory": str(temp_dir / "reports"),
        "download_timeout": 30,
        "max_retries": 3,
        "retry_delay_seconds": 1.0,
    }

