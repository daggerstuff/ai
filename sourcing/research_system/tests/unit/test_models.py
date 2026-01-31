"""Unit tests for core data models."""

from datetime import datetime

import pytest

from research_system.models import (
    AccessRequest,
    AcquiredDataset,
    DatasetEvaluation,
    DatasetSource,
    IntegrationPlan,
    ResearchLog,
    ResearchProgress,
    ResearchSession,
    TransformationSpec,
    WeeklyReport,
)


class TestDatasetSource:
    """Tests for DatasetSource model."""
    
    def test_valid_dataset_source(self):
        """Test creating a valid dataset source."""
        source = DatasetSource(
            source_id="TEST001",
            title="Test Dataset",
            authors=["Author A", "Author B"],
            publication_date=datetime(2024, 1, 1),
            source_type="journal",
            url="https://example.com/dataset",
            doi="10.1234/test",
            abstract="Test abstract",
            keywords=["test", "dataset"],
            open_access=True,
            data_availability="available",
            discovery_date=datetime.now(),
            discovery_method="pubmed_search"
        )
        
        is_valid, errors = source.validate()
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_source_type(self):
        """Test validation fails for invalid source type."""
        source = DatasetSource(
            source_id="TEST001",
            title="Test Dataset",
            authors=["Author A"],
            publication_date=datetime(2024, 1, 1),
            source_type="invalid_type",
            url="https://example.com/dataset",
            doi=None,
            abstract="Test abstract",
            keywords=["test"],
            open_access=True,
            data_availability="available",
            discovery_date=datetime.now(),
            discovery_method="pubmed_search"
        )
        
        is_valid, errors = source.validate()
        assert not is_valid
        assert any("source_type" in error for error in errors)


class TestDatasetEvaluation:
    """Tests for DatasetEvaluation model."""
    
    def test_calculate_overall_score(self):
        """Test overall score calculation."""
        evaluation = DatasetEvaluation(
            source_id="TEST001",
            therapeutic_relevance=9,
            therapeutic_relevance_notes="Excellent",
            data_structure_quality=8,
            data_structure_notes="Good",
            training_integration=7,
            integration_notes="Moderate",
            ethical_accessibility=9,
            ethical_notes="Excellent",
            overall_score=0.0,
            priority_tier="high",
            evaluation_date=datetime.now(),
            evaluator="test_user"
        )
        
        score = evaluation.calculate_overall_score()
        
        # Expected: 9*0.35 + 8*0.25 + 7*0.20 + 9*0.20 = 8.35
        assert score == 8.35
    
    def test_invalid_score_range(self):
        """Test validation fails for scores outside 1-10 range."""
        evaluation = DatasetEvaluation(
            source_id="TEST001",
            therapeutic_relevance=11,  # Invalid
            therapeutic_relevance_notes="Test",
            data_structure_quality=8,
            data_structure_notes="Test",
            training_integration=7,
            integration_notes="Test",
            ethical_accessibility=9,
            ethical_notes="Test",
            overall_score=8.0,
            priority_tier="high",
            evaluation_date=datetime.now(),
            evaluator="test_user"
        )
        
        is_valid, errors = evaluation.validate()
        assert not is_valid
        assert any("therapeutic_relevance" in error for error in errors)


class TestAccessRequest:
    """Tests for AccessRequest model."""
    
    def test_valid_access_request(self):
        """Test creating a valid access request."""
        request = AccessRequest(
            source_id="TEST001",
            access_method="direct",
            request_date=datetime.now(),
            status="pending",
            access_url="https://example.com/download",
            credentials_required=False,
            institutional_affiliation_required=False,
            estimated_access_date=None,
            notes="Test request"
        )
        
        is_valid, errors = request.validate()
        assert is_valid
        assert len(errors) == 0


class TestResearchProgress:
    """Tests for ResearchProgress model."""
    
    def test_valid_progress(self):
        """Test creating valid research progress."""
        progress = ResearchProgress(
            sources_identified=10,
            datasets_evaluated=5,
            access_established=3,
            datasets_acquired=2,
            integration_plans_created=2,
            last_updated=datetime.now()
        )
        
        is_valid, errors = progress.validate()
        assert is_valid
        assert len(errors) == 0
    
    def test_negative_counts_invalid(self):
        """Test validation fails for negative counts."""
        progress = ResearchProgress(
            sources_identified=-1,  # Invalid
            datasets_evaluated=5,
            access_established=3,
            datasets_acquired=2,
            integration_plans_created=2,
            last_updated=datetime.now()
        )
        
        is_valid, errors = progress.validate()
        assert not is_valid
        assert any("sources_identified" in error for error in errors)


class TestResearchSession:
    """Tests for ResearchSession model."""
    
    def test_valid_session(self):
        """Test creating a valid research session."""
        session = ResearchSession(
            session_id="SESSION001",
            start_date=datetime.now(),
            target_sources=["pubmed", "doaj"],
            search_keywords={"therapy": ["therapy transcript"]},
            weekly_targets={"sources_identified": 10},
            current_phase="discovery",
            progress_metrics={"sources_found": 5}
        )
        
        is_valid, errors = session.validate()
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_phase(self):
        """Test validation fails for invalid phase."""
        session = ResearchSession(
            session_id="SESSION001",
            start_date=datetime.now(),
            target_sources=["pubmed"],
            search_keywords={},
            weekly_targets={},
            current_phase="invalid_phase",
            progress_metrics={}
        )
        
        is_valid, errors = session.validate()
        assert not is_valid
        assert any("current_phase" in error for error in errors)
