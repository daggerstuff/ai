"""
Tests for Enhanced Mental Health Integrator

Tests the integration system with mock data and Tier 2 integration.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from enhanced_mental_health_integrator import (
    EnhancedMentalHealthIntegrator,
    DatasetIntegrationConfig,
    IntegrationProgress,
    IntegrationTelemetry,
)
from conversation_schema import Conversation, Message


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def temp_datasets_dir(tmp_path):
    """Create temporary datasets directory."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    return datasets_dir


@pytest.fixture
def sample_jsonl_data(tmp_path):
    """Create sample JSONL data file."""
    data_file = tmp_path / "sample_data.jsonl"
    
    sample_data = [
        {
            "messages": [
                {"role": "user", "content": "I'm feeling anxious"},
                {"role": "assistant", "content": "Let's talk about it"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "I'm stressed"},
                {"role": "assistant", "content": "Stress is normal"},
            ]
        },
    ]
    
    with open(data_file, "w") as f:
        for entry in sample_data:
            f.write(json.dumps(entry) + "\n")
    
    return data_file


class TestIntegrationProgress:
    """Test IntegrationProgress dataclass."""
    
    def test_acceptance_rate(self):
        progress = IntegrationProgress("test_dataset")
        progress.processed = 100
        progress.accepted = 75
        
        assert progress.acceptance_rate == 0.75
    
    def test_acceptance_rate_zero_processed(self):
        progress = IntegrationProgress("test_dataset")
        
        assert progress.acceptance_rate == 0.0
    
    def test_to_dict(self):
        progress = IntegrationProgress("test_dataset")
        progress.processed = 10
        progress.accepted = 8
        
        result = progress.to_dict()
        
        assert result["dataset_name"] == "test_dataset"
        assert result["processed"] == 10
        assert result["accepted"] == 8
        assert result["acceptance_rate"] == 0.8


class TestIntegrationTelemetry:
    """Test IntegrationTelemetry dataclass."""
    
    def test_overall_acceptance_rate(self):
        telemetry = IntegrationTelemetry()
        telemetry.total_conversations_processed = 100
        telemetry.total_conversations_accepted = 80
        
        assert telemetry.overall_acceptance_rate == 0.8
    
    def test_to_dict(self):
        telemetry = IntegrationTelemetry()
        telemetry.total_datasets = 5
        telemetry.successful_datasets = 4
        
        result = telemetry.to_dict()
        
        assert result["summary"]["total_datasets"] == 5
        assert result["summary"]["successful_datasets"] == 4


class TestEnhancedMentalHealthIntegrator:
    """Test EnhancedMentalHealthIntegrator."""
    
    def test_initialization(self, temp_output_dir, temp_datasets_dir):
        integrator = EnhancedMentalHealthIntegrator(
            output_dir=temp_output_dir,
            base_datasets_path=temp_datasets_dir,
            enable_tier2=False,
        )
        
        assert integrator.output_dir == temp_output_dir
        assert integrator.base_datasets_path == temp_datasets_dir
        assert integrator.tier2_loader is None
        assert integrator.adapter_registry is not None
    
    def test_add_dataset_config(self, temp_output_dir, temp_datasets_dir):
        integrator = EnhancedMentalHealthIntegrator(
            output_dir=temp_output_dir,
            base_datasets_path=temp_datasets_dir,
            enable_tier2=False,
        )
        
        config = DatasetIntegrationConfig(
            name="test_dataset",
            format_type="generic_chatml",
        )
        
        integrator.add_dataset_config(config)
        
        assert len(integrator.dataset_configs) == 1
        assert integrator.dataset_configs[0].name == "test_dataset"
    
    def test_integrate_single_dataset(
        self,
        temp_output_dir,
        temp_datasets_dir,
        sample_jsonl_data,
    ):
        integrator = EnhancedMentalHealthIntegrator(
            output_dir=temp_output_dir,
            base_datasets_path=temp_datasets_dir,
            enable_tier2=False,
            enable_progress_bar=False,
        )
        
        config = DatasetIntegrationConfig(
            name="test_dataset",
            source_path=sample_jsonl_data,
            format_type="generic_chatml",
            tier=3,
        )
        
        conversations = integrator._integrate_single_dataset(config)
        
        assert len(conversations) == 2
        assert all(isinstance(c, Conversation) for c in conversations)
        assert conversations[0].metadata["tier"] == 3
    
    def test_integrate_single_dataset_with_limit(
        self,
        temp_output_dir,
        temp_datasets_dir,
        sample_jsonl_data,
    ):
        integrator = EnhancedMentalHealthIntegrator(
            output_dir=temp_output_dir,
            base_datasets_path=temp_datasets_dir,
            enable_tier2=False,
            enable_progress_bar=False,
        )
        
        config = DatasetIntegrationConfig(
            name="test_dataset",
            source_path=sample_jsonl_data,
            format_type="generic_chatml",
        )
        
        conversations = integrator._integrate_single_dataset(config, max_conversations=1)
        
        assert len(conversations) == 1
    
    def test_integrate_single_dataset_missing_source(
        self,
        temp_output_dir,
        temp_datasets_dir,
    ):
        integrator = EnhancedMentalHealthIntegrator(
            output_dir=temp_output_dir,
            base_datasets_path=temp_datasets_dir,
            enable_tier2=False,
            enable_progress_bar=False,
        )
        
        config = DatasetIntegrationConfig(
            name="test_dataset",
            source_path=Path("nonexistent.jsonl"),
            format_type="generic_chatml",
        )
        
        conversations = integrator._integrate_single_dataset(config)
        
        assert len(conversations) == 0
    
    def test_save_conversations(self, temp_output_dir, temp_datasets_dir):
        integrator = EnhancedMentalHealthIntegrator(
            output_dir=temp_output_dir,
            base_datasets_path=temp_datasets_dir,
            enable_tier2=False,
        )
        
        conversations = [
            Conversation(
                messages=[
                    Message(role="user", content="Hello"),
                    Message(role="assistant", content="Hi"),
                ],
                metadata={"test": "data"},
            )
        ]
        
        output_file = temp_output_dir / "test.jsonl"
        integrator._save_conversations(conversations, output_file)
        
        assert output_file.exists()
        
        # Verify content
        with open(output_file) as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert "messages" in data
    
    def test_load_raw_data(self, temp_output_dir, temp_datasets_dir, sample_jsonl_data):
        integrator = EnhancedMentalHealthIntegrator(
            output_dir=temp_output_dir,
            base_datasets_path=temp_datasets_dir,
            enable_tier2=False,
        )
        
        raw_data = integrator._load_raw_data(sample_jsonl_data)
        
        assert len(raw_data) == 2
        assert all(isinstance(entry, dict) for entry in raw_data)
    
    def test_generate_integration_report(self, temp_output_dir, temp_datasets_dir):
        integrator = EnhancedMentalHealthIntegrator(
            output_dir=temp_output_dir,
            base_datasets_path=temp_datasets_dir,
            enable_tier2=False,
        )
        
        integrator.telemetry.total_datasets = 3
        integrator.telemetry.successful_datasets = 2
        
        report = integrator._generate_integration_report()
        
        assert "timestamp" in report
        assert "telemetry" in report
        assert report["telemetry"]["summary"]["total_datasets"] == 3
        assert report["telemetry"]["summary"]["successful_datasets"] == 2
    
    @patch("ai.dataset_pipeline.ingestion.tier_loaders.tier2_professional_loader.Tier2ProfessionalLoader")
    def test_integrate_tier2_datasets(
        self,
        mock_tier2_loader_class,
        temp_output_dir,
        temp_datasets_dir,
    ):
        # Mock the Tier 2 loader
        mock_loader = Mock()
        mock_loader.load_datasets.return_value = {
            "psych8k": [
                Conversation(
                    messages=[
                        Message(role="user", content="Test"),
                        Message(role="assistant", content="Response"),
                    ],
                    metadata={"source": "psych8k"},
                )
            ]
        }
        mock_tier2_loader_class.return_value = mock_loader
        
        integrator = EnhancedMentalHealthIntegrator(
            output_dir=temp_output_dir,
            base_datasets_path=temp_datasets_dir,
            enable_tier2=True,
        )
        
        conversations = integrator._integrate_tier2_datasets()
        
        assert len(conversations) == 1
        assert conversations[0].metadata["source"] == "psych8k"
        assert "tier2_psych8k" in integrator.telemetry.dataset_progress
    
    def test_integrate_all_datasets_no_tier2(
        self,
        temp_output_dir,
        temp_datasets_dir,
        sample_jsonl_data,
    ):
        integrator = EnhancedMentalHealthIntegrator(
            output_dir=temp_output_dir,
            base_datasets_path=temp_datasets_dir,
            enable_tier2=False,
            enable_progress_bar=False,
        )
        
        config = DatasetIntegrationConfig(
            name="test_dataset",
            source_path=sample_jsonl_data,
            format_type="generic_chatml",
        )
        integrator.add_dataset_config(config)
        
        report = integrator.integrate_all_datasets(include_tier2=False)
        
        assert "telemetry" in report
        assert integrator.telemetry.total_datasets >= 1
        assert integrator.telemetry.total_conversations_accepted >= 2
        
        # Check output file was created
        output_file = temp_output_dir / "integrated_conversations.jsonl"
        assert output_file.exists()
        
        # Check report file was created
        report_file = temp_output_dir / "integration_report.json"
        assert report_file.exists()


class TestDatasetIntegrationConfig:
    """Test DatasetIntegrationConfig dataclass."""
    
    def test_default_values(self):
        config = DatasetIntegrationConfig(name="test")
        
        assert config.name == "test"
        assert config.format_type == "generic_chatml"
        assert config.tier == 3
        assert config.quality_threshold == 0.7
        assert config.enabled is True
    
    def test_custom_values(self):
        config = DatasetIntegrationConfig(
            name="custom",
            format_type="psych8k",
            tier=2,
            quality_threshold=0.95,
            enabled=False,
        )
        
        assert config.name == "custom"
        assert config.format_type == "psych8k"
        assert config.tier == 2
        assert config.quality_threshold == 0.95
        assert config.enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
