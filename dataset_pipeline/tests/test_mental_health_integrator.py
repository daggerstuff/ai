"""
Tests for Mental Health Dataset Integration System
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from ai.dataset_pipeline.mental_health_integrator import (
    MentalHealthDatasetConfig,
    MentalHealthIntegrator,
)


class TestMentalHealthIntegrator:
    """Test suite for MentalHealthIntegrator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.integrator = MentalHealthIntegrator(output_dir=self.temp_dir)

        # Mock sample conversation data
        self.sample_conversation = {
            "conversation": [
                {"role": "client", "content": "I've been feeling anxious lately"},
                {"role": "therapist", "content": "Can you tell me more about what's making you feel anxious?"}
            ],
            "metadata": {
                "category": "mental_health",
                "subcategory": "anxiety",
                "therapeutic_approach": "cognitive_behavioral"
            }
        }

        # Mock dataset items
        self.mock_dataset_items = [
            {
                "input": "I'm feeling depressed",
                "output": "I understand you're going through a difficult time. Can you share what's been contributing to these feelings?",
                "instruction": "Provide therapeutic response"
            },
            {
                "conversations": [
                    {"from": "human", "value": "I have anxiety about work"},
                    {"from": "gpt", "value": "Work anxiety is very common. What specific aspects of work are causing you stress?"}
                ]
            },
            {
                "text": "Client: I can't sleep at night\nTherapist: Sleep difficulties can be very challenging. How long have you been experiencing this?"
            }
        ]

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test MentalHealthIntegrator initialization."""
        assert self.integrator.output_dir == self.temp_dir
        assert len(self.integrator.dataset_configs) == 4
        assert os.path.exists(self.temp_dir)

        # Check dataset configurations
        config_names = [config.name for config in self.integrator.dataset_configs]
        expected_names = ["mental_health_counseling", "psych8k", "psychology_10k", "addiction_counseling"]
        assert all(name in config_names for name in expected_names)

    def test_dataset_config_creation(self):
        """Test MentalHealthDatasetConfig creation."""
        config = MentalHealthDatasetConfig(
            name="test_dataset",
            hf_path="test/dataset",
            target_conversations=1000,
            priority=1,
            quality_threshold=0.8
        )

        assert config.name == "test_dataset"
        assert config.hf_path == "test/dataset"
        assert config.target_conversations == 1000
        assert config.priority == 1
        assert config.quality_threshold == 0.8
        assert config.therapeutic_accuracy_threshold == 0.75  # default
        assert config.emotional_authenticity_threshold == 0.7  # default

    @patch("ai.dataset_pipeline.mental_health_integrator.load_hf_dataset")
    def test_process_single_dataset_success(self, mock_load_hf):
        """Test successful processing of a single dataset."""
        # Mock dataset loading
        mock_dataset = Mock()
        mock_dataset.to_list.return_value = self.mock_dataset_items[:2]  # Use first 2 items
        mock_load_hf.return_value = mock_dataset

        # Mock quality assessments to return True
        with patch.object(self.integrator, "_assess_quality", return_value=True), \
             patch.object(self.integrator, "_assess_therapeutic_accuracy", return_value=True), \
             patch.object(self.integrator, "_assess_emotional_authenticity", return_value=True), \
             patch.object(self.integrator, "_standardize_conversation") as mock_standardize:

            # Mock standardization to return valid conversations
            mock_standardize.side_effect = [
                {
                    "conversation": [
                        {"role": "client", "content": "I'm feeling depressed"},
                        {"role": "therapist", "content": "I understand you're going through a difficult time."}
                    ],
                    "metadata": {"category": "mental_health", "subcategory": "depression"}
                },
                {
                    "conversation": [
                        {"role": "client", "content": "I have anxiety about work"},
                        {"role": "therapist", "content": "Work anxiety is very common."}
                    ],
                    "metadata": {"category": "mental_health", "subcategory": "anxiety"}
                }
            ]

            config = self.integrator.dataset_configs[0]  # Use first config
            result = self.integrator._process_single_dataset(config)

            assert len(result) == 2
            assert all("metadata" in conv for conv in result)
            assert all(conv["metadata"]["source_dataset"] == config.name for conv in result)

    @patch("ai.dataset_pipeline.mental_health_integrator.load_hf_dataset")
    def test_process_single_dataset_failure(self, mock_load_hf):
        """Test handling of dataset loading failure."""
        mock_load_hf.return_value = None

        config = self.integrator.dataset_configs[0]

        with pytest.raises(ValueError, match="Failed to load dataset"):
            self.integrator._process_single_dataset(config)

    def test_standardize_conversation_formats(self):
        """Test conversation standardization for different formats."""
        # Test input/output format
        item1 = {
            "input": "I'm feeling sad",
            "output": "I'm sorry to hear you're feeling sad. Can you tell me more?"
        }

        config = self.integrator.dataset_configs[0]

        with patch.object(self.integrator, "_detect_and_convert_format") as mock_convert:
            # Mock conversation object with messages
            mock_conversation = Mock()
            mock_conversation.messages = [
                Mock(role="client", content="I'm feeling sad"),
                Mock(role="therapist", content="I'm sorry to hear you're feeling sad.")
            ]
            mock_convert.return_value = mock_conversation

            result = self.integrator._standardize_conversation(item1, config)

            assert result is not None
            assert "metadata" in result
            assert result["metadata"]["category"] == "mental_health"
            assert "subcategory" in result["metadata"]

    def test_determine_subcategory(self):
        """Test subcategory determination based on dataset name."""
        item = {"content": "test"}

        # Test addiction counseling
        addiction_config = MentalHealthDatasetConfig(
            name="addiction_counseling", hf_path="test", target_conversations=100, priority=1
        )
        assert self.integrator._determine_subcategory(item, addiction_config) == "addiction_counseling"

        # Test psychology expertise
        psych_config = MentalHealthDatasetConfig(
            name="psych8k", hf_path="test", target_conversations=100, priority=1
        )
        assert self.integrator._determine_subcategory(item, psych_config) == "psychology_expertise"

        # Test general counseling
        counseling_config = MentalHealthDatasetConfig(
            name="mental_health_counseling", hf_path="test", target_conversations=100, priority=1
        )
        assert self.integrator._determine_subcategory(item, counseling_config) == "general_counseling"

    def test_determine_therapeutic_approach(self):
        """Test therapeutic approach determination."""
        # Test cognitive behavioral
        cbt_item = {"content": "Let's examine your thoughts and beliefs about this situation"}
        assert self.integrator._determine_therapeutic_approach(cbt_item) == "cognitive_behavioral"

        # Test mindfulness based
        mindful_item = {"content": "Focus on your feelings and be present in this moment"}
        assert self.integrator._determine_therapeutic_approach(mindful_item) == "mindfulness_based"

        # Test behavioral
        behavioral_item = {"content": "What actions can you take to achieve your goals?"}
        assert self.integrator._determine_therapeutic_approach(behavioral_item) == "behavioral"

        # Test integrative (default)
        general_item = {"content": "How are you doing today?"}
        assert self.integrator._determine_therapeutic_approach(general_item) == "integrative"

    def test_estimate_emotional_intensity(self):
        """Test emotional intensity estimation."""
        # High intensity
        crisis_item = {"content": "I'm having suicidal thoughts and this is a crisis"}
        intensity = self.integrator._estimate_emotional_intensity(crisis_item)
        assert intensity >= 0.8

        # Medium intensity
        anxious_item = {"content": "I'm feeling anxious and worried about work"}
        intensity = self.integrator._estimate_emotional_intensity(anxious_item)
        assert 0.5 <= intensity < 0.8

        # Low intensity
        neutral_item = {"content": "I had a good day today"}
        intensity = self.integrator._estimate_emotional_intensity(neutral_item)
        assert intensity < 0.5

    def test_quality_assessment(self):
        """Test quality assessment integration."""
        conversation = self.sample_conversation
        config = self.integrator.dataset_configs[0]

        with patch.object(self.integrator.quality_filter, "assess_conversation_quality") as mock_quality:
            # Test passing quality
            mock_quality.return_value = 0.8
            assert self.integrator._assess_quality(conversation, config) is True

            # Test failing quality
            mock_quality.return_value = 0.5
            assert self.integrator._assess_quality(conversation, config) is False

    def test_therapeutic_accuracy_assessment(self):
        """Test therapeutic accuracy assessment."""
        conversation = self.sample_conversation
        config = self.integrator.dataset_configs[0]

        with patch.object(self.integrator.therapeutic_assessor, "assess_therapeutic_accuracy") as mock_therapeutic:
            # Test passing accuracy
            mock_therapeutic.return_value = 0.85
            assert self.integrator._assess_therapeutic_accuracy(conversation, config) is True

            # Test failing accuracy
            mock_therapeutic.return_value = 0.6
            assert self.integrator._assess_therapeutic_accuracy(conversation, config) is False

    def test_emotional_authenticity_assessment(self):
        """Test emotional authenticity assessment."""
        conversation = self.sample_conversation
        config = self.integrator.dataset_configs[0]

        with patch.object(self.integrator.emotional_assessor, "assess_emotional_authenticity") as mock_emotional:
            # Test passing authenticity
            mock_emotional.return_value = 0.8
            assert self.integrator._assess_emotional_authenticity(conversation, config) is True

            # Test failing authenticity
            mock_emotional.return_value = 0.5
            assert self.integrator._assess_emotional_authenticity(conversation, config) is False

    def test_save_conversations(self):
        """Test conversation saving to JSONL."""
        conversations = [self.sample_conversation, self.sample_conversation]
        output_path = os.path.join(self.temp_dir, "test_conversations.jsonl")

        self.integrator._save_conversations(conversations, output_path)

        assert os.path.exists(output_path)

        # Verify content
        with open(output_path) as f:
            lines = f.readlines()
            assert len(lines) == 2

            for line in lines:
                loaded_conv = json.loads(line.strip())
                assert "conversation" in loaded_conv
                assert "metadata" in loaded_conv

    def test_analyze_conversation_categories(self):
        """Test conversation category analysis."""
        conversations = [
            {"metadata": {"subcategory": "anxiety"}},
            {"metadata": {"subcategory": "depression"}},
            {"metadata": {"subcategory": "anxiety"}},
            {"metadata": {}}  # Missing subcategory
        ]

        categories = self.integrator._analyze_conversation_categories(conversations)

        assert categories["anxiety"] == 2
        assert categories["depression"] == 1
        assert categories["unknown"] == 1

    def test_analyze_therapeutic_approaches(self):
        """Test therapeutic approach analysis."""
        conversations = [
            {"metadata": {"therapeutic_approach": "cognitive_behavioral"}},
            {"metadata": {"therapeutic_approach": "mindfulness_based"}},
            {"metadata": {"therapeutic_approach": "cognitive_behavioral"}},
            {"metadata": {}}  # Missing approach
        ]

        approaches = self.integrator._analyze_therapeutic_approaches(conversations)

        assert approaches["cognitive_behavioral"] == 2
        assert approaches["mindfulness_based"] == 1
        assert approaches["unknown"] == 1

    def test_analyze_emotional_intensity(self):
        """Test emotional intensity analysis."""
        conversations = [
            {"metadata": {"emotional_intensity": 0.2}},  # low
            {"metadata": {"emotional_intensity": 0.5}},  # medium
            {"metadata": {"emotional_intensity": 0.8}},  # high
            {"metadata": {"emotional_intensity": 0.6}},  # medium
            {"metadata": {}}  # missing intensity (defaults to 0.0)
        ]

        intensity_dist = self.integrator._analyze_emotional_intensity(conversations)

        assert intensity_dist["low"] == 2  # 0.2 and missing (0.0)
        assert intensity_dist["medium"] == 2  # 0.5 and 0.6
        assert intensity_dist["high"] == 1   # 0.8

    def test_generate_integration_report(self):
        """Test integration report generation."""
        dataset_results = {
            "test_dataset": {
                "conversations_processed": 100,
                "target": 1000,
                "success": True
            }
        }

        conversations = [self.sample_conversation] * 50
        processing_time = 120.5

        report = self.integrator._generate_integration_report(
            dataset_results, conversations, processing_time
        )

        assert "integration_summary" in report
        assert "quality_metrics" in report
        assert "dataset_results" in report
        assert "conversation_categories" in report
        assert "therapeutic_approaches" in report
        assert "emotional_intensity_distribution" in report
        assert "timestamp" in report
        assert "task" in report

        # Check specific values
        assert report["integration_summary"]["total_conversations"] == 50
        assert report["integration_summary"]["processing_time_seconds"] == 120.5
        assert report["task"] == "5.1-5.5: Mental Health Conversations Integration"

    @patch("ai.dataset_pipeline.mental_health_integrator.load_hf_dataset")
    def test_integrate_all_datasets_partial_success(self, mock_load_hf):
        """Test integration with some datasets failing."""
        # Mock one successful and one failed dataset load
        def side_effect(hf_path):
            if "mental_health_counseling" in hf_path:
                mock_dataset = Mock()
                mock_dataset.to_list.return_value = self.mock_dataset_items[:1]
                return mock_dataset
            return None  # Simulate failure

        mock_load_hf.side_effect = side_effect

        # Mock quality assessments
        with patch.object(self.integrator, "_assess_quality", return_value=True), \
             patch.object(self.integrator, "_assess_therapeutic_accuracy", return_value=True), \
             patch.object(self.integrator, "_assess_emotional_authenticity", return_value=True), \
             patch.object(self.integrator, "_standardize_conversation") as mock_standardize:

            mock_standardize.return_value = self.sample_conversation

            result = self.integrator.integrate_all_datasets()

            assert "integration_summary" in result
            assert result["integration_summary"]["datasets_processed"] >= 1
            assert self.integrator.integration_stats["processing_errors"] >= 1

            # Check that output files were created
            output_path = os.path.join(self.temp_dir, "integrated_mental_health_conversations.jsonl")
            report_path = os.path.join(self.temp_dir, "mental_health_integration_report.json")

            assert os.path.exists(output_path)
            assert os.path.exists(report_path)
