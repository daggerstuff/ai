"""
Tests for Consolidated Mental Health Dataset Processor
"""

import json
import os
import tempfile
from unittest.mock import mock_open, patch

import pytest
from consolidated_mental_health_processor import (
    ConsolidatedMentalHealthProcessor,
    ConsolidatedProcessingConfig,
)


class TestConsolidatedMentalHealthProcessor:
    """Test suite for ConsolidatedMentalHealthProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ConsolidatedProcessingConfig(
            input_file="test_input.jsonl",
            output_dir=self.temp_dir,
            target_conversations=100,
            batch_size=10,
        )
        self.processor = ConsolidatedMentalHealthProcessor(self.config)

        # Mock sample JSONL data
        self.sample_jsonl_data = [
            {
                "prompt": "You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description. I've been feeling anxious lately and can't sleep.",
                "response": "I understand you're experiencing anxiety and sleep difficulties. Let's explore some strategies that might help you manage these symptoms. First, establishing a regular sleep routine can be beneficial...",
            },
            {
                "prompt": "You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description. I'm having trouble with my relationships and feel isolated.",
                "response": "Relationship difficulties and feelings of isolation can be very challenging. It's important to recognize that you're not alone in experiencing these feelings. Let's work together to identify some strategies...",
            },
            {
                "prompt": "You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description. I'm dealing with depression and lack motivation.",
                "response": "Depression can significantly impact motivation and daily functioning. Thank you for sharing this with me. Let's explore some approaches that might help you regain a sense of purpose and energy...",
            },
        ]

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test ConsolidatedMentalHealthProcessor initialization."""
        assert self.processor.config.input_file == "test_input.jsonl"
        assert self.processor.config.output_dir == self.temp_dir
        assert self.processor.config.target_conversations == 100
        assert os.path.exists(self.temp_dir)

        # Check processing stats initialization
        assert self.processor.processing_stats["total_processed"] == 0
        assert self.processor.processing_stats["total_accepted"] == 0

    def test_config_defaults(self):
        """Test default configuration values."""
        default_processor = ConsolidatedMentalHealthProcessor()

        assert (
            default_processor.config.input_file
            == "ai/merged_mental_health_dataset.jsonl"
        )
        assert default_processor.config.target_conversations == 20000
        assert default_processor.config.quality_threshold == 0.75
        assert default_processor.config.therapeutic_accuracy_threshold == 0.8
        assert default_processor.config.emotional_authenticity_threshold == 0.75
        assert default_processor.config.batch_size == 1000

    def test_extract_client_content(self):
        """Test client content extraction from prompts."""
        # Test with system instruction
        prompt_with_system = "You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description. I'm feeling anxious about work."
        client_content = self.processor._extract_client_content(prompt_with_system)
        assert client_content == "I'm feeling anxious about work."

        # Test with different system instruction
        prompt_with_different_system = "You are a mental health counselor. Help the patient. I can't sleep at night."
        client_content = self.processor._extract_client_content(
            prompt_with_different_system
        )
        assert "I can't sleep at night" in client_content

        # Test without system instruction
        prompt_without_system = "I'm having relationship problems."
        client_content = self.processor._extract_client_content(prompt_without_system)
        assert client_content == "I'm having relationship problems."

    def test_determine_therapeutic_approach(self):
        """Test therapeutic approach determination."""
        # Test cognitive behavioral
        cbt_prompt = "I have negative thoughts"
        cbt_response = "Let's examine your thinking patterns and beliefs"
        approach = self.processor._determine_therapeutic_approach(
            cbt_prompt, cbt_response
        )
        assert approach == "cognitive_behavioral"

        # Test mindfulness based
        mindful_prompt = "I feel overwhelmed"
        mindful_response = "Let's practice mindfulness and present moment awareness"
        approach = self.processor._determine_therapeutic_approach(
            mindful_prompt, mindful_response
        )
        assert approach == "mindfulness_based"

        # Test behavioral
        behavioral_prompt = "I avoid social situations"
        behavioral_response = (
            "We can work on gradual exposure and behavior modification"
        )
        approach = self.processor._determine_therapeutic_approach(
            behavioral_prompt, behavioral_response
        )
        assert approach == "behavioral"

        # Test emotion focused
        emotion_prompt = "I'm struggling with my emotions"
        emotion_response = (
            "Let's validate your feelings and work on emotional regulation"
        )
        approach = self.processor._determine_therapeutic_approach(
            emotion_prompt, emotion_response
        )
        assert approach == "emotion_focused"

        # Test interpersonal
        interpersonal_prompt = "I have relationship issues"
        interpersonal_response = (
            "Let's explore your communication patterns and family dynamics"
        )
        approach = self.processor._determine_therapeutic_approach(
            interpersonal_prompt, interpersonal_response
        )
        assert approach == "interpersonal"

        # Test integrative (default)
        general_prompt = "I need help"
        general_response = "I'm here to support you"
        approach = self.processor._determine_therapeutic_approach(
            general_prompt, general_response
        )
        assert approach == "integrative"

    def test_estimate_emotional_intensity(self):
        """Test emotional intensity estimation."""
        # High intensity
        high_prompt = "I'm having suicidal thoughts and this is a crisis"
        high_response = "This is a serious emergency situation"
        intensity = self.processor._estimate_emotional_intensity(
            high_prompt, high_response
        )
        assert intensity >= 0.8

        # Medium intensity
        medium_prompt = "I'm feeling anxious and depressed"
        medium_response = "These are common symptoms we can work on"
        intensity = self.processor._estimate_emotional_intensity(
            medium_prompt, medium_response
        )
        assert 0.5 <= intensity < 0.8

        # Low intensity
        low_prompt = "I'm a bit concerned about my stress levels"
        low_response = "It's good that you're being proactive"
        intensity = self.processor._estimate_emotional_intensity(
            low_prompt, low_response
        )
        assert intensity < 0.5

        # Default intensity
        neutral_prompt = "I want to improve myself"
        neutral_response = "That's a great goal to have"
        intensity = self.processor._estimate_emotional_intensity(
            neutral_prompt, neutral_response
        )
        assert intensity == 0.4

    def test_standardize_consolidated_conversation(self):
        """Test conversation standardization."""
        item = self.sample_jsonl_data[0]

        result = self.processor._standardize_consolidated_conversation(item, 1)

        assert result is not None
        assert "conversation" in result
        assert "metadata" in result
        assert len(result["conversation"]) == 2
        assert result["conversation"][0]["role"] == "client"
        assert result["conversation"][1]["role"] == "therapist"
        assert result["metadata"]["category"] == "mental_health"
        assert result["metadata"]["subcategory"] == "consolidated_counseling"
        assert "therapeutic_approach" in result["metadata"]
        assert "emotional_intensity" in result["metadata"]

    def test_standardize_conversation_missing_fields(self):
        """Test standardization with missing fields."""
        # Missing response
        item_missing_response = {"prompt": "I need help"}
        result = self.processor._standardize_consolidated_conversation(
            item_missing_response, 1
        )
        assert result is None

        # Missing prompt
        item_missing_prompt = {"response": "I'm here to help"}
        result = self.processor._standardize_consolidated_conversation(
            item_missing_prompt, 1
        )
        assert result is None

        # Empty fields
        item_empty = {"prompt": "", "response": ""}
        result = self.processor._standardize_consolidated_conversation(item_empty, 1)
        assert result is None

    def test_process_batch(self):
        """Test batch processing."""
        batch = [(i + 1, item) for i, item in enumerate(self.sample_jsonl_data)]

        # Mock quality assessments to return True
        with (
            patch.object(self.processor, "_assess_quality", return_value=True),
            patch.object(
                self.processor, "_assess_therapeutic_accuracy", return_value=True
            ),
            patch.object(
                self.processor, "_assess_emotional_authenticity", return_value=True
            ),
        ):

            results = self.processor._process_batch(batch)

            assert len(results) == 3
            assert all("metadata" in conv for conv in results)
            assert all(
                conv["metadata"]["source_dataset"] == "consolidated_mental_health"
                for conv in results
            )
            assert self.processor.processing_stats["total_accepted"] == 3
            assert self.processor.processing_stats["total_processed"] == 3

    def test_process_batch_with_filtering(self):
        """Test batch processing with quality filtering."""
        batch = [(i + 1, item) for i, item in enumerate(self.sample_jsonl_data)]

        # Mock assessments to filter some conversations
        with (
            patch.object(
                self.processor, "_assess_quality", side_effect=[True, False, True]
            ),
            patch.object(
                self.processor, "_assess_therapeutic_accuracy", return_value=True
            ),
            patch.object(
                self.processor, "_assess_emotional_authenticity", return_value=True
            ),
        ):

            results = self.processor._process_batch(batch)

            assert len(results) == 2  # One filtered out by quality
            assert self.processor.processing_stats["quality_filtered"] == 1
            assert self.processor.processing_stats["total_accepted"] == 2

    def test_quality_assessment(self):
        """Test quality assessment integration."""
        conversation = {
            "conversation": [
                {"role": "client", "content": "I'm feeling anxious"},
                {"role": "therapist", "content": "Let's work on managing your anxiety"},
            ],
            "metadata": {"category": "mental_health"},
        }

        with patch.object(
            self.processor.quality_filter, "assess_conversation_quality"
        ) as mock_quality:
            # Test passing quality
            mock_quality.return_value = 0.8
            assert self.processor._assess_quality(conversation) is True

            # Test failing quality
            mock_quality.return_value = 0.6
            assert self.processor._assess_quality(conversation) is False

    def test_therapeutic_accuracy_assessment(self):
        """Test therapeutic accuracy assessment."""
        conversation = {
            "conversation": [
                {"role": "client", "content": "I'm depressed"},
                {"role": "therapist", "content": "Let's explore treatment options"},
            ],
            "metadata": {"category": "mental_health"},
        }

        with patch.object(
            self.processor.therapeutic_assessor, "assess_therapeutic_accuracy"
        ) as mock_therapeutic:
            # Test passing accuracy
            mock_therapeutic.return_value = 0.85
            assert self.processor._assess_therapeutic_accuracy(conversation) is True

            # Test failing accuracy
            mock_therapeutic.return_value = 0.7
            assert self.processor._assess_therapeutic_accuracy(conversation) is False

    def test_emotional_authenticity_assessment(self):
        """Test emotional authenticity assessment."""
        conversation = {
            "conversation": [
                {"role": "client", "content": "I feel sad"},
                {"role": "therapist", "content": "I understand your sadness"},
            ],
            "metadata": {"category": "mental_health"},
        }

        with patch.object(
            self.processor.emotional_assessor, "assess_emotional_authenticity"
        ) as mock_emotional:
            # Test passing authenticity
            mock_emotional.return_value = 0.8
            assert self.processor._assess_emotional_authenticity(conversation) is True

            # Test failing authenticity
            mock_emotional.return_value = 0.6
            assert self.processor._assess_emotional_authenticity(conversation) is False

    def test_save_conversations(self):
        """Test conversation saving to JSONL."""
        conversations = [
            {
                "conversation": [{"role": "client", "content": "test"}],
                "metadata": {"category": "mental_health"},
            }
        ]
        output_path = os.path.join(self.temp_dir, "test_output.jsonl")

        self.processor._save_conversations(conversations, output_path)

        assert os.path.exists(output_path)

        # Verify content
        with open(output_path) as f:
            lines = f.readlines()
            assert len(lines) == 1
            loaded_conv = json.loads(lines[0].strip())
            assert "conversation" in loaded_conv
            assert "metadata" in loaded_conv

    def test_analyze_therapeutic_approaches(self):
        """Test therapeutic approach analysis."""
        conversations = [
            {"metadata": {"therapeutic_approach": "cognitive_behavioral"}},
            {"metadata": {"therapeutic_approach": "mindfulness_based"}},
            {"metadata": {"therapeutic_approach": "cognitive_behavioral"}},
            {"metadata": {}},  # Missing approach
        ]

        approaches = self.processor._analyze_therapeutic_approaches(conversations)

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
            {"metadata": {}},  # missing intensity (defaults to 0.0)
        ]

        intensity_dist = self.processor._analyze_emotional_intensity(conversations)

        assert intensity_dist["low"] == 2  # 0.2 and missing (0.0)
        assert intensity_dist["medium"] == 2  # 0.5 and 0.6
        assert intensity_dist["high"] == 1  # 0.8

    def test_analyze_conversation_lengths(self):
        """Test conversation length analysis."""
        conversations = [
            {"metadata": {"conversation_length": 2}},
            {"metadata": {"conversation_length": 4}},
            {"metadata": {"conversation_length": 6}},
            {"metadata": {}},  # missing length (defaults to 0)
        ]

        length_stats = self.processor._analyze_conversation_lengths(conversations)

        assert length_stats["min"] == 0
        assert length_stats["max"] == 6
        assert length_stats["average"] == 3.0  # (2+4+6+0)/4

    def test_analyze_content_lengths(self):
        """Test content length analysis."""
        conversations = [
            {
                "metadata": {
                    "client_content_length": 50,
                    "therapist_response_length": 200,
                }
            },
            {
                "metadata": {
                    "client_content_length": 30,
                    "therapist_response_length": 150,
                }
            },
            {"metadata": {}},  # missing lengths
        ]

        content_stats = self.processor._analyze_content_lengths(conversations)

        assert "client_content" in content_stats
        assert "therapist_response" in content_stats
        assert content_stats["client_content"]["average"] == (50 + 30 + 0) / 3
        assert content_stats["therapist_response"]["average"] == (200 + 150 + 0) / 3

    def test_generate_processing_report(self):
        """Test processing report generation."""
        conversations = [
            {
                "metadata": {
                    "therapeutic_approach": "cognitive_behavioral",
                    "emotional_intensity": 0.6,
                    "conversation_length": 2,
                    "client_content_length": 50,
                    "therapist_response_length": 200,
                }
            }
        ]
        processing_time = 120.5

        report = self.processor._generate_processing_report(
            conversations, processing_time
        )

        assert "processing_summary" in report
        assert "quality_metrics" in report
        assert "conversation_analysis" in report
        assert "configuration" in report
        assert "timestamp" in report
        assert "task" in report

        # Check specific values
        assert report["processing_summary"]["total_conversations"] == 1
        assert report["processing_summary"]["processing_time_seconds"] == 120.5
        assert (
            report["task"]
            == "5.1: Process existing consolidated mental health dataset (86MB JSONL)"
        )

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    def test_process_consolidated_dataset_file_not_found(self, mock_exists, mock_file):
        """Test handling of missing input file."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError, match="Consolidated dataset not found"):
            self.processor.process_consolidated_dataset()

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    def test_process_consolidated_dataset_success(self, mock_exists, mock_file):
        """Test successful dataset processing."""
        mock_exists.return_value = True

        # Mock file content
        jsonl_lines = [json.dumps(item) + "\n" for item in self.sample_jsonl_data]
        mock_file.return_value.read_data = "".join(jsonl_lines)
        mock_file.return_value.__iter__ = lambda self: iter(jsonl_lines)

        # Mock quality assessments
        with (
            patch.object(self.processor, "_assess_quality", return_value=True),
            patch.object(
                self.processor, "_assess_therapeutic_accuracy", return_value=True
            ),
            patch.object(
                self.processor, "_assess_emotional_authenticity", return_value=True
            ),
        ):

            result = self.processor.process_consolidated_dataset()

            assert "processing_summary" in result
            assert result["processing_summary"]["total_conversations"] == 3
            assert self.processor.processing_stats["total_accepted"] == 3
