"""
Tests for Reasoning Enhancement Dataset Processor
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from ai.dataset_pipeline.reasoning_dataset_processor import (
    ReasoningDatasetConfig,
    ReasoningDatasetProcessor,
)


class TestReasoningDatasetProcessor:
    """Test suite for ReasoningDatasetProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = ReasoningDatasetProcessor(output_dir=self.temp_dir)

        # Mock sample reasoning conversation data
        self.sample_reasoning_conversation = {
            "conversation": [
                {"role": "client", "content": "I'm having trouble concentrating at work"},
                {"role": "therapist", "content": "Let's think through this step by step. First, can you identify when this started? Second, what specific situations trigger it? Third, what symptoms do you notice?"}
            ],
            "metadata": {
                "category": "reasoning_enhancement",
                "subcategory": "clinical_diagnosis",
                "reasoning_patterns": ["symptom_identification", "assessment_planning"],
                "reasoning_complexity": 0.8
            }
        }

        # Mock reasoning dataset items
        self.mock_reasoning_items = [
            {
                "input": "Patient presents with concentration issues",
                "output": "First, I need to assess the onset and duration. Second, identify potential causes like stress, sleep, or medical conditions. Third, evaluate impact on daily functioning. Therefore, I recommend a comprehensive assessment.",
                "reasoning_type": "clinical_diagnosis"
            },
            {
                "question": "How to support a neurodivergent client?",
                "answer": "Initially, understand their specific needs. Then, adapt communication style. Next, provide sensory accommodations. Finally, focus on their strengths and abilities.",
                "category": "neurodiversity"
            },
            {
                "scenario": "Client experiencing heartbreak",
                "response": "First, validate their emotional experience. Second, help them process the grief stages. Third, identify coping strategies. Subsequently, work on rebuilding self-worth.",
                "emotion_focus": "grief_processing"
            }
        ]

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test ReasoningDatasetProcessor initialization."""
        assert self.processor.output_dir == self.temp_dir
        assert len(self.processor.dataset_configs) == 4
        assert os.path.exists(self.temp_dir)

        # Check dataset configurations
        config_names = [config.name for config in self.processor.dataset_configs]
        expected_names = ["clinical_diagnosis_cot", "neurodivergent_cot", "heartbreak_cot", "mens_mental_health_cot"]
        assert all(name in config_names for name in expected_names)

        # Check reasoning types
        reasoning_types = [config.reasoning_type for config in self.processor.dataset_configs]
        expected_types = ["clinical_diagnosis", "neurodiversity_awareness", "emotional_intelligence", "gender_specific_mental_health"]
        assert all(rtype in reasoning_types for rtype in expected_types)

    def test_reasoning_dataset_config_creation(self):
        """Test ReasoningDatasetConfig creation."""
        config = ReasoningDatasetConfig(
            name="test_reasoning",
            hf_path="test/reasoning_dataset",
            target_conversations=500,
            priority=1,
            reasoning_type="clinical_diagnosis",
            quality_threshold=0.8
        )

        assert config.name == "test_reasoning"
        assert config.hf_path == "test/reasoning_dataset"
        assert config.target_conversations == 500
        assert config.priority == 1
        assert config.reasoning_type == "clinical_diagnosis"
        assert config.quality_threshold == 0.8
        assert config.clinical_accuracy_threshold == 0.8  # default
        assert config.reasoning_complexity_threshold == 0.7  # default

    def test_extract_reasoning_patterns_clinical(self):
        """Test reasoning pattern extraction for clinical diagnosis."""
        item = {
            "content": "The patient presents with symptoms of depression. I need to conduct a differential diagnosis to rule out other conditions. My assessment plan includes standardized tests and risk evaluation."
        }

        patterns = self.processor._extract_reasoning_patterns(item, "clinical_diagnosis")

        assert "symptom_identification" in patterns
        assert "differential_diagnosis" in patterns
        assert "assessment_planning" in patterns
        assert "risk_assessment" in patterns

    def test_extract_reasoning_patterns_neurodiversity(self):
        """Test reasoning pattern extraction for neurodiversity awareness."""
        item = {
            "content": "For this neurodivergent client, I need to provide communication adaptations and sensory accommodations. Let's focus on their executive function support and identify their unique strengths."
        }

        patterns = self.processor._extract_reasoning_patterns(item, "neurodiversity_awareness")

        assert "communication_adaptation" in patterns
        assert "sensory_considerations" in patterns
        assert "executive_function_support" in patterns
        assert "strength_identification" in patterns

    def test_extract_reasoning_patterns_emotional(self):
        """Test reasoning pattern extraction for emotional intelligence."""
        item = {
            "content": "I notice the client is struggling with emotion recognition and emotional regulation. We need to work on empathy development and self-awareness to improve their relationship management skills."
        }

        patterns = self.processor._extract_reasoning_patterns(item, "emotional_intelligence")

        assert "emotion_recognition" in patterns
        assert "emotional_regulation" in patterns
        assert "empathy_demonstration" in patterns
        assert "self_awareness" in patterns
        assert "relationship_management" in patterns

    def test_extract_reasoning_patterns_gender_specific(self):
        """Test reasoning pattern extraction for gender-specific mental health."""
        item = {
            "content": "This client faces societal pressure related to gender roles. We need cultural sensitivity and identity validation while building their support system and identifying appropriate resources."
        }

        patterns = self.processor._extract_reasoning_patterns(item, "gender_specific_mental_health")

        assert "societal_pressure_recognition" in patterns
        assert "gender_role_awareness" in patterns
        assert "cultural_sensitivity" in patterns
        assert "identity_validation" in patterns
        assert "support_system_building" in patterns
        assert "resource_identification" in patterns

    def test_calculate_reasoning_complexity(self):
        """Test reasoning complexity calculation."""
        # Simple patterns
        simple_patterns = ["general_reasoning"]
        complexity = self.processor._calculate_reasoning_complexity(simple_patterns)
        assert 0.1 <= complexity <= 0.4

        # Multiple patterns
        multiple_patterns = ["symptom_identification", "assessment_planning", "treatment_recommendation"]
        complexity = self.processor._calculate_reasoning_complexity(multiple_patterns)
        assert 0.5 <= complexity <= 0.8

        # High complexity patterns
        high_complexity_patterns = ["differential_diagnosis", "risk_assessment", "emotional_regulation"]
        complexity = self.processor._calculate_reasoning_complexity(high_complexity_patterns)
        assert complexity >= 0.8

        # Empty patterns
        empty_patterns = []
        complexity = self.processor._calculate_reasoning_complexity(empty_patterns)
        assert complexity == 0.3

    def test_extract_cot_steps(self):
        """Test chain-of-thought step extraction."""
        item = {
            "content": "First, I need to assess the symptoms. Second, consider differential diagnoses. Then, I will develop a treatment plan. Therefore, the next step is to implement interventions."
        }

        steps = self.processor._extract_cot_steps(item)

        assert len(steps) >= 3
        assert any("first" in step for step in steps)
        assert any("second" in step for step in steps)
        assert any("then" in step for step in steps)

    def test_assess_clinical_relevance(self):
        """Test clinical relevance assessment."""
        # High clinical relevance
        clinical_item = {
            "content": "The patient presents with symptoms of major depressive disorder. Clinical assessment reveals therapeutic intervention needs with psychiatric evaluation and counseling treatment."
        }
        relevance = self.processor._assess_clinical_relevance(clinical_item, "clinical_diagnosis")
        assert relevance >= 0.7

        # Medium clinical relevance
        medium_item = {
            "content": "The client is experiencing some emotional difficulties and may benefit from therapy sessions."
        }
        relevance = self.processor._assess_clinical_relevance(medium_item, "emotional_intelligence")
        assert 0.3 <= relevance < 0.7

        # Low clinical relevance
        low_item = {
            "content": "Today is a nice day and I feel good about life in general."
        }
        relevance = self.processor._assess_clinical_relevance(low_item, "emotional_intelligence")
        assert relevance <= 0.5

    def test_standardize_reasoning_conversation(self):
        """Test reasoning conversation standardization."""
        item = self.mock_reasoning_items[0]
        config = self.processor.dataset_configs[0]  # clinical_diagnosis_cot

        with patch.object(self.processor, "_detect_and_convert_format") as mock_convert:
            # Mock conversation object with messages
            mock_conversation = Mock()
            mock_conversation.messages = [
                Mock(role="client", content="Patient presents with concentration issues"),
                Mock(role="therapist", content="First, I need to assess the onset...")
            ]
            mock_convert.return_value = mock_conversation

            result = self.processor._standardize_reasoning_conversation(item, config)

            assert result is not None
            assert "metadata" in result
            assert result["metadata"]["category"] == "reasoning_enhancement"
            assert result["metadata"]["subcategory"] == "clinical_diagnosis"
            assert "reasoning_patterns" in result["metadata"]
            assert "reasoning_complexity" in result["metadata"]
            assert "chain_of_thought_steps" in result["metadata"]
            assert "clinical_relevance" in result["metadata"]

    @patch("ai.dataset_pipeline.reasoning_dataset_processor.load_hf_dataset")
    def test_process_single_dataset_success(self, mock_load_hf):
        """Test successful processing of a single reasoning dataset."""
        # Mock dataset loading
        mock_dataset = Mock()
        mock_dataset.to_list.return_value = self.mock_reasoning_items[:2]
        mock_load_hf.return_value = mock_dataset

        # Mock quality assessments to return True
        with patch.object(self.processor, "_assess_quality", return_value=True), \
             patch.object(self.processor, "_assess_clinical_accuracy", return_value=True), \
             patch.object(self.processor, "_assess_reasoning_complexity", return_value=True), \
             patch.object(self.processor, "_standardize_reasoning_conversation") as mock_standardize:

            # Mock standardization to return valid conversations
            mock_standardize.side_effect = [
                {
                    "conversation": [
                        {"role": "client", "content": "Patient presents with concentration issues"},
                        {"role": "therapist", "content": "First, I need to assess..."}
                    ],
                    "metadata": {
                        "category": "reasoning_enhancement",
                        "subcategory": "clinical_diagnosis",
                        "reasoning_patterns": ["symptom_identification"],
                        "reasoning_complexity": 0.8
                    }
                },
                {
                    "conversation": [
                        {"role": "client", "content": "How to support neurodivergent client?"},
                        {"role": "therapist", "content": "Initially, understand their needs..."}
                    ],
                    "metadata": {
                        "category": "reasoning_enhancement",
                        "subcategory": "neurodiversity_awareness",
                        "reasoning_patterns": ["accommodation_strategies"],
                        "reasoning_complexity": 0.7
                    }
                }
            ]

            config = self.processor.dataset_configs[0]  # clinical_diagnosis_cot
            result = self.processor._process_single_dataset(config)

            assert len(result) == 2
            assert all("metadata" in conv for conv in result)
            assert all(conv["metadata"]["source_dataset"] == config.name for conv in result)
            assert all(conv["metadata"]["reasoning_type"] == config.reasoning_type for conv in result)

    @patch("ai.dataset_pipeline.reasoning_dataset_processor.load_hf_dataset")
    def test_process_single_dataset_failure(self, mock_load_hf):
        """Test handling of reasoning dataset loading failure."""
        mock_load_hf.return_value = None

        config = self.processor.dataset_configs[0]

        with pytest.raises(ValueError, match="Failed to load reasoning dataset"):
            self.processor._process_single_dataset(config)

    def test_quality_assessment(self):
        """Test quality assessment integration."""
        conversation = self.sample_reasoning_conversation
        config = self.processor.dataset_configs[0]

        with patch.object(self.processor.quality_filter, "assess_conversation_quality") as mock_quality:
            # Test passing quality
            mock_quality.return_value = 0.8
            assert self.processor._assess_quality(conversation, config) is True

            # Test failing quality
            mock_quality.return_value = 0.6
            assert self.processor._assess_quality(conversation, config) is False

    def test_clinical_accuracy_assessment(self):
        """Test clinical accuracy assessment."""
        conversation = self.sample_reasoning_conversation
        config = self.processor.dataset_configs[0]  # clinical_diagnosis_cot

        with patch.object(self.processor.clinical_validator, "validate_clinical_accuracy") as mock_clinical:
            # Test passing accuracy
            mock_clinical.return_value = 0.85
            assert self.processor._assess_clinical_accuracy(conversation, config) is True

            # Test failing accuracy
            mock_clinical.return_value = 0.7
            assert self.processor._assess_clinical_accuracy(conversation, config) is False

    def test_reasoning_complexity_assessment(self):
        """Test reasoning complexity assessment."""
        # High complexity conversation
        high_complexity_conv = {
            "metadata": {"reasoning_complexity": 0.8}
        }
        config = self.processor.dataset_configs[0]
        assert self.processor._assess_reasoning_complexity(high_complexity_conv, config) is True

        # Low complexity conversation
        low_complexity_conv = {
            "metadata": {"reasoning_complexity": 0.5}
        }
        assert self.processor._assess_reasoning_complexity(low_complexity_conv, config) is False

        # Missing complexity
        missing_complexity_conv = {
            "metadata": {}
        }
        assert self.processor._assess_reasoning_complexity(missing_complexity_conv, config) is False

    def test_save_conversations(self):
        """Test conversation saving to JSONL."""
        conversations = [self.sample_reasoning_conversation, self.sample_reasoning_conversation]
        output_path = os.path.join(self.temp_dir, "test_reasoning_conversations.jsonl")

        self.processor._save_conversations(conversations, output_path)

        assert os.path.exists(output_path)

        # Verify content
        with open(output_path) as f:
            lines = f.readlines()
            assert len(lines) == 2

            for line in lines:
                loaded_conv = json.loads(line.strip())
                assert "conversation" in loaded_conv
                assert "metadata" in loaded_conv
                assert loaded_conv["metadata"]["category"] == "reasoning_enhancement"

    def test_analyze_reasoning_types(self):
        """Test reasoning type analysis."""
        conversations = [
            {"metadata": {"reasoning_type": "clinical_diagnosis"}},
            {"metadata": {"reasoning_type": "neurodiversity_awareness"}},
            {"metadata": {"reasoning_type": "clinical_diagnosis"}},
            {"metadata": {}}  # Missing reasoning type
        ]

        types = self.processor._analyze_reasoning_types(conversations)

        assert types["clinical_diagnosis"] == 2
        assert types["neurodiversity_awareness"] == 1
        assert types["unknown"] == 1

    def test_analyze_reasoning_patterns(self):
        """Test reasoning pattern analysis."""
        conversations = [
            {"metadata": {"reasoning_patterns": ["symptom_identification", "assessment_planning"]}},
            {"metadata": {"reasoning_patterns": ["empathy_demonstration"]}},
            {"metadata": {"reasoning_patterns": ["symptom_identification", "risk_assessment"]}},
            {"metadata": {}}  # Missing patterns
        ]

        patterns = self.processor._analyze_reasoning_patterns(conversations)

        assert patterns["symptom_identification"] == 2
        assert patterns["assessment_planning"] == 1
        assert patterns["empathy_demonstration"] == 1
        assert patterns["risk_assessment"] == 1

    def test_analyze_complexity_distribution(self):
        """Test complexity distribution analysis."""
        conversations = [
            {"metadata": {"reasoning_complexity": 0.2}},  # low
            {"metadata": {"reasoning_complexity": 0.5}},  # medium
            {"metadata": {"reasoning_complexity": 0.8}},  # high
            {"metadata": {"reasoning_complexity": 0.6}},  # medium
            {"metadata": {}}  # missing complexity (defaults to 0.0)
        ]

        complexity_dist = self.processor._analyze_complexity_distribution(conversations)

        assert complexity_dist["low"] == 2  # 0.2 and missing (0.0)
        assert complexity_dist["medium"] == 2  # 0.5 and 0.6
        assert complexity_dist["high"] == 1   # 0.8

    def test_analyze_clinical_relevance(self):
        """Test clinical relevance analysis."""
        conversations = [
            {"metadata": {"clinical_relevance": 0.3}},  # low
            {"metadata": {"clinical_relevance": 0.6}},  # medium
            {"metadata": {"clinical_relevance": 0.9}},  # high
            {"metadata": {"clinical_relevance": 0.5}},  # medium
            {"metadata": {}}  # missing relevance (defaults to 0.0)
        ]

        relevance_dist = self.processor._analyze_clinical_relevance(conversations)

        assert relevance_dist["low"] == 2   # 0.3 and missing (0.0)
        assert relevance_dist["medium"] == 2 # 0.6 and 0.5
        assert relevance_dist["high"] == 1   # 0.9

    def test_generate_processing_report(self):
        """Test processing report generation."""
        dataset_results = {
            "clinical_diagnosis_cot": {
                "conversations_processed": 100,
                "target": 2000,
                "reasoning_type": "clinical_diagnosis",
                "success": True
            }
        }

        conversations = [self.sample_reasoning_conversation] * 50
        processing_time = 180.5

        report = self.processor._generate_processing_report(
            dataset_results, conversations, processing_time
        )

        assert "processing_summary" in report
        assert "quality_metrics" in report
        assert "dataset_results" in report
        assert "reasoning_types" in report
        assert "reasoning_patterns" in report
        assert "complexity_distribution" in report
        assert "clinical_relevance_distribution" in report
        assert "timestamp" in report
        assert "task" in report

        # Check specific values
        assert report["processing_summary"]["total_conversations"] == 50
        assert report["processing_summary"]["target_conversations"] == 15000
        assert report["processing_summary"]["processing_time_seconds"] == 180.5
        assert report["task"] == "5.13-5.24: Reasoning Enhancement Integration"

    @patch("ai.dataset_pipeline.reasoning_dataset_processor.load_hf_dataset")
    def test_process_all_datasets_partial_success(self, mock_load_hf):
        """Test processing with some datasets failing."""
        # Mock one successful and others failed
        def side_effect(hf_path):
            if "clinical_diagnosis" in hf_path:
                mock_dataset = Mock()
                mock_dataset.to_list.return_value = self.mock_reasoning_items[:1]
                return mock_dataset
            return None  # Simulate failure

        mock_load_hf.side_effect = side_effect

        # Mock quality assessments
        with patch.object(self.processor, "_assess_quality", return_value=True), \
             patch.object(self.processor, "_assess_clinical_accuracy", return_value=True), \
             patch.object(self.processor, "_assess_reasoning_complexity", return_value=True), \
             patch.object(self.processor, "_standardize_reasoning_conversation") as mock_standardize:

            mock_standardize.return_value = self.sample_reasoning_conversation

            result = self.processor.process_all_datasets()

            assert "processing_summary" in result
            assert result["processing_summary"]["datasets_processed"] >= 1
            assert self.processor.processing_stats["processing_errors"] >= 1

            # Check that output files were created
            output_path = os.path.join(self.temp_dir, "processed_reasoning_conversations.jsonl")
            report_path = os.path.join(self.temp_dir, "reasoning_processing_report.json")

            assert os.path.exists(output_path)
            assert os.path.exists(report_path)
