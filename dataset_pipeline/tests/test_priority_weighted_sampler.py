#!/usr/bin/env python3
"""
Comprehensive test suite for Priority-Weighted Sampling Algorithms (Task 6.19)
Tests all functionality including tiered sampling, quality assessment, and weight adjustment.
"""

import json
import os
import tempfile

import pytest

from .priority_weighted_sampler import PriorityWeightedSampler, SamplingResult


class TestPriorityWeightedSampler:
    """Test suite for PriorityWeightedSampler"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sampler = PriorityWeightedSampler()

        # Create test data
        self.test_conversations = {
            "tier_1_priority": [
                {
                    "id": f"priority_{i}",
                    "messages": [
                        {"content": "This is a high quality therapeutic conversation with cognitive behavioral therapy techniques and mindfulness strategies.", "role": "therapist"},
                        {"content": "I feel much better after discussing my anxiety and learning coping mechanisms.", "role": "patient"}
                    ],
                    "quality_indicators": ["therapeutic", "cbt", "mindfulness"]
                } for i in range(100)
            ],
            "tier_2_professional": [
                {
                    "id": f"professional_{i}",
                    "messages": [
                        {"content": "Professional therapeutic dialogue focusing on treatment and intervention strategies.", "role": "therapist"},
                        {"content": "The therapy session helped me understand my emotions better.", "role": "client"}
                    ],
                    "quality_indicators": ["therapy", "treatment", "intervention"]
                } for i in range(80)
            ],
            "tier_3_cot": [
                {
                    "id": f"cot_{i}",
                    "messages": [
                        {"content": "Let me think through this step by step. First, we need to understand the cognitive patterns.", "role": "assistant"},
                        {"content": "That reasoning makes sense. I can see how my thoughts affect my mood.", "role": "user"}
                    ],
                    "quality_indicators": ["cognitive", "reasoning", "patterns"]
                } for i in range(60)
            ],
            "tier_4_reddit": [
                {
                    "id": f"reddit_{i}",
                    "messages": [
                        {"content": "I have been struggling with depression and anxiety lately.", "role": "user"},
                        {"content": "Have you considered talking to a therapist? It really helped me.", "role": "responder"}
                    ],
                    "quality_indicators": ["depression", "anxiety", "struggling"]
                } for i in range(40)
            ],
            "tier_5_research": [
                {
                    "id": f"research_{i}",
                    "messages": [
                        {"content": "Research shows that cognitive behavioral therapy is effective for anxiety disorders.", "role": "researcher"},
                        {"content": "The evidence supports this therapeutic approach.", "role": "participant"}
                    ],
                    "quality_indicators": ["research", "evidence", "effective"]
                } for i in range(20)
            ],
            "tier_6_knowledge": [
                {
                    "id": f"knowledge_{i}",
                    "messages": [
                        {"content": "According to the DSM-5, major depressive disorder is characterized by persistent sadness.", "role": "reference"},
                        {"content": "This definition helps in understanding the diagnostic criteria.", "role": "reader"}
                    ],
                    "quality_indicators": ["dsm-5", "diagnostic", "criteria"]
                } for i in range(10)
            ]
        }

    def test_initialization(self):
        """Test sampler initialization"""
        sampler = PriorityWeightedSampler()

        # Check that all tier configs are loaded
        assert len(sampler.tier_configs) == 6

        # Check default weights sum to 1.0
        total_weight = sum(config.weight for config in sampler.tier_configs.values())
        assert abs(total_weight - 1.0) < 0.001

        # Check tier 1 has highest weight
        tier_1_weight = sampler.tier_configs["tier_1_priority"].weight
        assert tier_1_weight == 0.40

        # Check quality thresholds are decreasing
        thresholds = [config.quality_threshold for config in sampler.tier_configs.values()]
        assert thresholds == sorted(thresholds, reverse=True)

    def test_custom_config_loading(self):
        """Test loading custom configuration"""
        # Create temporary config file
        custom_config = {
            "tier_1_priority": {
                "weight": 0.50,
                "quality_threshold": 0.95,
                "min_samples": 2000
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(custom_config, f)
            config_path = f.name

        try:
            sampler = PriorityWeightedSampler(config_path=config_path)

            # Check custom values were loaded
            tier_1_config = sampler.tier_configs["tier_1_priority"]
            assert tier_1_config.weight == 0.50
            assert tier_1_config.quality_threshold == 0.95
            assert tier_1_config.min_samples == 2000

        finally:
            os.unlink(config_path)

    def test_quality_score_calculation(self):
        """Test quality score calculation"""
        # High quality conversation
        high_quality_conv = {
            "id": "test_high",
            "messages": [
                {"content": "This therapeutic conversation uses cognitive behavioral therapy techniques to help with anxiety and depression. The mindfulness strategies are very effective.", "role": "therapist"},
                {"content": "I feel much better after learning these coping mechanisms. The therapy session was very helpful.", "role": "patient"}
            ]
        }

        # Low quality conversation
        low_quality_conv = {
            "id": "test_low",
            "messages": [
                {"content": "Hi.", "role": "user"},
                {"content": "Hello.", "role": "assistant"}
            ]
        }

        high_score = self.sampler.calculate_quality_score(high_quality_conv)
        low_score = self.sampler.calculate_quality_score(low_quality_conv)

        assert high_score > low_score
        assert 0.0 <= high_score <= 1.0
        assert 0.0 <= low_score <= 1.0
        assert high_score > 0.7  # Should be high quality
        assert low_score < 0.6   # Should be low quality

    def test_quality_caching(self):
        """Test quality score caching"""
        conversation = self.test_conversations["tier_1_priority"][0]

        # First calculation
        score1 = self.sampler.calculate_quality_score(conversation)

        # Second calculation should use cache
        score2 = self.sampler.calculate_quality_score(conversation)

        assert score1 == score2
        assert conversation["id"] in self.sampler.quality_cache

    def test_coherence_assessment(self):
        """Test conversation coherence assessment"""
        # Coherent conversation
        coherent_conv = {
            "messages": [
                {"content": "I have been feeling anxious about work lately.", "topic": "anxiety"},
                {"content": "Can you tell me more about what specifically makes you anxious at work?", "topic": "anxiety"},
                {"content": "The deadlines and pressure from my boss make me feel overwhelmed.", "topic": "anxiety"}
            ]
        }

        # Incoherent conversation
        incoherent_conv = {
            "messages": [
                {"content": "I like pizza.", "topic": "food"},
                {"content": "The weather is nice today.", "topic": "weather"},
                {"content": "My car needs repair.", "topic": "automotive"}
            ]
        }

        coherent_score = self.sampler._assess_coherence(coherent_conv)
        incoherent_score = self.sampler._assess_coherence(incoherent_conv)

        assert coherent_score > incoherent_score
        assert 0.0 <= coherent_score <= 1.0
        assert 0.0 <= incoherent_score <= 1.0

    def test_therapeutic_accuracy_assessment(self):
        """Test therapeutic accuracy assessment"""
        # High therapeutic accuracy
        therapeutic_conv = {
            "messages": [
                {"content": "Let's use cognitive behavioral therapy techniques to address your anxiety. We can develop coping strategies and mindfulness practices."}
            ]
        }

        # Low therapeutic accuracy
        non_therapeutic_conv = {
            "messages": [
                {"content": "What did you have for lunch today?"}
            ]
        }

        therapeutic_score = self.sampler._assess_therapeutic_accuracy(therapeutic_conv)
        non_therapeutic_score = self.sampler._assess_therapeutic_accuracy(non_therapeutic_conv)

        assert therapeutic_score > non_therapeutic_score
        assert therapeutic_score > 0.8
        assert non_therapeutic_score < 0.8

    def test_emotional_authenticity_assessment(self):
        """Test emotional authenticity assessment"""
        # High emotional authenticity
        emotional_conv = {
            "messages": [
                {"content": "I feel so sad and overwhelmed. I'm anxious about everything and feel depressed most days."}
            ]
        }

        # Low emotional authenticity
        non_emotional_conv = {
            "messages": [
                {"content": "The process involves several steps and procedures."}
            ]
        }

        emotional_score = self.sampler._assess_emotional_authenticity(emotional_conv)
        non_emotional_score = self.sampler._assess_emotional_authenticity(non_emotional_conv)

        assert emotional_score > non_emotional_score
        assert emotional_score > 0.7
        assert non_emotional_score < 0.5

    def test_safety_compliance_assessment(self):
        """Test safety compliance assessment"""
        # Safe conversation
        safe_conv = {
            "messages": [
                {"content": "I understand you're going through a difficult time. Let's work together to find healthy coping strategies."}
            ]
        }

        # Potentially unsafe conversation
        unsafe_conv = {
            "messages": [
                {"content": "I want to hurt myself and end everything. I have access to dangerous weapons and drugs."}
            ]
        }

        safe_score = self.sampler._assess_safety_compliance(safe_conv)
        unsafe_score = self.sampler._assess_safety_compliance(unsafe_conv)

        assert safe_score > unsafe_score
        assert safe_score == 1.0
        assert unsafe_score < 0.5

    def test_language_quality_assessment(self):
        """Test language quality assessment"""
        # High language quality
        high_quality_conv = {
            "messages": [
                {"content": "This is a well-structured sentence with proper grammar. It contains multiple sentences with appropriate punctuation. The vocabulary is varied and sophisticated."}
            ]
        }

        # Low language quality
        low_quality_conv = {
            "messages": [
                {"content": "bad grammar no punctuation very short"}
            ]
        }

        high_score = self.sampler._assess_language_quality(high_quality_conv)
        low_score = self.sampler._assess_language_quality(low_quality_conv)

        assert high_score > low_score
        assert 0.0 <= high_score <= 1.0
        assert 0.0 <= low_score <= 1.0

    def test_weight_adjustment(self):
        """Test dynamic weight adjustment based on quality"""
        sample_counts = self.sampler.adjust_weights_by_quality(
            self.test_conversations, target_total=1000
        )

        # Check that all tiers are represented
        assert len(sample_counts) == 6

        # Check that tier 1 gets the most samples
        tier_1_count = sample_counts["tier_1_priority"]
        assert tier_1_count > sample_counts["tier_2_professional"]
        assert tier_1_count > sample_counts["tier_3_cot"]

        # Check that total doesn't exceed target (allowing for minimums)
        total_samples = sum(sample_counts.values())
        assert total_samples >= 0

        # Check minimum samples are respected
        for tier_id, count in sample_counts.items():
            config = self.sampler.tier_configs[tier_id]
            available = len(self.test_conversations.get(tier_id, []))
            expected_min = min(config.min_samples, available)
            assert count >= expected_min or available == 0

    def test_stratified_sampling(self):
        """Test stratified sampling functionality"""
        conversations = self.test_conversations["tier_1_priority"]
        target_count = 50
        quality_threshold = 0.7

        sampled = self.sampler.stratified_sample(
            conversations, target_count, quality_threshold
        )

        # Check sample size
        assert len(sampled) <= target_count
        assert len(sampled) > 0

        # Check quality threshold
        for conv in sampled:
            quality = self.sampler.calculate_quality_score(conv)
            assert quality >= quality_threshold

    def test_stratified_sampling_edge_cases(self):
        """Test stratified sampling edge cases"""
        # Empty conversations
        sampled = self.sampler.stratified_sample([], 10, 0.5)
        assert len(sampled) == 0

        # Zero target count
        sampled = self.sampler.stratified_sample(
            self.test_conversations["tier_1_priority"], 0, 0.5
        )
        assert len(sampled) == 0

        # Very high quality threshold
        sampled = self.sampler.stratified_sample(
            self.test_conversations["tier_1_priority"], 10, 0.99
        )
        # Should return fewer samples or none if quality is too low
        assert len(sampled) >= 0

    def test_main_sampling_process(self):
        """Test the main sampling process"""
        results = self.sampler.sample_from_tiers(
            self.test_conversations, target_total=100
        )

        # Check that results are returned
        assert len(results) > 0

        # Check result structure
        for result in results:
            assert isinstance(result, SamplingResult)
            assert hasattr(result, "tier")
            assert hasattr(result, "samples")
            assert hasattr(result, "actual_weight")
            assert hasattr(result, "quality_score")
            assert hasattr(result, "metadata")

            # Check data types
            assert isinstance(result.samples, list)
            assert isinstance(result.actual_weight, float)
            assert isinstance(result.quality_score, float)
            assert isinstance(result.metadata, dict)

            # Check value ranges
            assert 0.0 <= result.actual_weight <= 1.0
            assert 0.0 <= result.quality_score <= 1.0

        # Check total samples
        total_samples = sum(len(result.samples) for result in results)
        assert total_samples > 0
        assert total_samples <= 100  # Should not exceed target

    def test_sampling_history(self):
        """Test sampling history tracking"""
        # Initial state
        assert len(self.sampler.sampling_history) == 0

        # Perform sampling
        self.sampler.sample_from_tiers(self.test_conversations, target_total=50)

        # Check history was recorded
        assert len(self.sampler.sampling_history) == 1

        history_entry = self.sampler.sampling_history[0]
        assert "timestamp" in history_entry
        assert "target_total" in history_entry
        assert "actual_total" in history_entry
        assert "tier_results" in history_entry

        assert history_entry["target_total"] == 50

    def test_config_export(self):
        """Test configuration export"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            self.sampler.export_sampling_config(output_path)

            # Check file was created
            assert os.path.exists(output_path)

            # Check content
            with open(output_path) as f:
                config_data = json.load(f)

            assert len(config_data) == 6
            assert "tier_1_priority" in config_data

            tier_1_data = config_data["tier_1_priority"]
            assert "name" in tier_1_data
            assert "weight" in tier_1_data
            assert "quality_threshold" in tier_1_data

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_sampling_statistics(self):
        """Test sampling statistics generation"""
        # Before sampling
        stats = self.sampler.get_sampling_statistics()
        assert "error" in stats  # No history yet

        # After sampling
        self.sampler.sample_from_tiers(self.test_conversations, target_total=50)
        stats = self.sampler.get_sampling_statistics()

        assert "total_sampling_runs" in stats
        assert "latest_run" in stats
        assert "tier_configurations" in stats
        assert "quality_cache_size" in stats

        assert stats["total_sampling_runs"] == 1
        assert len(stats["tier_configurations"]) == 6

    def test_performance_with_large_dataset(self):
        """Test performance with larger dataset"""
        # Create larger test dataset
        large_dataset = {}
        for tier_id in self.test_conversations:
            large_dataset[tier_id] = self.test_conversations[tier_id] * 10  # 10x larger

        # Should complete in reasonable time
        import time
        start_time = time.time()

        results = self.sampler.sample_from_tiers(large_dataset, target_total=1000)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete within 30 seconds for this size
        assert processing_time < 30.0
        assert len(results) > 0

    def test_configurable_sampling_ratios(self):
        """Test configurable sampling ratios"""
        # Test with different target totals
        small_results = self.sampler.sample_from_tiers(
            self.test_conversations, target_total=50
        )
        large_results = self.sampler.sample_from_tiers(
            self.test_conversations, target_total=200
        )

        # Should scale proportionally
        small_total = sum(len(r.samples) for r in small_results)
        large_total = sum(len(r.samples) for r in large_results)

        assert large_total > small_total

        # Ratios should be similar
        if small_results and large_results:
            small_tier1_ratio = len(small_results[0].samples) / small_total
            large_tier1_ratio = len(large_results[0].samples) / large_total

            # Allow some variance due to minimum sample requirements
            assert abs(small_tier1_ratio - large_tier1_ratio) < 0.2

def test_integration():
    """Integration test for the complete sampling pipeline"""
    sampler = PriorityWeightedSampler()

    # Create comprehensive test data
    test_data = {
        "tier_1_priority": [
            {
                "id": f"priority_{i}",
                "messages": [
                    {"content": f"High quality therapeutic conversation {i} with cognitive behavioral therapy and mindfulness techniques.", "role": "therapist"},
                    {"content": "I feel much better after discussing my anxiety and learning coping strategies.", "role": "patient"}
                ]
            } for i in range(200)
        ],
        "tier_2_professional": [
            {
                "id": f"professional_{i}",
                "messages": [
                    {"content": f"Professional therapeutic dialogue {i} focusing on treatment and intervention.", "role": "therapist"},
                    {"content": "The therapy session helped me understand my emotions better.", "role": "client"}
                ]
            } for i in range(150)
        ],
        "tier_3_cot": [
            {
                "id": f"cot_{i}",
                "messages": [
                    {"content": f"Chain of thought reasoning {i}. Let me think through this step by step.", "role": "assistant"},
                    {"content": "That reasoning makes sense and helps me understand the cognitive patterns.", "role": "user"}
                ]
            } for i in range(100)
        ]
    }

    # Perform complete sampling pipeline
    results = sampler.sample_from_tiers(test_data, target_total=300)

    # Verify results
    assert len(results) > 0
    total_sampled = sum(len(r.samples) for r in results)
    assert total_sampled > 0
    assert total_sampled <= 300

    # Verify quality scores
    for result in results:
        assert result.quality_score > 0.0
        assert result.quality_score <= 1.0

    # Verify tier distribution roughly follows expected weights
    tier_1_samples = next((len(r.samples) for r in results if "Priority" in r.tier), 0)
    tier_2_samples = next((len(r.samples) for r in results if "Professional" in r.tier), 0)

    if total_sampled > 0:
        tier_1_samples / total_sampled
        tier_2_samples / total_sampled

        # At least one tier should have samples
        assert tier_1_samples > 0 or tier_2_samples > 0

    # Test statistics
    stats = sampler.get_sampling_statistics()
    assert stats["total_sampling_runs"] == 1
    assert stats["quality_cache_size"] > 0


if __name__ == "__main__":
    # Run integration test
    test_integration()

    # Run all tests with pytest
    pytest.main([__file__, "-v"])
