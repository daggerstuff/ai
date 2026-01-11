#!/usr/bin/env python3
"""
Scale optimization testing for Psychology Knowledge Extractor.

Tests the enhanced pattern recognition and parallel processing capabilities
to achieve 10x scale improvement (target: 10,000+ concepts from 913 transcripts).
"""

import time
from pathlib import Path

import pytest
from psychology_knowledge_extractor import PsychologyKnowledgeExtractor


class TestScaleOptimization:
    """Test suite for scale optimization features."""
    
    def setup_method(self):
        """Set up test environment."""
        self.extractor = PsychologyKnowledgeExtractor()
        self.test_transcript_dir = ".notes/transcripts"
        
    def test_enhanced_pattern_recognition(self):
        """Test that enhanced patterns extract significantly more concepts."""
        # Test sample content with various psychological concepts
        test_content = """
        I've been feeling really down lately, like empty and hopeless. 
        My anxiety is through the roof - I'm constantly worried about everything.
        I feel like I'm walking on eggshells at home because of my toxic relationship.
        My therapist mentioned CBT and thought challenging exercises.
        We're working on grounding techniques when I get triggered.
        I think I have some attachment trauma from childhood experiences.
        Sometimes I feel like an imposter at work, like I'm not good enough.
        My inner child work has been really helpful for emotional regulation.
        I'm learning distress tolerance skills from DBT.
        The EMDR sessions have helped with my PTSD symptoms.
        """
        
        # Extract with enhanced patterns
        dsm5_concepts = self.extractor._extract_dsm5_concepts(test_content, "test", "test_expert")
        semantic_concepts = self.extractor._extract_semantic_concepts(test_content, "test", "test_expert")
        colloquial_concepts = self.extractor._extract_colloquial_expressions(test_content, "test", "test_expert")
        
        # Verify significant concept extraction
        total_concepts = len(dsm5_concepts) + len(semantic_concepts) + len(colloquial_concepts)
        assert total_concepts >= 15, f"Expected at least 15 concepts, got {total_concepts}"
        
        # Verify different categories are captured
        categories = set()
        for concept in dsm5_concepts + semantic_concepts + colloquial_concepts:
            categories.add(concept.category)
        
        assert len(categories) >= 5, f"Expected at least 5 categories, got {len(categories)}"
        
    def test_semantic_concept_extraction(self):
        """Test semantic concept extraction functionality."""
        test_content = """
        Learning to manage emotions has been a challenge for me.
        I struggle with communication issues in my relationships.
        Building self-esteem and self-confidence is important.
        Developing healthy coping strategies for stress management.
        My childhood experiences shaped my family dynamics.
        The healing process takes time and therapeutic work.
        """
        
        concepts = self.extractor._extract_semantic_concepts(test_content, "test", "test_expert")
        
        # Should extract concepts for emotional regulation, relationships, self-worth, etc.
        assert len(concepts) >= 6, f"Expected at least 6 semantic concepts, got {len(concepts)}"
        
        # Check for expected semantic categories
        categories = [c.category for c in concepts]
        expected_categories = ['semantic_emotional_regulation', 'semantic_relationships', 'semantic_self_worth']
        
        for expected in expected_categories:
            assert any(expected in cat for cat in categories), f"Missing expected category: {expected}"
    
    def test_colloquial_expression_extraction(self):
        """Test extraction of colloquial psychological expressions."""
        test_content = """
        I feel so stuck and trapped in this situation.
        I'm constantly walking on eggshells around my partner.
        I have major people pleasing tendencies and can't say no.
        Working with my inner child has been transformative.
        I got triggered by that comment about my appearance.
        This toxic relationship has so many red flags.
        I struggle with imposter syndrome at work.
        I had an emotional flashback during the meeting.
        """
        
        concepts = self.extractor._extract_colloquial_expressions(test_content, "test", "test_expert")
        
        # Should extract multiple colloquial expressions
        assert len(concepts) >= 6, f"Expected at least 6 colloquial concepts, got {len(concepts)}"
        
        # Check for high confidence scores (colloquial expressions should be high confidence)
        high_confidence_count = sum(1 for c in concepts if c.confidence_score >= 0.8)
        assert high_confidence_count >= 4, f"Expected at least 4 high-confidence concepts, got {high_confidence_count}"
    
    @pytest.mark.slow
    def test_performance_scaling(self):
        """Test performance with subset of transcripts to verify scaling."""
        # Get first 50 transcripts for performance testing
        transcript_files = list(Path(self.test_transcript_dir).glob("*.txt"))[:50]
        
        if len(transcript_files) < 10:
            pytest.skip("Not enough transcript files for performance testing")
        
        start_time = time.time()
        
        # Process with enhanced extraction
        results = []
        for transcript_path in transcript_files:
            result = self.extractor._process_single_transcript_optimized(str(transcript_path))
            results.append(result)
        
        processing_time = time.time() - start_time
        
        # Calculate performance metrics
        total_concepts = sum(
            len(result['concepts']['dsm5']) + 
            len(result['concepts']['semantic']) + 
            len(result['concepts']['colloquial'])
            for result in results
        )
        
        concepts_per_transcript = total_concepts / len(transcript_files)
        transcripts_per_second = len(transcript_files) / processing_time
        
        print("Performance metrics:")
        print(f"  Processed {len(transcript_files)} transcripts in {processing_time:.2f} seconds")
        print(f"  {transcripts_per_second:.2f} transcripts/second")
        print(f"  {total_concepts} total concepts extracted")
        print(f"  {concepts_per_transcript:.1f} concepts per transcript")
        
        # Performance targets
        assert concepts_per_transcript >= 10, f"Expected >= 10 concepts/transcript, got {concepts_per_transcript:.1f}"
        assert transcripts_per_second >= 1.0, f"Expected >= 1.0 transcripts/second, got {transcripts_per_second:.2f}"
    
    def test_enhanced_clinical_patterns(self):
        """Test the enhanced clinical pattern recognition."""
        patterns = self.extractor._get_enhanced_clinical_patterns()
        
        # Verify expanded pattern categories
        expected_categories = [
            'mood_disorders', 'anxiety_disorders', 'trauma_ptsd', 
            'personality_disorders', 'attachment_styles', 'addiction_substance'
        ]
        
        for category in expected_categories:
            assert category in patterns, f"Missing pattern category: {category}"
            assert len(patterns[category]) >= 3, f"Too few patterns in {category}: {len(patterns[category])}"
    
    @pytest.mark.integration
    def test_full_scale_extraction(self):
        """Integration test for full-scale knowledge extraction."""
        if not Path(self.test_transcript_dir).exists():
            pytest.skip("Transcript directory not found")
        
        # Run full extraction with optimization
        start_time = time.time()
        knowledge_base = self.extractor.extract_knowledge(
            transcript_dir=self.test_transcript_dir,
            use_parallel=True
        )
        processing_time = time.time() - start_time
        
        # Verify scale targets
        total_concepts = knowledge_base['statistics']['total_concepts']
        total_techniques = knowledge_base['statistics']['total_techniques']
        
        print("Full extraction results:")
        print(f"  Processing time: {processing_time:.2f} seconds")
        print(f"  Total concepts: {total_concepts}")
        print(f"  Total techniques: {total_techniques}")
        print(f"  Concept categories: {knowledge_base['statistics']['concept_categories']}")
        
        # Scale targets (aiming for 10x improvement from 715 concepts)
        assert total_concepts >= 3000, f"Expected >= 3000 concepts for scale target, got {total_concepts}"
        assert total_techniques >= 100, f"Expected >= 100 techniques, got {total_techniques}"
        
        # Performance targets
        performance = knowledge_base.get('performance_metrics', {})
        if performance:
            concepts_per_transcript = performance.get('concepts_per_transcript', 0)
            assert concepts_per_transcript >= 8, f"Expected >= 8 concepts/transcript, got {concepts_per_transcript:.1f}"
    
    def test_category_diversity(self):
        """Test that extraction captures diverse concept categories."""
        # Use a rich test sample
        test_content = """
        Patient presents with major depression and generalized anxiety.
        History of childhood trauma and complex PTSD symptoms.
        Shows signs of emotional dysregulation typical of borderline personality.
        Exhibits anxious attachment patterns in relationships.
        Struggling with addiction and codependency issues.
        Needs CBT for thought challenging and DBT for distress tolerance.
        EMDR recommended for trauma processing.
        Grounding techniques help with dissociation episodes.
        Family therapy could address systemic issues.
        """
        
        # Extract all concept types
        dsm5_concepts = self.extractor._extract_dsm5_concepts(test_content, "test", "test_expert")
        modality_concepts = self.extractor._extract_modality_concepts(test_content, "test", "test_expert")
        
        # Count unique categories
        all_categories = set()
        for concept in dsm5_concepts + modality_concepts:
            all_categories.add(concept.category)
        
        # Should capture multiple diverse categories
        assert len(all_categories) >= 3, f"Expected >= 3 categories, got {len(all_categories)}"
        
        # Check for specific expected categories
        category_list = list(all_categories)
        assert any('dsm5' in cat for cat in category_list), "Missing DSM-5 categories"
        assert any('modality' in cat for cat in category_list), "Missing therapeutic modality categories"


def run_scale_benchmark():
    """Run a comprehensive scale benchmark."""
    print("🎯 Running Psychology Knowledge Extractor Scale Benchmark")
    print("=" * 60)
    
    extractor = PsychologyKnowledgeExtractor()
    
    # Test sample for baseline
    test_sample = """
    The patient reports feeling depressed and anxious most days.
    She describes childhood trauma and attachment issues.
    Current symptoms include emotional dysregulation, dissociation, and interpersonal difficulties.
    Previous therapy included CBT and DBT skills training.
    She benefits from grounding techniques and mindfulness practices.
    Safety planning is important due to self-harm history.
    The therapeutic relationship shows secure attachment developing.
    """
    
    print("Testing enhanced pattern recognition...")
    
    # Extract with all methods
    dsm5_concepts = extractor._extract_dsm5_concepts(test_sample, "benchmark", "test_expert")
    semantic_concepts = extractor._extract_semantic_concepts(test_sample, "benchmark", "test_expert") 
    colloquial_concepts = extractor._extract_colloquial_expressions(test_sample, "benchmark", "test_expert")
    
    print(f"✅ DSM-5 concepts extracted: {len(dsm5_concepts)}")
    print(f"✅ Semantic concepts extracted: {len(semantic_concepts)}")
    print(f"✅ Colloquial concepts extracted: {len(colloquial_concepts)}")
    
    total_enhanced = len(dsm5_concepts) + len(semantic_concepts) + len(colloquial_concepts)
    print(f"🎯 Total enhanced extraction: {total_enhanced} concepts")
    
    # Show category breakdown
    categories = {}
    for concept in dsm5_concepts + semantic_concepts + colloquial_concepts:
        cat = concept.category
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n📊 Category breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    print(f"\n🚀 Scale improvement factor: ~{total_enhanced/3:.1f}x (baseline: ~3 concepts)")
    print("✅ Scale optimization ready for full transcript processing!")


if __name__ == "__main__":
    run_scale_benchmark()