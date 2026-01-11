"""
Test suite for Psychology Knowledge Extractor

Tests the extraction capabilities on sample transcripts and validates
the clinical concept identification and knowledge graph construction.
"""
from pathlib import Path

from ai.pixel.knowledge.psychology_knowledge_extractor import (
    ClinicalConcept,
    ExpertVoiceProfile,
    PsychologyKnowledgeExtractor,
    TherapeuticTechnique,
    extract_psychology_knowledge,
)


def test_psychology_extractor_initialization():
    """Test that the extractor initializes properly."""
    extractor = PsychologyKnowledgeExtractor()
    assert extractor.transcript_dir == Path(".notes/transcripts")
    assert len(extractor.dsm5_patterns) > 0
    assert len(extractor.therapeutic_modality_patterns) > 0


def test_dsm5_concept_extraction():
    """Test extraction of DSM-5 related concepts."""
    extractor = PsychologyKnowledgeExtractor()
    
    sample_text = """
    Complex PTSD often manifests through emotional dysregulation and dissociation. 
    People with attachment trauma frequently struggle with codependency in relationships.
    Narcissistic abuse creates trauma bonding that is difficult to break.
    """
    
    concepts = extractor._extract_dsm5_concepts(sample_text, "test_transcript", "Test Expert")
    
    # Should extract multiple DSM-5 concepts
    assert len(concepts) >= 3
    
    # Check for specific concepts
    concept_names = [c.name for c in concepts]
    assert "Complex PTSD" in concept_names
    assert "Emotional Dysregulation" in concept_names
    assert "Dissociation" in concept_names


def test_expert_source_identification():
    """Test identification of expert sources from file paths."""
    extractor = PsychologyKnowledgeExtractor()
    
    test_cases = [
        (Path(".notes/transcripts/Tim Fletcher/trauma_healing.txt"), "Tim Fletcher"),
        (Path(".notes/transcripts/DoctorRamani/narcissism.txt"), "Dr. Ramani"),
        (Path(".notes/transcripts/Crappy Childhood Fairy/healing_cptsd.txt"), "Crappy Childhood Fairy"),
        (Path(".notes/transcripts/some_expert/unknown.txt"), "Some Expert")
    ]
    
    for file_path, expected_expert in test_cases:
        identified_expert = extractor._identify_expert_source(file_path)
        assert expected_expert.lower() in identified_expert.lower()


def test_confidence_scoring():
    """Test confidence scoring for extracted concepts."""
    extractor = PsychologyKnowledgeExtractor()
    
    # High confidence: multiple mentions, clinical context
    high_context = "PTSD diagnosis requires clinical assessment. PTSD symptoms include flashbacks and hypervigilance."
    high_score = extractor._calculate_confidence("PTSD", high_context)
    
    # Low confidence: single mention, non-clinical context
    low_context = "Someone mentioned PTSD once in passing."
    low_score = extractor._calculate_confidence("PTSD", low_context)
    
    assert high_score > low_score
    assert 0.0 <= low_score <= 1.0
    assert 0.0 <= high_score <= 1.0


def test_clinical_concept_structure():
    """Test that ClinicalConcept dataclass works properly."""
    concept = ClinicalConcept(
        concept_id="test_001",
        name="Test Concept",
        category="dsm5",
        definition="A test clinical concept",
        source_transcript="test.txt",
        expert_source="Dr. Test",
        confidence_score=0.85,
        clinical_context="This is a test context"
    )
    
    assert concept.concept_id == "test_001"
    assert concept.confidence_score == 0.85
    assert len(concept.related_concepts) == 0  # Default empty list


def test_therapeutic_technique_structure():
    """Test that TherapeuticTechnique dataclass works properly."""
    technique = TherapeuticTechnique(
        technique_id="cbt_001",
        name="Cognitive Restructuring",
        modality="CBT",
        description="Identifying and challenging negative thought patterns",
        application_context=["Depression", "Anxiety"],
        contraindications=["Psychosis"],
        expert_quotes=["Challenge the thoughts that don't serve you"],
        effectiveness_indicators=["Reduced negative thinking"]
    )
    
    assert technique.modality == "CBT"
    assert "Depression" in technique.application_context
    assert len(technique.expert_quotes) == 1


def test_extract_psychology_knowledge_function():
    """Test the main extraction function with a small sample."""
    # Create a temporary test directory with sample content
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample transcript file
        test_transcript = Path(temp_dir) / "Tim Fletcher" / "test_trauma.txt"
        test_transcript.parent.mkdir(parents=True)
        
        sample_content = """
        Complex trauma often leads to emotional dysregulation in survivors.
        CBT and DBT are effective therapeutic modalities for treating PTSD.
        Dissociation is a common symptom that requires trauma-informed care.
        Safety planning is essential for clients with suicidal ideation.
        """
        
        with open(test_transcript, 'w') as f:
            f.write(sample_content)
        
        # Extract knowledge from the test directory
        knowledge = extract_psychology_knowledge(temp_dir)
        
        # Validate the structure
        assert "concepts" in knowledge
        assert "techniques" in knowledge
        assert "expert_profiles" in knowledge
        assert "knowledge_graph" in knowledge
        assert "statistics" in knowledge
        
        # Should extract some concepts
        assert knowledge["statistics"]["total_concepts"] > 0
        
        # Should identify Tim Fletcher as expert
        assert knowledge["statistics"]["total_experts"] >= 1


def test_modality_pattern_matching():
    """Test therapeutic modality pattern recognition."""
    extractor = PsychologyKnowledgeExtractor()
    
    test_texts = [
        "CBT is very effective for anxiety disorders.",
        "Dialectical behavior therapy helps with emotion regulation.",
        "EMDR can process traumatic memories.",
        "Trauma-informed care is essential for survivors."
    ]
    
    for text in test_texts:
        # Check that modality patterns match appropriately
        found_modality = False
        for pattern in extractor.therapeutic_modality_patterns.keys():
            import re
            if re.search(pattern, text, re.IGNORECASE):
                found_modality = True
                break
        assert found_modality, f"No modality pattern matched in: {text}"


def test_crisis_pattern_detection():
    """Test detection of crisis-related content."""
    extractor = PsychologyKnowledgeExtractor()
    
    crisis_texts = [
        "The client expressed suicidal ideation during the session.",
        "Self-harm behaviors require immediate safety planning.",
        "Crisis intervention was necessary due to harm to others risk.",
        "We developed a safety plan together."
    ]
    
    for text in crisis_texts:
        found_crisis = False
        for pattern in extractor.crisis_patterns:
            import re
            if re.search(pattern, text, re.IGNORECASE):
                found_crisis = True
                break
        assert found_crisis, f"No crisis pattern matched in: {text}"


def test_empathy_pattern_recognition():
    """Test recognition of empathetic therapeutic language."""
    extractor = PsychologyKnowledgeExtractor()
    
    empathy_texts = [
        "I hear you, that sounds really difficult.",
        "That makes sense given what you've been through.",
        "I understand how challenging this must be.",
        "It's understandable that you'd feel that way.",
        "You're not alone in this struggle."
    ]
    
    for text in empathy_texts:
        found_empathy = False
        for pattern in extractor.empathy_patterns:
            import re
            if re.search(pattern, text, re.IGNORECASE):
                found_empathy = True
                break
        assert found_empathy, f"No empathy pattern matched in: {text}"


def test_knowledge_base_statistics():
    """Test that statistics generation works correctly."""
    extractor = PsychologyKnowledgeExtractor()
    
    # Add some mock data
    extractor.concepts = {
        "c1": ClinicalConcept("c1", "PTSD", "dsm5", "Post-traumatic stress", "t1", "Expert1", 0.9),
        "c2": ClinicalConcept("c2", "CBT", "modality", "Cognitive therapy", "t2", "Expert2", 0.8),
    }
    
    extractor.techniques = {
        "t1": TherapeuticTechnique("t1", "Grounding", "Trauma", "Grounding technique", [], [], [], [])
    }
    
    extractor.expert_profiles = {
        "Expert1": ExpertVoiceProfile("Expert1", [], {}, {}, "", [])
    }
    
    stats = extractor._generate_statistics()
    
    assert stats["total_concepts"] == 2
    assert stats["total_techniques"] == 1
    assert stats["total_experts"] == 1
    assert "concept_categories" in stats
    assert stats["concept_categories"]["dsm5"] == 1
    assert stats["concept_categories"]["modality"] == 1


if __name__ == "__main__":
    # Run basic functionality test
    print("Testing Psychology Knowledge Extractor...")
    
    # Test with actual transcript directory if it exists
    if Path(".notes/transcripts").exists():
        print("Found transcript directory, testing extraction...")
        extractor = PsychologyKnowledgeExtractor()
        
        # Process just a few files for testing
        transcript_files = list(Path(".notes/transcripts").glob("**/*.txt"))[:5]
        print(f"Testing with {len(transcript_files)} sample transcripts")
        
        for transcript_file in transcript_files:
            try:
                extractor._process_transcript(transcript_file)
                print(f"✅ Processed: {transcript_file.name}")
            except Exception as e:
                print(f"❌ Error processing {transcript_file.name}: {e}")
        
        stats = extractor._generate_statistics()
        print(f"Extracted {stats['total_concepts']} concepts")
        print(f"Found {stats['total_experts']} experts")
        
        if stats['concept_categories']:
            print("Concept categories:", stats['concept_categories'])
    else:
        print("Transcript directory not found, running unit tests only")
    
    print("✅ Psychology Knowledge Extractor tests completed!")