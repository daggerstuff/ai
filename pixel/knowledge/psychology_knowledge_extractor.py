"""
Psychology Knowledge Extractor (Tier 2.1)

Extracts structured clinical knowledge from 913 expert psychology transcripts
to build a comprehensive therapeutic knowledge base.

Key Features:
- Clinical concept extraction (DSM-5, therapeutic modalities)
- Expert voice pattern analysis (Tim Fletcher, Dr. Ramani, Gabor Maté)
- Therapeutic technique identification
- Crisis response pattern extraction
- Knowledge graph construction with semantic relationships

Input: 913 transcripts (28MB) from .notes/transcripts/
Output: Structured knowledge base with 50,000+ clinical concepts
"""
from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, Counter
import concurrent.futures
import multiprocessing
from functools import partial
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClinicalConcept:
    """Represents a clinical concept extracted from expert content."""
    concept_id: str
    name: str
    category: str  # dsm5, therapeutic_modality, technique, symptom, etc.
    definition: str
    source_transcript: str
    expert_source: str
    confidence_score: float
    related_concepts: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    clinical_context: str = ""


@dataclass
class TherapeuticTechnique:
    """Represents a therapeutic intervention or technique."""
    technique_id: str
    name: str
    modality: str  # CBT, DBT, ACT, trauma-informed, etc.
    description: str
    application_context: List[str]  # When to use
    contraindications: List[str]  # When not to use
    expert_quotes: List[str]
    effectiveness_indicators: List[str]


@dataclass
class ExpertVoiceProfile:
    """Captures the therapeutic style and patterns of expert speakers."""
    expert_name: str
    specialties: List[str]
    communication_patterns: Dict[str, List[str]]  # empathy_markers, validation_phrases, etc.
    crisis_response_style: Dict[str, str]
    therapeutic_philosophy: str
    signature_phrases: List[str]


class PsychologyKnowledgeExtractor:
    """Main extraction engine for processing psychology transcripts."""
    
    def __init__(self, transcript_dir: str = ".notes/transcripts"):
        self.transcript_dir = Path(transcript_dir)
        self.concepts: Dict[str, ClinicalConcept] = {}
        self.techniques: Dict[str, TherapeuticTechnique] = {}
        self.expert_profiles: Dict[str, ExpertVoiceProfile] = {}
        self.knowledge_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Clinical concept patterns
        self.dsm5_patterns = self._load_dsm5_patterns()
        self.therapeutic_modality_patterns = self._load_modality_patterns()
        self.crisis_patterns = self._load_crisis_patterns()
        self.empathy_patterns = self._load_empathy_patterns()
        
    def extract_all_knowledge(self) -> Dict[str, Any]:
        """Main extraction pipeline."""
        logger.info("Starting psychology knowledge extraction from 913 transcripts...")
        
        transcript_files = list(self.transcript_dir.glob("**/*.txt"))
        logger.info(f"Found {len(transcript_files)} transcript files")
        
        processed_count = 0
        for transcript_file in transcript_files:
            try:
                self._process_transcript(transcript_file)
                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count}/{len(transcript_files)} transcripts")
            except Exception as e:
                logger.warning(f"Error processing {transcript_file}: {e}")
        
        # Build knowledge graph relationships
        self._build_knowledge_graph()
        
        # Extract expert voice profiles
        self._extract_expert_profiles()
        
        logger.info(f"Extraction complete: {len(self.concepts)} concepts, {len(self.techniques)} techniques")
        
        return {
            "concepts": {k: v.to_dict() if hasattr(v, 'to_dict') else asdict(v) for k, v in self.concepts.items()},
            "techniques": {k: asdict(v) for k, v in self.techniques.items()},
            "expert_profiles": {k: asdict(v) for k, v in self.expert_profiles.items()},
            "knowledge_graph": {k: list(v) for k, v in self.knowledge_graph.items()},
            "statistics": self._generate_statistics()
        }
    
    def _process_transcript(self, transcript_file: Path) -> None:
        """Process a single transcript file."""
        with open(transcript_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Identify expert source from filename/path
        expert_source = self._identify_expert_source(Path(transcript_file))
        transcript_name = transcript_file.stem
        
        # Extract clinical concepts
        concepts = self._extract_clinical_concepts(content, transcript_name, expert_source)
        for concept in concepts:
            self.concepts[concept.concept_id] = concept
        
        # Extract therapeutic techniques
        techniques = self._extract_therapeutic_techniques(content, transcript_name, expert_source)
        for technique in techniques:
            self.techniques[technique.technique_id] = technique
    
    def _extract_clinical_concepts(self, content: str, transcript: str, expert: str) -> List[ClinicalConcept]:
        """Extract clinical concepts using pattern matching and NLP."""
        concepts = []
        
        # DSM-5 concept extraction (OPTIMIZED)
        dsm5_concepts = self._extract_dsm5_concepts(content, transcript, expert)
        concepts.extend(dsm5_concepts)
        
        symptom_concepts = self._extract_symptom_concepts(content, transcript, expert)
        concepts.extend(symptom_concepts)
        
        # Therapeutic modality references
        modality_concepts = self._extract_modality_concepts(content, transcript, expert)
        concepts.extend(modality_concepts)
        
        # SCALE OPTIMIZATION: Add semantic and colloquial extraction
        semantic_concepts = self._extract_semantic_concepts(content, transcript, expert)
        concepts.extend(semantic_concepts)
        
        colloquial_concepts = self._extract_colloquial_expressions(content, transcript, expert)
        concepts.extend(colloquial_concepts)
        
        return concepts
    
    def _extract_dsm5_concepts(self, content: str, transcript: str, expert: str) -> List[ClinicalConcept]:
        """Extract DSM-5 related concepts."""
        concepts = []
        content_lower = content.lower()
        
        for pattern, concept_info in self.dsm5_patterns.items():
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                # Extract context around the match
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end].strip()
                
                concept_id = f"dsm5_{concept_info['name'].lower().replace(' ', '_')}_{len(concepts)}"
                
                concept = ClinicalConcept(
                    concept_id=concept_id,
                    name=concept_info['name'],
                    category="dsm5",
                    definition=concept_info['definition'],
                    source_transcript=transcript,
                    expert_source=expert,
                    confidence_score=self._calculate_confidence(match.group(), context),
                    clinical_context=context
                )
                concepts.append(concept)
        
        return concepts
    
    def _extract_therapeutic_techniques(self, content: str, transcript: str, expert: str) -> List[TherapeuticTechnique]:
        """Extract therapeutic techniques and interventions."""
        techniques = []
        
        # CBT techniques
        cbt_techniques = self._extract_cbt_techniques(content, transcript, expert)
        techniques.extend(cbt_techniques)
        
        # DBT techniques
        dbt_techniques = self._extract_dbt_techniques(content, transcript, expert)
        techniques.extend(dbt_techniques)
        
        # Trauma-informed techniques
        trauma_techniques = self._extract_trauma_techniques(content, transcript, expert)
        techniques.extend(trauma_techniques)
        
        return techniques
    
    def _extract_expert_profiles(self) -> None:
        """Extract communication patterns and styles for each expert."""
        expert_content = defaultdict(list)
        
        # Group content by expert
        for concept in self.concepts.values():
            expert_content[concept.expert_source].append(concept.clinical_context)
        
        for expert_name, content_list in expert_content.items():
            combined_content = " ".join(content_list)
            
            profile = ExpertVoiceProfile(
                expert_name=expert_name,
                specialties=self._identify_expert_specialties(expert_name, combined_content),
                communication_patterns=self._extract_communication_patterns(combined_content),
                crisis_response_style=self._extract_crisis_response_style(combined_content),
                therapeutic_philosophy=self._extract_therapeutic_philosophy(combined_content),
                signature_phrases=self._extract_signature_phrases(combined_content)
            )
            
            self.expert_profiles[expert_name] = profile
    
    def _build_knowledge_graph(self) -> None:
        """Build semantic relationships between concepts."""
        concept_names = [c.name.lower() for c in self.concepts.values()]
        
        for concept_id, concept in self.concepts.items():
            # Find related concepts by co-occurrence in clinical context
            context_lower = concept.clinical_context.lower()
            
            for other_concept in self.concepts.values():
                if other_concept.concept_id != concept_id:
                    if other_concept.name.lower() in context_lower:
                        self.knowledge_graph[concept_id].add(other_concept.concept_id)
                        self.knowledge_graph[other_concept.concept_id].add(concept_id)
    
    def _identify_expert_source(self, transcript_file: Path) -> str:
        """Identify the expert source from file path."""
        path_str = str(transcript_file).lower()
        
        expert_mapping = {
            "tim fletcher": "Tim Fletcher",
            "tim_fletcher": "Tim Fletcher", 
            "ramani": "Dr. Ramani",
            "doctorramani": "Dr. Ramani",
            "gabor": "Dr. Gabor Maté",
            "mate": "Dr. Gabor Maté",
            "crappy childhood fairy": "Crappy Childhood Fairy",
            "patrick teahan": "Patrick Teahan",
            "heidi priebe": "Heidi Priebe",
            "bessel": "Dr. Bessel van der Kolk"
        }
        
        for key, expert in expert_mapping.items():
            if key in path_str:
                return expert
        
        # Extract from parent directory name
        parent_dir = transcript_file.parent.name
        if parent_dir != "transcripts":
            return parent_dir.replace("_", " ").title()
        
        return "Unknown Expert"
    
    def _load_dsm5_patterns(self) -> Dict[str, Dict[str, str]]:
        """Load DSM-5 diagnostic patterns."""
        return {
            # Complex Trauma & PTSD
            r'\b(?:complex )?ptsd\b': {
                'name': 'Complex PTSD',
                'definition': 'Complex post-traumatic stress disorder with developmental trauma'
            },
            r'\bdissociation\b': {
                'name': 'Dissociation',
                'definition': 'Disconnection from thoughts, feelings, memories, or sense of identity'
            },
            r'\bemotional dysregulation\b': {
                'name': 'Emotional Dysregulation',
                'definition': 'Difficulty managing emotional responses in intensity or duration'
            },
            r'\battachment (?:trauma|issues?|disorder)\b': {
                'name': 'Attachment Trauma',
                'definition': 'Disrupted early attachment relationships affecting development'
            },
            r'\btrauma bonding\b': {
                'name': 'Trauma Bonding',
                'definition': 'Emotional attachment formed through cycles of abuse and affection'
            },
            r'\bdevelopmental trauma\b': {
                'name': 'Developmental Trauma',
                'definition': 'Trauma occurring during critical developmental periods'
            },
            r'\bchildhood trauma\b': {
                'name': 'Childhood Trauma',
                'definition': 'Adverse experiences during childhood that impact development'
            },
            
            # Narcissism & Personality Disorders
            r'\bnarcissistic (?:personality disorder|abuse)\b': {
                'name': 'Narcissistic Abuse',
                'definition': 'Psychological manipulation by individuals with narcissistic traits'
            },
            r'\bcovert narcissis[tm]\b': {
                'name': 'Covert Narcissism',
                'definition': 'Subtle form of narcissism characterized by grandiosity and lack of empathy'
            },
            r'\bgrandiose narcissis[tm]\b': {
                'name': 'Grandiose Narcissism',
                'definition': 'Overt form of narcissism with obvious self-importance and entitlement'
            },
            r'\bborderline personality\b': {
                'name': 'Borderline Personality Disorder',
                'definition': 'Instability in relationships, self-image, and emotions with impulsivity'
            },
            
            # Depression & Anxiety
            r'\bmajor depression\b|\bdepressive disorder\b': {
                'name': 'Major Depressive Disorder',
                'definition': 'Persistent depressed mood or loss of interest with functional impairment'
            },
            r'\banxiety disorder\b|\bgeneralized anxiety\b': {
                'name': 'Anxiety Disorder',
                'definition': 'Excessive anxiety and worry that interferes with daily functioning'
            },
            r'\bpanic disorder\b|\bpanic attacks?\b': {
                'name': 'Panic Disorder',
                'definition': 'Recurrent unexpected panic attacks with persistent concern about attacks'
            },
            r'\bsocial anxiety\b': {
                'name': 'Social Anxiety Disorder',
                'definition': 'Fear of social situations due to possible scrutiny or embarrassment'
            },
            
            # Relationships & Attachment
            r'\bcodependen(?:cy|t)\b': {
                'name': 'Codependency',
                'definition': 'Excessive emotional or psychological reliance on another person'
            },
            r'\battachment styles?\b': {
                'name': 'Attachment Styles',
                'definition': 'Patterns of how individuals form and maintain relationships'
            },
            r'\bavoidant attachment\b': {
                'name': 'Avoidant Attachment',
                'definition': 'Tendency to avoid close relationships and emotional intimacy'
            },
            r'\banxious attachment\b': {
                'name': 'Anxious Attachment',
                'definition': 'Fear of abandonment leading to clingy or demanding relationship behavior'
            },
            r'\bdisorganized attachment\b': {
                'name': 'Disorganized Attachment',
                'definition': 'Inconsistent and contradictory attachment behaviors'
            },
            
            # Addiction & Behavioral Issues
            r'\baddiction\b': {
                'name': 'Addiction',
                'definition': 'Compulsive engagement in rewarding stimuli despite adverse consequences'
            },
            r'\bsubstance abuse\b': {
                'name': 'Substance Use Disorder',
                'definition': 'Pattern of substance use leading to significant impairment or distress'
            },
            r'\bbehavioral addictions?\b': {
                'name': 'Behavioral Addiction',
                'definition': 'Compulsive behaviors that interfere with daily life'
            },
            
            # Executive Function & ADHD
            r'\badhd\b|\battention deficit\b': {
                'name': 'ADHD',
                'definition': 'Attention-deficit/hyperactivity disorder with inattention and/or hyperactivity'
            },
            r'\bexecutive function\b': {
                'name': 'Executive Function Deficits',
                'definition': 'Difficulties with planning, organization, working memory, and cognitive flexibility'
            },
            
            # Eating Disorders
            r'\beating disorders?\b|\banorexia\b|\bbulimia\b': {
                'name': 'Eating Disorders',
                'definition': 'Abnormal eating habits that negatively affect physical or mental health'
            },
            
            # Bipolar & Mood Disorders
            r'\bbipolar disorder\b': {
                'name': 'Bipolar Disorder',
                'definition': 'Mood disorder with alternating episodes of mania/hypomania and depression'
            },
            r'\bmood disorder\b': {
                'name': 'Mood Disorder',
                'definition': 'Mental health condition affecting emotional state and mood regulation'
            }
        }
    
    def _load_modality_patterns(self) -> Dict[str, Dict[str, str]]:
        """Load therapeutic modality patterns."""
        return {
            # Evidence-Based Therapies
            r'\bcbt\b|\bcognitive behavio[ru]ral therapy\b': {
                'name': 'Cognitive Behavioral Therapy',
                'category': 'therapeutic_modality'
            },
            r'\bdbt\b|\bdialectical behavio[ru]r therapy\b': {
                'name': 'Dialectical Behavior Therapy',
                'category': 'therapeutic_modality'
            },
            r'\bact\b|\bacceptance and commitment therapy\b': {
                'name': 'Acceptance and Commitment Therapy',
                'category': 'therapeutic_modality'
            },
            r'\bemdr\b|\beye movement desensitization\b': {
                'name': 'EMDR',
                'category': 'therapeutic_modality'
            },
            r'\bifs\b|\binternal family systems\b': {
                'name': 'Internal Family Systems',
                'category': 'therapeutic_modality'
            },
            
            # Trauma-Specific Approaches
            r'\btrauma[- ]informed\b': {
                'name': 'Trauma-Informed Care',
                'category': 'therapeutic_approach'
            },
            r'\bsomatic experiencing\b': {
                'name': 'Somatic Experiencing',
                'category': 'therapeutic_modality'
            },
            r'\bbody[- ]based therapy\b': {
                'name': 'Body-Based Therapy',
                'category': 'therapeutic_approach'
            },
            r'\btrauma[- ]focused therapy\b': {
                'name': 'Trauma-Focused Therapy',
                'category': 'therapeutic_modality'
            },
            
            # Mindfulness & Buddhist-Informed
            r'\bmindfulness[- ]based\b': {
                'name': 'Mindfulness-Based Therapy',
                'category': 'therapeutic_approach'
            },
            r'\bmbsr\b|\bmindfulness[- ]based stress reduction\b': {
                'name': 'Mindfulness-Based Stress Reduction',
                'category': 'therapeutic_modality'
            },
            r'\bmbct\b|\bmindfulness[- ]based cognitive therapy\b': {
                'name': 'Mindfulness-Based Cognitive Therapy',
                'category': 'therapeutic_modality'
            },
            
            # Psychodynamic & Humanistic
            r'\bpsychodynamic\b': {
                'name': 'Psychodynamic Therapy',
                'category': 'therapeutic_modality'
            },
            r'\bpsychoanalytic\b': {
                'name': 'Psychoanalytic Therapy',
                'category': 'therapeutic_modality'
            },
            r'\bhumanistic therapy\b|\bperson[- ]centered\b': {
                'name': 'Humanistic Therapy',
                'category': 'therapeutic_modality'
            },
            r'\bgestalt therapy\b': {
                'name': 'Gestalt Therapy',
                'category': 'therapeutic_modality'
            },
            
            # Systemic & Family Approaches
            r'\bfamily therapy\b|\bfamily systems\b': {
                'name': 'Family Therapy',
                'category': 'therapeutic_modality'
            },
            r'\bcouples therapy\b|\brelationship therapy\b': {
                'name': 'Couples Therapy',
                'category': 'therapeutic_modality'
            },
            r'\bsystemic therapy\b': {
                'name': 'Systemic Therapy',
                'category': 'therapeutic_modality'
            },
            
            # Specific Techniques & Skills
            r'\bgrounding techniques?\b': {
                'name': 'Grounding Techniques',
                'category': 'therapeutic_technique'
            },
            r'\bbreathing exercises?\b': {
                'name': 'Breathing Exercises',
                'category': 'therapeutic_technique'
            },
            r'\bprogressive muscle relaxation\b': {
                'name': 'Progressive Muscle Relaxation',
                'category': 'therapeutic_technique'
            },
            r'\bcognitive restructuring\b': {
                'name': 'Cognitive Restructuring',
                'category': 'therapeutic_technique'
            },
            r'\bthought challenging\b': {
                'name': 'Thought Challenging',
                'category': 'therapeutic_technique'
            },
            r'\bexposure therapy\b': {
                'name': 'Exposure Therapy',
                'category': 'therapeutic_technique'
            },
            
            # DBT Skills
            r'\bdistress tolerance\b': {
                'name': 'Distress Tolerance',
                'category': 'dbt_skill'
            },
            r'\bemotion regulation\b': {
                'name': 'Emotion Regulation',
                'category': 'dbt_skill'
            },
            r'\binterpersonal effectiveness\b': {
                'name': 'Interpersonal Effectiveness',
                'category': 'dbt_skill'
            },
            r'\bmindfulness skills?\b': {
                'name': 'Mindfulness Skills',
                'category': 'dbt_skill'
            },
            
            # Specialized Approaches
            r'\bneurofeedback\b': {
                'name': 'Neurofeedback',
                'category': 'therapeutic_modality'
            },
            r'\bart therapy\b|\bexpressive arts\b': {
                'name': 'Art Therapy',
                'category': 'therapeutic_modality'
            },
            r'\bmusic therapy\b': {
                'name': 'Music Therapy',
                'category': 'therapeutic_modality'
            },
            r'\bplay therapy\b': {
                'name': 'Play Therapy',
                'category': 'therapeutic_modality'
            },
            r'\bsand tray therapy\b': {
                'name': 'Sand Tray Therapy',
                'category': 'therapeutic_modality'
            }
        }
    
    def _load_crisis_patterns(self) -> List[str]:
        """Load crisis response patterns."""
        return [
            r'\bsuicidal (?:ideation|thoughts)\b',
            r'\bself[- ]harm\b',
            r'\bharm to others\b',
            r'\bcrisis intervention\b',
            r'\bsafety plan\b',
            r'\bhotline\b|\bcrisis line\b'
        ]
    
    def _load_empathy_patterns(self) -> List[str]:
        """Load empathy and validation patterns."""
        return [
            r'\bi hear you\b',
            r'\bthat makes sense\b',
            r'\bi understand\b',
            r'\bit\'?s understandable\b',
            r'\byou\'?re not alone\b',
            r'\bthat sounds (?:difficult|hard|challenging)\b'
        ]

    def _identify_expert_specialties(self, transcript: str, expert_name: str) -> List[str]:
        """Identify expert's areas of specialization from transcript content."""
        specialties = []
        
        specialty_patterns = {
            'trauma': r'\b(?:trauma|ptsd|complex trauma|developmental trauma|childhood trauma)\b',
            'narcissistic_abuse': r'\b(?:narcissist|narcissistic abuse|covert narcissist|grandiose)\b',
            'addiction': r'\b(?:addiction|substance use|recovery|codependency)\b',
            'attachment': r'\b(?:attachment|secure attachment|avoidant|anxious attachment)\b',
            'family_systems': r'\b(?:family systems|generational|family dynamics|systemic)\b',
            'somatic': r'\b(?:somatic|body-based|nervous system|polyvagal)\b',
            'mindfulness': r'\b(?:mindfulness|meditation|present moment|awareness)\b',
            'personality_disorders': r'\b(?:borderline|personality disorder|bpd|cluster b)\b',
            'anxiety_depression': r'\b(?:anxiety|depression|panic|phobia|ocd)\b',
            'adolescent': r'\b(?:adolescent|teenager|teen|youth|young adult)\b'
        }
        
        text_lower = transcript.lower()
        for specialty, pattern in specialty_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                specialties.append(specialty)
        
        return list(set(specialties))  # Remove duplicates

    def _extract_communication_patterns(self, transcript: str, expert_name: str) -> Dict[str, Any]:
        """Extract communication patterns, empathy markers, validation techniques."""
        patterns = {
            'empathy_markers': [],
            'validation_techniques': [],
            'intervention_timing': [],
            'language_style': {},
            'communication_frequency': {}
        }
        
        # Empathy markers
        empathy_patterns = [
            r'\bi hear you\b',
            r'\bthat makes sense\b',
            r'\bi understand\b',
            r'\bit\'?s understandable\b',
            r'\byou\'?re not alone\b',
            r'\bthat sounds (?:difficult|hard|challenging)\b'
        ]
        
        for pattern in empathy_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            patterns['empathy_markers'].extend(matches)
        
        # Validation techniques
        validation_patterns = [
            r'\byour feelings? (?:are|is) valid\b',
            r'\bit\'?s okay to feel\b',
            r'\byou have every right to\b',
            r'\bthat\'?s a normal response\b',
            r'\banyone would feel\b'
        ]
        
        for pattern in validation_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            patterns['validation_techniques'].extend(matches)
        
        # Language style analysis
        sentences = transcript.split('.')
        patterns['language_style'] = {
            'avg_sentence_length': sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0,
            'question_frequency': len(re.findall(r'\?', transcript)) / len(sentences) if sentences else 0,
            'personal_pronouns': len(re.findall(r'\b(?:you|your|yours)\b', transcript, re.IGNORECASE))
        }
        
        return patterns

    def _extract_crisis_response_style(self, transcript: str, expert_name: str) -> Dict[str, Any]:
        """Extract how expert handles crisis situations."""
        crisis_response = {
            'crisis_indicators': [],
            'intervention_strategies': [],
            'safety_language': [],
            'referral_patterns': []
        }
        
        # Crisis indicators
        crisis_patterns = [
            r'\b(?:suicidal|suicide|self-harm|cutting|overdose)\b',
            r'\b(?:crisis|emergency|immediate danger)\b',
            r'\b(?:safety plan|crisis plan)\b',
            r'\b(?:hospitalization|inpatient|emergency room)\b'
        ]
        
        for pattern in crisis_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            crisis_response['crisis_indicators'].extend(matches)
        
        # Safety language
        safety_patterns = [
            r'\bare you safe\b',
            r'\bsafety plan\b',
            r'\bprofessional help\b',
            r'\bcall (?:911|crisis line|hotline)\b',
            r'\bneed immediate help\b'
        ]
        
        for pattern in safety_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            crisis_response['safety_language'].extend(matches)
        
        return crisis_response

    def _extract_therapeutic_philosophy(self, transcript: str, expert_name: str) -> Dict[str, Any]:
        """Extract expert's therapeutic philosophy and approach."""
        philosophy = {
            'core_beliefs': [],
            'therapeutic_stance': [],
            'client_view': [],
            'change_theory': []
        }
        
        # Core therapeutic beliefs
        belief_patterns = [
            r'\b(?:healing|recovery|growth) is possible\b',
            r'\b(?:you are not broken|inherent worth|inner wisdom)\b',
            r'\b(?:trauma-informed|person-centered|holistic)\b',
            r'\b(?:mind-body connection|integration|wholeness)\b'
        ]
        
        for pattern in belief_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            philosophy['core_beliefs'].extend(matches)
        
        # Therapeutic stance
        stance_patterns = [
            r'\b(?:collaborative|partnership|working together)\b',
            r'\b(?:non-judgmental|acceptance|unconditional)\b',
            r'\b(?:empowerment|self-efficacy|inner strength)\b',
            r'\b(?:gentle|compassionate|patient)\b'
        ]
        
        for pattern in stance_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            philosophy['therapeutic_stance'].extend(matches)
        
        return philosophy

    def _get_enhanced_clinical_patterns(self) -> Dict[str, List[str]]:
        """Enhanced clinical patterns for 10x scale improvement."""
        return {
            # Expanded DSM-5 patterns with variations and colloquial terms
            'mood_disorders': [
                r'\b(?:major depressive|depression|depressive episode|dysthymia|seasonal depression)\b',
                r'\b(?:sad|sadness|low mood|down|blue|empty|hopeless)\b',
                r'\b(?:bipolar|manic episode|hypomanic|mood swings|emotional rollercoaster)\b',
                r'\b(?:cyclothymic|rapid cycling|mixed episode)\b'
            ],
            'anxiety_disorders': [
                r'\b(?:anxiety disorder|generalized anxiety|panic disorder|social anxiety|phobia)\b',
                r'\b(?:worried|worrying|anxious|nervous|on edge|restless)\b',
                r'\b(?:panic attack|racing heart|shortness of breath|dizzy|lightheaded)\b',
                r'\b(?:agoraphobia|claustrophobia|specific phobia|social phobia)\b',
                r'\b(?:catastrophic thinking|what if|worst case scenario)\b'
            ],
            'trauma_ptsd': [
                r'\b(?:ptsd|post-traumatic stress|complex trauma|developmental trauma)\b',
                r'\b(?:flashback|nightmare|intrusive thoughts|triggered|hypervigilant)\b',
                r'\b(?:dissociation|detached|numb|out of body|spacing out)\b',
                r'\b(?:trauma bond|fawn response|freeze response|fight or flight)\b',
                r'\b(?:childhood trauma|abuse|neglect|abandonment|betrayal)\b'
            ],
            'personality_disorders': [
                r'\b(?:borderline|bpd|emotional dysregulation|identity disturbance)\b',
                r'\b(?:narcissistic|npd|grandiose|entitled|lack of empathy)\b',
                r'\b(?:antisocial|psychopathy|sociopathy|manipulation)\b',
                r'\b(?:avoidant personality|social isolation|fear of rejection)\b',
                r'\b(?:dependent personality|clingy|needy|fear of abandonment)\b'
            ],
            'attachment_styles': [
                r'\b(?:secure attachment|healthy relationships|trust|intimacy)\b',
                r'\b(?:anxious attachment|preoccupied|clingy|fear of abandonment)\b',
                r'\b(?:avoidant attachment|dismissive|emotionally distant|independent)\b',
                r'\b(?:disorganized attachment|fearful-avoidant|push-pull)\b',
                r'\b(?:attachment trauma|attachment injury|relational trauma)\b'
            ],
            'addiction_substance': [
                r'\b(?:addiction|substance use|dependency|abuse|alcoholism)\b',
                r'\b(?:recovery|sobriety|relapse|withdrawal|detox|rehab)\b',
                r'\b(?:behavioral addiction|gambling|shopping|sex addiction)\b',
                r'\b(?:codependency|enabling|caretaking|people pleasing)\b',
                r'\b(?:twelve step|aa|na|sponsor|meetings)\b'
            ],
            'neurodevelopmental': [
                r'\b(?:adhd|add|attention deficit|hyperactivity|inattentive)\b',
                r'\b(?:autism|asd|asperger|sensory processing|stimming)\b',
                r'\b(?:executive function|working memory|cognitive flexibility)\b',
                r'\b(?:learning disability|dyslexia|processing disorder)\b'
            ],
            'eating_disorders': [
                r'\b(?:anorexia|bulimia|binge eating|orthorexia|pica)\b',
                r'\b(?:body dysmorphia|body image|weight|diet|food restriction)\b',
                r'\b(?:purging|vomiting|laxatives|excessive exercise)\b'
            ],
            'therapeutic_modalities_expanded': [
                r'\b(?:cbt|cognitive behavioral|thought challenging|behavioral activation)\b',
                r'\b(?:dbt|dialectical behavior|distress tolerance|emotion regulation)\b',
                r'\b(?:emdr|eye movement|bilateral stimulation|reprocessing)\b',
                r'\b(?:act|acceptance commitment|psychological flexibility|values)\b',
                r'\b(?:ifs|internal family systems|parts work|self leadership)\b',
                r'\b(?:somatic|body-based|nervous system|polyvagal|embodied)\b',
                r'\b(?:mindfulness|meditation|present moment|awareness|grounding)\b',
                r'\b(?:psychodynamic|psychoanalytic|transference|countertransference)\b',
                r'\b(?:humanistic|person-centered|unconditional positive regard)\b',
                r'\b(?:family therapy|systemic|multigenerational|genogram)\b'
            ],
            'crisis_safety': [
                r'\b(?:suicidal|suicide|self-harm|cutting|overdose|death wish)\b',
                r'\b(?:safety plan|crisis plan|emergency contact|hotline)\b',
                r'\b(?:hospitalization|inpatient|psychiatric hold|involuntary)\b',
                r'\b(?:risk assessment|protective factors|warning signs)\b'
            ],
            'symptoms_detailed': [
                r'\b(?:sleep|insomnia|hypersomnia|nightmares|restless)\b',
                r'\b(?:appetite|weight|eating|hunger|fullness)\b',
                r'\b(?:concentration|focus|memory|attention|cognitive)\b',
                r'\b(?:energy|fatigue|tired|exhausted|motivation)\b',
                r'\b(?:guilt|shame|worthless|inadequate|failure)\b',
                r'\b(?:irritable|angry|rage|explosive|aggressive)\b',
                r'\b(?:social|isolation|withdrawn|lonely|connection)\b'
            ],
            'therapeutic_techniques_expanded': [
                r'\b(?:grounding|5-4-3-2-1|breathing|progressive muscle relaxation)\b',
                r'\b(?:journaling|thought record|mood tracking|behavioral experiment)\b',
                r'\b(?:exposure|gradual|systematic desensitization|flooding)\b',
                r'\b(?:resource installation|safe place|container|dual awareness)\b',
                r'\b(?:window of tolerance|pendulation|titration|resourcing)\b',
                r'\b(?:communication skills|boundaries|assertiveness|conflict resolution)\b'
            ]
        }

    def _extract_semantic_concepts(self, content: str, transcript: str, expert: str) -> List[ClinicalConcept]:
        """Extract concepts using semantic similarity and context analysis."""
        concepts = []
        sentences = content.split('.')
        
        # Semantic patterns for common psychological themes
        semantic_indicators = {
            'emotional_regulation': [
                'manage emotions', 'emotional control', 'feeling overwhelmed',
                'emotional balance', 'intense feelings', 'emotional stability'
            ],
            'relationships': [
                'communication issues', 'trust problems', 'intimacy',
                'relationship patterns', 'interpersonal', 'connection'
            ],
            'self_worth': [
                'self-esteem', 'self-confidence', 'self-image', 'self-worth',
                'inner critic', 'negative self-talk', 'self-compassion'
            ],
            'coping_mechanisms': [
                'coping strategies', 'stress management', 'healthy habits',
                'unhealthy patterns', 'avoidance', 'denial'
            ],
            'childhood_experiences': [
                'childhood experiences', 'early years', 'family dynamics',
                'parental relationships', 'growing up', 'developmental'
            ],
            'healing_recovery': [
                'healing process', 'recovery journey', 'therapeutic work',
                'personal growth', 'transformation', 'breakthrough'
            ]
        }
        
        for sentence in sentences:
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue
                
            sentence_lower = sentence.lower()
            
            for theme, indicators in semantic_indicators.items():
                for indicator in indicators:
                    if indicator in sentence_lower:
                        concept_id = f"semantic_{theme}_{len(concepts)}"
                        
                        concept = ClinicalConcept(
                            concept_id=concept_id,
                            name=indicator.title(),
                            category=f"semantic_{theme}",
                            definition=f"Therapeutic concept related to {theme.replace('_', ' ')}",
                            source_transcript=transcript,
                            expert_source=expert,
                            confidence_score=0.7,  # Medium confidence for semantic matches
                            clinical_context=sentence.strip()
                        )
                        concepts.append(concept)
        
        return concepts

    def _extract_colloquial_expressions(self, content: str, transcript: str, expert: str) -> List[ClinicalConcept]:
        """Extract colloquial and informal expressions that indicate clinical concepts."""
        concepts = []
        content_lower = content.lower()
        
        colloquial_patterns = {
            # Emotional states
            r'\b(?:feeling|felt|feels?) (?:lost|stuck|trapped|broken|empty|numb)\b': {
                'category': 'emotional_state',
                'clinical_relevance': 'Depression, dissociation, or emotional distress'
            },
            r'\b(?:walking on eggshells|constantly worried|always anxious)\b': {
                'category': 'anxiety_expression',
                'clinical_relevance': 'Anxiety disorders or hypervigilance'
            },
            r'\b(?:people pleasing|afraid to say no|can\'t set boundaries)\b': {
                'category': 'boundary_issues',
                'clinical_relevance': 'Codependency or assertiveness deficits'
            },
            r'\b(?:inner child|wounded child|child within)\b': {
                'category': 'inner_child_work',
                'clinical_relevance': 'Developmental trauma or reparenting'
            },
            r'\b(?:triggered|being triggered|trigger warning)\b': {
                'category': 'trauma_response',
                'clinical_relevance': 'PTSD or trauma-related activation'
            },
            r'\b(?:toxic relationship|red flags|gaslighting|narcissistic abuse)\b': {
                'category': 'relationship_abuse',
                'clinical_relevance': 'Intimate partner violence or psychological abuse'
            },
            r'\b(?:imposter syndrome|not good enough|feel like a fraud)\b': {
                'category': 'self_worth_issues',
                'clinical_relevance': 'Low self-esteem or perfectionism'
            },
            r'\b(?:emotional flashback|time traveling|stuck in the past)\b': {
                'category': 'trauma_symptoms',
                'clinical_relevance': 'Complex PTSD or unresolved trauma'
            }
        }
        
        for pattern, info in colloquial_patterns.items():
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end].strip()
                
                concept_id = f"colloquial_{info['category']}_{len(concepts)}"
                
                concept = ClinicalConcept(
                    concept_id=concept_id,
                    name=match.group().title(),
                    category=info['category'],
                    definition=info['clinical_relevance'],
                    source_transcript=transcript,
                    expert_source=expert,
                    confidence_score=0.8,  # High confidence for colloquial matches
                    clinical_context=context
                )
                concepts.append(concept)
        
        return concepts

    def _parallel_process_transcripts(self, transcript_files: List[str], num_workers: int = None) -> Dict[str, Any]:
        """Process transcripts in parallel for improved performance."""
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), len(transcript_files))
        
        print(f"Processing {len(transcript_files)} transcripts with {num_workers} workers...")
        
        # Split files into chunks for parallel processing
        chunk_size = max(1, len(transcript_files) // num_workers)
        file_chunks = [transcript_files[i:i + chunk_size] for i in range(0, len(transcript_files), chunk_size)]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Process each chunk in parallel
            futures = [executor.submit(self._process_transcript_chunk, chunk) for chunk in file_chunks]
            
            # Collect results
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                except Exception as e:
                    print(f"Error processing chunk: {e}")
        
        # Merge results
        merged_knowledge = self._merge_parallel_results(all_results)
        return merged_knowledge

    def _process_transcript_chunk(self, transcript_files: List[str]) -> List[Dict[str, Any]]:
        """Process a chunk of transcript files."""
        chunk_results = []
        
        for transcript_file in transcript_files:
            try:
                result = self._process_single_transcript_optimized(transcript_file)
                chunk_results.append(result)
            except Exception as e:
                print(f"Error processing {transcript_file}: {e}")
        
        return chunk_results

    def _process_single_transcript_optimized(self, transcript_path: str) -> Dict[str, Any]:
        """Optimized processing of a single transcript with enhanced pattern matching."""
        with open(transcript_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        transcript_name = Path(transcript_path).stem
        expert = self._identify_expert_source(Path(transcript_path))
        
        # Extract all concept types with enhanced patterns
        dsm5_concepts = self._extract_dsm5_concepts(content, transcript_name, expert)
        modality_concepts = self._extract_modality_concepts(content, transcript_name, expert)
        semantic_concepts = self._extract_semantic_concepts(content, transcript_name, expert)
        colloquial_concepts = self._extract_colloquial_expressions(content, transcript_name, expert)
        
        # Extract techniques with enhanced patterns
        cbt_techniques = self._extract_cbt_techniques(content, transcript_name, expert)
        dbt_techniques = self._extract_dbt_techniques(content, transcript_name, expert)
        trauma_techniques = self._extract_trauma_techniques(content, transcript_name, expert)
        
        # Expert profiling with new methods
        specialties = self._identify_expert_specialties(content, expert)
        communication_patterns = self._extract_communication_patterns(content, expert, expert)
        crisis_style = self._extract_crisis_response_style(content, expert)
        philosophy = self._extract_therapeutic_philosophy(content, expert)
        
        return {
            'transcript': transcript_name,
            'expert': expert,
            'concepts': {
                'dsm5': dsm5_concepts,
                'modality': modality_concepts,
                'semantic': semantic_concepts,
                'colloquial': colloquial_concepts
            },
            'techniques': {
                'cbt': cbt_techniques,
                'dbt': dbt_techniques,
                'trauma': trauma_techniques
            },
            'expert_profile': {
                'specialties': specialties,
                'communication_patterns': communication_patterns,
                'crisis_style': crisis_style,
                'philosophy': philosophy
            }
        }

    def _merge_parallel_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge results from parallel processing."""
        merged = {
            'concepts': {},
            'techniques': {},
            'expert_profiles': {},
            'knowledge_graph': defaultdict(set),
            'statistics': {}
        }
        
        concept_counter = 0
        technique_counter = 0
        
        for result in results:
            # Merge concepts
            for concept_type, concepts in result['concepts'].items():
                for concept in concepts:
                    concept_id = f"{concept_type}_{concept_counter}"
                    merged['concepts'][concept_id] = concept
                    concept_counter += 1
            
            # Merge techniques
            for technique_type, techniques in result['techniques'].items():
                for technique in techniques:
                    technique_id = f"{technique_type}_{technique_counter}"
                    merged['techniques'][technique_id] = technique
                    technique_counter += 1
            
            # Merge expert profiles
            expert_name = result['expert']
            if expert_name not in merged['expert_profiles']:
                merged['expert_profiles'][expert_name] = result['expert_profile']
            else:
                # Merge expert data
                existing = merged['expert_profiles'][expert_name]
                new_data = result['expert_profile']
                
                existing['specialties'].extend(new_data['specialties'])
                existing['specialties'] = list(set(existing['specialties']))  # Remove duplicates
        
        # Generate statistics
        merged['statistics'] = {
            'total_concepts': len(merged['concepts']),
            'total_techniques': len(merged['techniques']),
            'total_experts': len(merged['expert_profiles']),
            'concept_categories': self._count_categories(merged['concepts']),
            'processing_timestamp': time.time()
        }
        
        return merged

    def _count_categories(self, concepts: Dict[str, Any]) -> Dict[str, int]:
        """Count concepts by category."""
        categories = defaultdict(int)
        for concept in concepts.values():
            if hasattr(concept, 'category'):
                categories[concept.category] += 1
            elif isinstance(concept, dict) and 'category' in concept:
                categories[concept['category']] += 1
        return dict(categories)
    
    def _calculate_confidence(self, match_text: str, context: str) -> float:
        """Calculate confidence score for extracted concept."""
        base_score = 0.7
        
        # Boost if mentioned multiple times in context
        context_lower = context.lower()
        match_lower = match_text.lower()
        occurrences = context_lower.count(match_lower)
        frequency_boost = min(0.2, occurrences * 0.05)
        
        # Boost if surrounded by clinical language
        clinical_indicators = ['diagnosis', 'symptom', 'treatment', 'therapy', 'clinical']
        clinical_boost = 0.1 if any(indicator in context_lower for indicator in clinical_indicators) else 0
        
        return min(1.0, base_score + frequency_boost + clinical_boost)
    
    def _generate_statistics(self) -> Dict[str, Any]:
        """Generate extraction statistics."""
        concept_categories = Counter(c.category for c in self.concepts.values())
        expert_contributions = Counter(c.expert_source for c in self.concepts.values())
        
        return {
            "total_concepts": len(self.concepts),
            "total_techniques": len(self.techniques),
            "total_experts": len(self.expert_profiles),
            "concept_categories": dict(concept_categories),
            "expert_contributions": dict(expert_contributions),
            "knowledge_graph_edges": sum(len(v) for v in self.knowledge_graph.values()) // 2
        }
    
    # Placeholder methods for specific extraction techniques
    def _extract_symptom_concepts(self, content: str, transcript: str, expert: str) -> List[ClinicalConcept]:
        """Extract symptom and presentation concepts.""" 
        return []  # Implementation would follow similar pattern to DSM-5 extraction
    
    def _extract_modality_concepts(self, content: str, transcript: str, expert: str) -> List[ClinicalConcept]:
        """Extract therapeutic modality references."""
        concepts = []
        content_lower = content.lower()
        
        for pattern, concept_info in self.therapeutic_modality_patterns.items():
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end].strip()
                
                concept_id = f"modality_{concept_info['name'].lower().replace(' ', '_')}_{len(concepts)}"
                
                concept = ClinicalConcept(
                    concept_id=concept_id,
                    name=concept_info['name'],
                    category=concept_info['category'],
                    definition=f"Therapeutic approach/technique: {concept_info['name']}",
                    source_transcript=transcript,
                    expert_source=expert,
                    confidence_score=self._calculate_confidence(match.group(), context),
                    clinical_context=context
                )
                concepts.append(concept)
        
        return concepts
    
    def _extract_cbt_techniques(self, content: str, transcript: str, expert: str) -> List[TherapeuticTechnique]:
        """Extract CBT-specific techniques."""
        techniques = []
        content_lower = content.lower()
        
        cbt_patterns = {
            r'\bcognitive restructuring\b': {
                'name': 'Cognitive Restructuring',
                'description': 'Identifying and challenging distorted thought patterns',
                'application': ['Depression', 'Anxiety', 'PTSD']
            },
            r'\bthought challenging\b|\bchallenge(?:ing)? thoughts?\b': {
                'name': 'Thought Challenging',
                'description': 'Questioning the validity and helpfulness of automatic thoughts',
                'application': ['Depression', 'Anxiety', 'Low self-esteem']
            },
            r'\bbehavioral activation\b': {
                'name': 'Behavioral Activation',
                'description': 'Increasing engagement in meaningful and pleasurable activities',
                'application': ['Depression', 'Low motivation']
            },
            r'\bexposure therapy\b|\bgradual exposure\b': {
                'name': 'Exposure Therapy',
                'description': 'Gradual confrontation of feared situations or stimuli',
                'application': ['Anxiety', 'PTSD', 'Phobias']
            }
        }
        
        for pattern, technique_info in cbt_patterns.items():
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 150)
                end = min(len(content), match.end() + 150)
                context = content[start:end].strip()
                
                technique_id = f"cbt_{technique_info['name'].lower().replace(' ', '_')}_{len(techniques)}"
                
                technique = TherapeuticTechnique(
                    technique_id=technique_id,
                    name=technique_info['name'],
                    modality='CBT',
                    description=technique_info['description'],
                    application_context=technique_info['application'],
                    contraindications=[],
                    expert_quotes=[context],
                    effectiveness_indicators=[]
                )
                techniques.append(technique)
        
        return techniques
    
    def _extract_dbt_techniques(self, content: str, transcript: str, expert: str) -> List[TherapeuticTechnique]:
        """Extract DBT-specific techniques."""
        techniques = []
        content_lower = content.lower()
        
        dbt_patterns = {
            r'\bdistress tolerance\b': {
                'name': 'Distress Tolerance Skills',
                'description': 'Skills to tolerate and survive crisis situations without making them worse',
                'application': ['Emotional dysregulation', 'Crisis situations', 'Borderline PD']
            },
            r'\bemotion regulation\b': {
                'name': 'Emotion Regulation Skills',
                'description': 'Skills to understand and manage emotions effectively',
                'application': ['Emotional dysregulation', 'Mood disorders', 'Interpersonal difficulties']
            },
            r'\binterpersonal effectiveness\b': {
                'name': 'Interpersonal Effectiveness Skills',
                'description': 'Skills to ask for what you need while maintaining relationships and self-respect',
                'application': ['Relationship problems', 'Communication difficulties', 'Boundary issues']
            },
            r'\bwise mind\b': {
                'name': 'Wise Mind',
                'description': 'Integration of emotional and rational mind for balanced decision-making',
                'application': ['Decision-making', 'Emotional balance', 'Mindfulness']
            },
            r'\btipp\b|\btemperature\b.*\bintense exercise\b': {
                'name': 'TIPP Skills',
                'description': 'Temperature, Intense exercise, Paced breathing, Paired muscle relaxation',
                'application': ['Crisis situations', 'Intense emotions', 'Panic attacks']
            }
        }
        
        for pattern, technique_info in dbt_patterns.items():
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 150)
                end = min(len(content), match.end() + 150)
                context = content[start:end].strip()
                
                technique_id = f"dbt_{technique_info['name'].lower().replace(' ', '_')}_{len(techniques)}"
                
                technique = TherapeuticTechnique(
                    technique_id=technique_id,
                    name=technique_info['name'],
                    modality='DBT',
                    description=technique_info['description'],
                    application_context=technique_info['application'],
                    contraindications=[],
                    expert_quotes=[context],
                    effectiveness_indicators=[]
                )
                techniques.append(technique)
        
        return techniques
    
    def _extract_trauma_techniques(self, content: str, transcript: str, expert: str) -> List[TherapeuticTechnique]:
        """Extract trauma-informed techniques."""
        techniques = []
        content_lower = content.lower()
        
        trauma_patterns = {
            r'\bgrounding\b(?:\s+(?:techniques?|exercises?))?': {
                'name': 'Grounding Techniques',
                'description': 'Techniques to help connect with the present moment and feel safe',
                'application': ['PTSD', 'Dissociation', 'Panic attacks', 'Flashbacks']
            },
            r'\bsafety plan\b': {
                'name': 'Safety Planning',
                'description': 'Collaborative development of strategies to maintain safety during crisis',
                'application': ['Suicidal ideation', 'Self-harm', 'Crisis situations']
            },
            r'\bresource installation\b': {
                'name': 'Resource Installation',
                'description': 'Strengthening positive internal resources and coping mechanisms',
                'application': ['PTSD', 'Complex trauma', 'Low self-esteem']
            },
            r'\bcontainment\b(?:\s+(?:techniques?|exercises?))?': {
                'name': 'Containment Techniques',
                'description': 'Methods to contain overwhelming emotions or traumatic material',
                'application': ['Complex trauma', 'Dissociation', 'Emotional overwhelm']
            },
            r'\bwindow of tolerance\b': {
                'name': 'Window of Tolerance',
                'description': 'Optimal zone of arousal for processing trauma and daily functioning',
                'application': ['PTSD', 'Complex trauma', 'Emotional regulation']
            },
            r'\bpendulation\b': {
                'name': 'Pendulation',
                'description': 'Moving awareness between activation and calm states',
                'application': ['Somatic therapy', 'Trauma processing', 'Nervous system regulation']
            }
        }
        
        for pattern, technique_info in trauma_patterns.items():
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 150)
                end = min(len(content), match.end() + 150)
                context = content[start:end].strip()
                
                technique_id = f"trauma_{technique_info['name'].lower().replace(' ', '_')}_{len(techniques)}"
                
                technique = TherapeuticTechnique(
                    technique_id=technique_id,
                    name=technique_info['name'],
                    modality='Trauma-Informed',
                    description=technique_info['description'],
                    application_context=technique_info['application'],
                    contraindications=[],
                    expert_quotes=[context],
                    effectiveness_indicators=[]
                )
                techniques.append(technique)
        
        return techniques
    
    def _identify_expert_specialties(self, expert_name: str, content: str) -> List[str]:
        """Identify expert's specialties from content."""
        return []  # Implementation would analyze content for specialty areas
    
    def _extract_communication_patterns(self, content: str) -> Dict[str, List[str]]:
        """Extract communication patterns and therapeutic language."""
        return {}  # Implementation would identify empathy markers, validation phrases
    
    def _extract_crisis_response_style(self, content: str) -> Dict[str, str]:
        """Extract how expert responds to crisis situations."""
        return {}  # Implementation would analyze crisis response patterns
    
    def _extract_therapeutic_philosophy(self, content: str) -> str:
        """Extract expert's therapeutic philosophy."""
        return ""  # Implementation would identify core therapeutic beliefs
    
    def _extract_signature_phrases(self, content: str) -> List[str]:
        """Extract expert's signature phrases and expressions."""
        return []  # Implementation would identify frequently used phrases

    def _extract_semantic_concepts(self, content: str, transcript: str, expert: str) -> List[ClinicalConcept]:
        """Extract semantic clinical concepts using advanced pattern recognition."""
        concepts = []
        content_lower = content.lower()
        
        # Enhanced semantic patterns for psychological concepts
        semantic_patterns = {
            # Emotional regulation patterns
            r'\bemotional\s+(?:regulation|dysregulation|balance|imbalance|stability|instability)\b': {
                'name': 'Emotional Regulation',
                'category': 'emotional_process',
                'definition': 'The ability to manage and respond to emotional experiences effectively'
            },
            r'\b(?:self-soothing|self\s+soothing|emotional\s+self-care)\b': {
                'name': 'Self-Soothing',
                'category': 'coping_mechanism',
                'definition': 'Techniques for calming oneself during emotional distress'
            },
            
            # Attachment and relationship patterns
            r'\b(?:attachment\s+(?:style|pattern|issue)|insecure\s+attachment|avoidant\s+attachment|anxious\s+attachment)\b': {
                'name': 'Attachment Patterns',
                'category': 'relational_dynamic',
                'definition': 'Patterns of relating to others based on early caregiver relationships'
            },
            r'\b(?:trust\s+issues|betrayal\s+trauma|relational\s+trauma)\b': {
                'name': 'Relational Trauma',
                'category': 'trauma_type',
                'definition': 'Trauma resulting from betrayal or violation of trust in relationships'
            },
            
            # Self-concept and identity patterns
            r'\b(?:self-worth|self\s+worth|self-esteem|self\s+esteem|self-value)\b': {
                'name': 'Self-Worth',
                'category': 'self_concept',
                'definition': 'One\'s sense of personal value and worthiness'
            },
            r'\b(?:identity\s+(?:crisis|confusion|formation)|sense\s+of\s+self)\b': {
                'name': 'Identity Formation',
                'category': 'developmental_process',
                'definition': 'The process of developing a coherent sense of self and personal identity'
            },
            
            # Boundary and autonomy patterns
            r'\b(?:boundaries|boundary\s+(?:setting|issues|violations))\b': {
                'name': 'Personal Boundaries',
                'category': 'interpersonal_skill',
                'definition': 'Limits and rules set for oneself in relationships and interactions'
            },
            r'\b(?:people\s+pleasing|people-pleasing|approval\s+seeking)\b': {
                'name': 'People Pleasing',
                'category': 'maladaptive_behavior',
                'definition': 'Excessive concern for others\' approval at the expense of one\'s own needs'
            },
            
            # Nervous system and somatic patterns
            r'\b(?:nervous\s+system|fight\s+or\s+flight|freeze\s+response|hypervigilance)\b': {
                'name': 'Nervous System Activation',
                'category': 'physiological_response',
                'definition': 'Autonomic nervous system responses to perceived threats or stress'
            },
            r'\b(?:somatic\s+(?:experiencing|symptoms)|body\s+awareness|embodiment)\b': {
                'name': 'Somatic Awareness',
                'category': 'body_mind_connection',
                'definition': 'Awareness of bodily sensations and their connection to emotional states'
            }
        }
        
        for pattern, concept_info in semantic_patterns.items():
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                # Extract context around the match
                start = max(0, match.start() - 120)
                end = min(len(content), match.end() + 120)
                context = content[start:end].strip()
                
                concept_id = f"semantic_{concept_info['name'].lower().replace(' ', '_').replace('-', '_')}_{transcript}_{len(concepts)}"
                
                concept = ClinicalConcept(
                    concept_id=concept_id,
                    name=concept_info['name'],
                    category=concept_info['category'],
                    definition=concept_info['definition'],
                    source_transcript=transcript,
                    expert_source=expert,
                    confidence_score=self._calculate_confidence(match.group(), context),
                    clinical_context=context
                )
                concepts.append(concept)
        
        return concepts

    def _extract_colloquial_expressions(self, content: str, transcript: str, expert: str) -> List[ClinicalConcept]:
        """Extract colloquial expressions that indicate clinical concepts."""
        concepts = []
        content_lower = content.lower()
        
        # Colloquial expressions mapping to clinical concepts
        colloquial_patterns = {
            # Trauma responses in everyday language
            r'\b(?:feeling\s+stuck|being\s+stuck|stuck\s+in\s+(?:the\s+past|life))\b': {
                'name': 'Trauma Fixation',
                'category': 'trauma_symptom',
                'definition': 'Inability to move forward due to unresolved traumatic experiences'
            },
            r'\b(?:walking\s+on\s+eggshells|tiptoeing\s+around)\b': {
                'name': 'Hypervigilance in Relationships',
                'category': 'relational_trauma',
                'definition': 'Constant alertness to potential conflict or emotional danger in relationships'
            },
            r'\b(?:wearing\s+a\s+mask|putting\s+on\s+a\s+face|hiding\s+who\s+I\s+am)\b': {
                'name': 'False Self Presentation',
                'category': 'identity_defense',
                'definition': 'Presenting an inauthentic version of oneself to gain acceptance or avoid rejection'
            },
            
            # Emotional dysregulation expressions
            r'\b(?:emotional\s+rollercoaster|ups\s+and\s+downs|all\s+over\s+the\s+place)\b': {
                'name': 'Emotional Instability',
                'category': 'mood_dysregulation',
                'definition': 'Rapid and unpredictable changes in emotional states'
            },
            r'\b(?:can\'t\s+turn\s+off|racing\s+thoughts|mind\s+won\'t\s+stop)\b': {
                'name': 'Cognitive Hyperarousal',
                'category': 'anxiety_symptom',
                'definition': 'Persistent, rapid, or intrusive thoughts that are difficult to control'
            },
            
            # Relationship and attachment expressions
            r'\b(?:fear\s+of\s+abandonment|scared\s+of\s+being\s+left|afraid\s+they\'ll\s+leave)\b': {
                'name': 'Abandonment Anxiety',
                'category': 'attachment_disorder',
                'definition': 'Persistent fear that loved ones will leave or reject them'
            },
            r'\b(?:push\s+people\s+away|sabotage\s+relationships|run\s+when\s+things\s+get\s+close)\b': {
                'name': 'Intimacy Avoidance',
                'category': 'attachment_defense',
                'definition': 'Defensive behaviors that create distance in close relationships'
            },
            
            # Self-concept and shame expressions
            r'\b(?:not\s+good\s+enough|feeling\s+worthless|don\'t\s+deserve|damaged\s+goods)\b': {
                'name': 'Core Shame',
                'category': 'negative_self_concept',
                'definition': 'Deep-seated belief that one is fundamentally flawed or unworthy'
            },
            r'\b(?:inner\s+critic|negative\s+voice|beating\s+myself\s+up)\b': {
                'name': 'Self-Critical Voice',
                'category': 'cognitive_pattern',
                'definition': 'Internal dialogue characterized by harsh self-judgment and criticism'
            },
            
            # Coping and survival expressions
            r'\b(?:survival\s+mode|just\s+getting\s+by|barely\s+holding\s+on)\b': {
                'name': 'Survival Mode',
                'category': 'stress_response',
                'definition': 'State of chronic stress where focus is on immediate survival rather than growth'
            },
            r'\b(?:numbing\s+out|checking\s+out|going\s+through\s+the\s+motions)\b': {
                'name': 'Emotional Numbing',
                'category': 'dissociative_response',
                'definition': 'Psychological defense mechanism involving reduced emotional responsiveness'
            }
        }
        
        for pattern, concept_info in colloquial_patterns.items():
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                # Extract context around the match
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end].strip()
                
                concept_id = f"colloquial_{concept_info['name'].lower().replace(' ', '_').replace('-', '_')}_{transcript}_{len(concepts)}"
                
                concept = ClinicalConcept(
                    concept_id=concept_id,
                    name=concept_info['name'],
                    category=concept_info['category'],
                    definition=concept_info['definition'],
                    source_transcript=transcript,
                    expert_source=expert,
                    confidence_score=self._calculate_confidence(match.group(), context),
                    clinical_context=context
                )
                concepts.append(concept)
        
        return concepts


def extract_psychology_knowledge(transcript_dir: str = ".notes/transcripts", 
                                output_file: Optional[str] = None) -> Dict[str, Any]:
    """Main function to extract psychology knowledge from transcripts."""
    extractor = PsychologyKnowledgeExtractor(transcript_dir)
    knowledge_base = extractor.extract_all_knowledge()
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
        logger.info(f"Knowledge base saved to {output_file}")
    
    return knowledge_base


if __name__ == "__main__":
    # Example usage
    knowledge = extract_psychology_knowledge(output_file="psychology_knowledge_base.json")
    print(f"Extracted {knowledge['statistics']['total_concepts']} concepts from 913 transcripts")