"""
Automated labeler for predictable categories with confidence scoring.
Implements automated labeling for therapeutic responses, crisis detection, and other primary tasks.
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import re
import uuid
from datetime import datetime
from .label_taxonomy import (
    LabelBundle, TherapeuticResponseLabel, CrisisLabel, TherapyModalityLabel,
    MentalHealthConditionLabel, DemographicLabel, LabelMetadata, LabelProvenanceType,
    TherapeuticResponseType, CrisisLevelType, TherapyModalityType, MentalHealthConditionType,
    DemographicType
)
from .conversation_schema import Conversation, Message
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    """Confidence score with explanation"""
    score: float  # 0.0 to 1.0
    explanation: str
    model_confidence: float  # Raw model confidence before adjustment
    rule_based_confidence: float  # Confidence from rule-based heuristics
    context_dependent: bool = False  # Whether confidence depends on context


class AutomatedLabeler:
    """
    Automated labeler for predictable categories in therapeutic conversations.
    Implements rule-based and model-based approaches for various label types.
    """

    def __init__(self):
        self.therapeutic_patterns = self._initialize_therapeutic_patterns()
        self.crisis_keywords = self._initialize_crisis_keywords()
        self.modality_keywords = self._initialize_modality_keywords()
        self.condition_keywords = self._initialize_condition_keywords()
        
    def _initialize_therapeutic_patterns(self) -> Dict[TherapeuticResponseType, List[str]]:
        """Initialize regex patterns for therapeutic response types"""
        return {
            TherapeuticResponseType.REFLECTION: [
                r'\b(?:you said|you mentioned|it sounds like|it seems like|you feel|you seem to|from what you said)\b',
                r'\b(?:I hear you saying|you\'re describing|what I\'m hearing is)\b',
                r'(?i)\bright(?:\'s|s|)\s+that\b|you(?:\'ve| have)\s+explained',
            ],
            TherapeuticResponseType.EMPATHY: [
                r'\b(?:I understand|I can see|I imagine|must be difficult|that sounds|that must)\b',
                r'(?i)\bsounds tough|I(?:\'m| am)\s+sorry you\'re going through',
                r'\b(?:that\'s hard|I can only imagine|how difficult that must be)\b',
            ],
            TherapeuticResponseType.CHALLENGE: [
                r'\b(?:have you considered|what if|have you thought about|another way to look at)\b',
                r'\b(?:I wonder if|perhaps|could it be|what would happen if)\b',
                r'(?i)\bare you sure|is that really true',
            ],
            TherapeuticResponseType.EDUCATION: [
                r'\b(?:this is called|it\'s known as|a technique called|the term|refers to)\b',
                r'\b(?:research shows|studies indicate|evidence suggests)\b',
                r'\b(?:here\'s some information|let me explain|it works by)\b',
            ],
            TherapeuticResponseType.REFRAME: [
                r'\b(?:another way to view|different perspective|instead of|looking at it differently)\b',
                r'(?i)\bwhat if we think of|considered from another angle',
                r'\b(?:it could also be seen|flipping the script|reversing the perspective)\b',
            ],
            TherapeuticResponseType.PROBING: [
                r'\b(?:can you tell me|what happened|how did that make|can you describe|walk me through)\b',
                r'(?i)\bhelp me understand|I\'d like to know more about',
                r'\b(?:what else|anything else|go deeper|tell me more)\b',
            ],
            TherapeuticResponseType.SUPPORT: [
                r'\b(?:I\'m here|you\'re doing great|that\'s a good step|I\'m proud of you|you\'re strong)\b',
                r'(?i)\byou\'re not alone|I believe in you|that takes courage',
                r'\b(?:that\'s a lot to handle|I appreciate your honesty|thank you for sharing)\b',
            ],
            TherapeuticResponseType.CONFRONTATION: [
                r'\b(?:I notice a pattern|you seem to|there\'s a contradiction|what I see is)\b',
                r'(?i)\byou said one thing but|your actions suggest|there seems to be a disconnect',
                r'\b(?:I want to point out|I need to address|it appears)\b',
            ],
            TherapeuticResponseType.INTERPRETATION: [
                r'\b(?:this might mean|could suggest|may indicate|what this tells me|underlying issue)\b',
                r'(?i)\bwhat I think is happening|the root cause|deeper meaning',
                r'\b(?:this could be about|related to|connected to)\b',
            ],
            TherapeuticResponseType.PSYCHOEDUCATION: [
                r'\b(?:common with|typical of|many people|normal response|part of the process)\b',
                r'(?i)\bmany people experience|it\'s normal|typical of recovery',
                r'\b(?:this is part of|a common experience|expected during)\b',
            ],
            TherapeuticResponseType.GOAL_SETTING: [
                r'\b(?:let\'s work toward|next step|how can we achieve|what would help|action plan)\b',
                r'(?i)\bwhat are your goals|let\'s set a target|working toward',
                r'\b(?:steps to take|moving forward|next phase)\b',
            ],
            TherapeuticResponseType.SUMMARIZATION: [
                r'\b(?:to summarize|in summary|so far|what I\'ve heard|key points are)\b',
                r'(?i)\bto recap|what we\'ve covered|the main themes',
                r'\b(?:you\'ve shared|main points|highlights of our session)\b',
            ]
        }

    def _initialize_crisis_keywords(self) -> Dict[CrisisLevelType, List[str]]:
        """Initialize crisis detection keywords by severity level"""
        return {
            CrisisLevelType.IMMEDIATE_RISK: [
                r'\b(?:kill myself|kill myself|end it all|no reason to live|can\'t go on|suicide plan|lethal means)\b',
                r'\b(?:hurt someone|seriously harm|cause serious damage|violence plan|attack|assault plan)\b',
                r'(?i)\bactive plan with method|specific lethal plan|in the next few hours|immediate threat',
            ],
            CrisisLevelType.HIGH_RISK: [
                r'\b(?:want to die|thoughts of dying|don\'t want to live|suicidal thoughts|suicide ideation)\b',
                r'\b(?:harm myself|hurt myself|self-harm|cut|overdose|intentional injury)\b',
                r'(?i)\baccess to means|lethal plan|not safe|plan made|method chosen',
            ],
            CrisisLevelType.MEDIUM_RISK: [
                r'\b(?:hopeless|nothing matters|worthless|no point|can\'t handle|overwhelmed|can\'t cope)\b',
                r'\b(?:escape|get away|leave|disappear|won\'t be here|not around|gone)\b',
                r'(?i)\bpast attempts|previous ideation|history of self-harm|recent escalation',
            ],
            CrisisLevelType.LOW_RISK: [
                r'\b(?:stressed|anxious|overwhelmed|sad|depressed|struggling|difficulty|hard time)\b',
                r'\b(?:not coping well|challenging|tough period|going through|dealing with)\b',
                r'(?i)\bfeeling down|emotional pain|mental health concerns',
            ],
            CrisisLevelType.NO_RISK: [
                r'\b(?:getting better|coping well|stable|feeling better|progress|improvement|good day)\b',
                r'\b(?:support system|coping strategies|resources|safe|not at risk|no plans)\b',
                r'(?i)\bno intention|not serious|just venting|looking for help',
            ]
        }

    def _initialize_modality_keywords(self) -> Dict[TherapyModalityType, List[str]]:
        """Initialize keywords for therapy modalities"""
        return {
            TherapyModalityType.CBT: [
                r'\b(?:thought pattern|cognitive distortion|thinking trap|cognitive restructuring)\b',
                r'(?i)\bchallenging thoughts|thought record|behavioral experiment|activity scheduling',
                r'\b(?:identify thoughts|think differently|cognitive flexibility|thoughts cause feelings)\b',
            ],
            TherapyModalityType.DBT: [
                r'\b(?:mindfulness|acceptance|distress tolerance|emotion regulation|interpersonal effectiveness)\b',
                r'(?i)\bopposite action|radical acceptance|check the facts|wise mind',
                r'\b(?:validate|validation|dialectical|balance acceptance)\b',
            ],
            TherapyModalityType.PSYCHODYNAMIC: [
                r'\b(?:unconscious|repressed|childhood|trauma|defense mechanism|transference)\b',
                r'(?i)\bunresolved conflict|past relationships|deep feelings|unconscious motivation',
                r'\b(?:patterns from past|early experiences|relationship patterns|internal conflicts)\b',
            ],
            TherapyModalityType.HUMANISTIC: [
                r'\b(?:unconditional positive regard|client-centered|authentic|genuineness|congruence)\b',
                r'(?i)\bclient strengths|personal growth|self-actualization|non-directive',
                r'\b(?:person-centered|empathetic understanding|personal responsibility|inherent worth)\b',
            ],
            TherapyModalityType.SOLUTION_FOCUSED: [
                r'\b(?:exception finding|miracle question|scaling questions|coping questions|best hope)\b',
                r'(?i)\bwhat\'s working|exceptions to problem|future goals|solution-oriented',
                r'\b(?:strengths-based|resources you have|when was it not a problem|positive change)\b',
            ],
            TherapyModalityType.FAMILY_SYSTEMS: [
                r'\b(?:family dynamics|systemic|family of origin|family rules|multigenerational|boundaries)\b',
                r'(?i)\brelationship patterns|family structure|extended family|systemic thinking',
                r'\b(?:enmeshment|disengagement|family roles|family communication)\b',
            ],
            TherapyModalityType.MOTIVATIONAL_INTERVIEWING: [
                r'\b(?:change talk|sustain talk|ambivalence|roll with resistance|decisional balance)\b',
                r'(?i)\bimportance ruler|confidence ruler|open questions|affirmations',
                r'\b(?:evoking change|readiness to change|motivational interviewing spirit)\b',
            ],
        }

    def _initialize_condition_keywords(self) -> Dict[MentalHealthConditionType, List[str]]:
        """Initialize keywords for mental health conditions"""
        return {
            MentalHealthConditionType.DEPRESSION: [
                r'\b(?:depressed|depression|hopeless|worthless|no energy|anhedonia|anhedonic|sleep|appetite|fatigue)\b',
                r'(?i)\blost interest|feeling down|guilt|worthlessness|concentration problems',
                r'\b(?:motivation|drive|pleasure|joy|happy|sad|crying|withdrawn)\b',
            ],
            MentalHealthConditionType.ANXIETY: [
                r'\b(?:anxious|anxiety|worry|worried|nervous|panic|phobia|phobic|fear|tension|stress|overthinking)\b',
                r'(?i)\bracing heart|sweating|shaking|breathing|restless|on edge',
                r'\b(?:anticipating|worries|fears|catastrophizing|avoidance|hypervigilance)\b',
            ],
            MentalHealthConditionType.PTSD: [
                r'\b(?:trauma|ptsd|flashbacks|nightmares|intrusive|hypervigilant|hypervigilance|trigger|avoidance)\b',
                r'(?i)\bstress disorder|traumatic event|re-experiencing|numbing|dissociation',
                r'\b(?:startle|hypervigilant|intrusive memories|emotional numbing)\b',
            ],
            MentalHealthConditionType.OCD: [
                r'\b(?:obsession|compulsion|compulsive|intrusive thoughts|checking|cleaning|contamination|repetitive)\b',
                r'(?i)\brumination|rituals|compulsions|obsessive|repetitive behaviors',
                r'\b(?:inappropriate thoughts|reassurance seeking|perfectionism|contamination fears)\b',
            ],
            MentalHealthConditionType.BIPOLAR: [
                r'\b(?:bipolar|mania|manic|hypomania|hypomanic|depression|mood swings|energy|irritability)\b',
                r'(?i)\bmanic episode|depressive episode|mood disorder|energy levels',
                r'\b(?:sleeping less|grandiose|racing thoughts|impulsivity|euphoria)\b',
            ],
            MentalHealthConditionType.EATING_DISORDER: [
                r'\b(?:eating disorder|anorexia|bulimia|binge|restriction|body image|weight|food|purging|calories)\b',
                r'(?i)\bcontrol weight|body image|food restriction|binge eating',
                r'\b(?:starvation|purging|food|eating|body dysmorphia|weight loss)\b',
            ],
            MentalHealthConditionType.SUBSTANCE_ABUSE: [
                r'\b(?:substance|addiction|alcohol|drugs|dependence|abuse|substance use|recovery|relapse)\b',
                r'(?i)\bdrug use|alcohol abuse|substance dependence|using substances',
                r'\b(?:craving|withdrawal|tolerance|substance seeking|using to cope)\b',
            ],
            MentalHealthConditionType.ADHD: [
                r'\b(?:adhd|attention|hyperactive|impulsive|inattentive|fidgeting|focus|concentration|distractible)\b',
                r'(?i)\battention deficit|hyperactivity|impulsivity|attention problems',
                r'\b(?:distracted|focus issues|concentration problems|fidgeting|organization)\b',
            ],
        }

    def calculate_confidence(self, matches: int, total_relevant_terms: int, 
                           pattern_specificity: float = 1.0) -> ConfidenceScore:
        """Calculate confidence based on pattern matches and context"""
        if total_relevant_terms == 0:
            return ConfidenceScore(score=0.0, explanation="No relevant terms found", 
                                 model_confidence=0.0, rule_based_confidence=0.0)
        
        # Base confidence from match ratio
        base_confidence = min(1.0, matches / max(total_relevant_terms, 1) * 1.5)  # Boost for multiple matches
        
        # Adjust for pattern specificity (how unique the matched terms are)
        adjusted_confidence = base_confidence * pattern_specificity
        
        # Cap at 1.0
        final_confidence = min(1.0, adjusted_confidence)
        
        explanation = f"Matched {matches} of {total_relevant_terms} relevant terms with specificity factor {pattern_specificity}. Base confidence {base_confidence:.2f}, final confidence {final_confidence:.2f}"
        
        return ConfidenceScore(
            score=final_confidence,
            explanation=explanation,
            model_confidence=0.0,  # Placeholder for ML model confidence
            rule_based_confidence=final_confidence,
            context_dependent=True
        )

    def detect_therapeutic_responses(self, conversation: Conversation) -> List[TherapeuticResponseLabel]:
        """Detect therapeutic response types in a conversation"""
        labels = []
        text_content = self._extract_text_from_conversation(conversation)
        
        for response_type, patterns in self.therapeutic_patterns.items():
            matches = 0
            for pattern in patterns:
                matches += len(re.findall(pattern, text_content, re.IGNORECASE))
            
            if matches > 0:
                total_patterns = len(patterns)
                confidence = self.calculate_confidence(matches, total_patterns, 
                                                     pattern_specificity=0.8)  # Adjust based on pattern uniqueness
                
                label = TherapeuticResponseLabel(
                    response_type=response_type,
                    metadata=LabelMetadata(
                        confidence=confidence.score,
                        confidence_explanation=confidence.explanation,
                        provenance=LabelProvenanceType.AUTOMATED_MODEL
                    )
                )
                labels.append(label)
        
        return labels

    def detect_crisis_level(self, conversation: Conversation) -> CrisisLabel:
        """Detect crisis level in a conversation"""
        text_content = self._extract_text_from_conversation(conversation)
        
        # Count matches for each crisis level
        crisis_counts = {}
        for level, keywords in self.crisis_keywords.items():
            count = 0
            for keyword in keywords:
                count += len(re.findall(keyword, text_content, re.IGNORECASE))
            crisis_counts[level] = count
        
        # Determine the highest level with matches
        current_level = CrisisLevelType.NO_RISK
        max_matches = 0
        
        # Process in order of severity (highest first)
        for level in [CrisisLevelType.IMMEDIATE_RISK, CrisisLevelType.HIGH_RISK, 
                     CrisisLevelType.MEDIUM_RISK, CrisisLevelType.LOW_RISK, CrisisLevelType.NO_RISK]:
            if crisis_counts.get(level, 0) > max_matches:
                current_level = level
                max_matches = crisis_counts[level]
        
        # Calculate confidence based on match strength
        if max_matches > 0:
            # Higher severity levels get higher confidence when matched
            severity_factor = {
                CrisisLevelType.NO_RISK: 0.5,
                CrisisLevelType.LOW_RISK: 0.7,
                CrisisLevelType.MEDIUM_RISK: 0.8,
                CrisisLevelType.HIGH_RISK: 0.9,
                CrisisLevelType.IMMEDIATE_RISK: 1.0
            }
            
            confidence = min(1.0, max_matches * 0.2 * severity_factor[current_level])
            explanation = f"Detected {max_matches} crisis-related terms at {current_level.value} level with confidence {confidence:.2f}"
        else:
            confidence = 1.0  # High confidence in no risk when nothing detected
            explanation = "No crisis-related terms detected, confidence in no risk assessment high"
        
        return CrisisLabel(
            crisis_level=current_level,
            metadata=LabelMetadata(
                confidence=confidence,
                confidence_explanation=explanation,
                provenance=LabelProvenanceType.AUTOMATED_MODEL
            )
        )

    def detect_therapy_modality(self, conversation: Conversation) -> Optional[TherapyModalityLabel]:
        """Detect therapy modality being used"""
        text_content = self._extract_text_from_conversation(conversation)
        
        modality_matches = {}
        for modality, keywords in self.modality_keywords.items():
            matches = 0
            for keyword in keywords:
                matches += len(re.findall(keyword, text_content, re.IGNORECASE))
            modality_matches[modality] = matches
        
        # Find the modality with most matches
        primary_modality = max(modality_matches, key=modality_matches.get)
        max_matches = modality_matches[primary_modality]
        
        if max_matches == 0:
            return None  # No clear modality detected
        
        # Calculate confidence
        total_keywords = sum(len(keywords) for keywords in self.modality_keywords.values())
        confidence_score = self.calculate_confidence(max_matches, total_keywords // len(self.modality_keywords), 0.6)
        
        return TherapyModalityLabel(
            modality=primary_modality,
            metadata=LabelMetadata(
                confidence=confidence_score.score,
                confidence_explanation=confidence_score.explanation,
                provenance=LabelProvenanceType.AUTOMATED_MODEL
            )
        )

    def detect_mental_health_conditions(self, conversation: Conversation) -> Optional[MentalHealthConditionLabel]:
        """Detect mental health conditions mentioned in conversation"""
        text_content = self._extract_text_from_conversation(conversation)
        
        detected_conditions = []
        condition_matches = {}
        
        for condition, keywords in self.condition_keywords.items():
            matches = 0
            for keyword in keywords:
                matches += len(re.findall(keyword, text_content, re.IGNORECASE))
            if matches > 0:
                condition_matches[condition] = matches
                detected_conditions.append(condition)
        
        if not detected_conditions:
            return None
        
        # Calculate overall confidence based on matches
        total_matches = sum(condition_matches.values())
        max_condition_matches = max(condition_matches.values())
        
        confidence_score = min(1.0, total_matches * 0.1)  # Scale confidence based on total matches
        explanation = f"Detected {len(detected_conditions)} conditions with {total_matches} total keyword matches"
        
        primary_condition = max(condition_matches, key=condition_matches.get) if condition_matches else None
        
        return MentalHealthConditionLabel(
            conditions=detected_conditions,
            primary_condition=primary_condition,
            metadata=LabelMetadata(
                confidence=confidence_score,
                confidence_explanation=explanation,
                provenance=LabelProvenanceType.AUTOMATED_MODEL
            )
        )

    def detect_demographics(self, conversation: Conversation) -> Optional[DemographicLabel]:
        """Basic demographic detection based on conversation content"""
        text_content = self._extract_text_from_conversation(conversation)
        
        # Simple heuristics for demographic detection
        demographics = []
        
        # Age indicators
        if re.search(r'\b(?:child|kid|youngster|juvenile|minor|underage)\b', text_content, re.IGNORECASE):
            demographics.append(DemographicType.AGE_CHILD)
        elif re.search(r'\b(?:teen|teenager|adolescent|high school|middle school)\b', text_content, re.IGNORECASE):
            demographics.append(DemographicType.AGE_TEEN)
        elif re.search(r'\b(?:elderly|senior|retired|grandparent|old)\b', text_content, re.IGNORECASE):
            demographics.append(DemographicType.AGE_ELDERLY)
        
        # Economic indicators
        if re.search(r'\b(?:low income|welfare|food stamps|financial aid|struggling financially|can\'t afford|poor)\b', text_content, re.IGNORECASE):
            demographics.append(DemographicType.SOCIOECONOMIC_LOW)
        elif re.search(r'\b(?:affluent|wealthy|rich|well-off|well-to-do|financially stable)\b', text_content, re.IGNORECASE):
            demographics.append(DemographicType.SOCIOECONOMIC_HIGH)
        
        if not demographics:
            return None
        
        # Calculate confidence based on number of demographic indicators found
        confidence = min(1.0, len(demographics) * 0.3)
        explanation = f"Detected {len(demographics)} demographic indicators"
        
        return DemographicLabel(
            demographics=demographics,
            metadata=LabelMetadata(
                confidence=confidence,
                confidence_explanation=explanation,
                provenance=LabelProvenanceType.AUTOMATED_MODEL
            )
        )

    def _extract_text_from_conversation(self, conversation: Conversation) -> str:
        """Extract all text content from a conversation"""
        text_parts = []
        for message in conversation.messages:
            text_parts.append(message.content)
        return " ".join(text_parts)

    def label_conversation(self, conversation: Conversation) -> LabelBundle:
        """Apply all automated labels to a conversation"""
        logger.info(f"Starting automated labeling for conversation {conversation.conversation_id}")
        
        # Apply all label types
        therapeutic_labels = self.detect_therapeutic_responses(conversation)
        crisis_label = self.detect_crisis_level(conversation)
        therapy_modality = self.detect_therapy_modality(conversation)
        mental_health_conditions = self.detect_mental_health_conditions(conversation)
        demographic_label = self.detect_demographics(conversation)
        
        # Create label bundle
        label_bundle = LabelBundle(
            conversation_id=conversation.conversation_id,
            therapeutic_response_labels=therapeutic_labels,
            crisis_label=crisis_label,
            therapy_modality_label=therapy_modality,
            mental_health_condition_label=mental_health_conditions,
            demographic_label=demographic_label
        )
        
        logger.info(f"Completed automated labeling for conversation {conversation.conversation_id}")
        logger.info(f"Generated {len(therapeutic_labels)} therapeutic response labels, "
                   f"crisis level: {crisis_label.crisis_level.value if crisis_label else 'None'}, "
                   f"modality: {therapy_modality.modality.value if therapy_modality else 'None'}")
        
        return label_bundle


# Example usage and testing
def create_default_automated_labeler() -> AutomatedLabeler:
    """Create a default instance of the automated labeler"""
    return AutomatedLabeler()


def test_automated_labeler():
    """Test the automated labeler with sample data"""
    from .conversation_schema import Conversation, Message
    
    # Create a sample conversation for testing
    conversation = Conversation()
    conversation.add_message("therapist", "I hear you saying that you've been feeling really down lately and having trouble finding joy in things you used to enjoy. That sounds very difficult.")
    conversation.add_message("client", "Yes, I just feel empty all the time and I don't know how to get out of this hole.")
    conversation.add_message("therapist", "You mentioned feeling like you're in a 'hole'. Can you tell me more about what that feels like for you?")
    
    # Create and run the labeler
    labeler = AutomatedLabeler()
    labels = labeler.label_conversation(conversation)
    
    print(f"Labels for conversation {conversation.conversation_id}:")
    print(f"Therapeutic responses: {len(labels.therapeutic_response_labels)}")
    for label in labels.therapeutic_response_labels:
        print(f"  - {label.response_type.value} (confidence: {label.metadata.confidence:.2f})")
    
    if labels.crisis_label:
        print(f"Crisis level: {labels.crisis_label.crisis_level.value} (confidence: {labels.crisis_label.metadata.confidence:.2f})")
    
    if labels.therapy_modality_label:
        print(f"Therapy modality: {labels.therapy_modality_label.modality.value} (confidence: {labels.therapy_modality_label.metadata.confidence:.2f})")
    
    if labels.mental_health_condition_label:
        conditions = [c.value for c in labels.mental_health_condition_label.conditions]
        print(f"Mental health conditions: {conditions} (confidence: {labels.mental_health_condition_label.metadata.confidence:.2f})")
    
    if labels.demographic_label:
        demographics = [d.value for d in labels.demographic_label.demographics]
        print(f"Demographics: {demographics} (confidence: {labels.demographic_label.metadata.confidence:.2f})")


if __name__ == "__main__":
    test_automated_labeler()