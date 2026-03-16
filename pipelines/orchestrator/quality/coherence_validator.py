#!/usr/bin/env python3
"""
Conversation Coherence Validation using Chain-of-Thought Reasoning
Validates logical flow and therapeutic reasoning coherence in conversations.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoherenceLevel(Enum):
    """Levels of conversation coherence."""

    HIGHLY_COHERENT = "highly_coherent"
    MODERATELY_COHERENT = "moderately_coherent"
    MINIMALLY_COHERENT = "minimally_coherent"
    INCOHERENT = "incoherent"


class ReasoningType(Enum):
    """Types of therapeutic reasoning."""

    LOGICAL_FLOW = "logical_flow"
    THERAPEUTIC_REASONING = "therapeutic_reasoning"
    INTERVENTION_SEQUENCE = "intervention_sequence"
    CONSISTENCY = "consistency"
    CONTEXTUAL_RELEVANCE = "contextual_relevance"


@dataclass
class CoherenceIssue:
    """Individual coherence issue."""

    issue_type: ReasoningType
    severity: str
    description: str
    location: str
    suggestion: str
    confidence: float


@dataclass
class CoherenceResult:
    """Complete coherence validation result."""

    conversation_id: str
    overall_coherence: CoherenceLevel
    coherence_score: float
    reasoning_scores: dict[ReasoningType, float] = field(default_factory=dict)
    coherence_issues: list[CoherenceIssue] = field(default_factory=list)
    cot_analysis: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class CoherenceValidator:
    """
    Conversation coherence validation using Chain-of-Thought reasoning.
    """

    def __init__(self):
        """Initialize the coherence validator."""
        self.validation_history: list[CoherenceResult] = []
        self.reasoning_patterns = self._load_reasoning_patterns()
        self.cot_templates = self._load_cot_templates()

    def _load_reasoning_patterns(self) -> dict[ReasoningType, dict[str, Any]]:
        """Load therapeutic reasoning patterns."""
        return {
            ReasoningType.LOGICAL_FLOW: {
                "indicators": [
                    "because",
                    "therefore",
                    "since",
                    "as a result",
                    "this leads to",
                    "consequently",
                    "due to",
                    "given that",
                    "it follows that",
                ],
                "transitions": [
                    "first",
                    "next",
                    "then",
                    "finally",
                    "meanwhile",
                    "however",
                    "on the other hand",
                    "in addition",
                    "furthermore",
                ],
                "weight": 0.25,
            },
            ReasoningType.THERAPEUTIC_REASONING: {
                "indicators": [
                    "this suggests",
                    "indicates that",
                    "pattern shows",
                    "evidence points to",
                    "assessment reveals",
                    "clinical picture",
                    "therapeutic goal",
                    "intervention rationale",
                    "research shows",
                    "studies indicate",
                    "clinical evidence",
                    "based on your symptoms",
                    "given your presentation",
                    "considering your history",
                    "this approach works because",
                    "the mechanism behind",
                    "therapeutic rationale",
                    "clinical reasoning",
                ],
                "clinical_terms": [
                    "symptoms",
                    "diagnosis",
                    "treatment",
                    "intervention",
                    "assessment",
                    "therapeutic",
                    "clinical",
                    "evidence-based",
                    "outcome",
                    "prognosis",
                    "etiology",
                    "pathophysiology",
                    "comorbidity",
                    "differential diagnosis",
                    "treatment plan",
                    "therapeutic alliance",
                    "case formulation",
                    "psychoeducation",
                ],
                "therapeutic_modalities": [
                    "cognitive behavioral therapy",
                    "cbt",
                    "dialectical behavior therapy",
                    "dbt",
                    "acceptance and commitment therapy",
                    "act",
                    "mindfulness-based",
                    "psychodynamic",
                    "humanistic",
                    "gestalt",
                    "emdr",
                    "exposure therapy",
                    "systematic desensitization",
                    "cognitive restructuring",
                    "behavioral activation",
                ],
                "clinical_reasoning_patterns": [
                    "if.*then",
                    "because.*therefore",
                    "given that.*we can",
                    "since.*it follows",
                    "research indicates.*so",
                    "evidence suggests.*thus",
                    "studies show.*which means",
                    "this is effective because",
                    "the rationale is",
                    "clinically speaking",
                ],
                "weight": 0.30,
            },
            ReasoningType.INTERVENTION_SEQUENCE: {
                "indicators": [
                    "before we",
                    "after you",
                    "once you've",
                    "when you're ready",
                    "building on",
                    "next step",
                    "progression",
                    "sequence",
                ],
                "sequence_markers": [
                    "step 1",
                    "step 2",
                    "phase",
                    "stage",
                    "level",
                    "tier",
                    "beginning",
                    "intermediate",
                    "advanced",
                ],
                "weight": 0.20,
            },
            ReasoningType.CONSISTENCY: {
                "indicators": [
                    "as mentioned",
                    "consistent with",
                    "aligns with",
                    "supports",
                    "contradicts",
                    "inconsistent",
                    "conflicts with",
                ],
                "consistency_markers": [
                    "previously",
                    "earlier",
                    "before",
                    "initially",
                    "originally",
                    "throughout",
                    "consistently",
                    "repeatedly",
                ],
                "weight": 0.15,
            },
            ReasoningType.CONTEXTUAL_RELEVANCE: {
                "indicators": [
                    "in your situation",
                    "given your",
                    "specific to",
                    "relevant to",
                    "applies to you",
                    "in this context",
                    "for your case",
                ],
                "context_markers": [
                    "personal",
                    "individual",
                    "specific",
                    "unique",
                    "particular",
                    "tailored",
                    "customized",
                    "personalized",
                ],
                "weight": 0.10,
            },
        }

    def _load_cot_templates(self) -> dict[str, str]:
        """Load Chain-of-Thought reasoning templates."""
        return {
            "problem_identification": "Given that {client_presents}, this suggests {clinical_assessment}",
            "intervention_rationale": "Because {assessment_finding}, the appropriate intervention is {intervention} which should lead to {expected_outcome}",
            "logical_progression": "First {step1}, then {step2}, which will result in {outcome}",
            "evidence_based": "Research shows that {intervention} is effective for {condition} because {mechanism}",
            "therapeutic_alliance": "By {therapist_action}, this builds {alliance_component} which facilitates {therapeutic_process}",
        }

    def validate_coherence(self, conversation: dict[str, Any]) -> CoherenceResult:
        """
        Validate conversation coherence using CoT reasoning.

        Args:
            conversation: Conversation data to validate

        Returns:
            CoherenceResult with detailed assessment
        """
        conversation_id = conversation.get("id", "unknown")
        logger.info(f"Validating coherence for conversation {conversation_id}")

        content = str(conversation.get("content", ""))
        turns = conversation.get("turns", [])

        # Analyze reasoning types
        reasoning_scores = self._analyze_reasoning_types(content, turns)

        # Perform CoT analysis
        cot_analysis = self._perform_cot_analysis(content, turns)

        # Identify coherence issues
        coherence_issues = self._identify_coherence_issues(content, reasoning_scores)

        # Calculate overall coherence score
        coherence_score = self._calculate_coherence_score(reasoning_scores, coherence_issues)

        # Determine coherence level
        overall_coherence = self._determine_coherence_level(coherence_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(coherence_issues, reasoning_scores)

        result = CoherenceResult(
            conversation_id=conversation_id,
            overall_coherence=overall_coherence,
            coherence_score=coherence_score,
            reasoning_scores=reasoning_scores,
            coherence_issues=coherence_issues,
            cot_analysis=cot_analysis,
            recommendations=recommendations,
        )

        self.validation_history.append(result)
        return result

    def _analyze_reasoning_types(
        self, content: str, turns: list[dict]
    ) -> dict[ReasoningType, float]:
        """
        Analyze different types of reasoning in the conversation.

        Returns:
            Dictionary mapping ReasoningType to a normalized float score.
        Raises:
            ValueError: If content is not a string or turns is not a list.
        """
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        if not isinstance(turns, list):
            raise ValueError("turns must be a list of dicts")

        scores: dict[ReasoningType, float] = {}
        for reasoning_type, patterns in self.reasoning_patterns.items():
            if reasoning_type == ReasoningType.LOGICAL_FLOW:
                scores[reasoning_type] = self._score_logical_flow(content, turns, patterns)
            elif reasoning_type == ReasoningType.THERAPEUTIC_REASONING:
                scores[reasoning_type] = self._score_therapeutic_reasoning(content, patterns)
            elif reasoning_type == ReasoningType.INTERVENTION_SEQUENCE:
                scores[reasoning_type] = self._score_intervention_sequence(content, turns, patterns)
            elif reasoning_type == ReasoningType.CONSISTENCY:
                scores[reasoning_type] = self._score_consistency(content, turns, patterns)
            elif reasoning_type == ReasoningType.CONTEXTUAL_RELEVANCE:
                scores[reasoning_type] = self._score_contextual_relevance(content, patterns)
        return scores

    def _count_matches(self, content: str, terms: list[str]) -> int:
        """Helper to count term matches in content_lower."""
        return sum(term in content.lower() for term in terms)

    def _score_logical_flow(self, content, turns, patterns):
        score = 0.1
        score += min(0.5, self._count_matches(content, patterns.get("indicators", [])) * 0.15)
        score += min(0.3, self._count_matches(content, patterns.get("transitions", [])) * 0.15)
        score += min(0.4, self._count_matches(content, patterns.get("clinical_terms", [])) * 0.08)
        score += self._analyze_turn_reasoning(turns, ReasoningType.LOGICAL_FLOW) * 0.4
        return min(1.0, score)

    def _score_therapeutic_reasoning(self, content, patterns):
        score = 0.1
        score += min(0.5, self._count_matches(content, patterns.get("indicators", [])) * 0.15)
        score += min(0.4, self._count_matches(content, patterns.get("clinical_terms", [])) * 0.08)
        score += min(
            0.3,
            self._count_matches(
                content,
                [
                    "cbt",
                    "cognitive behavioral",
                    "therapy",
                    "therapeutic",
                    "intervention",
                    "treatment",
                    "technique",
                    "approach",
                ],
            )
            * 0.08,
        )
        score += min(
            0.25, self._count_matches(content, patterns.get("therapeutic_modalities", [])) * 0.12
        )
        if "clinical_reasoning_patterns" in patterns:
            reasoning_pattern_matches = len(
                [
                    pattern
                    for pattern in patterns["clinical_reasoning_patterns"]
                    if re.search(pattern, content.lower())
                ]
            )
            score += min(0.2, reasoning_pattern_matches * 0.1)
        score += min(
            0.15,
            self._count_matches(
                content,
                [
                    "research shows",
                    "studies indicate",
                    "evidence suggests",
                    "meta-analysis",
                    "randomized controlled trial",
                    "systematic review",
                    "evidence-based practice",
                    "clinical guidelines",
                    "best practices",
                ],
            )
            * 0.08,
        )
        score += min(
            0.15,
            self._count_matches(
                content,
                [
                    "assessment reveals",
                    "clinical presentation",
                    "diagnostic criteria",
                    "symptom profile",
                    "risk factors",
                    "protective factors",
                    "case formulation",
                    "treatment planning",
                    "therapeutic goals",
                ],
            )
            * 0.08,
        )
        score += min(
            0.1,
            self._count_matches(
                content,
                [
                    "therapeutic alliance",
                    "rapport building",
                    "psychoeducation",
                    "skill building",
                    "homework assignment",
                    "between sessions",
                    "progress monitoring",
                    "treatment adherence",
                    "therapeutic relationship",
                ],
            )
            * 0.05,
        )
        return min(1.0, score)

    def _score_intervention_sequence(self, content, turns, patterns):
        score = 0.1
        score += min(0.5, self._count_matches(content, patterns.get("indicators", [])) * 0.15)
        score += min(0.3, self._count_matches(content, patterns.get("sequence_markers", [])) * 0.15)
        score += self._analyze_turn_reasoning(turns, ReasoningType.INTERVENTION_SEQUENCE) * 0.4
        return min(1.0, score)

    def _score_consistency(self, content, turns, patterns):
        score = 0.1
        score += min(0.5, self._count_matches(content, patterns.get("indicators", [])) * 0.15)
        score += min(
            0.3, self._count_matches(content, patterns.get("consistency_markers", [])) * 0.15
        )
        if len(turns) >= 2:
            score += 0.2
        if any(term in content.lower() for term in ["cbt", "cognitive", "behavioral", "anxiety"]):
            score += 0.2
        return min(1.0, score)

    def _score_contextual_relevance(self, content, patterns):
        score = 0.1
        score += min(0.5, self._count_matches(content, patterns.get("indicators", [])) * 0.15)
        score += min(0.3, self._count_matches(content, patterns.get("context_markers", [])) * 0.15)
        client_specific = ["you", "your", "you're", "you've"]
        score += min(0.3, self._count_matches(content, client_specific) * 0.05)
        if any(
            problem in content.lower() for problem in ["anxiety", "depression", "stress", "worry"]
        ):
            score += 0.2
        return min(1.0, score)

    def _analyze_turn_reasoning(self, turns: list[dict], reasoning_type: ReasoningType) -> float:
        """Analyze reasoning across conversation turns."""
        if len(turns) < 2:
            return 0.0

        reasoning_score = 0.0

        for i in range(1, len(turns)):
            current_turn = turns[i].get("text", "").lower()

            if reasoning_type == ReasoningType.LOGICAL_FLOW:
                # Check for logical connections between turns
                if any(
                    connector in current_turn for connector in ["because", "since", "therefore"]
                ):
                    reasoning_score += 0.3  # Increased from 0.2

                # Check for building on previous content
                if any(
                    ref in current_turn for ref in ["you mentioned", "as you said", "building on"]
                ):
                    reasoning_score += 0.3  # Increased from 0.2

                # Check for sequential flow indicators
                if any(seq in current_turn for seq in ["first", "then", "next", "after"]):
                    reasoning_score += 0.2

            elif reasoning_type == ReasoningType.INTERVENTION_SEQUENCE:
                # Check for sequential progression
                if any(seq in current_turn for seq in ["next", "then", "after", "once", "first"]):
                    reasoning_score += 0.3  # Increased from 0.2

                # Check for skill building progression
                if any(
                    prog in current_turn
                    for prog in ["practice", "try", "work on", "develop", "identify", "challenge"]
                ):
                    reasoning_score += 0.2  # Increased from 0.1

        return min(1.0, reasoning_score)

    def _perform_cot_analysis(self, content: str, turns: list[dict]) -> dict[str, Any]:
        """Perform Chain-of-Thought analysis."""
        analysis = {
            "reasoning_chains": self._identify_reasoning_chains(content),
            "logical_gaps": [],
            "therapeutic_flow": {},
            "evidence_integration": 0.0,
        }

        # Identify reasoning chains
        reasoning_chains = self._identify_reasoning_chains(content)
        analysis["reasoning_chains"] = reasoning_chains

        # Identify logical gaps
        logical_gaps = self._identify_logical_gaps(content, turns)
        analysis["logical_gaps"] = logical_gaps

        # Analyze therapeutic flow
        therapeutic_flow = self._analyze_therapeutic_flow(turns)
        analysis["therapeutic_flow"] = therapeutic_flow

        # Assess evidence integration
        evidence_score = self._assess_evidence_integration(content)
        analysis["evidence_integration"] = evidence_score

        return analysis

    def _identify_reasoning_chains(self, content: str) -> list[dict[str, Any]]:
        """Identify chains of reasoning in the conversation."""
        chains = []

        # Look for explicit reasoning patterns
        reasoning_patterns = [
            r"because (.+?), (?:therefore|so|thus) (.+)",
            r"since (.+?), (?:we can|this means|it follows) (.+)",
            r"given that (.+?), (?:the best|we should|I recommend) (.+)",
        ]

        for pattern in reasoning_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            chains.extend(
                [
                    {
                        "premise": match.group(1).strip(),
                        "conclusion": match.group(2).strip(),
                        "pattern": pattern,
                        "strength": 0.8,
                    }
                    for match in matches
                ]
            )

        # Look for implicit reasoning in therapeutic context
        therapeutic_patterns = [
            r"(?:you're experiencing|symptoms include) (.+?)[.,] (?:so|therefore) (.+)",
            r"(?:assessment shows|this indicates) (.+?)[.,] (?:which means|suggesting) (.+)",
        ]

        for pattern in therapeutic_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            chains.extend(
                [
                    {
                        "premise": match.group(1).strip(),
                        "conclusion": match.group(2).strip(),
                        "pattern": "therapeutic_reasoning",
                        "strength": 0.7,
                    }
                    for match in matches
                ]
            )

        return chains

    def _identify_logical_gaps(self, content: str, turns: list[dict]) -> list[dict[str, Any]]:
        """Identify logical gaps in reasoning."""
        gaps = []

        # Check for unsupported conclusions
        conclusion_markers = ["therefore", "so", "thus", "consequently", "as a result"]

        for marker in conclusion_markers:
            if marker in content.lower():
                # Look for supporting premises
                marker_pos = content.lower().find(marker)
                preceding_text = content[:marker_pos].lower()

                # Check if there's adequate support
                support_indicators = ["because", "since", "given", "due to", "as"]
                has_support = any(
                    indicator in preceding_text[-100:] for indicator in support_indicators
                )

                if not has_support:
                    gaps.append(
                        {
                            "type": "unsupported_conclusion",
                            "location": f"Near '{marker}'",
                            "description": "Conclusion drawn without clear supporting evidence",
                            "severity": "moderate",
                        }
                    )

        # Check for topic jumps
        if len(turns) >= 3:
            for i in range(2, len(turns)):
                current_topics = self._extract_topics(turns[i].get("text", ""))
                previous_topics = self._extract_topics(turns[i - 1].get("text", ""))

                # Check for topic continuity
                topic_overlap = len(set(current_topics) & set(previous_topics))
                if topic_overlap == 0 and len(current_topics) > 0 and len(previous_topics) > 0:
                    gaps.append(
                        {
                            "type": "topic_discontinuity",
                            "location": f"Turn {i}",
                            "description": "Abrupt topic change without transition",
                            "severity": "low",
                        }
                    )

        return gaps

    def _extract_topics(self, text: str) -> list[str]:
        """Extract main topics from text."""
        # Simple topic extraction based on key therapeutic terms
        topic_keywords = {
            "anxiety": ["anxiety", "anxious", "worry", "fear", "panic"],
            "depression": ["depression", "depressed", "sad", "hopeless", "mood"],
            "trauma": ["trauma", "ptsd", "abuse", "assault", "flashback"],
            "relationships": ["relationship", "partner", "family", "friends", "social"],
            "coping": ["coping", "manage", "handle", "deal with", "strategies"],
            "therapy": ["therapy", "treatment", "session", "therapeutic", "counseling"],
        }

        text_lower = text.lower()
        return [
            topic
            for topic, keywords in topic_keywords.items()
            if any(keyword in text_lower for keyword in keywords)
        ]

    def _analyze_therapeutic_flow(self, turns: list[dict]) -> dict[str, Any]:
        """Analyze the therapeutic flow of the conversation."""
        flow = {
            "assessment_to_intervention": 0.0,
            "skill_building_progression": 0.0,
            "therapeutic_alliance": 0.0,
            "client_engagement": 0.0,
        }

        if len(turns) < 2:
            return flow

        # Analyze assessment to intervention flow
        assessment_found = False

        for turn in turns:
            text = turn.get("text", "").lower()

            # Check for assessment
            if any(term in text for term in ["assess", "evaluate", "understand", "tell me about"]):
                assessment_found = True

            # Check for intervention (after assessment)
            if assessment_found and any(
                term in text for term in ["try", "practice", "work on", "technique"]
            ):
                flow["assessment_to_intervention"] = 0.8
                break

        # Analyze skill building progression
        skill_mentions = []
        for i, turn in enumerate(turns):
            text = turn.get("text", "").lower()
            if any(skill in text for skill in ["skill", "technique", "strategy", "tool"]):
                skill_mentions.append(i)

        if len(skill_mentions) >= 2:
            # Check for progression
            flow["skill_building_progression"] = 0.7

        # Analyze therapeutic alliance indicators
        alliance_score = 0.0
        for turn in turns:
            text = turn.get("text", "").lower()
            alliance_indicators = [
                "understand",
                "support",
                "together",
                "partnership",
                "safe",
                "comfortable",
                "trust",
                "validate",
                "hear you",
            ]
            alliance_count = len(
                [indicator for indicator in alliance_indicators if indicator in text]
            )
            alliance_score += min(0.2, alliance_count * 0.1)

        flow["therapeutic_alliance"] = min(1.0, alliance_score)

        return flow

    def _assess_evidence_integration(self, content: str) -> float:
        """Assess integration of evidence-based practices."""
        evidence_score = 0.0
        content_lower = content.lower()

        # Check for evidence-based language
        evidence_indicators = [
            "research shows",
            "studies indicate",
            "evidence suggests",
            "proven effective",
            "evidence-based",
            "clinical trials",
            "meta-analysis",
            "systematic review",
        ]

        evidence_count = len(
            [indicator for indicator in evidence_indicators if indicator in content_lower]
        )
        evidence_score += min(0.4, evidence_count * 0.2)

        # Check for specific therapeutic modalities
        modalities = [
            "cognitive behavioral",
            "cbt",
            "dialectical behavior",
            "dbt",
            "acceptance commitment",
            "act",
            "emdr",
            "mindfulness-based",
        ]

        modality_count = len([modality for modality in modalities if modality in content_lower])
        evidence_score += min(0.3, modality_count * 0.15)

        # Check for outcome references
        outcome_indicators = [
            "effective for",
            "helps with",
            "reduces",
            "improves",
            "outcome",
            "result",
            "benefit",
            "success rate",
        ]

        outcome_count = len(
            [indicator for indicator in outcome_indicators if indicator in content_lower]
        )
        evidence_score += min(0.3, outcome_count * 0.1)

        return min(1.0, evidence_score)

    def _identify_coherence_issues(
        self, content: str, reasoning_scores: dict[ReasoningType, float]
    ) -> list[CoherenceIssue]:
        """Identify specific coherence issues."""
        issues = []

        # Check for low reasoning scores with much more reasonable thresholds
        for reasoning_type, score in reasoning_scores.items():
            if score < 0.1:  # Only flag extremely low scores as high severity
                issues.append(
                    CoherenceIssue(
                        issue_type=reasoning_type,
                        severity="high",
                        description=f"Poor {reasoning_type.value.replace('_', ' ')} in conversation",
                        location="Overall conversation",
                        suggestion=f"Improve {reasoning_type.value.replace('_', ' ')} with clearer connections",
                        confidence=0.8,
                    )
                )
            elif score < 0.2:  # Very low threshold for moderate issues
                issues.append(
                    CoherenceIssue(
                        issue_type=reasoning_type,
                        severity="moderate",
                        description=f"Weak {reasoning_type.value.replace('_', ' ')} in conversation",
                        location="Overall conversation",
                        suggestion=f"Strengthen {reasoning_type.value.replace('_', ' ')} with better transitions",
                        confidence=0.6,
                    )
                )

        # Check for specific coherence problems - more sophisticated contradiction detection
        content_lower = content.lower()

        # Look for actual contradictions, not just presence of negative and positive words
        contradiction_patterns = [
            (
                r"(always|never|definitely|certainly)\s+.*?\s+(not|never|don't|can't)",
                "absolute_contradiction",
            ),
            (
                r"(should|must|need to)\s+.*?\s+(shouldn't|mustn't|don't need)",
                "directive_contradiction",
            ),
            (
                r"(effective|helpful|works)\s+.*?\s+(ineffective|unhelpful|doesn't work)",
                "effectiveness_contradiction",
            ),
        ]

        issues.extend(
            [
                CoherenceIssue(
                    issue_type=ReasoningType.CONSISTENCY,
                    severity="moderate",
                    description=f"Potential {contradiction_type.replace('_', ' ')} detected",
                    location="Content analysis",
                    suggestion="Review for contradictory statements and clarify",
                    confidence=0.7,
                )
                for pattern, contradiction_type in contradiction_patterns
                if re.search(pattern, content_lower)
            ]
        )

        return issues

    def _calculate_coherence_score(
        self, reasoning_scores: dict[ReasoningType, float], coherence_issues: list[CoherenceIssue]
    ) -> float:
        """Calculate overall coherence score."""
        # Weighted average of reasoning scores
        weighted_score = sum(
            score * self.reasoning_patterns[reasoning_type]["weight"]
            for reasoning_type, score in reasoning_scores.items()
        )

        # Apply more reasonable penalty for coherence issues
        issue_penalty = 0.0
        for issue in coherence_issues:
            if issue.severity == "high":
                issue_penalty += 0.04  # Reduced from 0.05
            elif issue.severity == "moderate":
                issue_penalty += 0.02  # Reduced from 0.03
            else:
                issue_penalty += 0.01  # Same

        # Cap the penalty to prevent scores from going to 0
        issue_penalty = min(issue_penalty, 0.25)  # Reduced from 0.3

        final_score = max(0.1, weighted_score - issue_penalty)  # Minimum score of 0.1
        return min(1.0, final_score)

    def _determine_coherence_level(self, coherence_score: float) -> CoherenceLevel:
        """Determine overall coherence level."""
        if coherence_score >= 0.7:
            return CoherenceLevel.HIGHLY_COHERENT
        if coherence_score >= 0.5:
            return CoherenceLevel.MODERATELY_COHERENT
        if coherence_score >= 0.3:
            return CoherenceLevel.MINIMALLY_COHERENT
        return CoherenceLevel.INCOHERENT

    def _generate_recommendations(
        self, coherence_issues: list[CoherenceIssue], reasoning_scores: dict[ReasoningType, float]
    ) -> list[str]:
        """Generate recommendations for improving coherence."""
        # Address specific issues
        recommendations = [
            issue.suggestion for issue in coherence_issues if issue.severity in ["high", "moderate"]
        ]

        # Address low reasoning scores
        for reasoning_type, score in reasoning_scores.items():
            if score < 0.5:
                if reasoning_type == ReasoningType.LOGICAL_FLOW:
                    recommendations.append(
                        "Use more logical connectors (because, therefore, since)"
                    )
                elif reasoning_type == ReasoningType.THERAPEUTIC_REASONING:
                    recommendations.append("Provide clearer clinical rationale for interventions")
                elif reasoning_type == ReasoningType.INTERVENTION_SEQUENCE:
                    recommendations.append(
                        "Establish clearer progression in therapeutic interventions"
                    )
                elif reasoning_type == ReasoningType.CONSISTENCY:
                    recommendations.append("Ensure consistency throughout the conversation")
                elif reasoning_type == ReasoningType.CONTEXTUAL_RELEVANCE:
                    recommendations.append("Make interventions more specific to client context")

        # General recommendations
        if len([s for s in reasoning_scores.values() if s < 0.6]) >= 3:
            recommendations.append("Overall coherence needs improvement - focus on clear reasoning")

        return list(set(recommendations))  # Remove duplicates

    def get_validation_summary(self) -> dict[str, Any]:
        """Get validation summary statistics."""
        if not self.validation_history:
            return {"message": "No validations performed yet"}

        total_validations = len(self.validation_history)

        # Coherence level distribution
        coherence_distribution = {
            level.value: len(
                [result for result in self.validation_history if result.overall_coherence == level]
            )
            for level in CoherenceLevel
        }

        # Average coherence score
        avg_coherence_score = (
            sum(r.coherence_score for r in self.validation_history) / total_validations
        )

        # Average reasoning scores
        avg_reasoning_scores = {}
        for reasoning_type in ReasoningType:
            scores = [r.reasoning_scores.get(reasoning_type, 0.0) for r in self.validation_history]
            avg_reasoning_scores[reasoning_type.value] = sum(scores) / len(scores)

        # Most common issues
        all_issues = [
            issue for result in self.validation_history for issue in result.coherence_issues
        ]
        issue_counts = {}
        for issue in all_issues:
            issue_type = issue.issue_type.value
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        return {
            "total_validations": total_validations,
            "coherence_distribution": coherence_distribution,
            "average_coherence_score": avg_coherence_score,
            "average_reasoning_scores": avg_reasoning_scores,
            "common_coherence_issues": dict(
                sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            "last_validation": self.validation_history[-1].timestamp.isoformat(),
        }


def main():
    """Example usage of the CoherenceValidator."""
    validator = CoherenceValidator()

    # Example conversations
    sample_conversations = [
        {
            "id": "coherent_001",
            "content": "Since you're experiencing anxiety symptoms, I recommend we start with cognitive behavioral techniques. First, we'll identify your thought patterns, then challenge negative thoughts, which should help reduce your anxiety levels.",
            "turns": [
                {"speaker": "user", "text": "I'm having anxiety attacks."},
                {
                    "speaker": "therapist",
                    "text": "Since you're experiencing anxiety, let's start with CBT techniques.",
                },
                {
                    "speaker": "therapist",
                    "text": "First, we'll identify thought patterns, then challenge negative thoughts.",
                },
            ],
        },
        {
            "id": "incoherent_001",
            "content": "You have depression. Try meditation. Also, your childhood affects everything. Let's talk about your job.",
            "turns": [
                {"speaker": "user", "text": "I feel sad sometimes."},
                {"speaker": "therapist", "text": "You have depression. Try meditation."},
                {
                    "speaker": "therapist",
                    "text": "Your childhood affects everything. Let's talk about your job.",
                },
            ],
        },
    ]

    # Validate conversations
    for conversation in sample_conversations:
        logger.info(f"\n=== Validating {conversation['id']} ===")
        result = validator.validate_coherence(conversation)

        logger.info(f"Overall Coherence: {result.overall_coherence.value}")
        logger.info(f"Coherence Score: {result.coherence_score:.3f}")
        logger.info(f"Reasoning Chains: {len(result.cot_analysis['reasoning_chains'])}")
        logger.info(f"Logical Gaps: {len(result.cot_analysis['logical_gaps'])}")
        logger.info(f"Issues: {len(result.coherence_issues)}")

        if result.coherence_issues:
            logger.info("Top Issues:")
            for issue in result.coherence_issues[:3]:
                logger.info(f"  - {issue.severity.upper()}: {issue.description}")

        if result.recommendations:
            logger.info("Recommendations:")
            for rec in result.recommendations[:3]:
                logger.info(f"  - {rec}")

    # Print summary
    logger.info("\n=== VALIDATION SUMMARY ===")
    summary = validator.get_validation_summary()
    logger.info(f"Total Validations: {summary['total_validations']}")
    logger.info(f"Average Coherence Score: {summary['average_coherence_score']:.3f}")
    logger.info(f"Coherence Distribution: {summary['coherence_distribution']}")


if __name__ == "__main__":
    main()
