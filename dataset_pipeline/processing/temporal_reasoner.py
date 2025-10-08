#!/usr/bin/env python3
"""
Temporal Reasoning Integration System for Task 6.17
Integrates temporal reasoning capabilities for time-based therapeutic planning.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalRelation(Enum):
    """Types of temporal relations."""
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    OVERLAPS = "overlaps"
    MEETS = "meets"
    STARTS = "starts"
    FINISHES = "finishes"
    EQUALS = "equals"
    CONTAINS = "contains"
    CONCURRENT = "concurrent"


class TimeScale(Enum):
    """Time scales for analysis."""
    IMMEDIATE = "immediate"  # Minutes to hours
    SHORT_TERM = "short_term"  # Days to weeks
    MEDIUM_TERM = "medium_term"  # Weeks to months
    LONG_TERM = "long_term"  # Months to years
    CHRONIC = "chronic"  # Years to lifetime


class TemporalPattern(Enum):
    """Temporal patterns in mental health."""
    EPISODIC = "episodic"  # Recurring episodes
    PROGRESSIVE = "progressive"  # Worsening over time
    CYCLICAL = "cyclical"  # Regular cycles
    SEASONAL = "seasonal"  # Seasonal patterns
    TRIGGERED = "triggered"  # Event-triggered
    CHRONIC_STABLE = "chronic_stable"  # Stable chronic condition
    RECOVERY = "recovery"  # Recovery trajectory


@dataclass
class TemporalEvent:
    """Temporal event in conversation."""
    event_id: str
    event_type: str
    description: str
    timestamp: datetime | None
    duration: timedelta | None
    confidence: float
    temporal_markers: list[str]
    context: str


@dataclass
class TemporalSequence:
    """Sequence of temporal events."""
    sequence_id: str
    events: list[TemporalEvent]
    temporal_relations: list[tuple[str, TemporalRelation, str]]
    pattern: TemporalPattern
    time_scale: TimeScale
    sequence_confidence: float


@dataclass
class TemporalAnalysis:
    """Complete temporal analysis result."""
    conversation_id: str
    temporal_events: list[TemporalEvent]
    temporal_sequences: list[TemporalSequence]
    dominant_pattern: TemporalPattern
    time_horizon: TimeScale
    temporal_coherence: float
    therapeutic_timeline: dict[str, Any]
    recommendations: list[str]


class TemporalReasoner:
    """
    Temporal reasoning integration system for therapeutic planning.
    """

    def __init__(self):
        """Initialize the temporal reasoner."""
        self.temporal_markers = self._load_temporal_markers()
        self.mental_health_patterns = self._load_mental_health_temporal_patterns()
        self.therapeutic_timelines = self._load_therapeutic_timelines()

        logger.info("TemporalReasoner initialized successfully")

    def _load_temporal_markers(self) -> dict[str, list[str]]:
        """Load temporal markers for different time scales."""
        return {
            "immediate": [
                "now", "right now", "currently", "at the moment", "today",
                "this morning", "this afternoon", "tonight", "just now"
            ],
            "short_term": [
                "yesterday", "tomorrow", "this week", "last week", "next week",
                "recently", "soon", "in a few days", "a couple days ago"
            ],
            "medium_term": [
                "this month", "last month", "next month", "a few weeks ago",
                "in a few weeks", "lately", "these days", "for a while"
            ],
            "long_term": [
                "this year", "last year", "next year", "months ago",
                "in months", "for years", "since childhood", "growing up"
            ],
            "chronic": [
                "always", "never", "forever", "all my life", "since I was young",
                "for as long as I can remember", "constantly", "continuously"
            ],
            "frequency": [
                "daily", "weekly", "monthly", "yearly", "often", "sometimes",
                "rarely", "occasionally", "frequently", "regularly"
            ],
            "duration": [
                "for hours", "for days", "for weeks", "for months", "for years",
                "briefly", "temporarily", "permanently", "ongoing"
            ],
            "sequence": [
                "first", "then", "next", "after", "before", "during", "while",
                "meanwhile", "subsequently", "previously", "finally"
            ]
        }

    def _load_mental_health_temporal_patterns(self) -> dict[str, dict[str, Any]]:
        """Load temporal patterns for mental health conditions."""
        return {
            "depression": {
                "typical_patterns": [TemporalPattern.EPISODIC, TemporalPattern.CHRONIC_STABLE],
                "episode_duration": {"min": "weeks", "max": "months"},
                "recovery_time": {"typical": "months", "range": "weeks to years"},
                "seasonal_component": True,
                "triggers": ["life_events", "stress", "seasonal_changes"]
            },
            "anxiety": {
                "typical_patterns": [TemporalPattern.TRIGGERED, TemporalPattern.CHRONIC_STABLE],
                "episode_duration": {"min": "minutes", "max": "hours"},
                "recovery_time": {"typical": "hours to days", "range": "minutes to weeks"},
                "seasonal_component": False,
                "triggers": ["specific_situations", "stress", "physical_symptoms"]
            },
            "bipolar": {
                "typical_patterns": [TemporalPattern.CYCLICAL, TemporalPattern.EPISODIC],
                "episode_duration": {"min": "days", "max": "months"},
                "recovery_time": {"typical": "weeks", "range": "days to months"},
                "seasonal_component": True,
                "triggers": ["sleep_disruption", "stress", "medication_changes"]
            },
            "ptsd": {
                "typical_patterns": [TemporalPattern.TRIGGERED, TemporalPattern.CHRONIC_STABLE],
                "episode_duration": {"min": "minutes", "max": "days"},
                "recovery_time": {"typical": "months to years", "range": "weeks to lifetime"},
                "seasonal_component": False,
                "triggers": ["trauma_reminders", "anniversaries", "stress"]
            }
        }

    def _load_therapeutic_timelines(self) -> dict[str, dict[str, Any]]:
        """Load therapeutic intervention timelines."""
        return {
            "crisis_intervention": {
                "immediate": ["safety_assessment", "crisis_stabilization"],
                "short_term": ["safety_planning", "support_mobilization"],
                "medium_term": ["therapy_initiation", "medication_evaluation"],
                "long_term": ["ongoing_therapy", "relapse_prevention"]
            },
            "depression_treatment": {
                "immediate": ["risk_assessment", "symptom_stabilization"],
                "short_term": ["therapy_engagement", "medication_trial"],
                "medium_term": ["symptom_improvement", "functional_recovery"],
                "long_term": ["maintenance_therapy", "relapse_prevention"]
            },
            "anxiety_treatment": {
                "immediate": ["symptom_management", "coping_strategies"],
                "short_term": ["exposure_therapy", "skill_building"],
                "medium_term": ["generalization", "independence"],
                "long_term": ["maintenance", "booster_sessions"]
            },
            "trauma_treatment": {
                "immediate": ["stabilization", "safety_establishment"],
                "short_term": ["trauma_processing_prep", "resource_building"],
                "medium_term": ["trauma_processing", "integration"],
                "long_term": ["post_traumatic_growth", "maintenance"]
            }
        }

    def analyze_temporal_patterns(self, conversation: dict[str, Any]) -> TemporalAnalysis:
        """Analyze temporal patterns in conversation."""
        try:
            content = self._extract_content(conversation)

            # Extract temporal events
            temporal_events = self._extract_temporal_events(content, conversation.get("conversation_id", "unknown"))

            # Identify temporal sequences
            temporal_sequences = self._identify_temporal_sequences(temporal_events)

            # Determine dominant pattern
            dominant_pattern = self._determine_dominant_pattern(temporal_sequences)

            # Determine time horizon
            time_horizon = self._determine_time_horizon(temporal_events)

            # Calculate temporal coherence
            temporal_coherence = self._calculate_temporal_coherence(temporal_sequences)

            # Generate therapeutic timeline
            therapeutic_timeline = self._generate_therapeutic_timeline(
                dominant_pattern, time_horizon, temporal_events
            )

            # Generate recommendations
            recommendations = self._generate_temporal_recommendations(
                dominant_pattern, time_horizon, temporal_events
            )

            analysis = TemporalAnalysis(
                conversation_id=conversation.get("conversation_id", "unknown"),
                temporal_events=temporal_events,
                temporal_sequences=temporal_sequences,
                dominant_pattern=dominant_pattern,
                time_horizon=time_horizon,
                temporal_coherence=temporal_coherence,
                therapeutic_timeline=therapeutic_timeline,
                recommendations=recommendations
            )

            logger.info(f"Temporal analysis completed: {len(temporal_events)} events, "
                       f"pattern: {dominant_pattern.value}, coherence: {temporal_coherence:.2f}")

            return analysis

        except Exception as e:
            logger.error(f"Error in temporal analysis: {e}")
            return self._get_default_temporal_analysis(conversation.get("conversation_id", "unknown"))

    def _extract_content(self, conversation: dict[str, Any]) -> str:
        """Extract content from conversation."""
        content = ""
        if "turns" in conversation:
            for turn in conversation["turns"]:
                if isinstance(turn, dict) and "content" in turn:
                    content += turn["content"] + " "
        elif "content" in conversation:
            content = conversation["content"]
        return content.strip()

    def _extract_temporal_events(self, content: str, conversation_id: str) -> list[TemporalEvent]:
        """Extract temporal events from content."""
        events = []
        sentences = content.split(".")

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            # Look for temporal markers
            found_markers = []
            detected_time_scale = None

            for time_scale, markers in self.temporal_markers.items():
                for marker in markers:
                    if marker.lower() in sentence.lower():
                        found_markers.append(marker)
                        if not detected_time_scale:
                            detected_time_scale = time_scale

            if found_markers:
                # Extract event type and description
                event_type = self._classify_event_type(sentence)

                event = TemporalEvent(
                    event_id=f"{conversation_id}_event_{i}",
                    event_type=event_type,
                    description=sentence,
                    timestamp=self._extract_timestamp(sentence),
                    duration=self._extract_duration(sentence),
                    confidence=len(found_markers) / 10.0,  # Simple confidence based on marker count
                    temporal_markers=found_markers,
                    context=detected_time_scale or "unknown"
                )

                events.append(event)

        return events

    def _classify_event_type(self, sentence: str) -> str:
        """Classify the type of event described in sentence."""
        sentence_lower = sentence.lower()

        # Mental health events
        if any(word in sentence_lower for word in ["depressed", "sad", "down", "hopeless"]):
            return "depression_episode"
        if any(word in sentence_lower for word in ["anxious", "worried", "panic", "fear"]):
            return "anxiety_episode"
        if any(word in sentence_lower for word in ["therapy", "counseling", "treatment"]):
            return "therapeutic_intervention"
        if any(word in sentence_lower for word in ["medication", "pills", "prescription"]):
            return "medication_event"
        if any(word in sentence_lower for word in ["crisis", "emergency", "hospital"]):
            return "crisis_event"
        if any(word in sentence_lower for word in ["better", "improved", "recovery"]):
            return "improvement_event"
        if any(word in sentence_lower for word in ["worse", "deteriorated", "relapse"]):
            return "deterioration_event"
        return "general_event"

    def _extract_timestamp(self, sentence: str) -> datetime | None:
        """Extract timestamp from sentence (simplified)."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated NLP for date/time extraction

        sentence_lower = sentence.lower()
        now = datetime.now()

        if "today" in sentence_lower:
            return now.replace(hour=12, minute=0, second=0, microsecond=0)
        if "yesterday" in sentence_lower:
            return (now - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
        if "last week" in sentence_lower:
            return now - timedelta(weeks=1)
        if "last month" in sentence_lower:
            return now - timedelta(days=30)
        if "last year" in sentence_lower:
            return now - timedelta(days=365)

        return None

    def _extract_duration(self, sentence: str) -> timedelta | None:
        """Extract duration from sentence (simplified)."""
        sentence_lower = sentence.lower()

        if "for hours" in sentence_lower:
            return timedelta(hours=3)  # Average
        if "for days" in sentence_lower:
            return timedelta(days=3)
        if "for weeks" in sentence_lower:
            return timedelta(weeks=3)
        if "for months" in sentence_lower:
            return timedelta(days=90)
        if "for years" in sentence_lower:
            return timedelta(days=1095)

        return None

    def _identify_temporal_sequences(self, events: list[TemporalEvent]) -> list[TemporalSequence]:
        """Identify temporal sequences from events."""
        if len(events) < 2:
            return []

        sequences = []

        # Group events by type and temporal proximity
        event_groups = self._group_events_by_type(events)

        for event_type, type_events in event_groups.items():
            if len(type_events) >= 2:
                # Create sequence for this event type
                relations = self._determine_temporal_relations(type_events)
                pattern = self._identify_pattern_in_sequence(type_events)
                time_scale = self._determine_sequence_time_scale(type_events)

                sequence = TemporalSequence(
                    sequence_id=f"seq_{event_type}_{len(sequences)}",
                    events=type_events,
                    temporal_relations=relations,
                    pattern=pattern,
                    time_scale=time_scale,
                    sequence_confidence=sum(e.confidence for e in type_events) / len(type_events)
                )

                sequences.append(sequence)

        return sequences

    def _group_events_by_type(self, events: list[TemporalEvent]) -> dict[str, list[TemporalEvent]]:
        """Group events by type."""
        groups = {}
        for event in events:
            if event.event_type not in groups:
                groups[event.event_type] = []
            groups[event.event_type].append(event)

        return groups

    def _determine_temporal_relations(self, events: list[TemporalEvent]) -> list[tuple[str, TemporalRelation, str]]:
        """Determine temporal relations between events."""
        relations = []

        for i in range(len(events) - 1):
            event1 = events[i]
            event2 = events[i + 1]

            # Simple temporal relation determination
            if event1.timestamp and event2.timestamp:
                if event1.timestamp < event2.timestamp:
                    relation = TemporalRelation.BEFORE
                elif event1.timestamp > event2.timestamp:
                    relation = TemporalRelation.AFTER
                else:
                    relation = TemporalRelation.CONCURRENT
            else:
                # Default to sequence order
                relation = TemporalRelation.BEFORE

            relations.append((event1.event_id, relation, event2.event_id))

        return relations

    def _identify_pattern_in_sequence(self, events: list[TemporalEvent]) -> TemporalPattern:
        """Identify temporal pattern in sequence of events."""
        if len(events) < 2:
            return TemporalPattern.CHRONIC_STABLE

        # Analyze event types for pattern recognition
        event_types = [e.event_type for e in events]

        # Look for episodic patterns (alternating states)
        if any("episode" in et for et in event_types):
            return TemporalPattern.EPISODIC

        # Look for recovery patterns
        if any("improvement" in et for et in event_types) and any("deterioration" in et for et in event_types):
            return TemporalPattern.RECOVERY

        # Look for triggered patterns
        if any("crisis" in et for et in event_types):
            return TemporalPattern.TRIGGERED

        # Default to chronic stable
        return TemporalPattern.CHRONIC_STABLE

    def _determine_sequence_time_scale(self, events: list[TemporalEvent]) -> TimeScale:
        """Determine time scale of sequence."""
        contexts = [e.context for e in events if e.context != "unknown"]

        if not contexts:
            return TimeScale.MEDIUM_TERM

        # Determine dominant time scale
        context_counts = {}
        for context in contexts:
            context_counts[context] = context_counts.get(context, 0) + 1

        dominant_context = max(context_counts, key=context_counts.get)

        context_to_scale = {
            "immediate": TimeScale.IMMEDIATE,
            "short_term": TimeScale.SHORT_TERM,
            "medium_term": TimeScale.MEDIUM_TERM,
            "long_term": TimeScale.LONG_TERM,
            "chronic": TimeScale.CHRONIC
        }

        return context_to_scale.get(dominant_context, TimeScale.MEDIUM_TERM)

    def _determine_dominant_pattern(self, sequences: list[TemporalSequence]) -> TemporalPattern:
        """Determine dominant temporal pattern."""
        if not sequences:
            return TemporalPattern.CHRONIC_STABLE

        # Count pattern occurrences
        pattern_counts = {}
        for seq in sequences:
            pattern_counts[seq.pattern] = pattern_counts.get(seq.pattern, 0) + 1

        return max(pattern_counts, key=pattern_counts.get)

    def _determine_time_horizon(self, events: list[TemporalEvent]) -> TimeScale:
        """Determine overall time horizon."""
        if not events:
            return TimeScale.MEDIUM_TERM

        contexts = [e.context for e in events if e.context != "unknown"]

        # Determine the longest time scale mentioned
        scale_priority = {
            "chronic": 5,
            "long_term": 4,
            "medium_term": 3,
            "short_term": 2,
            "immediate": 1
        }

        max_scale = "medium_term"
        max_priority = 0

        for context in contexts:
            priority = scale_priority.get(context, 0)
            if priority > max_priority:
                max_priority = priority
                max_scale = context

        context_to_scale = {
            "immediate": TimeScale.IMMEDIATE,
            "short_term": TimeScale.SHORT_TERM,
            "medium_term": TimeScale.MEDIUM_TERM,
            "long_term": TimeScale.LONG_TERM,
            "chronic": TimeScale.CHRONIC
        }

        return context_to_scale.get(max_scale, TimeScale.MEDIUM_TERM)

    def _calculate_temporal_coherence(self, sequences: list[TemporalSequence]) -> float:
        """Calculate temporal coherence score."""
        if not sequences:
            return 0.5

        # Calculate based on sequence confidence and consistency
        avg_confidence = sum(seq.sequence_confidence for seq in sequences) / len(sequences)

        # Consistency bonus if patterns are similar
        patterns = [seq.pattern for seq in sequences]
        pattern_consistency = len(set(patterns)) / len(patterns) if patterns else 1.0
        consistency_bonus = 1.0 - pattern_consistency

        coherence = (avg_confidence + consistency_bonus) / 2.0
        return min(1.0, coherence)

    def _generate_therapeutic_timeline(self, pattern: TemporalPattern,
                                     time_horizon: TimeScale,
                                     events: list[TemporalEvent]) -> dict[str, Any]:
        """Generate therapeutic timeline based on temporal analysis."""
        # Determine appropriate therapeutic approach based on pattern
        if pattern == TemporalPattern.TRIGGERED:
            approach = "crisis_intervention"
        elif pattern in [TemporalPattern.EPISODIC, TemporalPattern.CYCLICAL]:
            approach = "depression_treatment"  # Default for episodic patterns
        elif any("anxiety" in e.event_type for e in events):
            approach = "anxiety_treatment"
        elif any("trauma" in e.description.lower() for e in events):
            approach = "trauma_treatment"
        else:
            approach = "depression_treatment"  # Default

        timeline = self.therapeutic_timelines.get(approach, {})

        return {
            "approach": approach,
            "pattern": pattern.value,
            "time_horizon": time_horizon.value,
            "phases": timeline,
            "estimated_duration": self._estimate_treatment_duration(pattern, time_horizon),
            "key_milestones": self._generate_milestones(pattern, time_horizon)
        }

    def _estimate_treatment_duration(self, pattern: TemporalPattern, time_horizon: TimeScale) -> str:
        """Estimate treatment duration based on pattern and time horizon."""
        if pattern == TemporalPattern.TRIGGERED and time_horizon == TimeScale.IMMEDIATE:
            return "1-3 months"
        if pattern == TemporalPattern.EPISODIC:
            return "6-12 months"
        if pattern == TemporalPattern.CHRONIC_STABLE:
            return "12+ months"
        if pattern == TemporalPattern.RECOVERY:
            return "3-6 months"
        return "6-12 months"

    def _generate_milestones(self, pattern: TemporalPattern, time_horizon: TimeScale) -> list[str]:
        """Generate key therapeutic milestones."""
        milestones = []

        if pattern == TemporalPattern.TRIGGERED:
            milestones = [
                "Crisis stabilization (1-2 weeks)",
                "Safety plan implementation (2-4 weeks)",
                "Trigger identification (1-2 months)",
                "Coping strategy mastery (2-3 months)"
            ]
        elif pattern == TemporalPattern.EPISODIC:
            milestones = [
                "Episode pattern recognition (1-2 months)",
                "Early warning sign identification (2-3 months)",
                "Relapse prevention planning (3-4 months)",
                "Long-term maintenance (6+ months)"
            ]
        elif pattern == TemporalPattern.CHRONIC_STABLE:
            milestones = [
                "Symptom stabilization (1-3 months)",
                "Functional improvement (3-6 months)",
                "Quality of life enhancement (6-12 months)",
                "Maintenance and monitoring (ongoing)"
            ]

        return milestones

    def _generate_temporal_recommendations(self, pattern: TemporalPattern,
                                         time_horizon: TimeScale,
                                         events: list[TemporalEvent]) -> list[str]:
        """Generate temporal-based recommendations."""
        recommendations = []

        # Pattern-based recommendations
        if pattern == TemporalPattern.TRIGGERED:
            recommendations.extend([
                "Develop comprehensive safety plan",
                "Identify and address specific triggers",
                "Establish crisis support network",
                "Practice grounding and coping techniques"
            ])
        elif pattern == TemporalPattern.EPISODIC:
            recommendations.extend([
                "Track mood and symptom patterns",
                "Develop early warning system",
                "Create episode management plan",
                "Consider maintenance therapy"
            ])
        elif pattern == TemporalPattern.CYCLICAL:
            recommendations.extend([
                "Monitor cyclical patterns",
                "Adjust treatment timing to cycles",
                "Prepare for predictable difficult periods",
                "Optimize medication timing"
            ])

        # Time horizon recommendations
        if time_horizon == TimeScale.IMMEDIATE:
            recommendations.append("Focus on immediate stabilization and safety")
        elif time_horizon == TimeScale.CHRONIC:
            recommendations.append("Develop long-term management strategies")

        return recommendations

    def _get_default_temporal_analysis(self, conversation_id: str) -> TemporalAnalysis:
        """Get default temporal analysis when processing fails."""
        return TemporalAnalysis(
            conversation_id=conversation_id,
            temporal_events=[],
            temporal_sequences=[],
            dominant_pattern=TemporalPattern.CHRONIC_STABLE,
            time_horizon=TimeScale.MEDIUM_TERM,
            temporal_coherence=0.5,
            therapeutic_timeline={
                "approach": "general_support",
                "pattern": "stable",
                "phases": {"immediate": ["assessment"], "short_term": ["support"]},
                "estimated_duration": "3-6 months"
            },
            recommendations=["Comprehensive assessment needed", "Establish therapeutic relationship"]
        )

    def get_temporal_summary(self, analysis: TemporalAnalysis) -> dict[str, Any]:
        """Get summary of temporal analysis."""
        return {
            "conversation_id": analysis.conversation_id,
            "num_events": len(analysis.temporal_events),
            "num_sequences": len(analysis.temporal_sequences),
            "dominant_pattern": analysis.dominant_pattern.value,
            "time_horizon": analysis.time_horizon.value,
            "temporal_coherence": analysis.temporal_coherence,
            "therapeutic_approach": analysis.therapeutic_timeline.get("approach", "unknown"),
            "estimated_duration": analysis.therapeutic_timeline.get("estimated_duration", "unknown"),
            "key_recommendations": analysis.recommendations[:3],  # Top 3 recommendations
            "events_by_type": {
                event_type: len([e for e in analysis.temporal_events if e.event_type == event_type])
                for event_type in {e.event_type for e in analysis.temporal_events}
            }
        }


def main():
    """Test the temporal reasoner."""
    reasoner = TemporalReasoner()

    # Test conversation with temporal elements
    test_conversation = {
        "conversation_id": "test_001",
        "turns": [
            {
                "speaker": "user",
                "content": "I've been feeling depressed for the past few months. It started after I lost my job last year. I had a panic attack yesterday and couldn't sleep for hours. This has been going on for weeks now."
            },
            {
                "speaker": "assistant",
                "content": "I understand you've been struggling with depression and anxiety. Let's work on developing some coping strategies."
            }
        ]
    }

    # Analyze temporal patterns
    analysis = reasoner.analyze_temporal_patterns(test_conversation)


    for _event in analysis.temporal_events:
        pass

    for _seq in analysis.temporal_sequences:
        pass

    timeline = analysis.therapeutic_timeline

    if "key_milestones" in timeline:
        for _milestone in timeline["key_milestones"]:
            pass

    for _i, _rec in enumerate(analysis.recommendations, 1):
        pass

    # Get summary
    reasoner.get_temporal_summary(analysis)


if __name__ == "__main__":
    main()
