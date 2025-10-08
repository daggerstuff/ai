#!/usr/bin/env python3
"""
Task 6.0 Phase 3 Demonstration: Multi-Modal Integration & Advanced Analytics

This demonstration showcases the completed Phase 3 components:
- Task 6.13: Audio Emotion Recognition Integration (IEMOCAP) ✅
- Task 6.14: Multi-Modal Mental Disorder Analysis Pipeline (MODMA) ✅
- Task 6.15: Emotion Cause Extraction and Intervention Mapping (RECCON) ✅

Strategic Achievement: Complete multi-modal therapeutic intelligence system
with advanced analytics capabilities for the 2.59M+ conversation ecosystem.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

# Import our Phase 3 components
from audio_emotion_integration import MultiModalTherapeuticAnalyzer
from emotion_cause_extraction import EmotionCauseExtractor
from multimodal_disorder_analysis import MODMAAnalyzer


class Phase3MultiModalDemo:
    """Comprehensive demonstration of Phase 3 multi-modal capabilities."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize all Phase 3 components
        self.multimodal_analyzer = MultiModalTherapeuticAnalyzer()
        self.modma_analyzer = MODMAAnalyzer()
        self.emotion_extractor = EmotionCauseExtractor()

        # Demo conversations representing complex multi-modal scenarios
        self.demo_conversations = self._create_demo_conversations()

    def _create_demo_conversations(self) -> list:
        """Create comprehensive demo conversations for multi-modal testing."""
        return [
            {
                "id": "multimodal_depression_complex",
                "scenario": "Complex Depression with Multi-Modal Indicators",
                "messages": [
                    {
                        "role": "client",
                        "content": "I feel so hopeless and my voice keeps breaking when I talk. I've been feeling this way since my relationship ended three months ago. I keep thinking that I'll never find love again and that I'm fundamentally unlovable. Sometimes I feel angry at myself for being so weak, and other times I just feel completely empty."
                    },
                    {
                        "role": "therapist",
                        "content": "I can hear the pain in your voice and I want you to know that what you're experiencing after such a significant loss is understandable. Let's explore these thoughts about being unlovable and work on some strategies to help you process this grief."
                    }
                ]
            },
            {
                "id": "multimodal_anxiety_panic",
                "scenario": "Anxiety with Panic Features and Audio Cues",
                "messages": [
                    {
                        "role": "client",
                        "content": "I'm so anxious right now I can barely breathe. My heart is racing and I feel like I'm going to die. This started happening after I had to give a presentation at work last week. Now I'm terrified of speaking in public again because I keep thinking everyone will see how nervous I am and judge me."
                    },
                    {
                        "role": "therapist",
                        "content": "I can hear how distressed you are right now. Let's start with some breathing exercises to help you feel more grounded, and then we can work on understanding what happened during that presentation."
                    }
                ]
            },
            {
                "id": "multimodal_trauma_complex",
                "scenario": "Complex Trauma with Dissociative Features",
                "messages": [
                    {
                        "role": "client",
                        "content": "Sometimes I feel like I'm not really here, like I'm watching myself from outside my body. This happens especially when I'm reminded of what happened to me as a child. My voice gets really quiet and I can't seem to speak normally. I feel scared and angry at the same time, but mostly I just feel numb."
                    },
                    {
                        "role": "therapist",
                        "content": "Thank you for sharing something so difficult. What you're describing sounds like dissociation, which is a common response to trauma. Let's work on some grounding techniques to help you feel more present and safe in your body."
                    }
                ]
            },
            {
                "id": "multimodal_bipolar_mixed",
                "scenario": "Bipolar Disorder with Mixed Episode Features",
                "messages": [
                    {
                        "role": "client",
                        "content": "I don't know what's wrong with me. One minute I'm talking really fast and feel like I can do anything, and the next minute I'm crying and want to disappear. My thoughts are racing but I also feel hopeless. I haven't been sleeping much because my mind won't shut off, but I also don't have energy to do anything."
                    },
                    {
                        "role": "therapist",
                        "content": "It sounds like you're experiencing some very intense and conflicting emotions right now. This kind of mixed state can be really distressing. Let's talk about what's been happening and work on some strategies to help stabilize your mood."
                    }
                ]
            },
            {
                "id": "multimodal_relationship_conflict",
                "scenario": "Relationship Conflict with Emotional Dysregulation",
                "messages": [
                    {
                        "role": "client",
                        "content": "I had another huge fight with my partner last night and I completely lost it. I was screaming and crying at the same time. I feel so ashamed because I know my behavior was over the top, but in the moment I felt so hurt and angry that they weren't listening to me. Now I'm terrified they're going to leave me because I can't control my emotions."
                    },
                    {
                        "role": "therapist",
                        "content": "I can hear how much distress this is causing you. It takes courage to acknowledge when our emotional reactions feel overwhelming. Let's explore what was happening for you in that moment and work on some skills for managing intense emotions."
                    }
                ]
            }
        ]

    async def run_comprehensive_multimodal_analysis(self, conversation: dict) -> dict:
        """Run comprehensive multi-modal analysis on a conversation."""

        # Step 1: Multi-Modal Therapeutic Analysis (Audio + Text)
        multimodal_result = self.multimodal_analyzer.analyze_conversation(conversation)


        # Step 2: MODMA Analysis (Multi-Modal Disorder Analysis)
        modma_result = self.modma_analyzer.analyze_conversation(conversation)


        # Step 3: Emotion Cause Extraction and Intervention Mapping
        emotion_result = self.emotion_extractor.extract_emotion_causes(conversation)


        # Step 4: Integration and Synthesis
        integrated_analysis = self._integrate_multimodal_results(
            multimodal_result, modma_result, emotion_result
        )


        # Display key insights
        for _insight in integrated_analysis["key_insights"][:3]:
            pass

        for _rec in integrated_analysis["integrated_recommendations"][:3]:
            pass

        # Compile comprehensive analysis
        return {
            "conversation_id": conversation["id"],
            "scenario": conversation["scenario"],
            "multimodal_analysis": {
                "overall_confidence": multimodal_result.overall_confidence,
                "emotion_text_alignment": multimodal_result.emotion_text_alignment,
                "audio_emotion": multimodal_result.audio_analysis.dominant_emotion.value,
                "audio_intensity": multimodal_result.audio_analysis.intensity_level.value,
                "therapeutic_insights": multimodal_result.therapeutic_insights,
                "intervention_recommendations": multimodal_result.intervention_recommendations
            },
            "modma_analysis": {
                "primary_disorder": modma_result.primary_disorder.value,
                "disorder_confidence": modma_result.disorder_confidence,
                "analysis_confidence": modma_result.analysis_confidence.value,
                "cross_modal_consistency": modma_result.cross_modal_consistency,
                "modality_contributions": modma_result.modality_contributions,
                "treatment_recommendations": modma_result.treatment_recommendations
            },
            "emotion_cause_analysis": {
                "analysis_confidence": emotion_result.analysis_confidence,
                "identified_emotions": [e.value for e in emotion_result.identified_emotions],
                "emotion_causes": len(emotion_result.emotion_causes),
                "intervention_mappings": len(emotion_result.intervention_mappings),
                "therapeutic_focus_areas": emotion_result.therapeutic_focus_areas,
                "treatment_sequence": emotion_result.treatment_sequence
            },
            "integrated_analysis": integrated_analysis,
            "analysis_timestamp": datetime.now().isoformat()
        }


    def _integrate_multimodal_results(self, multimodal_result, modma_result, emotion_result) -> dict:
        """Integrate results from all three analysis systems."""

        # Determine consensus disorder
        disorders = [
            multimodal_result.text_analysis["condition_recognition"]["primary_condition"],
            modma_result.primary_disorder.value
        ]
        consensus_disorder = max(set(disorders), key=disorders.count)

        # Calculate integration confidence
        confidences = [
            multimodal_result.overall_confidence,
            modma_result.disorder_confidence,
            emotion_result.analysis_confidence
        ]
        integration_confidence = sum(confidences) / len(confidences)

        # Risk assessment integration
        risk_indicators = []
        risk_indicators.extend(multimodal_result.risk_indicators)
        risk_indicators.extend(modma_result.diagnostic_evidence.get("text_indicators", []))

        risk_level = "low"
        if len(risk_indicators) > 3:
            risk_level = "high"
        elif len(risk_indicators) > 1:
            risk_level = "moderate"

        # Treatment priority assessment
        treatment_priority = "routine"
        if risk_level == "high" or any("crisis" in str(r).lower() for r in risk_indicators):
            treatment_priority = "urgent"
        elif risk_level == "moderate":
            treatment_priority = "priority"

        # Generate key insights
        key_insights = []

        # Audio-text alignment insights
        if multimodal_result.emotion_text_alignment < 0.5:
            key_insights.append("Significant emotional incongruence detected between audio and text")

        # Cross-modal consistency insights
        if modma_result.cross_modal_consistency > 0.8:
            key_insights.append("High cross-modal consistency supports diagnostic confidence")

        # Emotion cause insights
        if emotion_result.cause_interaction_patterns["interaction_complexity"] == "complex":
            key_insights.append("Complex emotion-cause interactions require integrated treatment approach")

        # Integration-specific insights
        if integration_confidence > 0.7:
            key_insights.append("Strong multi-modal convergence supports treatment recommendations")

        # Integrate treatment recommendations
        all_recommendations = []
        all_recommendations.extend(multimodal_result.intervention_recommendations)
        all_recommendations.extend(modma_result.treatment_recommendations)
        all_recommendations.extend(emotion_result.therapeutic_focus_areas)

        # Remove duplicates and prioritize
        unique_recommendations = list(set(all_recommendations))
        integrated_recommendations = unique_recommendations[:5]  # Top 5

        return {
            "consensus_disorder": consensus_disorder,
            "integration_confidence": integration_confidence,
            "risk_assessment": {
                "level": risk_level,
                "indicators": risk_indicators[:5],  # Top 5
                "indicator_count": len(risk_indicators)
            },
            "treatment_priority": treatment_priority,
            "key_insights": key_insights,
            "integrated_recommendations": integrated_recommendations,
            "analysis_convergence": {
                "multimodal_confidence": multimodal_result.overall_confidence,
                "modma_confidence": modma_result.disorder_confidence,
                "emotion_confidence": emotion_result.analysis_confidence,
                "cross_modal_consistency": modma_result.cross_modal_consistency
            }
        }

    async def run_batch_multimodal_analysis(self) -> list:
        """Run batch multi-modal analysis on all demo conversations."""

        analyses = []

        for conversation in self.demo_conversations:
            analysis = await self.run_comprehensive_multimodal_analysis(conversation)
            analyses.append(analysis)

            # Small delay for readability
            await asyncio.sleep(0.5)

        return analyses

    def generate_phase3_statistics(self, analyses: list) -> dict:
        """Generate comprehensive Phase 3 system statistics."""

        # Disorder consensus analysis
        consensus_disorders = [a["integrated_analysis"]["consensus_disorder"] for a in analyses]
        disorder_distribution = {}
        for disorder in consensus_disorders:
            disorder_distribution[disorder] = disorder_distribution.get(disorder, 0) + 1

        for disorder, count in disorder_distribution.items():
            (count / len(analyses)) * 100

        # Integration confidence analysis
        integration_confidences = [a["integrated_analysis"]["integration_confidence"] for a in analyses]
        avg_integration_confidence = sum(integration_confidences) / len(integration_confidences)


        # Component confidence analysis
        multimodal_confidences = [a["multimodal_analysis"]["overall_confidence"] for a in analyses]
        modma_confidences = [a["modma_analysis"]["disorder_confidence"] for a in analyses]
        emotion_confidences = [a["emotion_cause_analysis"]["analysis_confidence"] for a in analyses]


        # Risk assessment analysis
        risk_levels = [a["integrated_analysis"]["risk_assessment"]["level"] for a in analyses]
        risk_distribution = {}
        for level in risk_levels:
            risk_distribution[level] = risk_distribution.get(level, 0) + 1

        for level, count in risk_distribution.items():
            (count / len(analyses)) * 100

        # Treatment priority analysis
        treatment_priorities = [a["integrated_analysis"]["treatment_priority"] for a in analyses]
        priority_distribution = {}
        for priority in treatment_priorities:
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1

        for priority, count in priority_distribution.items():
            (count / len(analyses)) * 100

        # Cross-modal consistency analysis
        consistency_scores = [a["modma_analysis"]["cross_modal_consistency"] for a in analyses]
        avg_consistency = sum(consistency_scores) / len(consistency_scores)


        return {
            "total_analyses": len(analyses),
            "disorder_distribution": disorder_distribution,
            "integration_confidence": {
                "average": avg_integration_confidence,
                "individual_scores": integration_confidences
            },
            "component_confidences": {
                "multimodal": sum(multimodal_confidences)/len(multimodal_confidences),
                "modma": sum(modma_confidences)/len(modma_confidences),
                "emotion": sum(emotion_confidences)/len(emotion_confidences)
            },
            "risk_distribution": risk_distribution,
            "priority_distribution": priority_distribution,
            "cross_modal_consistency": {
                "average": avg_consistency,
                "high_consistency_cases": sum(1 for s in consistency_scores if s > 0.8)
            }
        }

    def export_phase3_results(self, analyses: list, statistics: dict, output_dir: str = "data/phase3_results"):
        """Export comprehensive Phase 3 results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export comprehensive analyses
        analyses_file = output_path / "phase3_multimodal_analyses.json"
        with open(analyses_file, "w") as f:
            json.dump(analyses, f, indent=2)

        # Export statistics
        stats_file = output_path / "phase3_system_statistics.json"
        with open(stats_file, "w") as f:
            json.dump(statistics, f, indent=2)

        # Export component-specific statistics
        multimodal_stats = self.multimodal_analyzer.get_analysis_statistics()
        modma_stats = self.modma_analyzer.get_analysis_statistics()
        emotion_stats = self.emotion_extractor.get_extraction_statistics()

        component_stats_file = output_path / "phase3_component_statistics.json"
        with open(component_stats_file, "w") as f:
            json.dump({
                "multimodal_stats": multimodal_stats,
                "modma_stats": modma_stats,
                "emotion_stats": emotion_stats
            }, f, indent=2)


        return [str(f) for f in output_path.glob("*.json")]


async def main():
    """Main demonstration function for Phase 3."""

    # Create demo instance
    demo = Phase3MultiModalDemo()

    # Run comprehensive multi-modal analysis
    analyses = await demo.run_batch_multimodal_analysis()

    # Generate statistics
    statistics = demo.generate_phase3_statistics(analyses)

    # Export results
    demo.export_phase3_results(analyses, statistics)

    # Final summary







if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Run demonstration
    asyncio.run(main())
