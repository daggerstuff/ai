#!/usr/bin/env python3
"""
Advanced Analytics System - Example Usage

This script demonstrates the complete advanced analytics system including:
1. Real-time streaming data processing
2. Advanced pattern recognition
3. Complexity scoring
4. Integrated analytics orchestration
"""

import asyncio
import json
import logging
from pathlib import Path

# Import our analytics components
from analytics_orchestrator import AdvancedAnalyticsEngine, StreamingAnalyticsOrchestrator
from complexity_scorer import ComplexityScorer
from pattern_recognition_engine import PatternRecognitionEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_conversations():
    """Create sample conversations for testing."""
    return [
        {
            "id": "basic_conversation",
            "messages": [
                {"role": "client", "content": "I feel sad today."},
                {"role": "therapist", "content": "I understand. Can you tell me more about what's making you feel sad?"},
                {"role": "client", "content": "Work has been stressful."},
                {"role": "therapist", "content": "Work stress can be difficult. What specifically at work is causing you stress?"}
            ]
        },
        {
            "id": "intermediate_conversation",
            "messages": [
                {"role": "client", "content": "I've been experiencing anxiety attacks lately. They seem to come out of nowhere and I feel like I can't breathe."},
                {"role": "therapist", "content": "Panic attacks can be very frightening. When you say they come out of nowhere, have you noticed any patterns or triggers? Sometimes there are subtle cues our body picks up on before we're consciously aware."},
                {"role": "client", "content": "Now that you mention it, they often happen when I'm in crowded places or when I have to speak in meetings."},
                {"role": "therapist", "content": "That's a really important observation. It sounds like social situations might be a trigger for you. Let's explore some coping strategies you can use when you notice those early warning signs."}
            ]
        },
        {
            "id": "advanced_conversation",
            "messages": [
                {"role": "client", "content": "I've been reflecting on our previous sessions about my attachment patterns. I think I'm starting to see how my childhood experiences with my emotionally unavailable father are playing out in my romantic relationships."},
                {"role": "therapist", "content": "That's a profound insight. The fact that you're making these connections between your early attachment experiences and current relational patterns shows significant therapeutic progress. Can you help me understand what specific patterns you're noticing?"},
                {"role": "client", "content": "I realize I have this anxious attachment style where I constantly seek reassurance from my partner, but then I also push them away when they get too close. It's like I'm terrified of both abandonment and intimacy simultaneously."},
                {"role": "therapist", "content": "You've just articulated something that many people struggle to put into words - that paradox of craving connection while simultaneously fearing it. This kind of ambivalent attachment often develops as an adaptive response to inconsistent caregiving. Now that you have this awareness, we can work on developing more secure attachment behaviors."}
            ]
        },
        {
            "id": "expert_conversation",
            "messages": [
                {"role": "client", "content": "I've been practicing the mindfulness techniques we discussed, and I'm noticing something interesting about my dissociative episodes. There seems to be a somatic component - I feel this disconnection starting in my chest before my mind goes blank."},
                {"role": "therapist", "content": "That's a remarkable level of somatic awareness you're developing. The fact that you're tracking the embodied precursors to dissociation suggests your window of tolerance is expanding. From a polyvagal theory perspective, what you're describing sounds like you're catching the dorsal vagal shutdown before it fully engages."},
                {"role": "client", "content": "Yes, exactly. And when I notice that chest sensation, I've been using the bilateral stimulation technique you taught me. It seems to help me stay more present and connected to my body."},
                {"role": "therapist", "content": "This is beautiful integration of somatic awareness with self-regulation skills. You're essentially rewiring your nervous system's response to trauma triggers. The bilateral stimulation is helping activate your parasympathetic nervous system and maintain dual awareness - staying present while processing difficult material. This kind of neuroplasticity-informed healing is exactly what we're aiming for."}
            ]
        }
    ]


async def demonstrate_streaming_analytics():
    """Demonstrate streaming analytics capabilities."""

    # Create sample data directory
    streaming_dir = Path("data/streaming")
    streaming_dir.mkdir(parents=True, exist_ok=True)

    # Create sample conversations as files
    conversations = create_sample_conversations()

    for i, conv in enumerate(conversations):
        file_path = streaming_dir / f"conversation_{i+1}.json"
        with open(file_path, "w") as f:
            json.dump(conv, f, indent=2)

    # Create orchestrator
    orchestrator = StreamingAnalyticsOrchestrator()


    # Process files directly for demonstration
    for conv in conversations:
        orchestrator.analytics_engine.analyze_conversation(conv)

    # Get real-time analytics
    orchestrator.get_real_time_analytics()


def demonstrate_pattern_recognition():
    """Demonstrate pattern recognition capabilities."""

    engine = PatternRecognitionEngine()
    conversations = create_sample_conversations()

    for conv in conversations:
        analysis = engine.analyze_conversation(conv)


        for _pattern in analysis.patterns[:3]:  # Show top 3
            pass

        for _rec in analysis.recommendations[:2]:  # Show top 2
            pass


def demonstrate_complexity_scoring():
    """Demonstrate complexity scoring capabilities."""

    scorer = ComplexityScorer()
    conversations = create_sample_conversations()

    for conv in conversations:
        scorer.score_conversation_complexity(conv)




def demonstrate_integrated_analytics():
    """Demonstrate integrated analytics capabilities."""

    engine = AdvancedAnalyticsEngine()
    conversations = create_sample_conversations()


    results = []
    for conv in conversations:
        result = engine.analyze_conversation(conv)
        results.append(result)

    # Show results summary

    for _result in results:
        pass

    # Show system summary
    summary = engine.get_analytics_summary()

    sophistication_dist = summary["system_metrics"]["sophistication_distribution"]
    for _level, _count in sophistication_dist.items():
        pass


def create_configuration_files():
    """Create configuration files for the analytics system."""

    # Analytics configuration
    analytics_config = {
        "streaming": {
            "queue_size": 10000,
            "processing_queue_size": 1000,
            "worker_threads": 4,
            "batch_size": 100,
            "quality_threshold": 0.6,
            "processing_timeout": 30.0,
            "metrics_update_interval": 5.0
        },
        "analytics": {
            "worker_threads": 6,
            "enable_real_time": True,
            "store_results": True,
            "max_results_history": 10000
        },
        "data_sources": [
            {
                "type": "file_watcher",
                "config": {
                    "directory": "data/streaming",
                    "patterns": ["*.jsonl", "*.json"]
                }
            },
            {
                "type": "websocket",
                "config": {
                    "uri": "ws://localhost:8765",
                    "reconnect_interval": 5.0
                }
            }
        ],
        "pattern_recognition": {
            "enable_therapeutic_techniques": True,
            "enable_flow_analysis": True,
            "enable_clinical_detection": True,
            "confidence_threshold": 0.3
        },
        "complexity_scoring": {
            "enable_linguistic_analysis": True,
            "enable_therapeutic_depth": True,
            "enable_emotional_complexity": True,
            "readability_weight": 0.3,
            "vocabulary_weight": 0.3,
            "syntactic_weight": 0.2,
            "semantic_weight": 0.2
        },
        "alerts": {
            "enable_quality_alerts": True,
            "enable_sophistication_alerts": True,
            "enable_pattern_alerts": True,
            "low_quality_threshold": 0.3,
            "high_quality_threshold": 0.8
        }
    }

    config_path = Path("analytics_config.json")
    with open(config_path, "w") as f:
        json.dump(analytics_config, f, indent=2)


    # Streaming configuration
    streaming_config = {
        "queue_size": 10000,
        "processing_queue_size": 1000,
        "worker_threads": 4,
        "batch_size": 100,
        "quality_threshold": 0.6,
        "processing_timeout": 30.0,
        "metrics_update_interval": 5.0,
        "data_sources": []
    }

    streaming_config_path = Path("streaming_config.json")
    with open(streaming_config_path, "w") as f:
        json.dump(streaming_config, f, indent=2)


    # Dashboard configuration
    dashboard_config = {
        "port": 5000,
        "debug": False,
        "update_interval": 1.0,
        "max_alerts": 100,
        "max_metrics_history": 1000,
        "enable_websocket": True,
        "cors_origins": ["*"]
    }

    dashboard_config_path = Path("dashboard_config.json")
    with open(dashboard_config_path, "w") as f:
        json.dump(dashboard_config, f, indent=2)



async def main():
    """Main demonstration function."""

    # Create configuration files
    create_configuration_files()

    # Demonstrate individual components
    demonstrate_pattern_recognition()
    demonstrate_complexity_scoring()
    demonstrate_integrated_analytics()

    # Demonstrate streaming analytics
    await demonstrate_streaming_analytics()




if __name__ == "__main__":
    asyncio.run(main())
