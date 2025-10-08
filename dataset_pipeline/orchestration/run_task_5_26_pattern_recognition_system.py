#!/usr/bin/env python3
"""
Task 5.26: Create Advanced Therapeutic Reasoning Pattern Recognition System
Analyzes and categorizes reasoning patterns across all Phase 3 CoT datasets.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np


class TherapeuticReasoningPatternRecognizer:
    """Advanced system for recognizing and categorizing therapeutic reasoning patterns."""

    def __init__(self, output_dir: str = "ai/data/processed/phase_3_cot_reasoning/task_5_26_pattern_recognition"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Pattern categories
        self.reasoning_patterns = {
            "diagnostic": [
                "assess", "evaluate", "diagnose", "identify", "determine", "examine",
                "symptoms", "criteria", "differential", "screening", "assessment"
            ],
            "treatment_planning": [
                "plan", "strategy", "approach", "intervention", "treatment", "therapy",
                "goals", "objectives", "outcomes", "progress", "monitoring"
            ],
            "therapeutic_alliance": [
                "rapport", "trust", "relationship", "alliance", "connection", "empathy",
                "understanding", "validation", "support", "collaboration"
            ],
            "cognitive_restructuring": [
                "thoughts", "beliefs", "cognitive", "thinking", "reframe", "challenge",
                "perspective", "interpretation", "assumptions", "distortions"
            ],
            "emotional_processing": [
                "emotions", "feelings", "emotional", "affect", "mood", "expression",
                "regulation", "processing", "awareness", "acceptance"
            ],
            "behavioral_analysis": [
                "behavior", "actions", "patterns", "habits", "responses", "triggers",
                "consequences", "reinforcement", "modification", "change"
            ],
            "systemic_thinking": [
                "system", "family", "relationships", "context", "environment", "social",
                "cultural", "interpersonal", "dynamics", "interactions"
            ],
            "risk_assessment": [
                "risk", "safety", "danger", "harm", "suicide", "crisis", "emergency",
                "protective", "factors", "warning", "signs"
            ]
        }

        # Complexity indicators
        self.complexity_indicators = {
            "high": [
                "complex", "multifaceted", "intricate", "sophisticated", "nuanced",
                "comprehensive", "integrated", "synthesize", "multidimensional"
            ],
            "medium": [
                "consider", "analyze", "evaluate", "compare", "balance", "weigh",
                "multiple", "various", "different", "several"
            ],
            "low": [
                "simple", "basic", "straightforward", "direct", "clear", "obvious",
                "single", "one", "primary", "main"
            ]
        }

        # Quality indicators
        self.quality_indicators = {
            "evidence_based": [
                "research", "evidence", "studies", "literature", "empirical", "validated",
                "proven", "effective", "efficacy", "outcomes"
            ],
            "ethical": [
                "ethical", "ethics", "boundaries", "confidentiality", "consent", "professional",
                "standards", "guidelines", "appropriate", "responsible"
            ],
            "culturally_sensitive": [
                "cultural", "culture", "diversity", "inclusive", "sensitive", "aware",
                "respectful", "appropriate", "context", "background"
            ],
            "trauma_informed": [
                "trauma", "informed", "safety", "trustworthy", "collaborative", "empowerment",
                "choice", "cultural", "humility", "resilience"
            ]
        }

    def analyze_phase_3_patterns(self) -> dict[str, Any]:
        """Analyze reasoning patterns across all Phase 3 CoT datasets."""

        # Load Phase 3 data
        phase_3_file = "data/processed/phase_3_cot_reasoning/phase_3_cot_reasoning_consolidated.jsonl"

        if not os.path.exists(phase_3_file):
            return self.create_error_report("Phase 3 data file not found")

        conversations = self.load_conversations(phase_3_file)

        # Analyze patterns
        pattern_analysis = self.analyze_reasoning_patterns(conversations)

        complexity_analysis = self.analyze_complexity_patterns(conversations)

        quality_analysis = self.analyze_quality_patterns(conversations)

        chain_analysis = self.analyze_reasoning_chains(conversations)

        insights = self.generate_pattern_insights(pattern_analysis, complexity_analysis, quality_analysis, chain_analysis)

        # Create comprehensive report
        report = {
            "task": "5.26: Advanced Therapeutic Reasoning Pattern Recognition System",
            "analysis_summary": {
                "total_conversations_analyzed": len(conversations),
                "analysis_timestamp": datetime.now().isoformat(),
                "patterns_identified": len(pattern_analysis["pattern_distribution"]),
                "complexity_levels": len(complexity_analysis["complexity_distribution"]),
                "quality_dimensions": len(quality_analysis["quality_distribution"])
            },
            "pattern_analysis": pattern_analysis,
            "complexity_analysis": complexity_analysis,
            "quality_analysis": quality_analysis,
            "reasoning_chain_analysis": chain_analysis,
            "insights_and_recommendations": insights
        }

        # Save report
        report_path = os.path.join(self.output_dir, "pattern_recognition_system_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Save pattern database
        pattern_db_path = os.path.join(self.output_dir, "therapeutic_reasoning_pattern_database.json")
        pattern_database = self.create_pattern_database(conversations, pattern_analysis)
        with open(pattern_db_path, "w", encoding="utf-8") as f:
            json.dump(pattern_database, f, indent=2, ensure_ascii=False)


        return report

    def load_conversations(self, file_path: str) -> list[dict[str, Any]]:
        """Load conversations from JSONL file."""
        conversations = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                try:
                    conv = json.loads(line.strip())
                    conversations.append(conv)
                except:
                    continue
        return conversations

    def analyze_reasoning_patterns(self, conversations: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze therapeutic reasoning patterns in conversations."""
        pattern_counts = defaultdict(int)
        pattern_combinations = defaultdict(int)
        dataset_patterns = defaultdict(lambda: defaultdict(int))

        for conv in conversations:
            # Get conversation text
            text = self.extract_conversation_text(conv)
            metadata = conv.get("metadata", {})
            source_dataset = metadata.get("source_dataset", "unknown")

            # Identify patterns in this conversation
            conv_patterns = []
            for pattern_type, keywords in self.reasoning_patterns.items():
                if self.contains_pattern(text, keywords):
                    pattern_counts[pattern_type] += 1
                    dataset_patterns[source_dataset][pattern_type] += 1
                    conv_patterns.append(pattern_type)

            # Track pattern combinations
            if len(conv_patterns) > 1:
                combo = "+".join(sorted(conv_patterns))  # Convert tuple to string
                pattern_combinations[combo] += 1

        return {
            "pattern_distribution": dict(pattern_counts),
            "pattern_combinations": dict(pattern_combinations),
            "dataset_pattern_distribution": dict(dataset_patterns),
            "total_patterns_identified": sum(pattern_counts.values())
        }

    def analyze_complexity_patterns(self, conversations: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze complexity patterns in reasoning."""
        complexity_counts = defaultdict(int)
        complexity_by_dataset = defaultdict(lambda: defaultdict(int))
        complexity_scores = []

        for conv in conversations:
            text = self.extract_conversation_text(conv)
            metadata = conv.get("metadata", {})
            source_dataset = metadata.get("source_dataset", "unknown")

            # Calculate complexity score
            complexity_score = self.calculate_complexity_score(text)
            complexity_scores.append(complexity_score)

            # Categorize complexity
            if complexity_score >= 0.7:
                complexity_level = "high"
            elif complexity_score >= 0.4:
                complexity_level = "medium"
            else:
                complexity_level = "low"

            complexity_counts[complexity_level] += 1
            complexity_by_dataset[source_dataset][complexity_level] += 1

        # Calculate statistics without numpy
        if complexity_scores:
            mean_score = sum(complexity_scores) / len(complexity_scores)
            sorted_scores = sorted(complexity_scores)
            median_score = sorted_scores[len(sorted_scores) // 2]
            min_score = min(complexity_scores)
            max_score = max(complexity_scores)
            # Simple standard deviation calculation
            variance = sum((x - mean_score) ** 2 for x in complexity_scores) / len(complexity_scores)
            std_score = variance ** 0.5
        else:
            mean_score = median_score = min_score = max_score = std_score = 0.0

        return {
            "complexity_distribution": dict(complexity_counts),
            "complexity_by_dataset": dict(complexity_by_dataset),
            "complexity_statistics": {
                "mean": mean_score,
                "median": median_score,
                "std": std_score,
                "min": min_score,
                "max": max_score
            }
        }

    def analyze_quality_patterns(self, conversations: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze quality patterns in therapeutic reasoning."""
        quality_counts = defaultdict(int)
        quality_by_dataset = defaultdict(lambda: defaultdict(int))

        for conv in conversations:
            text = self.extract_conversation_text(conv)
            metadata = conv.get("metadata", {})
            source_dataset = metadata.get("source_dataset", "unknown")

            # Check for quality indicators
            for quality_type, keywords in self.quality_indicators.items():
                if self.contains_pattern(text, keywords):
                    quality_counts[quality_type] += 1
                    quality_by_dataset[source_dataset][quality_type] += 1

        return {
            "quality_distribution": dict(quality_counts),
            "quality_by_dataset": dict(quality_by_dataset),
            "total_quality_indicators": sum(quality_counts.values())
        }

    def analyze_reasoning_chains(self, conversations: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze reasoning chain patterns."""
        chain_lengths = []
        chain_patterns = defaultdict(int)

        for conv in conversations:
            reasoning_chain = conv.get("reasoning_chain", [])
            chain_length = len(reasoning_chain)
            chain_lengths.append(chain_length)

            # Analyze chain structure
            if chain_length == 0:
                chain_patterns["no_chain"] += 1
            elif chain_length <= 2:
                chain_patterns["simple_chain"] += 1
            elif chain_length <= 5:
                chain_patterns["moderate_chain"] += 1
            else:
                chain_patterns["complex_chain"] += 1

        return {
            "chain_length_distribution": dict(chain_patterns),
            "chain_statistics": {
                "mean_length": float(np.mean(chain_lengths)) if chain_lengths else 0.0,
                "median_length": float(np.median(chain_lengths)) if chain_lengths else 0.0,
                "max_length": int(np.max(chain_lengths)) if chain_lengths else 0,
                "total_chains": len([c for c in chain_lengths if c > 0])
            }
        }

    def extract_conversation_text(self, conversation: dict[str, Any]) -> str:
        """Extract all text content from a conversation."""
        text_parts = []

        # Extract conversation messages
        for msg in conversation.get("conversation", []):
            content = msg.get("content", "")
            text_parts.append(content)

        # Extract reasoning chain
        for step in conversation.get("reasoning_chain", []):
            if isinstance(step, str):
                text_parts.append(step)
            elif isinstance(step, dict):
                text_parts.append(step.get("content", ""))

        return " ".join(text_parts).lower()

    def contains_pattern(self, text: str, keywords: list[str]) -> bool:
        """Check if text contains any of the pattern keywords."""
        return any(keyword in text for keyword in keywords)

    def calculate_complexity_score(self, text: str) -> float:
        """Calculate complexity score based on indicators."""
        high_count = sum(1 for keyword in self.complexity_indicators["high"] if keyword in text)
        medium_count = sum(1 for keyword in self.complexity_indicators["medium"] if keyword in text)
        low_count = sum(1 for keyword in self.complexity_indicators["low"] if keyword in text)

        # Weight the counts
        weighted_score = (high_count * 1.0) + (medium_count * 0.6) + (low_count * 0.2)

        # Normalize by text length (approximate)
        word_count = len(text.split())
        normalized_score = weighted_score / max(word_count / 100, 1)  # Per 100 words

        return min(normalized_score, 1.0)  # Cap at 1.0

    def generate_pattern_insights(self, pattern_analysis: dict, complexity_analysis: dict,
                                quality_analysis: dict, chain_analysis: dict) -> dict[str, Any]:
        """Generate insights and recommendations from pattern analysis."""
        insights = {
            "key_findings": [],
            "recommendations": [],
            "pattern_strengths": [],
            "areas_for_improvement": []
        }

        # Analyze pattern distribution
        patterns = pattern_analysis["pattern_distribution"]
        most_common_pattern = max(patterns, key=patterns.get) if patterns else None
        least_common_pattern = min(patterns, key=patterns.get) if patterns else None

        if most_common_pattern:
            insights["key_findings"].append(f"Most prevalent reasoning pattern: {most_common_pattern} ({patterns[most_common_pattern]} occurrences)")

        if least_common_pattern:
            insights["key_findings"].append(f"Least prevalent reasoning pattern: {least_common_pattern} ({patterns[least_common_pattern]} occurrences)")

        # Analyze complexity
        complexity_stats = complexity_analysis["complexity_statistics"]
        insights["key_findings"].append(f"Average complexity score: {complexity_stats['mean']:.3f}")

        if complexity_stats["mean"] < 0.5:
            insights["areas_for_improvement"].append("Overall reasoning complexity is below optimal level")
            insights["recommendations"].append("Increase complexity of reasoning examples in training data")

        # Analyze quality indicators
        quality_dist = quality_analysis["quality_distribution"]
        if quality_dist.get("evidence_based", 0) < 1000:
            insights["areas_for_improvement"].append("Limited evidence-based reasoning examples")
            insights["recommendations"].append("Incorporate more research-backed therapeutic approaches")

        # Analyze reasoning chains
        chain_stats = chain_analysis["chain_statistics"]
        if chain_stats["mean_length"] < 3:
            insights["areas_for_improvement"].append("Reasoning chains are relatively short")
            insights["recommendations"].append("Develop longer, more detailed reasoning chains")

        return insights

    def create_pattern_database(self, conversations: list[dict[str, Any]],
                              pattern_analysis: dict) -> dict[str, Any]:
        """Create a searchable database of reasoning patterns."""
        database = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_conversations": len(conversations),
                "total_patterns": len(pattern_analysis["pattern_distribution"])
            },
            "pattern_examples": defaultdict(list),
            "pattern_templates": {},
            "search_index": defaultdict(list)
        }

        # Extract examples for each pattern
        for i, conv in enumerate(conversations):
            text = self.extract_conversation_text(conv)
            metadata = conv.get("metadata", {})

            for pattern_type, keywords in self.reasoning_patterns.items():
                if self.contains_pattern(text, keywords):
                    example = {
                        "conversation_id": i,
                        "source_dataset": metadata.get("source_dataset", "unknown"),
                        "reasoning_type": metadata.get("reasoning_type", "unknown"),
                        "sample_text": text[:500] + "..." if len(text) > 500 else text,
                        "quality_score": metadata.get("reasoning_quality_score", 0.0)
                    }
                    database["pattern_examples"][pattern_type].append(example)
                    database["search_index"][pattern_type].append(i)

        return dict(database)

    def create_error_report(self, error_message: str) -> dict[str, Any]:
        """Create error report for failed analysis."""
        return {
            "task": "5.26: Advanced Therapeutic Reasoning Pattern Recognition System",
            "status": "FAILED",
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }

def process_pattern_recognition():
    """Main function to run the pattern recognition system."""
    recognizer = TherapeuticReasoningPatternRecognizer()
    return recognizer.analyze_phase_3_patterns()

if __name__ == "__main__":
    process_pattern_recognition()
