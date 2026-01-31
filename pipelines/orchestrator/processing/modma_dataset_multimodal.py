"""
MODMA-Dataset Multi-modal Mental Disorder Analysis

Integrates MODMA-Dataset for multi-modal mental disorder analysis.
Note: Filename includes exact dataset name with hyphen.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from logger import get_logger

logger = get_logger(__name__)

@dataclass
class MODMAMultimodalEntry:
    """MODMA multi-modal mental disorder entry."""
    entry_id: str
    text_modality: dict[str, Any] = field(default_factory=dict)
    audio_modality: dict[str, Any] = field(default_factory=dict)
    visual_modality: dict[str, Any] = field(default_factory=dict)
    disorder_analysis: dict[str, Any] = field(default_factory=dict)
    multimodal_features: dict[str, Any] = field(default_factory=dict)

class MODMADatasetMultimodal:
    """Processes MODMA-Dataset for multi-modal mental disorder analysis."""

    def __init__(self, dataset_path: str = "./MODMA-Dataset",
                 output_dir: str = "./processed_modma"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = get_logger(__name__)

        # Mental disorders for analysis
        self.mental_disorders = {
            "depression": {
                "text_indicators": ["hopeless", "worthless", "sad", "empty", "tired", "sleep problems"],
                "audio_indicators": ["slow speech", "low pitch", "monotone", "long pauses", "quiet voice"],
                "visual_indicators": ["downward gaze", "slumped posture", "minimal facial expression", "reduced eye contact"]
            },
            "anxiety": {
                "text_indicators": ["worried", "nervous", "panic", "fear", "restless", "overwhelmed"],
                "audio_indicators": ["rapid speech", "higher pitch", "trembling voice", "frequent pauses", "breathless"],
                "visual_indicators": ["fidgeting", "tense posture", "wide eyes", "frequent blinking", "restless movement"]
            },
            "bipolar": {
                "text_indicators": ["mood swings", "manic", "depressed", "energy changes", "impulsive", "grandiose"],
                "audio_indicators": ["variable speech rate", "pitch changes", "volume fluctuations", "pressured speech"],
                "visual_indicators": ["animated gestures", "posture changes", "eye contact variations", "energy shifts"]
            },
            "ptsd": {
                "text_indicators": ["trauma", "flashbacks", "nightmares", "avoidance", "hypervigilant", "triggered"],
                "audio_indicators": ["startled responses", "voice tension", "speech disruptions", "emotional breaks"],
                "visual_indicators": ["hyperalertness", "scanning behavior", "startle response", "defensive posture"]
            },
            "schizophrenia": {
                "text_indicators": ["hallucinations", "delusions", "disorganized", "paranoid", "confused", "voices"],
                "audio_indicators": ["disorganized speech", "word salad", "tangential", "circumstantial", "neologisms"],
                "visual_indicators": ["inappropriate affect", "bizarre behavior", "catatonic features", "poor eye contact"]
            }
        }

        # Multimodal analysis features
        self.multimodal_features = {
            "text_features": ["sentiment", "emotion", "linguistic_complexity", "coherence", "topic_modeling"],
            "audio_features": ["prosody", "voice_quality", "speech_rate", "pause_patterns", "emotional_tone"],
            "visual_features": ["facial_expressions", "body_language", "eye_movements", "gesture_patterns", "posture_analysis"],
            "fusion_features": ["cross_modal_correlation", "temporal_alignment", "feature_integration", "multimodal_attention"]
        }

        logger.info("MODMADatasetMultimodal initialized")

    def process_modma_dataset(self) -> dict[str, Any]:
        """Process MODMA-Dataset multi-modal mental disorder analysis."""
        start_time = datetime.now()

        result = {
            "success": False,
            "entries_processed": 0,
            "modalities_analyzed": [],
            "disorders_covered": [],
            "multimodal_features_extracted": 0,
            "quality_metrics": {},
            "issues": [],
            "output_path": None
        }

        try:
            # Check if dataset exists, create mock if not
            if not self.dataset_path.exists():
                self._create_mock_modma_data()
                result["issues"].append("Created mock MODMA dataset for testing")

            # Load MODMA entries
            modma_entries = self._load_modma_entries()

            # Process each entry
            processed_entries = []
            modalities_found = set()
            disorders_found = set()
            total_features = 0

            for entry_data in modma_entries:
                processed_entry = self._process_modma_entry(entry_data)
                if processed_entry:
                    processed_entries.append(processed_entry)

                    # Track modalities
                    if processed_entry.text_modality:
                        modalities_found.add("text")
                    if processed_entry.audio_modality:
                        modalities_found.add("audio")
                    if processed_entry.visual_modality:
                        modalities_found.add("visual")

                    # Track disorders
                    disorder = processed_entry.disorder_analysis.get("primary_disorder")
                    if disorder:
                        disorders_found.add(disorder)

                    # Count features
                    total_features += len(processed_entry.multimodal_features)

            # Quality assessment
            quality_metrics = self._assess_modma_quality(processed_entries)

            # Save processed data
            output_path = self._save_modma_processed(processed_entries, quality_metrics)

            # Update result
            result.update({
                "success": True,
                "entries_processed": len(processed_entries),
                "modalities_analyzed": list(modalities_found),
                "disorders_covered": list(disorders_found),
                "multimodal_features_extracted": total_features,
                "quality_metrics": quality_metrics,
                "output_path": str(output_path),
                "processing_time": (datetime.now() - start_time).total_seconds()
            })

            logger.info(f"Successfully processed MODMA dataset: {len(processed_entries)} entries")

        except Exception as e:
            result["issues"].append(f"Processing failed: {e!s}")
            logger.error(f"MODMA dataset processing failed: {e}")

        return result

    def _create_mock_modma_data(self):
        """Create mock MODMA dataset."""
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        entries = []

        # Generate entries for each disorder
        for disorder, indicators in self.mental_disorders.items():
            for i in range(20):  # 20 entries per disorder
                entry = {
                    "entry_id": f"modma_{disorder}_{i:03d}",
                    "participant_id": f"participant_{disorder}_{i:03d}",
                    "session_info": {
                        "session_type": "clinical_interview",
                        "duration_minutes": 30 + (i % 20),
                        "setting": "controlled_clinical"
                    },
                    "text_data": {
                        "transcript": f"Patient exhibits symptoms consistent with {disorder}. Reports {indicators['text_indicators'][i % len(indicators['text_indicators'])]} and related concerns.",
                        "sentiment_score": np.random.uniform(-1, 1),
                        "emotion_labels": [indicators["text_indicators"][i % len(indicators["text_indicators"])]],
                        "linguistic_features": {
                            "word_count": 150 + (i * 10),
                            "complexity_score": np.random.uniform(0.3, 0.9),
                            "coherence_score": np.random.uniform(0.4, 0.95)
                        }
                    },
                    "audio_data": {
                        "features": {
                            "mfcc": np.random.random(13).tolist(),
                            "pitch_mean": float(np.random.normal(150, 50)),
                            "pitch_std": float(np.random.uniform(10, 40)),
                            "speech_rate": float(np.random.normal(150, 30)),
                            "pause_ratio": float(np.random.uniform(0.1, 0.4)),
                            "voice_quality": indicators["audio_indicators"][i % len(indicators["audio_indicators"])]
                        },
                        "prosodic_features": {
                            "intonation_pattern": "falling" if disorder == "depression" else "variable",
                            "stress_pattern": "reduced" if disorder == "depression" else "normal",
                            "rhythm_regularity": np.random.uniform(0.3, 0.9)
                        }
                    },
                    "visual_data": {
                        "facial_expressions": {
                            "happiness": float(np.random.uniform(0, 0.3)) if disorder == "depression" else float(np.random.uniform(0.2, 0.8)),
                            "sadness": float(np.random.uniform(0.4, 0.9)) if disorder == "depression" else float(np.random.uniform(0, 0.4)),
                            "anger": float(np.random.uniform(0, 0.6)),
                            "fear": float(np.random.uniform(0.3, 0.8)) if disorder == "anxiety" else float(np.random.uniform(0, 0.4)),
                            "surprise": float(np.random.uniform(0, 0.5)),
                            "disgust": float(np.random.uniform(0, 0.3))
                        },
                        "body_language": {
                            "posture": indicators["visual_indicators"][i % len(indicators["visual_indicators"])],
                            "eye_contact_ratio": float(np.random.uniform(0.2, 0.8)),
                            "gesture_frequency": float(np.random.uniform(0.1, 0.7)),
                            "movement_energy": float(np.random.uniform(0.2, 0.9))
                        },
                        "micro_expressions": {
                            "detected_count": np.random.randint(5, 25),
                            "authenticity_score": float(np.random.uniform(0.4, 0.9))
                        }
                    },
                    "disorder_labels": {
                        "primary_disorder": disorder,
                        "severity_score": float(np.random.uniform(0.3, 0.9)),
                        "confidence_score": float(np.random.uniform(0.6, 0.95)),
                        "comorbidity_indicators": []
                    },
                    "metadata": {
                        "recording_quality": "high",
                        "annotation_confidence": float(np.random.uniform(0.7, 0.95)),
                        "clinical_validation": True,
                        "multimodal_alignment": "synchronized"
                    }
                }
                entries.append(entry)

        # Save entries
        with open(self.dataset_path / "modma_entries.jsonl", "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        # Create metadata
        metadata = {
            "dataset_name": "MODMA-Dataset",
            "description": "Multi-modal mental disorder analysis dataset",
            "total_entries": len(entries),
            "modalities": ["text", "audio", "visual"],
            "disorders": list(self.mental_disorders.keys()),
            "multimodal_features": self.multimodal_features,
            "data_collection": {
                "setting": "clinical_controlled",
                "participants": len(entries),
                "sessions_per_participant": 1,
                "synchronization": "multimodal_aligned"
            },
            "created_at": datetime.now().isoformat()
        }

        with open(self.dataset_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_modma_entries(self) -> list[dict[str, Any]]:
        """Load MODMA entries."""
        entries = []

        data_file = self.dataset_path / "modma_entries.jsonl"
        if data_file.exists():
            with open(data_file) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num}: {e}")

        return entries

    def _process_modma_entry(self, entry_data: dict[str, Any]) -> MODMAMultimodalEntry | None:
        """Process a MODMA entry."""
        try:
            entry_id = entry_data.get("entry_id", f"modma_{hash(str(entry_data))%10000}")

            # Extract multimodal features
            multimodal_features = {}

            # Text features
            text_data = entry_data.get("text_data", {})
            if text_data:
                multimodal_features.update({
                    "text_sentiment": text_data.get("sentiment_score", 0),
                    "text_complexity": text_data.get("linguistic_features", {}).get("complexity_score", 0),
                    "text_coherence": text_data.get("linguistic_features", {}).get("coherence_score", 0)
                })

            # Audio features
            audio_data = entry_data.get("audio_data", {})
            if audio_data:
                audio_features = audio_data.get("features", {})
                multimodal_features.update({
                    "audio_pitch_mean": audio_features.get("pitch_mean", 0),
                    "audio_speech_rate": audio_features.get("speech_rate", 0),
                    "audio_pause_ratio": audio_features.get("pause_ratio", 0)
                })

            # Visual features
            visual_data = entry_data.get("visual_data", {})
            if visual_data:
                facial_expr = visual_data.get("facial_expressions", {})
                body_lang = visual_data.get("body_language", {})
                multimodal_features.update({
                    "visual_happiness": facial_expr.get("happiness", 0),
                    "visual_sadness": facial_expr.get("sadness", 0),
                    "visual_eye_contact": body_lang.get("eye_contact_ratio", 0),
                    "visual_gesture_freq": body_lang.get("gesture_frequency", 0)
                })

            return MODMAMultimodalEntry(
                entry_id=entry_id,
                text_modality=text_data,
                audio_modality=audio_data,
                visual_modality=visual_data,
                disorder_analysis=entry_data.get("disorder_labels", {}),
                multimodal_features=multimodal_features
            )

        except Exception as e:
            logger.error(f"Error processing MODMA entry: {e}")
            return None

    def _assess_modma_quality(self, entries: list[MODMAMultimodalEntry]) -> dict[str, float]:
        """Assess quality of MODMA dataset."""
        if not entries:
            return {"overall_quality": 0.0}

        # Modality coverage
        text_coverage = sum(1 for e in entries if e.text_modality) / len(entries)
        audio_coverage = sum(1 for e in entries if e.audio_modality) / len(entries)
        visual_coverage = sum(1 for e in entries if e.visual_modality) / len(entries)

        # Multimodal completeness
        complete_multimodal = sum(1 for e in entries if e.text_modality and e.audio_modality and e.visual_modality) / len(entries)

        # Feature richness
        avg_features = sum(len(e.multimodal_features) for e in entries) / len(entries)
        feature_richness = min(1.0, avg_features / 10)  # Normalize to 10 expected features

        # Disorder coverage
        disorders_covered = {e.disorder_analysis.get("primary_disorder") for e in entries if e.disorder_analysis.get("primary_disorder")}
        disorder_coverage = len(disorders_covered) / len(self.mental_disorders)

        return {
            "overall_quality": (text_coverage + audio_coverage + visual_coverage + complete_multimodal + feature_richness + disorder_coverage) / 6,
            "text_modality_coverage": text_coverage,
            "audio_modality_coverage": audio_coverage,
            "visual_modality_coverage": visual_coverage,
            "complete_multimodal_entries": complete_multimodal,
            "feature_richness": feature_richness,
            "disorder_coverage": disorder_coverage,
            "average_features_per_entry": avg_features
        }


    def _save_modma_processed(self, entries: list[MODMAMultimodalEntry],
                            quality_metrics: dict[str, float]) -> Path:
        """Save processed MODMA dataset."""
        output_file = self.output_dir / "modma_dataset_multimodal_processed.json"

        # Convert to serializable format
        entries_data = []
        for entry in entries:
            entry_dict = {
                "entry_id": entry.entry_id,
                "text_modality": entry.text_modality,
                "audio_modality": entry.audio_modality,
                "visual_modality": entry.visual_modality,
                "disorder_analysis": entry.disorder_analysis,
                "multimodal_features": entry.multimodal_features
            }
            entries_data.append(entry_dict)

        output_data = {
            "dataset_info": {
                "name": "MODMA-Dataset Multi-modal Analysis",
                "description": "Multi-modal mental disorder analysis with text, audio, and visual modalities",
                "total_entries": len(entries),
                "processed_at": datetime.now().isoformat()
            },
            "quality_metrics": quality_metrics,
            "mental_disorders": self.mental_disorders,
            "multimodal_features": self.multimodal_features,
            "entries": entries_data
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"MODMA dataset processed data saved: {output_file}")
        return output_file


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = MODMADatasetMultimodal()

    # Process MODMA dataset
    result = processor.process_modma_dataset()

    # Show results
    if result["success"]:
        pass
    else:
        pass

