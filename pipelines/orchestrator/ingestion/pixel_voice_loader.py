#!/usr/bin/env python3
"""
Pixel Voice Data Loader for Training Pipeline Integration
Loads voice-derived therapeutic dialogues from the Pixel Voice pipeline
"""

import json
from dataclasses import dataclass
from pathlib import Path

from ai.pipelines.orchestrator.utils.logger import get_logger

logger = get_logger("dataset_pipeline.pixel_voice_loader")


@dataclass
class VoiceDialoguePair:
    """Structured voice-derived dialogue pair"""

    turn_1: str
    turn_2: str
    personality_markers: dict
    emotional_patterns: dict
    validation_scores: dict
    source_url: str | None = None
    transcription_quality: float = 0.0
    naturalness_score: float = 0.0

    def to_training_format(self) -> dict:
        """Convert to standard training format"""
        # Create conversational text
        text = f"Therapist: {self.turn_1}\nClient: {self.turn_2}"
        voice_signature = (
            self.personality_markers.get("signature")
            or self.personality_markers.get("speaker")
            or "tim_fletcher_voice_profile"
        )

        # Extract empathy and safety scores from validation_scores
        # Empathy: average of empathy scores from both turns
        validation = self.validation_scores or {}
        emp1 = validation.get("empathy_turn_1", [{}])[0].get("score", 0.5)
        emp2 = validation.get("empathy_turn_2", [{}])[0].get("score", 0.5)
        empathy_score = (emp1 + emp2) / 2.0

        # Safety: derived from toxicity scores (safety = 1 - toxicity)
        # Lower toxicity = higher safety
        tox1 = validation.get("toxicity_turn_1", [{}])[0].get("score", 0.3)
        tox2 = validation.get("toxicity_turn_2", [{}])[0].get("score", 0.3)
        avg_toxicity = (tox1 + tox2) / 2.0
        safety_score = max(0.0, min(1.0, 1.0 - avg_toxicity))

        return {
            "text": text,
            "prompt": self.turn_1,
            "response": self.turn_2,
            "metadata": {
                "source": "pixel_voice",
                "personality_markers": self.personality_markers,
                "emotional_patterns": self.emotional_patterns,
                "validation_scores": self.validation_scores,
                "transcription_quality": self.transcription_quality,
                "naturalness_score": self.naturalness_score,
                "source_url": self.source_url,
                "is_voice_derived": True,
                "is_edge_case": False,
                "stage": "stage4_voice_persona",
                "voice_signature": voice_signature,
                "quality_profile": "voice",
                "empathy_score": empathy_score,
                "safety_score": safety_score,
            },
        }


@dataclass
class VoiceConfig:
    """Configuration for PixelVoiceLoader"""

    pipeline_dir: str = "ai/pipelines/voice"


class PixelVoiceLoader:
    """Loader for Pixel Voice pipeline therapeutic dialogue data"""

    def __init__(
        self, config: VoiceConfig | None = None, file_path: Path | None = None
    ):
        self.config = config or VoiceConfig()

        # Allow override
        if file_path:
            path = Path(file_path)
            if path.is_dir():
                # Allow for pipeline directory to be passed
                candidates = [
                    path / "data/therapeutic_pairs/therapeutic_pairs.json",
                    path / "therapeutic_pairs.json",
                ]
                self.therapeutic_pairs_file = candidates[0]  # Default
                for candidate in candidates:
                    if candidate.exists():
                        self.therapeutic_pairs_file = candidate
                        break
            else:
                self.therapeutic_pairs_file = path
        else:
            self.therapeutic_pairs_file = Path(
                "ai/data/tim_fletcher_voice/therapeutic_pairs.json"
            )

        self.voice_profile_path = Path(
            "ai/data/tim_fletcher_voice/tim_fletcher_voice_profile.json"
        )
        self.dialogue_pairs_file = Path(
            "ai/data/tim_fletcher_voice/dialogue_pairs_validated.json"
        )  # Assuming this is the intended replacement for the old dialogue_pairs_file

    def load_therapeutic_pairs(self) -> list[VoiceDialoguePair]:
        """Load therapeutic dialogue pairs"""
        if not self.therapeutic_pairs_file.exists():
            logger.warning(
                f"Therapeutic pairs file not found: {self.therapeutic_pairs_file}"
            )
            logger.info("Trying alternative: validated dialogue pairs")
            return self._load_validated_pairs()

        try:
            with open(self.therapeutic_pairs_file) as f:
                data = json.load(f)

            pairs = []
            for item in data:
                try:
                    pair = VoiceDialoguePair(
                        turn_1=item.get("turn_1", ""),
                        turn_2=item.get("turn_2", ""),
                        personality_markers=item.get("personality", {}),
                        emotional_patterns=item.get("emotions", {}),
                        validation_scores=item.get("validation", {}),
                        source_url=item.get("source_url"),
                        transcription_quality=item.get("transcription_quality", 0.0),
                        naturalness_score=item.get("naturalness_score", 0.0),
                    )
                    pairs.append(pair)
                except Exception as e:
                    logger.error(f"Error parsing dialogue pair: {e}")
                    continue

            logger.info(f"Loaded {len(pairs)} therapeutic dialogue pairs")
            return pairs

        except Exception as e:
            logger.error(f"Failed to load therapeutic pairs: {e}")
            return []

    def _load_validated_pairs(self) -> list[VoiceDialoguePair]:
        """Load validated dialogue pairs as fallback"""
        if not self.dialogue_pairs_file.exists():
            logger.warning(
                f"Validated pairs file not found: {self.dialogue_pairs_file}"
            )
            return []

        try:
            with open(self.dialogue_pairs_file) as f:
                data = json.load(f)

            pairs = []
            for item in data:
                try:
                    # Filter for therapeutic quality
                    if not self._is_therapeutic_quality(item):
                        continue

                    pair = VoiceDialoguePair(
                        turn_1=item.get("turn_1", ""),
                        turn_2=item.get("turn_2", ""),
                        personality_markers=item.get("personality", {}),
                        emotional_patterns=item.get("emotions", {}),
                        validation_scores=item.get("validation", {}),
                        source_url=item.get("source_url"),
                        transcription_quality=item.get("transcription_quality", 0.0),
                        naturalness_score=item.get("naturalness_score", 0.0),
                    )
                    pairs.append(pair)
                except Exception as e:
                    logger.error(f"Error parsing validated pair: {e}")
                    continue

            logger.info(f"Loaded {len(pairs)} validated dialogue pairs")
            return pairs

        except Exception as e:
            logger.error(f"Failed to load validated pairs: {e}")
            return []

    def _is_therapeutic_quality(self, item: dict) -> bool:
        """Check if dialogue pair meets therapeutic quality standards"""
        try:
            validation = item.get("validation", {})

            # Check empathy scores
            emp1 = validation.get("empathy_turn_1", [{}])[0].get("score", 0)
            emp2 = validation.get("empathy_turn_2", [{}])[0].get("score", 0)

            # Check toxicity scores
            tox1 = validation.get("toxicity_turn_1", [{}])[0].get("score", 1)
            tox2 = validation.get("toxicity_turn_2", [{}])[0].get("score", 1)

            # Therapeutic criteria
            has_empathy = emp1 >= 0.7 or emp2 >= 0.7
            low_toxicity = tox1 <= 0.3 and tox2 <= 0.3

            return has_empathy and low_toxicity

        except Exception:
            return False

    def get_statistics(self) -> dict:
        """Get statistics about loaded voice data"""
        pairs = self.load_therapeutic_pairs()

        if not pairs:
            return {
                "total_pairs": 0,
                "avg_transcription_quality": 0.0,
                "avg_naturalness_score": 0.0,
                "personality_markers": {},
                "emotional_patterns": {},
            }

        # Calculate averages
        avg_transcription = sum(p.transcription_quality for p in pairs) / len(pairs)
        avg_naturalness = sum(p.naturalness_score for p in pairs) / len(pairs)

        # Count personality markers
        personality_counts = {}
        for pair in pairs:
            for marker, value in pair.personality_markers.items():
                if marker not in personality_counts:
                    personality_counts[marker] = []
                personality_counts[marker].append(value)

        # Count emotional patterns
        emotion_counts = {}
        for pair in pairs:
            for emotion, value in pair.emotional_patterns.items():
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = []
                emotion_counts[emotion].append(value)

        return {
            "total_pairs": len(pairs),
            "avg_transcription_quality": avg_transcription,
            "avg_naturalness_score": avg_naturalness,
            "personality_markers": {k: len(v) for k, v in personality_counts.items()},
            "emotional_patterns": {k: len(v) for k, v in emotion_counts.items()},
            "file_path": str(self.therapeutic_pairs_file),
        }

    def convert_to_training_format(
        self, pairs: list[VoiceDialoguePair] | None = None
    ) -> list[dict]:
        """Convert voice dialogue pairs to standard training format"""
        if pairs is None:
            pairs = self.load_therapeutic_pairs()

        training_data = [pair.to_training_format() for pair in pairs]
        logger.info(f"Converted {len(training_data)} voice pairs to training format")
        return training_data

    def check_pipeline_output_exists(self) -> bool:
        """Check if Pixel Voice pipeline has been run and output exists"""
        return self.therapeutic_pairs_file.exists() or self.dialogue_pairs_file.exists()

    def get_pipeline_instructions(self) -> str:
        """Get instructions for running the Pixel Voice pipeline"""
        return """
To generate Pixel Voice training data:

1. Navigate to the Pixel Voice pipeline:
   cd ai/pipelines/voice/

2. Ensure you have audio/transcript data:
   - YouTube transcripts in data/transcripts/
   - Or audio files in data/audio/

3. Run the full pipeline:
   python run_full_pipeline.py

   This will:
   - Process audio quality
   - Transcribe audio
   - Filter transcription quality
   - Extract personality features
   - Cluster emotions
   - Construct dialogue pairs
   - Validate pairs
   - Generate therapeutic pairs

4. Output will be saved to:
   data/therapeutic_pairs/therapeutic_pairs.json

5. Then this loader will automatically find and load the data

Alternative - Run individual stages:
   python batch_transcribe.py
   python feature_extraction.py
   python dialogue_pair_constructor.py
   python generate_therapeutic_pairs.py
"""


def load_pixel_voice_training_data(pipeline_dir: str | None = None) -> list[dict]:
    """
    Convenience function to load Pixel Voice training data

    Args:
        pipeline_dir: Optional path to Pixel Voice pipeline directory

    Returns:
        List of training examples in standard format
    """
    loader = (
        PixelVoiceLoader(file_path=Path(pipeline_dir))
        if pipeline_dir
        else PixelVoiceLoader()
    )

    if not loader.check_pipeline_output_exists():
        logger.warning("Pixel Voice training data not found!")
        logger.info(loader.get_pipeline_instructions())
        return []

    return loader.convert_to_training_format()


if __name__ == "__main__":
    # Test the loader
    loader = PixelVoiceLoader()

    logger.info("Pixel Voice Training Data Loader")
    logger.info("=" * 60)

    if not loader.check_pipeline_output_exists():
        logger.warning("\n❌ Pixel Voice training data not found!")
        logger.info(loader.get_pipeline_instructions())
    else:
        logger.info("\n✅ Pixel Voice training data found!")

        # Load and show statistics
        stats = loader.get_statistics()
        logger.info("\n📊 Statistics:")
        logger.info(f"   Total pairs: {stats['total_pairs']}")
        logger.info(
            f"   Avg transcription quality: {stats['avg_transcription_quality']:.2f}"
        )
        logger.info(f"   Avg naturalness score: {stats['avg_naturalness_score']:.2f}")

        if stats["personality_markers"]:
            logger.info("\n👤 Personality Markers:")
            for marker, count in list(stats["personality_markers"].items())[:5]:
                logger.info(f"   {marker}: {count}")

        if stats["emotional_patterns"]:
            logger.info("\n😊 Emotional Patterns:")
            for emotion, count in list(stats["emotional_patterns"].items())[:5]:
                logger.info(f"   {emotion}: {count}")

        # Load training data
        training_data = loader.convert_to_training_format()
        logger.info(f"\n✅ Loaded {len(training_data)} training examples")

        if training_data:
            logger.info("\n📝 Sample example:")
            sample = training_data[0]
            logger.info(f"   Source: {sample['metadata']['source']}")
            logger.info(
                f"   Transcription quality: "
                f"{sample['metadata']['transcription_quality']:.2f}"
            )
            logger.info(f"   Text: {sample['text'][:200]}...")
