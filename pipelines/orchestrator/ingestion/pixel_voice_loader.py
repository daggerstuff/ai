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
        self._cached_pairs: list[VoiceDialoguePair] | None = None

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
        if self._cached_pairs is not None:
            return list(self._cached_pairs)

        if self.therapeutic_pairs_file.suffix == ".jsonl":
            pairs = self._load_jsonl_pairs()
            self._cached_pairs = list(pairs)
            return pairs

        if not self.therapeutic_pairs_file.exists():
            logger.warning(
                f"Therapeutic pairs file not found: {self.therapeutic_pairs_file}"
            )
            logger.info("Trying alternative: validated dialogue pairs")
            pairs = self._load_validated_pairs()
            self._cached_pairs = list(pairs)
            return pairs

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
            self._cached_pairs = list(pairs)
            return pairs

        except Exception as e:
            logger.error(f"Failed to load therapeutic pairs: {e}")
            return []

    def _load_jsonl_pairs(self) -> list[VoiceDialoguePair]:
        """Load JSONL conversation exports and normalize them into dialogue pairs."""
        if not self.therapeutic_pairs_file.exists():
            logger.warning(
                f"Therapeutic JSONL file not found: {self.therapeutic_pairs_file}"
            )
            return []

        pairs = []
        skipped_count = 0
        try:
            with open(self.therapeutic_pairs_file) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        pair = self._pair_from_conversation_export(item)
                        if pair is None:
                            pair = self._pair_from_stage4_persona_record(item)
                        if pair is not None:
                            pairs.append(pair)
                    except Exception as e:
                        skipped_count += 1
                        logger.error(
                            "Error parsing JSONL dialogue pair line %s: %s | raw=%s",
                            line_num,
                            e,
                            line[:200],
                        )
                        continue

            logger.info(
                "Loaded %s therapeutic dialogue pairs from JSONL export (%s skipped)",
                len(pairs),
                skipped_count,
            )
            return pairs

        except Exception as e:
            logger.error(f"Failed to load therapeutic JSONL pairs: {e}")
            return []

    def _pair_from_conversation_export(
        self, item: dict
    ) -> VoiceDialoguePair | None:
        """Convert message-based exports into the legacy two-turn pair shape."""
        messages = item.get("messages")
        if not isinstance(messages, list):
            return None

        user_turn = ""
        assistant_turn = ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content", "")
            if role == "user" and not user_turn:
                user_turn = content
            elif role == "assistant" and not assistant_turn:
                assistant_turn = content
            if user_turn and assistant_turn:
                break

        if not user_turn or not assistant_turn:
            return None

        metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        personality_markers = metadata.get("personality_markers", {})
        quality_scores = metadata.get("quality_scores", {})
        empathy_score = float(quality_scores.get("empathy", 0.5))
        raw_safety = quality_scores.get("safety")
        raw_toxicity = quality_scores.get("toxicity")
        if raw_toxicity is not None:
            toxicity_score = max(0.0, min(1.0, float(raw_toxicity)))
        elif raw_safety is not None:
            toxicity_score = max(0.0, min(1.0, 1.0 - float(raw_safety)))
        else:
            toxicity_score = 0.2

        validation_scores = {
            "empathy_turn_1": [{"score": empathy_score}],
            "empathy_turn_2": [{"score": empathy_score}],
            "toxicity_turn_1": [{"score": toxicity_score}],
            "toxicity_turn_2": [{"score": toxicity_score}],
        }

        provenance = metadata.get("provenance") or {}

        return VoiceDialoguePair(
            turn_1=user_turn,
            turn_2=assistant_turn,
            personality_markers=personality_markers,
            emotional_patterns=metadata.get("emotional_patterns", {}),
            validation_scores=validation_scores,
            source_url=provenance.get("original_path"),
            transcription_quality=float(quality_scores.get("therapeutic_value", 0.0)),
            naturalness_score=float(quality_scores.get("empathy", 0.0)),
        )

    def _pair_from_stage4_persona_record(
        self, item: dict
    ) -> VoiceDialoguePair | None:
        """Convert stage-native persona JSONL records into a two-turn pair."""
        data = item.get("data")
        if not isinstance(data, dict):
            return None

        raw_text = data.get("text")
        source_name = str(item.get("source", "stage4_voice_persona"))
        character_name = str(data.get("name") or data.get("character_name") or "").strip()
        description = str(data.get("description") or data.get("scenario") or "").strip()
        user_turn = ""
        assistant_turn = ""

        if isinstance(raw_text, str) and "<|user|>" in raw_text:
            user_turn = self._extract_tagged_segment(raw_text, ("<|user|>",), ("<|assistant|>",))
            assistant_turn = self._extract_tagged_segment(
                raw_text,
                ("<|assistant|>",),
                ("<|end|>", "</s>", "<|user|>"),
            )
        elif source_name == "google/Synthetic-Persona-Chat":
            conversation = data.get("Best Generated Conversation", "")
            if isinstance(conversation, str):
                user_turn, assistant_turn = self._pair_from_speaker_transcript(
                    conversation,
                    "User 1:",
                    "User 2:",
                )
        elif source_name == "nazlicanto/persona-based-chat":
            dialogue = data.get("dialogue")
            if isinstance(dialogue, list):
                user_turn, assistant_turn = self._pair_from_dialogue_list(
                    dialogue,
                    "Persona A:",
                    "Persona B:",
                )

        if not user_turn or not assistant_turn:
            return None

        personality_markers = {
            "signature": character_name or source_name,
            "speaker": source_name,
        }
        if description:
            personality_markers["description"] = description[:500]

        validation_scores = {
            "empathy_turn_1": [{"score": 0.75}],
            "empathy_turn_2": [{"score": 0.75}],
            "toxicity_turn_1": [{"score": 0.15}],
            "toxicity_turn_2": [{"score": 0.15}],
        }

        return VoiceDialoguePair(
            turn_1=user_turn,
            turn_2=assistant_turn,
            personality_markers=personality_markers,
            emotional_patterns={},
            validation_scores=validation_scores,
            source_url=None,
            transcription_quality=0.8,
            naturalness_score=0.75,
        )

    @staticmethod
    def _extract_tagged_segment(
        text: str,
        start_tags: tuple[str, ...],
        end_tags: tuple[str, ...],
    ) -> str:
        """Extract the first tagged segment from roleplay-style records."""
        start_index = -1
        matched_start = ""
        for start_tag in start_tags:
            candidate_index = text.find(start_tag)
            if candidate_index == -1:
                continue
            if start_index == -1 or candidate_index < start_index:
                start_index = candidate_index
                matched_start = start_tag
        if start_index == -1:
            return ""
        start_index += len(matched_start)
        end_index = len(text)
        for end_tag in end_tags:
            candidate_index = text.find(end_tag, start_index)
            if candidate_index == -1:
                continue
            end_index = min(end_index, candidate_index)
        return text[start_index:end_index].replace("</s>", "").strip()

    @staticmethod
    def _pair_from_speaker_transcript(
        transcript: str,
        first_speaker: str,
        second_speaker: str,
    ) -> tuple[str, str]:
        first_turn = ""
        second_turn = ""
        for raw_line in transcript.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if not first_turn and line.startswith(first_speaker):
                first_turn = line[len(first_speaker) :].strip()
                continue
            if first_turn and not second_turn and line.startswith(second_speaker):
                second_turn = line[len(second_speaker) :].strip()
                break
        return first_turn, second_turn

    @staticmethod
    def _pair_from_dialogue_list(
        dialogue: list[object],
        first_speaker: str,
        second_speaker: str,
    ) -> tuple[str, str]:
        first_turn = ""
        second_turn = ""
        for entry in dialogue:
            if not isinstance(entry, str):
                continue
            line = entry.strip()
            if not first_turn and line.startswith(first_speaker):
                first_turn = line[len(first_speaker) :].strip()
                continue
            if first_turn and not second_turn and line.startswith(second_speaker):
                second_turn = line[len(second_speaker) :].strip()
                break
        return first_turn, second_turn

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
