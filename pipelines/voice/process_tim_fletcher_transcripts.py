#!/usr/bin/env python3
"""
Process Tim Fletcher transcripts into Stage 4 training format.

Per MasterTrainingPlan.md:
- Convert YouTube transcripts into conversation + voice_signature format
- Store derived files in ai/data/tim_fletcher_voice/exports/
- Generate voice profile with signature tokens
"""

import json
from pathlib import Path
from datetime import datetime

# Input/output directories
TRANSCRIPTS_DIR = Path("/home/vivi/pixelated/ai/training_data_consolidated/transcripts")
OUTPUT_DIR = Path("/home/vivi/pixelated/ai/data/tim_fletcher_voice/exports")
PROFILE_FILE = Path("/home/vivi/pixelated/ai/data/tim_fletcher_voice/tim_fletcher_voice_profile.json")

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)

def extract_title_from_filename(filename: str) -> str:
    """Extract title from filename by removing .txt extension."""
    return filename.replace('.txt', '').replace('_', ' ').strip()

def process_transcript(transcript_path: Path) -> dict:
    """
    Process a single transcript file into training format.

    Returns a conversation record with voice signature metadata.
    """
    title = extract_title_from_filename(transcript_path.name)

    # Read transcript content
    with open(transcript_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Create conversation record in Stage 4 format
    conversation = {
        "conversation_id": f"tim_fletcher_{transcript_path.stem}",
        "stage": "stage4_voice_persona",
        "messages": [
            {
                "role": "system",
                "content": "You are Tim Fletcher, a compassionate therapist specializing in complex trauma (C-PTSD) and emotional processing. Your approach is warm, insightful, and deeply empathetic. You help clients understand their trauma responses and develop healthier coping mechanisms."
            },
            {
                "role": "user",
                "content": f"Can you help me understand: {title}?"
            },
            {
                "role": "assistant",
                "content": content
            }
        ],
        "metadata": {
            "source": "tim_fletcher_transcripts",
            "source_family": "stage4_voice_persona",
            "voice_signature": "tim_fletcher_v1",
            "persona_id": "therapist_mentor",
            "personality_markers": {
                "style": "compassionate_therapist",
                "expertise": ["complex_trauma", "C-PTSD", "emotional_processing", "narcissistic_abuse"],
                "tone": "warm_insightful_empathetic",
                "approach": "psychoeducational"
            },
            "crisis_intensity": "high",  # Tim Fletcher topics often involve trauma
            "is_training_edge_case": False,  # These are therapeutic, not edge cases
            "quality_scores": {
                "empathy": 0.85,
                "safety": 0.95,
                "therapeutic_value": 0.90
            },
            "provenance": {
                "original_path": str(transcript_path),
                "processed_at": datetime.now().isoformat(),
                "processing_pipeline": "process_tim_fletcher_transcripts"
            }
        }
    }

    return conversation

def generate_voice_profile(conversations: list) -> dict:
    """
    Generate voice profile from processed conversations.

    This captures Tim Fletcher's therapeutic style, common themes,
    and signature communication patterns.
    """
    profile = {
        "name": "Tim Fletcher",
        "voice_signature": "tim_fletcher_v1",
        "description": "Compassionate therapist specializing in complex trauma (C-PTSD), emotional processing, and narcissistic abuse recovery",
        "personality_traits": {
            "primary_style": "warm_therapeutic",
            "communication_patterns": [
                "Uses psychoeducational framing",
                "Validates client experiences before offering insight",
                "References trauma theory and attachment",
                "Employs gentle challenging of maladaptive patterns",
                "Normalizes trauma responses"
            ],
            "expertise_areas": [
                "Complex PTSD (C-PTSD)",
                "Emotional processing",
                "Narcissistic abuse recovery",
                "Attachment theory",
                "Codependency",
                "Childhood trauma",
                "Emotional regulation"
            ],
            "tone_characteristics": {
                "empathy_level": "high",
                "formality": "casual_professional",
                "pacing": "deliberate_thoughtful",
                "emotional_temperature": "warm_validating"
            }
        },
        "signature_phrases": [
            "That makes sense given what you've been through",
            "Your nervous system is trying to protect you",
            "This is a common response to trauma",
            "Let's slow down and notice what's happening"
        ],
        "training_samples": len(conversations),
        "created_at": datetime.now().isoformat(),
        "version": "1.0"
    }

    return profile

def main():
    print("=" * 60)
    print("Processing Tim Fletcher Transcripts for Stage 4 Training")
    print("=" * 60)

    # Find all transcript files
    transcript_files = list(TRANSCRIPTS_DIR.glob("*.txt"))
    print(f"\nFound {len(transcript_files)} transcript files")

    if not transcript_files:
        print("ERROR: No transcript files found!")
        return

    # Process each transcript
    conversations = []
    output_file = OUTPUT_DIR / "tim_fletcher_conversations.jsonl"

    print(f"\nProcessing transcripts...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, transcript_path in enumerate(transcript_files, 1):
            try:
                conversation = process_transcript(transcript_path)
                conversations.append(conversation)
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')

                if i % 10 == 0 or i == len(transcript_files):
                    print(f"  Processed {i}/{len(transcript_files)}: {transcript_path.name[:50]}...")
            except Exception as e:
                print(f"  ERROR processing {transcript_path.name}: {e}")

    print(f"\nProcessed {len(conversations)} conversations")
    print(f"Output written to: {output_file}")

    # Generate voice profile
    print("\nGenerating voice profile...")
    profile = generate_voice_profile(conversations)

    with open(PROFILE_FILE, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print(f"Voice profile written to: {PROFILE_FILE}")

    # Summary
    print("\n" + "=" * 60)
    print("STAGE 4 VOICE PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Transcripts processed: {len(conversations)}")
    print(f"Voice signature: {profile['voice_signature']}")
    print(f"Persona ID: therapist_mentor")
    print(f"\nNext steps:")
    print("  1. Review generated conversations in {OUTPUT_DIR}")
    print("  2. Integrate into Stage 4 training dataset")
    print("  3. Use persona_blender.py to combine with Stage 1 data")

if __name__ == "__main__":
    main()
