#!/usr/bin/env python3
"""
Process ALL persona transcripts from Google Drive into Stage 4 training format.

Per MasterTrainingPlan.md:
- Convert YouTube transcripts into conversation + voice_signature format
- Store derived files in ai/data/<persona_name>_voice/exports/
- Generate voice profiles with signature tokens for each speaker

This script processes ALL speakers in tier4_voice_persona, not just Tim Fletcher.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Base directories
PERSONA_BASE = Path("/home/vivi/mnt/gdrive/backups/S3-Complete/archive/gdrive/tier4_voice_persona")
OUTPUT_BASE = Path("/home/vivi/pixelated/ai/data")

# Persona configurations with expertise and style markers
PERSONA_CONFIGS: Dict[str, dict] = {
    "Tim Fletcher": {
        "voice_signature": "tim_fletcher_v1",
        "description": "Compassionate therapist specializing in complex trauma (C-PTSD) and emotional processing",
        "expertise": ["complex_trauma", "C-PTSD", "emotional_processing", "narcissistic_abuse", "attachment_theory"],
        "tone": "warm_insightful_empathetic",
        "approach": "psychoeducational",
        "style_notes": "Uses psychoeducational framing, validates before offering insight, references trauma theory"
    },
    "DoctorRamani": {
        "voice_signature": "doctor_ramani_v1",
        "description": "Clinical psychologist specializing in narcissistic abuse, personality disorders, and mental health education",
        "expertise": ["narcissistic_abuse", "personality_disorders", "mental_health_education", "toxic_relationships", "boundaries"],
        "tone": "authoritative_educational_compassionate",
        "approach": "clinical_education",
        "style_notes": "Direct clinical perspective, educational framing, emphasizes validation and no-contact strategies"
    },
    "Patrick Teahan": {
        "voice_signature": "patrick_teahan_v1",
        "description": "Therapist and content creator focusing on mental health awareness and therapeutic insights",
        "expertise": ["mental_health_awareness", "therapy_insights", "emotional_regulation", "self_awareness"],
        "tone": "conversational_insightful_warm",
        "approach": "conversational_therapeutic",
        "style_notes": "Conversational style, practical insights, relatable examples"
    },
    "Crappy Childhood Fairy": {
        "voice_signature": "ccf_v1",
        "description": "Trauma recovery specialist focusing on childhood trauma and CPTSD healing",
        "expertise": ["childhood_trauma", "CPTSD", "trauma_recovery", "inner_child_work", "healing_journey"],
        "tone": "nurturing_empowering_gentle",
        "approach": "trauma_informed_compassionate",
        "style_notes": "Nurturing approach, emphasizes empowerment, inner child work, gentle healing"
    },
    "Therapy in a Nutshell": {
        "voice_signature": "therapy_nutshell_v1",
        "description": "Mental health education with practical tools and skills for everyday challenges",
        "expertise": ["mental_health_tools", "coping_skills", "anxiety_management", "depression_tools", "skill_building"],
        "tone": "practical_encouraging_clear",
        "approach": "skill_based_educational",
        "style_notes": "Practical skills, clear explanations, actionable tools, evidence-based"
    },
    "Heidi Priebe": {
        "voice_signature": "heidi_priebe_v1",
        "description": "Therapist and educator specializing in attachment, trauma, and family systems",
        "expertise": ["attachment_theory", "family_systems", "trauma", "boundaries", "healing_relationships"],
        "tone": "warm_insightful_systemic",
        "approach": "attachment_informed_systemic",
        "style_notes": "Systems perspective, attachment-focused, warm and relatable"
    },
    "Doc Snipes": {
        "voice_signature": "doc_snipes_v1",
        "description": "Clinical counselor focusing on codependency, validation, and mental health education",
        "expertise": ["codependency", "validation", "mental_health_education", "addiction_recovery", "family_dynamics"],
        "tone": "professional_educational_supportive",
        "approach": "clinical_educational",
        "style_notes": "Professional clinical perspective, educational focus on validation and healthy dynamics"
    }
}


def extract_title_from_filename(filename: str) -> str:
    """Extract title from filename by removing .txt extension."""
    return filename.replace('.txt', '').replace('_', ' ').strip()


def process_transcript(transcript_path: Path, persona_name: str, config: dict) -> dict:
    """
    Process a single transcript file into training format.

    Returns a conversation record with voice signature metadata.
    """
    title = extract_title_from_filename(transcript_path.name)

    # Read transcript content
    with open(transcript_path, encoding='utf-8') as f:
        content = f.read().strip()

    # Create conversation record in Stage 4 format
    conversation = {
        "conversation_id": f"{persona_name.lower().replace(' ', '_')}_{transcript_path.stem}",
        "stage": "stage4_voice_persona",
        "messages": [
            {
                "role": "system",
                "content": f"You are {persona_name}. {config['description']}. Your approach is {config['tone'].replace('_', ', ')}."
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
            "source": f"{persona_name.lower().replace(' ', '_')}_transcripts",
            "source_family": "stage4_voice_persona",
            "voice_signature": config["voice_signature"],
            "persona_id": persona_name.lower().replace(' ', '_'),
            "personality_markers": {
                "style": config["tone"],
                "expertise": config["expertise"],
                "tone": config["tone"],
                "approach": config["approach"],
                "style_notes": config["style_notes"]
            },
            "crisis_intensity": "high",  # Therapeutic content often involves trauma
            "is_training_edge_case": False,  # These are therapeutic, not edge cases
            "quality_scores": {
                "empathy": 0.85,
                "safety": 0.95,
                "therapeutic_value": 0.90
            },
            "provenance": {
                "original_path": str(transcript_path),
                "processed_at": datetime.now().isoformat(),
                "processing_pipeline": "process_all_personas"
            }
        }
    }

    return conversation


def generate_voice_profile(persona_name: str, config: dict, conversation_count: int) -> dict:
    """
    Generate voice profile from processed conversations.
    """
    profile = {
        "name": persona_name,
        "voice_signature": config["voice_signature"],
        "description": config["description"],
        "personality_traits": {
            "primary_style": config["tone"],
            "communication_patterns": [
                config["style_notes"],
                f"Expertise in: {', '.join(config['expertise'])}",
                f"Approach: {config['approach']}"
            ],
            "expertise_areas": config["expertise"],
            "tone_characteristics": {
                "empathy_level": "high",
                "formality": "professional_yet_accessible",
                "pacing": "deliberate_thoughtful",
                "emotional_temperature": config["tone"].split('_')[0]
            }
        },
        "training_samples": conversation_count,
        "created_at": datetime.now().isoformat(),
        "version": "1.0"
    }

    return profile


def process_persona_folder(persona_name: str) -> dict:
    """
    Process all transcripts for a single persona.

    Returns summary statistics.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {persona_name}")
    print(f"{'='*60}")

    config = PERSONA_CONFIGS.get(persona_name, {
        "voice_signature": f"{persona_name.lower().replace(' ', '_')}_v1",
        "description": f"Therapist and content creator",
        "expertise": ["mental_health"],
        "tone": "professional",
        "approach": "therapeutic",
        "style_notes": "Professional therapeutic approach"
    })

    persona_dir = PERSONA_BASE / persona_name
    output_dir = OUTPUT_BASE / f"{persona_name.lower().replace(' ', '_')}_voice" / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all transcript files
    transcript_files = list(persona_dir.glob("*.txt"))
    print(f"Found {len(transcript_files)} transcript files")

    if not transcript_files:
        return {"persona": persona_name, "processed": 0, "output": str(output_dir)}

    # Process each transcript
    conversations = []
    output_file = output_dir / f"{persona_name.lower().replace(' ', '_')}_conversations.jsonl"

    print(f"Processing transcripts...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, transcript_path in enumerate(transcript_files, 1):
            try:
                conversation = process_transcript(transcript_path, persona_name, config)
                conversations.append(conversation)
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')

                if i % 5 == 0 or i == len(transcript_files):
                    print(f"  Processed {i}/{len(transcript_files)}: {transcript_path.name[:50]}...")
            except Exception as e:
                print(f"  ERROR processing {transcript_path.name}: {e}")

    print(f"\nProcessed {len(conversations)} conversations")
    print(f"Output written to: {output_file}")

    # Generate voice profile
    print(f"\nGenerating voice profile for {persona_name}...")
    profile = generate_voice_profile(persona_name, config, len(conversations))

    profile_file = output_dir.parent / f"{persona_name.lower().replace(' ', '_')}_voice_profile.json"
    with open(profile_file, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print(f"Voice profile written to: {profile_file}")

    return {
        "persona": persona_name,
        "processed": len(conversations),
        "output": str(output_file),
        "profile": str(profile_file)
    }


def main():
    print("="*60)
    print("Processing ALL Persona Transcripts for Stage 4 Training")
    print("="*60)

    # Get list of persona folders
    persona_folders = [d.name for d in PERSONA_BASE.iterdir() if d.is_dir()]
    print(f"\nFound {len(persona_folders)} persona folders:")
    for p in persona_folders:
        print(f"  - {p}")

    # Process each persona
    results = []
    for persona_name in persona_folders:
        result = process_persona_folder(persona_name)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("STAGE 4 VOICE PROCESSING COMPLETE")
    print("="*60)

    total_conversations = 0
    for r in results:
        print(f"  {r['persona']}: {r['processed']} conversations")
        total_conversations += r['processed']

    print(f"\nTotal: {total_conversations} conversations across {len(results)} personas")
    print(f"\nNext steps:")
    print("  1. Review generated conversations in ai/data/")
    print("  2. Integrate into Stage 4 training dataset")
    print("  3. Use persona_blender.py to combine with Stage 1 data")


if __name__ == "__main__":
    main()
