"""
Bootstrap-only dataset routing for local/source acquisition helpers.

These mappings are bounded acquisition hints, not canonical production stage
policy and not proof of corpus sufficiency.
"""

from __future__ import annotations

HF_BOOTSTRAP_STAGE_TARGETS: dict[str, dict[str, str]] = {
    "Amod/mental_health_counseling_conversations": {
        "target": "stage1_foundation",
        "split": "train",
    },
    "heliosbrahma/mental_health_chatbot_dataset": {
        "target": "stage1_foundation",
        "split": "train",
    },
    "fadodr/mental_health_therapy": {
        "target": "stage2_specialist_addiction",
        "split": "train",
    },
    "yenopoya/thousand-voices-trauma": {
        "target": "stage2_specialist_ptsd",
        "split": "train",
    },
    "Kanakmi/mental-disorders": {
        "target": "stage2_specialist_personality",
        "split": "train",
    },
    "AIMH/SWMH": {"target": "stage3_edge_crisis", "split": "train"},
}

HF_STREAMING_BOOTSTRAP_TARGETS: dict[str, dict[str, str]] = {
    "Amod/mental_health_counseling_conversations": {"target": "tier2_professional"},
    "heliosbrahma/mental_health_chatbot_dataset": {"target": "tier1_foundation"},
    "fadodr/mental_health_therapy": {"target": "tier2_professional"},
    "yenopoya/thousand-voices-trauma": {"target": "tier2_professional"},
    "Kanakmi/mental-disorders": {"target": "tier2_professional"},
    "Cartinoe5930/CoT-Clinical-Reasoning": {"target": "tier3_cot_reasoning"},
    "Cartinoe5930/CoT-Emotional-Reasoning": {"target": "tier3_cot_reasoning"},
    "WizardLM/WizardLM-70b-V1.0-evol-cot": {"target": "tier3_cot_reasoning"},
    "jondurbin/bagel-dpo-34b-v0.2": {"target": "tier3_cot_reasoning"},
    "google/Synthetic-Persona-Chat": {"target": "tier4_voice_persona"},
    "nazlicanto/persona-based-chat": {"target": "tier4_voice_persona"},
    "hieunguyenminh/roleplay": {"target": "tier4_voice_persona"},
    "NousResearch/CharacterCodex": {"target": "tier4_voice_persona"},
    "HuggingFaceH4/ultrafeedback_binarized": {"target": "tier5_research"},
    "argilla/ultrafeedback-multi-binarized": {"target": "tier5_research"},
}
