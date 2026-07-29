#!/usr/bin/env python3
"""
Mental Health Instruction Dataset Builder

Curates instruction-following training data for mental health prediction tasks,
following the Mental-LLM methodology (arXiv 2307.14385).

Task types:
- symptom_classification
- severity_estimation
- therapy_response_generation
- risk_assessment
- empathy_scoring
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MentalHealthTaskType(str, Enum):
    SYMPTOM_CLASSIFICATION = "symptom_classification"
    SEVERITY_ESTIMATION = "severity_estimation"
    THERAPY_RESPONSE_GENERATION = "therapy_response_generation"
    RISK_ASSESSMENT = "risk_assessment"
    EMPATHY_SCORING = "empathy_scoring"


class DemographicGroup(str, Enum):
    AGE_YOUNG = "age_18_25"
    AGE_MIDDLE = "age_26_45"
    AGE_OLDER = "age_46_plus"
    GENDER_MALE = "gender_male"
    GENDER_FEMALE = "gender_female"
    GENDER_NONBINARY = "gender_nonbinary"
    SES_LOW = "ses_low"
    SES_MIDDLE = "ses_middle"
    SES_HIGH = "ses_high"
    ETHNICITY_DIVERSE = "ethnicity_diverse"


@dataclass
class MentalHealthInstruction:
    """Single instruction-following example for mental health IFT."""

    id: str
    task_type: str
    instruction: str
    input: str
    output: str
    demographic_tags: list[str]
    diagnostic_tag: str | None = None
    linguistic_style: str | None = None
    source: str | None = None
    clinical_reviewed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_alpaca(self) -> dict[str, Any]:
        """Convert to Alpaca-style format used by many SFT trainers."""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "task_type": self.task_type,
            "demographic_tags": self.demographic_tags,
            "diagnostic_tag": self.diagnostic_tag,
            "linguistic_style": self.linguistic_style,
            "source": self.source,
            "clinical_reviewed": self.clinical_reviewed,
        }

    def to_chat(self, system_prompt: str | None = None) -> dict[str, Any]:
        """Convert to chat-completion format."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"{self.instruction}\n\nInput:\n{self.input}"})
        messages.append({"role": "assistant", "content": self.output})
        return {
            "messages": messages,
            "task_type": self.task_type,
            "demographic_tags": self.demographic_tags,
            "diagnostic_tag": self.diagnostic_tag,
            "linguistic_style": self.linguistic_style,
            "source": self.source,
            "clinical_reviewed": self.clinical_reviewed,
        }


# Task-specific instruction templates aligned with Mental-LLM tasks.
INSTRUCTION_TEMPLATES: dict[MentalHealthTaskType, list[str]] = {
    MentalHealthTaskType.SYMPTOM_CLASSIFICATION: [
        "Identify the primary mental health symptoms described in the following text. Return a comma-separated list of symptoms.",
        "Classify the mental health symptoms present in the input. Be specific and concise.",
        "Read the passage and list the psychological symptoms the person is experiencing.",
    ],
    MentalHealthTaskType.SEVERITY_ESTIMATION: [
        "Estimate the severity of the described mental health symptoms on a scale of 1-10, where 1 is minimal and 10 is severe. Explain your reasoning briefly.",
        "Rate the severity of distress described below using a 1-10 scale and justify the rating.",
        "How severe do the symptoms appear? Provide a numeric severity score (1-10) and a short rationale.",
    ],
    MentalHealthTaskType.THERAPY_RESPONSE_GENERATION: [
        "Respond as a compassionate therapist using evidence-based techniques. Validate the person's experience and offer one concrete coping strategy.",
        "Generate a therapeutic response that demonstrates empathy and suggests a CBT or DBT-informed technique.",
        "Write a supportive therapy response. Include validation, normalization, and a gentle question or skill suggestion.",
    ],
    MentalHealthTaskType.RISK_ASSESSMENT: [
        "Assess the level of risk described in the input (none, low, moderate, high, imminent). Include uncertainty quantification and recommended next steps.",
        "Evaluate suicide or self-harm risk based on the text. Provide a risk level and explain what information is missing.",
        "Determine the urgency of the situation. State the risk level and whether immediate professional help is warranted.",
    ],
    MentalHealthTaskType.EMPATHY_SCORING: [
        "Score the empathy of the following therapist response across cognitive, affective, and compassionate dimensions (each 1-5). Provide a brief explanation.",
        "Evaluate how empathetic the response is. Rate cognitive empathy, affective empathy, and compassionate action from 1-5.",
        "Assess the empathy in the response. Give sub-scores for understanding feelings, sharing feelings, and offering support.",
    ],
}


# Seed vignettes for synthetic clinical data generation.
SEED_VIGNETTES: list[dict[str, Any]] = [
    {
        "text": "I can't sleep and I keep thinking everyone would be better off without me. I lost my job two months ago and I feel like a failure.",
        "symptoms": ["insomnia", "hopelessness", "suicidal ideation", "worthlessness"],
        "severity": 8,
        "risk": "high",
        "demographic_tags": ["age_26_45", "gender_male", "ses_low"],
        "diagnostic_tag": "major_depressive_disorder",
        "linguistic_style": "formal",
    },
    {
        "text": "My heart races before meetings and I avoid going out because I'm scared people will judge me.",
        "symptoms": ["anxiety", "panic", "social avoidance"],
        "severity": 6,
        "risk": "low",
        "demographic_tags": ["age_18_25", "gender_female", "ses_middle"],
        "diagnostic_tag": "social_anxiety_disorder",
        "linguistic_style": "informal",
    },
    {
        "text": "I keep replaying the accident in my head. Loud noises make me freeze. I don't feel safe anywhere.",
        "symptoms": ["intrusive thoughts", "hypervigilance", "emotional numbing"],
        "severity": 7,
        "risk": "moderate",
        "demographic_tags": ["age_46_plus", "gender_nonbinary", "ses_middle"],
        "diagnostic_tag": "ptsd",
        "linguistic_style": "formal",
    },
    {
        "text": "Sometimes I get so mad I break things. Afterwards I hate myself and I don't know why I do it.",
        "symptoms": ["anger dysregulation", "impulsivity", "shame"],
        "severity": 5,
        "risk": "moderate",
        "demographic_tags": ["age_18_25", "gender_female", "ses_low"],
        "diagnostic_tag": "emotion_dysregulation",
        "linguistic_style": "informal",
    },
    {
        "text": "I haven't eaten properly in days. Food feels disgusting and I can't stand my body.",
        "symptoms": ["restricted eating", "body image disturbance", "low mood"],
        "severity": 7,
        "risk": "moderate",
        "demographic_tags": ["age_18_25", "gender_male", "ses_high"],
        "diagnostic_tag": "eating_disorder",
        "linguistic_style": "formal",
    },
]


class MentalHealthInstructionDatasetBuilder:
    """Builds instruction datasets for mental health IFT."""

    def __init__(self, seed: int | None = 42):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.examples: list[MentalHealthInstruction] = []

    def build_from_seed_vignettes(self, augment_per_vignette: int = 3) -> list[MentalHealthInstruction]:
        """Generate instruction examples from seed vignettes with demographic augmentation."""
        for vignette in SEED_VIGNETTES:
            base_text = vignette["text"]
            for _ in range(augment_per_vignette):
                # Symptom classification
                self.examples.append(
                    MentalHealthInstruction(
                        id=str(uuid.uuid4()),
                        task_type=MentalHealthTaskType.SYMPTOM_CLASSIFICATION.value,
                        instruction=random.choice(INSTRUCTION_TEMPLATES[MentalHealthTaskType.SYMPTOM_CLASSIFICATION]),
                        input=base_text,
                        output=", ".join(vignette["symptoms"]),
                        demographic_tags=vignette["demographic_tags"],
                        diagnostic_tag=vignette["diagnostic_tag"],
                        linguistic_style=vignette["linguistic_style"],
                        source="synthetic_vignette",
                    )
                )

                # Severity estimation
                self.examples.append(
                    MentalHealthInstruction(
                        id=str(uuid.uuid4()),
                        task_type=MentalHealthTaskType.SEVERITY_ESTIMATION.value,
                        instruction=random.choice(INSTRUCTION_TEMPLATES[MentalHealthTaskType.SEVERITY_ESTIMATION]),
                        input=base_text,
                        output=f"Severity score: {vignette['severity']}/10. Rationale: symptoms significantly impair daily functioning and include "
                        f"{', '.join(vignette['symptoms'][:2])}.",
                        demographic_tags=vignette["demographic_tags"],
                        diagnostic_tag=vignette["diagnostic_tag"],
                        linguistic_style=vignette["linguistic_style"],
                        source="synthetic_vignette",
                    )
                )

                # Risk assessment
                self.examples.append(
                    MentalHealthInstruction(
                        id=str(uuid.uuid4()),
                        task_type=MentalHealthTaskType.RISK_ASSESSMENT.value,
                        instruction=random.choice(INSTRUCTION_TEMPLATES[MentalHealthTaskType.RISK_ASSESSMENT]),
                        input=base_text,
                        output=f"Risk level: {vignette['risk']}. Uncertainty: moderate; clinical interview required. "
                        f"Next step: {'immediate safety evaluation' if vignette['risk'] == 'high' else 'schedule follow-up with licensed clinician'}.",
                        demographic_tags=vignette["demographic_tags"],
                        diagnostic_tag=vignette["diagnostic_tag"],
                        linguistic_style=vignette["linguistic_style"],
                        source="synthetic_vignette",
                    )
                )

                # Therapy response generation
                self.examples.append(
                    MentalHealthInstruction(
                        id=str(uuid.uuid4()),
                        task_type=MentalHealthTaskType.THERAPY_RESPONSE_GENERATION.value,
                        instruction=random.choice(
                            INSTRUCTION_TEMPLATES[MentalHealthTaskType.THERAPY_RESPONSE_GENERATION]
                        ),
                        input=base_text,
                        output=self._generate_therapeutic_response(vignette),
                        demographic_tags=vignette["demographic_tags"],
                        diagnostic_tag=vignette["diagnostic_tag"],
                        linguistic_style=vignette["linguistic_style"],
                        source="synthetic_vignette",
                    )
                )

                # Empathy scoring (input is a therapist response, output is score)
                sample_response = self._generate_therapeutic_response(vignette)
                self.examples.append(
                    MentalHealthInstruction(
                        id=str(uuid.uuid4()),
                        task_type=MentalHealthTaskType.EMPATHY_SCORING.value,
                        instruction=random.choice(INSTRUCTION_TEMPLATES[MentalHealthTaskType.EMPATHY_SCORING]),
                        input=sample_response,
                        output="Cognitive empathy: 4/5, Affective empathy: 4/5, Compassionate action: 4/5. "
                        "The response validates the user's experience and offers a concrete next step.",
                        demographic_tags=vignette["demographic_tags"],
                        diagnostic_tag=vignette["diagnostic_tag"],
                        linguistic_style=vignette["linguistic_style"],
                        source="synthetic_vignette",
                    )
                )

        logger.info(f"Built {len(self.examples)} instruction examples from {len(SEED_VIGNETTES)} seed vignettes")
        return self.examples

    def _generate_therapeutic_response(self, vignette: dict[str, Any]) -> str:
        """Generate a simple evidence-based therapeutic response for a vignette."""
        if vignette["risk"] == "high":
            return (
                "I'm really sorry you're feeling this way. Your pain matters, and you don't have to carry it alone. "
                "If you're thinking about hurting yourself, please reach out to a crisis line or emergency services right now. "
                "Would you be willing to tell me more about what's been hardest today?"
            )
        return (
            "Thank you for sharing that with me. It makes sense that you're feeling overwhelmed given what you've described. "
            "One small step we can practice together is naming the emotion and noticing where you feel it in your body. "
            "What comes up when you try that?"
        )

    def add_conversation_turns(
        self, conversations: list[dict[str, Any]], source: str = "conversation"
    ) -> list[MentalHealthInstruction]:
        """Convert therapy conversation turns into instruction examples."""
        for conv in conversations:
            client = conv.get("client") or conv.get("user_message") or ""
            therapist = conv.get("therapist") or conv.get("assistant_message") or ""
            if not client or not therapist:
                continue

            tags = conv.get("demographic_tags", [])
            diagnostic_tag = conv.get("diagnostic_tag")
            linguistic_style = conv.get("linguistic_style", "formal")

            self.examples.append(
                MentalHealthInstruction(
                    id=str(uuid.uuid4()),
                    task_type=MentalHealthTaskType.THERAPY_RESPONSE_GENERATION.value,
                    instruction=random.choice(INSTRUCTION_TEMPLATES[MentalHealthTaskType.THERAPY_RESPONSE_GENERATION]),
                    input=client,
                    output=therapist,
                    demographic_tags=tags,
                    diagnostic_tag=diagnostic_tag,
                    linguistic_style=linguistic_style,
                    source=source,
                )
            )

        logger.info(f"Added {len(self.examples)} examples after conversation ingestion")
        return self.examples

    def stratified_split(
        self, train_ratio: float = 0.9
    ) -> tuple[list[MentalHealthInstruction], list[MentalHealthInstruction]]:
        """Split dataset while preserving task-type distribution."""
        by_task: dict[str, list[MentalHealthInstruction]] = {}
        for ex in self.examples:
            by_task.setdefault(ex.task_type, []).append(ex)

        train: list[MentalHealthInstruction] = []
        val: list[MentalHealthInstruction] = []
        for task_examples in by_task.values():
            split_idx = int(len(task_examples) * train_ratio)
            train.extend(task_examples[:split_idx])
            val.extend(task_examples[split_idx:])

        random.shuffle(train)
        random.shuffle(val)
        return train, val

    def save(self, output_dir: str | Path, format: str = "alpaca") -> tuple[Path, Path]:
        """Save train/val splits to disk in the requested format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train, val = self.stratified_split()

        if format == "alpaca":
            train_path = output_dir / "train_alpaca.json"
            val_path = output_dir / "val_alpaca.json"
            _write_jsonl(train_path, [ex.to_alpaca() for ex in train])
            _write_jsonl(val_path, [ex.to_alpaca() for ex in val])
        elif format == "chat":
            system_prompt = (
                "You are a mental health assistant trained to provide supportive, evidence-based responses. "
                "You are not a substitute for professional care."
            )
            train_path = output_dir / "train_chat.json"
            val_path = output_dir / "val_chat.json"
            _write_jsonl(train_path, [ex.to_chat(system_prompt) for ex in train])
            _write_jsonl(val_path, [ex.to_chat(system_prompt) for ex in val])
        elif format == "raw":
            train_path = output_dir / "train_raw.json"
            val_path = output_dir / "val_raw.json"
            _write_jsonl(train_path, [ex.to_dict() for ex in train])
            _write_jsonl(val_path, [ex.to_dict() for ex in val])
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved {len(train)} train and {len(val)} validation examples to {output_dir}")
        return train_path, val_path


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_default_dataset(
    output_dir: str | Path = "./ai/data/mental_health_ift", min_examples: int = 10000
) -> tuple[Path, Path]:
    """Build a default mental health IFT dataset, augmenting seed data to reach min_examples."""
    builder = MentalHealthInstructionDatasetBuilder()
    builder.build_from_seed_vignettes(augment_per_vignette=max(1, min_examples // (len(SEED_VIGNETTES) * 5)))

    # If still below threshold, repeat examples with varied instructions.
    while len(builder.examples) < min_examples:
        for ex in list(builder.examples):
            if len(builder.examples) >= min_examples:
                break
            templates = INSTRUCTION_TEMPLATES.get(MentalHealthTaskType(ex.task_type), [ex.instruction])
            new_instruction = random.choice([t for t in templates if t != ex.instruction] or templates)
            builder.examples.append(
                MentalHealthInstruction(
                    id=str(uuid.uuid4()),
                    task_type=ex.task_type,
                    instruction=new_instruction,
                    input=ex.input,
                    output=ex.output,
                    demographic_tags=ex.demographic_tags,
                    diagnostic_tag=ex.diagnostic_tag,
                    linguistic_style=ex.linguistic_style,
                    source=f"{ex.source}_augmented",
                )
            )

    return builder.save(output_dir, format="alpaca")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_path, val_path = build_default_dataset()
    print(f"Train: {train_path}")
    print(f"Val: {val_path}")
