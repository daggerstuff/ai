"""
Data Augmentation Pipeline for Pixel LLM Training
Implements augmentation strategies for mental health conversations:
- Context expansion
- Synthetic crisis scenario generation
- Therapeutic dialogue variations
- Semantic paraphrasing
- Demographic diversity injection
- Safety-aware augmentation
"""

import json
import logging
import random
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation"""

    augmentation_probability: float = 0.3  # Probability of augmenting each record
    context_expansion_enabled: bool = True
    crisis_scenario_generation_enabled: bool = True
    dialogue_variation_enabled: bool = True
    semantic_paraphrase_enabled: bool = True
    demographic_diversity_enabled: bool = True
    preserve_crisis_keywords: bool = True  # Never modify crisis-related keywords
    seed: int = 42
    max_augmentations_per_record: int = 3  # Max augmented versions per record


class ContextExpander:
    """Expands therapeutic context in conversations"""

    THERAPEUTIC_CONTEXTS = [
        "This is a therapy session focused on cognitive behavioral therapy (CBT) techniques.",
        "This conversation demonstrates dialectical behavior therapy (DBT) principles.",
        "This session uses acceptance and commitment therapy (ACT) approaches.",
        "This dialogue illustrates motivational interviewing techniques.",
        "This conversation demonstrates psychodynamic therapy principles.",
        "This session uses solution-focused brief therapy (SFBT) techniques.",
    ]

    EMOTIONAL_CONTEXTS = [
        "The client is experiencing significant emotional distress.",
        "The client is in a vulnerable emotional state.",
        "The client is processing difficult emotions.",
        "The client is working through trauma-related responses.",
        "The client is managing anxiety and worry.",
        "The client is addressing depressive symptoms.",
    ]

    THERAPEUTIC_GOALS = [
        "The goal is to build coping skills.",
        "The goal is to increase emotional awareness.",
        "The goal is to develop healthy boundaries.",
        "The goal is to process past experiences.",
        "The goal is to reduce symptom severity.",
        "The goal is to improve emotional regulation.",
    ]

    def expand(self, record: dict) -> dict:
        """Expand context in a record"""
        expanded = record.copy()

        # Add therapeutic context if not present
        if "therapeutic_context" not in expanded:
            expanded["therapeutic_context"] = random.choice(self.THERAPEUTIC_CONTEXTS)

        # Add emotional context
        if "emotional_context" not in expanded:
            expanded["emotional_context"] = random.choice(self.EMOTIONAL_CONTEXTS)

        # Add therapeutic goal
        if "therapeutic_goal" not in expanded:
            expanded["therapeutic_goal"] = random.choice(self.THERAPEUTIC_GOALS)

        # Enhance instructions with context
        if "instructions" in expanded:
            original_instructions = expanded["instructions"]
            context_addition = (
                f"\n\nTherapeutic Context: {expanded['therapeutic_context']} "
                f"Emotional Context: {expanded['emotional_context']} "
                f"Goal: {expanded['therapeutic_goal']}"
            )
            expanded["instructions"] = original_instructions + context_addition

        return expanded


class CrisisScenarioGenerator:
    """Generates synthetic crisis scenarios for training"""

    CRISIS_TYPES = [
        "suicidality",
        "self_harm",
        "acute_anxiety",
        "severe_depression",
        "psychotic_symptoms",
        "substance_abuse_crisis",
        "domestic_violence",
        "trauma_flashback",
    ]

    DIFFICULTY_LEVELS = ["mild", "moderate", "severe", "extreme"]

    CRISIS_INDICATORS = {
        "suicidality": [
            "active suicidal ideation",
            "detailed suicide plan",
            "access to means",
            "recent suicide attempt",
        ],
        "self_harm": [
            "urges to self-harm",
            "recent self-injury",
            "escalating self-harm behavior",
            "self-harm as coping mechanism",
        ],
        "acute_anxiety": [
            "panic attack symptoms",
            "severe anxiety escalation",
            "dissociative symptoms",
            "physical panic symptoms",
        ],
        "severe_depression": [
            "hopelessness",
            "anhedonia",
            "suicidal ideation",
            "inability to function",
        ],
    }

    def generate(self, base_record: dict) -> dict:
        """Generate a synthetic crisis scenario"""
        crisis_record = base_record.copy()

        # Select crisis type
        crisis_type = random.choice(self.CRISIS_TYPES)
        difficulty = random.choice(self.DIFFICULTY_LEVELS)

        crisis_record["prompt_id"] = f"{crisis_type}_{random.randint(1000, 9999)}"
        crisis_record["category"] = "crisis_scenarios"
        crisis_record["scenario_type"] = crisis_type
        crisis_record["difficulty"] = difficulty

        # Add crisis-specific instructions
        crisis_instructions = self._generate_crisis_instructions(crisis_type, difficulty)
        if "instructions" in crisis_record:
            crisis_record["instructions"] = (
                crisis_record["instructions"] + "\n\n" + crisis_instructions
            )
        else:
            crisis_record["instructions"] = crisis_instructions

        # Add metadata
        if "metadata" not in crisis_record:
            crisis_record["metadata"] = {}

        crisis_record["metadata"]["crisis_type"] = crisis_type
        crisis_record["metadata"]["difficulty_level"] = difficulty
        crisis_record["metadata"]["generated_by"] = "crisis_scenario_generator"

        return crisis_record

    def _generate_crisis_instructions(self, crisis_type: str, difficulty: str) -> str:
        """Generate crisis-specific instructions"""
        base_instruction = (
            f"This is a {difficulty} difficulty crisis scenario involving {crisis_type}. "
            f"The therapist must demonstrate appropriate crisis intervention techniques. "
        )

        if crisis_type in self.CRISIS_INDICATORS:
            indicators = random.sample(self.CRISIS_INDICATORS[crisis_type], k=2)
            base_instruction += f"Key indicators: {', '.join(indicators)}. "

        if difficulty == "extreme":
            base_instruction += (
                "This is an extremely challenging scenario. "
                "The outcome may be tragic regardless of intervention. "
                "Focus on demonstrating appropriate risk assessment and safety planning."
            )
        elif difficulty == "severe":
            base_instruction += (
                "This is a severe scenario requiring immediate intervention. "
                "Demonstrate crisis de-escalation and safety planning."
            )

        return base_instruction


class DialogueVariationGenerator:
    """Generates variations of therapeutic dialogues"""

    DIALOGUE_VARIATIONS = {
        "empathic_response": [
            "I hear you. That sounds really difficult.",
            "I can see how that would be challenging for you.",
            "That must be really hard to experience.",
            "I understand this is painful for you.",
        ],
        "validation": [
            "Your feelings are completely valid.",
            "It makes sense that you feel this way.",
            "That's a natural response to what you're experiencing.",
            "Your emotions are understandable given the circumstances.",
        ],
        "exploration": [
            "Can you tell me more about that?",
            "What does that feel like for you?",
            "How long have you been experiencing this?",
            "When did you first notice this?",
        ],
        "coping_strategy": [
            "What coping strategies have helped you in the past?",
            "Have you tried any techniques that provided relief?",
            "What do you think might help right now?",
            "Let's explore some coping options together.",
        ],
    }

    def generate_variation(self, record: dict) -> dict:
        """Generate a dialogue variation of a record"""
        variation = record.copy()

        # Add dialogue variation metadata
        if "dialogue_variations" not in variation:
            variation["dialogue_variations"] = {}

        for variation_type, options in self.DIALOGUE_VARIATIONS.items():
            variation["dialogue_variations"][variation_type] = random.choice(options)

        # Add to instructions
        if "instructions" in variation:
            variation_note = (
                "\n\nDialogue Variation: Incorporate the following therapeutic responses: "
                + ", ".join(variation["dialogue_variations"].values())
            )
            variation["instructions"] = variation["instructions"] + variation_note

        return variation


class SemanticParaphraser:
    """Generates semantic paraphrases while preserving meaning and safety"""

    PARAPHRASE_PATTERNS = {
        "suicidal_ideation": [
            ("suicidal ideation", "thoughts of ending one's life"),
            ("suicidal thoughts", "persistent thoughts about suicide"),
            ("wants to die", "expresses desire to end life"),
            ("kill myself", "harm myself fatally"),
        ],
        "self_harm": [
            ("self-harm", "self-injurious behavior"),
            ("cutting", "self-injury through cutting"),
            ("self-injury", "deliberate self-harm"),
        ],
        "depression": [
            ("depressed", "experiencing depressive symptoms"),
            ("depression", "major depressive disorder"),
            ("sad", "experiencing sadness and low mood"),
            ("hopeless", "experiencing hopelessness"),
        ],
        "anxiety": [
            ("anxious", "experiencing anxiety symptoms"),
            ("panic attack", "acute anxiety episode"),
            ("worried", "experiencing worry and concern"),
        ],
    }

    THERAPEUTIC_PARAPHRASES = [
        ("I understand", "I can see"),
        ("That sounds difficult", "That seems challenging"),
        ("Tell me more", "Can you elaborate"),
        ("How does that make you feel", "What's your experience with that"),
        ("Let's explore", "Let's examine"),
    ]

    def paraphrase_text(self, text: str, preserve_crisis_keywords: bool = True) -> str:
        """Paraphrase text while preserving critical safety information"""
        paraphrased = text

        # Apply therapeutic paraphrases
        for original, replacement in self.THERAPEUTIC_PARAPHRASES:
            if random.random() < 0.4:  # 40% chance to apply
                paraphrased = re.sub(
                    rf"\b{re.escape(original)}\b",
                    replacement,
                    paraphrased,
                    flags=re.IGNORECASE,
                    count=1,
                )

        # Apply domain-specific paraphrases (with safety checks)
        if not preserve_crisis_keywords:
            for category, patterns in self.PARAPHRASE_PATTERNS.items():
                for original, replacement in patterns:
                    if random.random() < 0.3:  # 30% chance
                        paraphrased = re.sub(
                            rf"\b{re.escape(original)}\b",
                            replacement,
                            paraphrased,
                            flags=re.IGNORECASE,
                            count=1,
                        )

        return paraphrased


class DemographicDiversityInjector:
    """Injects demographic diversity into scenarios"""

    DEMOGRAPHIC_VARIATIONS = {
        "age_groups": [
            {"label": "adolescent", "descriptors": ["teen", "young", "school-age"]},
            {"label": "young_adult", "descriptors": ["college-age", "early career"]},
            {"label": "middle_aged", "descriptors": ["mid-career", "established"]},
            {"label": "older_adult", "descriptors": ["senior", "retired", "elderly"]},
        ],
        "cultural_contexts": [
            "from an urban background",
            "from a rural community",
            "from a multicultural family",
            "from an immigrant family",
            "from a first-generation background",
        ],
        "socioeconomic": [
            "experiencing financial stress",
            "with stable financial resources",
            "navigating economic uncertainty",
            "with limited access to resources",
        ],
    }

    def inject_diversity(self, record: dict) -> dict:
        """Inject demographic diversity markers"""
        diverse = record.copy()

        if "metadata" not in diverse:
            diverse["metadata"] = {}

        # Add demographic context
        diverse["metadata"]["demographic_context"] = {
            "age_group": random.choice(self.DEMOGRAPHIC_VARIATIONS["age_groups"])["label"],
            "cultural_context": random.choice(self.DEMOGRAPHIC_VARIATIONS["cultural_contexts"]),
            "socioeconomic": random.choice(self.DEMOGRAPHIC_VARIATIONS["socioeconomic"]),
        }

        # Enhance instructions with diversity context
        if "instructions" in diverse:
            diversity_note = (
                f"\n\nDemographic Context: Client is {diverse['metadata']['demographic_context']['age_group']}, "
                f"{diverse['metadata']['demographic_context']['cultural_context']}, "
                f"and {diverse['metadata']['demographic_context']['socioeconomic']}."
            )
            diverse["instructions"] = diverse["instructions"] + diversity_note

        return diverse


class DataAugmentationPipeline:
    """Main data augmentation pipeline"""

    def __init__(self, config: AugmentationConfig = None):
        self.config = config or AugmentationConfig()
        random.seed(self.config.seed)

        self.context_expander = ContextExpander()
        self.crisis_generator = CrisisScenarioGenerator()
        self.dialogue_generator = DialogueVariationGenerator()
        self.semantic_paraphraser = SemanticParaphraser()
        self.demographic_injector = DemographicDiversityInjector()

        logger.info("Initialized DataAugmentationPipeline")
        logger.info(f"  Context expansion: {self.config.context_expansion_enabled}")
        logger.info(f"  Crisis scenarios: {self.config.crisis_scenario_generation_enabled}")
        logger.info(f"  Dialogue variations: {self.config.dialogue_variation_enabled}")
        logger.info(f"  Semantic paraphrasing: {self.config.semantic_paraphrase_enabled}")
        logger.info(f"  Demographic diversity: {self.config.demographic_diversity_enabled}")
        logger.info(f"  Preserve crisis keywords: {self.config.preserve_crisis_keywords}")

    def augment_record(self, record: dict) -> list[dict]:
        """Augment a single record, returning original + augmented versions"""
        augmented_records = [record]  # Always include original

        # Apply augmentations based on probability
        if random.random() < self.config.augmentation_probability:
            augmentation_count = 0

            if (
                self.config.context_expansion_enabled
                and augmentation_count < self.config.max_augmentations_per_record
            ):
                expanded = self.context_expander.expand(record)
                augmented_records.append(expanded)
                augmentation_count += 1

            if (
                self.config.crisis_scenario_generation_enabled
                and augmentation_count < self.config.max_augmentations_per_record
            ):
                crisis = self.crisis_generator.generate(record)
                augmented_records.append(crisis)
                augmentation_count += 1

            if (
                self.config.dialogue_variation_enabled
                and augmentation_count < self.config.max_augmentations_per_record
            ):
                variation = self.dialogue_generator.generate_variation(record)
                augmented_records.append(variation)
                augmentation_count += 1

            if (
                self.config.semantic_paraphrase_enabled
                and augmentation_count < self.config.max_augmentations_per_record
            ):
                paraphrased = record.copy()
                if "instructions" in paraphrased:
                    paraphrased["instructions"] = self.semantic_paraphraser.paraphrase_text(
                        paraphrased["instructions"],
                        preserve_crisis_keywords=self.config.preserve_crisis_keywords,
                    )
                    paraphrased["metadata"] = paraphrased.get("metadata", {})
                    paraphrased["metadata"]["augmentation_type"] = "semantic_paraphrase"
                    augmented_records.append(paraphrased)
                    augmentation_count += 1

            if (
                self.config.demographic_diversity_enabled
                and augmentation_count < self.config.max_augmentations_per_record
            ):
                diverse = self.demographic_injector.inject_diversity(record)
                diverse["metadata"] = diverse.get("metadata", {})
                diverse["metadata"]["augmentation_type"] = "demographic_diversity"
                augmented_records.append(diverse)
                augmentation_count += 1

        return augmented_records

    def augment_dataset(self, records: list[dict]) -> list[dict]:
        """Augment entire dataset"""
        augmented = []
        total_original = len(records)

        for i, record in enumerate(records):
            if (i + 1) % 10000 == 0:
                logger.info(f"Augmented {i + 1}/{total_original} records")

            augmented_versions = self.augment_record(record)
            augmented.extend(augmented_versions)

        logger.info(
            f"Augmentation complete: {total_original} original records -> "
            f"{len(augmented)} total records ({len(augmented) - total_original} added)"
        )

        return augmented

    def augment_jsonl_file(
        self, input_path: str, output_path: str, sample_size: int = None
    ) -> dict:
        """Augment records from JSONL file and write to output"""
        records = []
        logger.info(f"Loading records from {input_path}")

        with open(input_path) as f:
            for i, line in enumerate(f):
                if sample_size and i >= sample_size:
                    break
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON at line {i + 1}: {e}")

        logger.info(f"Loaded {len(records)} records")

        # Augment
        augmented_records = self.augment_dataset(records)

        # Write output
        logger.info(f"Writing augmented records to {output_path}")
        with open(output_path, "w") as f:
            for record in augmented_records:
                f.write(json.dumps(record) + "\n")

        stats = {
            "original_records": len(records),
            "augmented_records": len(augmented_records),
            "records_added": len(augmented_records) - len(records),
            "augmentation_ratio": len(augmented_records) / len(records),
        }

        logger.info(f"Augmentation stats: {stats}")
        return stats
