import logging
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import sys

# Adjust import for project structure
try:
    from ai.common.llm_client import LLMClient
except ImportError:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(project_root))
    from ai.common.llm_client import LLMClient

logger = logging.getLogger(__name__)

class CrisisScenarioExpander:
    """
    Generates high-intensity, "Nightmare Fuel" edge cases for Stage 3 training.
    Focuses on: Addiction, CPTSD, Active Suicidality, Psychosis, Severe Dissociation.
    """

    CATEGORIES = [
        # Suicide/Self-Harm
        "Active Suicidal Ideation with Plan",
        "Recent Suicide Attempt (Post-Emergency)",
        "Self-Harm Urges (Cutting)",

        # Addiction
        "Opioid Overdose in progress",
        "Alcohol Withdrawal Seizure",
        "Relapse after 5 years sober",

        # Trauma/Dissociation
        "Severe Dissociative Identity Disorder switch",
        "CPTSD Flashback (Violent)",
        "Childhood Sexual Abuse Disclosure",

        # Psychosis
        "Paranoid Psychosis (Command Hallucinations)",
        "First Psychotic Break",

        # Mood Disorders
        "Manic Episode with Risk Taking",
        "Severe Depressive Episode (Catatonic)",

        # Personality Disorders
        "Borderline Splitting Episode",
        "Narcissistic Rage Escalation",

        # Other
        "Domestic Violence (Active)",
        "Eating Disorder Medical Emergency",
        "Grief (Child Death)"
    ]

    def __init__(self):
        self.llm = LLMClient(driver="mock")
        self.output_path = Path("ai/training_ready/datasets/stage3_edge_crisis")
        self.output_path.mkdir(parents=True, exist_ok=True)

    def generate_scenario(self, category: str) -> Dict[str, Any]:
        """
        Generates a single high-intensity scenario.
        """
        prompt = f"""
        Generate a realistic, high-intensity crisis dialogue between a person in crisis and a crisis counselor.
        Situation: {category}
        The user should be extremely distressed, potentially incoherent or hostile.
        The counselor should remain calm but urgent.
        Length: 10 turns.
        """
        # In real usage, this would hit OpenAI/Anthropic
        # For mock, we simulate:
        content = self.llm.generate(prompt)

        return {
            "id": f"crisis_{random.randint(1000, 9999)}",
            "category": category,
            "transcript": content,
            "intensity": "severe",
            "tags": ["nightmare_fuel", "stress_test", category.lower().replace(" ", "_")]
        }

    def generate_batch(self, count: int = 10) -> List[Dict]:
        """Generates a batch of crisis scenarios."""
        scenarios = []
        for i in range(count):
            category = random.choice(self.CATEGORIES)
            scenario = self.generate_scenario(category)
            scenarios.append(scenario)

        # Export
        output_file = self.output_path / f"nightmare_scenarios_batch_{random.randint(100,999)}.jsonl"
        with open(output_file, "w") as f:
            for s in scenarios:
                f.write(json.dumps(s) + "\n")

        logger.info(f"Generated {len(scenarios)} nightmare scenarios to {output_file}")
        return scenarios, str(output_file)
