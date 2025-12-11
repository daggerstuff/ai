import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CrisisScenarioExpander:
    """
    Expander for nightmare fuel crisis scenarios.
    Task 1.5 in Mental Health Datasets Expansion.
    """

    def __init__(self, output_base_path: str = "ai/training_ready/datasets"):
        self.output_base_path = Path(output_base_path)
        self.crisis_scenarios_path = self.output_base_path / "stage3_edge" / "crisis_scenarios"
        self._ensure_directories()

        logger.info(f"Initialized CrisisScenarioExpander. Output path: {self.crisis_scenarios_path}")

    def _ensure_directories(self):
        """Ensure output directories exist."""
        self.crisis_scenarios_path.mkdir(parents=True, exist_ok=True)

    def define_crisis_categories(self) -> list[str]:
        """
        Returns the expanded list of 50+ crisis categories.
        """
        # Subset of the full 50+ list for simulation
        return [
            # High Risk
            "Suicidality: Active Ideation with Plan",
            "Suicidality: Passive Ideation",
            "Homicidal Ideation: Targeted",
            "Psychosis: Command Hallucinations",
            "Psychosis: Paranoid Delusions",
            # Trauma
            "Domestic Violence: Acute Crisis",
            "Sexual Assault: Recent Disclosure",
            "Child Abuse: Mandated Reporting Trigger",
            # Substance
            "Opioid Overdose Risk",
            "Alcohol Withdrawal: Delirium Tremens Risk",
            # Personality
            "BPD: Severe Dissociation",
            "BPD: Fear of Abandonment/Rage",
            # Eating Disorders
            "Anorexia: Medical Instability",
            # Other
            "Manic Episode: High Risk Behavior",
            "Severe Panic Attack: ER Presentation"
        ]

    def generate_scenarios(self, categories: list[str]) -> list[dict]:
        """
        Simulates generation of scenarios for each category.
        In production, this might use LLMs or templates.
        """
        logger.info(f"Generating scenarios for {len(categories)} categories (Simulation)...")

        scenarios = []
        for cat in categories:
            scenarios.append({
                "category": cat,
                "severity": "High" if "Risk" in cat or "Active" in cat else "Moderate",
                "scenario_id": f"CRISIS-{abs(hash(cat)) % 10000}",
                "context": f"Simulated context for {cat}...",
                "safety_protocol_required": True,
                "guard_rails": "OFF" # As per spec requirements for nightmare fuel
            })

        return scenarios

    def export_data(self, data: list[dict]):
        """Exports the crisis scenarios."""
        output_file = self.crisis_scenarios_path / "crisis_expansion_batch_001.json"

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Successfully exported {len(data)} crisis scenarios to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise

    def run_expansion_pipeline(self):
        """Main execution method."""
        logger.info("Starting Crisis Scenario Expansion Pipeline...")
        categories = self.define_crisis_categories()
        scenarios = self.generate_scenarios(categories)
        output_path = self.export_data(scenarios)
        logger.info("Crisis Scenario Expansion Pipeline Completed.")
        return output_path
