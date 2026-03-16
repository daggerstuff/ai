import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ResearchInstrumentCollector:
    """
    Collector for validated mental health assessment tools and screening instruments.
    Task 1.2 in Mental Health Datasets Expansion.
    """

    def __init__(self, output_base_path: str = "ai/training/ready_packages/datasets"):
        self.output_base_path = Path(output_base_path)
        self.clinical_instruments_path = self.output_base_path / "stage2_reasoning" / "clinical_instruments"
        self._ensure_directories()

        logger.info(f"Initialized ResearchInstrumentCollector. Output path: {self.clinical_instruments_path}")

    def _ensure_directories(self):
        """Ensure output directories exist."""
        self.clinical_instruments_path.mkdir(parents=True, exist_ok=True)

    def collect_instruments(self) -> list[dict]:
        """
        Collects validated research instruments.
        Currently uses an expanded internal database.
        """
        logger.info("Collecting research instruments...")

        # Expanded internal database of common clinical instruments
        # Note: 'public_domain' is an assumption for common tools, usage in commercial apps requires specific verification.
        return [
            {
                "name": "PHQ-9",
                "full_name": "Patient Health Questionnaire-9",
                "type": "Depression Screening",
                "validation_status": "gold_standard",
                "compliance_check": "public_domain",
                "items": [
                    "Little interest or pleasure in doing things",
                    "Feeling down, depressed, or hopeless"
                ]
            },
            {
                "name": "GAD-7",
                "full_name": "Generalized Anxiety Disorder-7",
                "type": "Anxiety Screening",
                "validation_status": "gold_standard",
                "compliance_check": "public_domain",
                "items": [
                    "Feeling nervous, anxious or on edge",
                    "Not being able to stop or control worrying"
                ]
            },
            {
                "name": "BDI-II",
                "full_name": "Beck Depression Inventory-II",
                "type": "Depression Severity",
                "validation_status": "gold_standard",
                "compliance_check": "licensed", # Usually requires license
                "items": []
            },
            {
                "name": "PCL-5",
                "full_name": "PTSD Checklist for DSM-5",
                "type": "PTSD Screening",
                "validation_status": "gold_standard",
                "compliance_check": "public_domain",
                "items": ["Repeated, disturbing, and unwanted memories of the stressful experience?"]
            },
            {
                "name": "AUDIT",
                "full_name": "Alcohol Use Disorders Identification Test",
                "type": "Substance Use",
                "validation_status": "gold_standard",
                "compliance_check": "public_domain",
                "items": ["How often do you have a drink containing alcohol?"]
            },
            {
                "name": "ISI",
                "full_name": "Insomnia Severity Index",
                "type": "Sleep Disorder",
                "validation_status": "gold_standard",
                "compliance_check": "licensed", # Often requires permission for commercial use
                "items": ["Difficulty falling asleep"]
            },
            {
                "name": "K-10",
                "full_name": "Kessler Psychological Distress Scale",
                "type": "Global Distress",
                "validation_status": "gold_standard",
                "compliance_check": "public_domain",
                "items": ["In the past 30 days, how often did you feel tired out for no good reason?"]
            },
            {
                "name": "MDQ",
                "full_name": "Mood Disorder Questionnaire",
                "type": "Bipolar Screening",
                "validation_status": "gold_standard",
                "compliance_check": "public_domain",
                "items": ["Has there ever been a period of time when you were not your usual self and..."]
            },
             {
                "name": "EPDS",
                "full_name": "Edinburgh Postnatal Depression Scale",
                "type": "Postnatal Depression",
                "validation_status": "gold_standard",
                "compliance_check": "public_domain",
                "items": ["I have been able to laugh and see the funny side of things"]
            }
        ]


    def validate_compliance(self, instruments: list[dict]) -> list[dict]:
        """
        Validates copyright and usage rights.
        """
        validated = []
        for inst in instruments:
            # Placeholder logic for validation
            if inst.get("compliance_check") in ["public_domain", "licensed"]:
                validated.append(inst)
            else:
                logger.warning(f"Skipping instrument {inst.get('name')} due to compliance check failure.")
        return validated

    def export_data(self, data: list[dict]):
        """Exports the collected instruments to the stage directory."""
        output_file = self.clinical_instruments_path / "validated_instruments_batch_001.json"

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Successfully exported {len(data)} instruments to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise

    def run_collection_pipeline(self):
        """Main execution method."""
        logger.info("Starting Research Instrument Collection Pipeline...")
        raw_data = self.collect_instruments()
        validated_data = self.validate_compliance(raw_data)
        output_path = self.export_data(validated_data)
        logger.info("Research Instrument Collection Pipeline Completed.")
        return output_path
