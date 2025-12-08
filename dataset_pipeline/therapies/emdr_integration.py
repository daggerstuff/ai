import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class EMDRIntegration:
    """
    Generates structured EMDR (Eye Movement Desensitization and Reprocessing) content.
    Includes Standard 8-Phase Protocol, Treatment Planning, and Bilateral Stimulation logs.
    Task 3.3 in Mental Health Datasets Expansion.
    """

    def __init__(self, output_base_path: str = "ai/training_ready/datasets"):
        self.output_base_path = Path(output_base_path)
        self.emdr_content_path = self.output_base_path / "stage2_reasoning" / "emdr_content"
        self._ensure_directories()

        logger.info(f"Initialized EMDRIntegration. Output path: {self.emdr_content_path}")

    def _ensure_directories(self):
        """Ensure output directories exist."""
        self.emdr_content_path.mkdir(parents=True, exist_ok=True)

    def generate_phase_protocol(self, phase_number: int) -> dict:
        """
        Returns the standard script/steps for a specific EMDR phase.
        """
        phases = {
            1: {
                "name": "History Taking",
                "focus": "Identify target memories and readiness.",
                "steps": [
                    "Gather history",
                    "Identify possible targets",
                    "Assess internal/external safety/resources",
                ],
            },
            2: {
                "name": "Preparation",
                "focus": "Establish therapeutic relationship and teach resourcing.",
                "steps": [
                    "Explain EMDR theory",
                    "Teach Safe/Calm Place",
                    "Install resources (BLS)",
                ],
            },
            3: {
                "name": "Assessment",
                "focus": "Activate target memory components.",
                "steps": [
                    "Select Target Image",
                    "Identify Negative Cognition (NC)",
                    "Identify Positive Cognition (PC)",
                    "Rate VOC (1-7)",
                    "Identify Emotions/Sensations",
                    "Rate SUDs (0-10)",
                ],
            },
            4: {
                "name": "Desensitization",
                "focus": "Process memory to SUDs 0.",
                "steps": [
                    "Focus on target",
                    "Apply BLS (sets of 24+)",
                    "Check in ('What do you notice?')",
                    "Continue until channels clear",
                ],
            },
            5: {
                "name": "Installation",
                "focus": "Link Positive Cognition to target.",
                "steps": [
                    "Focus on Target + PC",
                    "Rate VOC",
                    "Apply BLS to strengthen",
                    "Check for blocking beliefs",
                ],
            },
            6: {
                "name": "Body Scan",
                "focus": "Clear residual somatic sensation.",
                "steps": ["Mental scan of body", "Process any tension with BLS"],
            },
            7: {
                "name": "Closure",
                "focus": "Ensure patient leaves stable.",
                "steps": [
                    "Debrief",
                    "Containment exercise if incomplete",
                    "Journaling instructions",
                ],
            },
            8: {
                "name": "Reevaluation",
                "focus": "Check progress at start of next session.",
                "steps": [
                    "Re-access target",
                    "Check SUDs/VOC",
                    "Resume processing or move to new target",
                ],
            },
        }

        phase_info = phases.get(phase_number, {})

        return {
            "type": "EMDR_Phase_Protocol",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "phase": phase_number,
                "name": phase_info.get("name"),
                "focus": phase_info.get("focus"),
                "steps": phase_info.get("steps", []),
            },
            "validation_status": "clinical_standard",
        }

    def generate_treatment_plan(
        self, target_memory: str, negative_cognition: str, positive_cognition: str
    ) -> dict:
        """
        Creates a structured EMDR treatment plan (Targeting Sequence Plan).
        """
        return {
            "type": "EMDR_Treatment_Plan",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "presenting_issue": "Anxiety/Trauma",
                "target_memory": target_memory,
                "negative_cognition_nc": negative_cognition,  # "I am powerless"
                "positive_cognition_pc": positive_cognition,  # "I am in control now"
                "past_events": ["[List touchstone events]"],
                "present_triggers": ["[List current triggers]"],
                "future_template": ["[Desired future behavior]"],
            },
            "validation_status": "template_verified",
        }

    def generate_resourcing_script(self, technique: str = "Safe Place") -> dict:
        """
        Generates instructions for a specific resourcing technique.
        """
        scripts = {
            "Safe Place": "Identify a place (real or imagined) where you feel calm and safe. Focus on the sensory details...",
            "Container": "Visualize a secure container where you can put disturbing thoughts/feelings...",
            "Light Stream": "Visualize a healing light entering the top of your head...",
        }

        return {
            "type": "EMDR_Resource_Script",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "technique": technique,
                "script_summary": scripts.get(technique, "Follow standard protocol."),
                "bls_instruction": "Slow, short sets (4-6 taps) to install positive resource.",
            },
            "validation_status": "template_verified",
        }

    def generate_bilateral_stimulation_log(self) -> dict:
        """
        A template for logging a processing session.
        """
        return {
            "type": "EMDR_BLS_Log",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "target": "Index Trauma A",
                "starting_suds": 8,
                "sets": [
                    {"set_num": 1, "duration": "30 sec", "client_report": "Image became sharper"},
                    {"set_num": 2, "duration": "45 sec", "client_report": "Felt sadness in chest"},
                    {
                        "set_num": 3,
                        "duration": "45 sec",
                        "client_report": "Sadness moving/changing",
                    },
                ],
                "ending_suds": 4,
                "incomplete_session": True,
                "closure_technique": "Container",
            },
            "validation_status": "simulation_example",
        }

    def generate_batch_content(self, count: int = 10) -> list[dict]:
        """
        Generates a batch of mixed EMDR content.
        """
        logger.info(f"Generating {count} EMDR simulated records...")
        batch = []

        for i in range(count):
            item_type = i % 4
            if item_type == 0:
                item = self.generate_phase_protocol(phase_number=random.randint(1, 8))
            elif item_type == 1:
                item = self.generate_treatment_plan("Car Accident", "I am unsafe", "I am safe now")
            elif item_type == 2:
                item = self.generate_resourcing_script(random.choice(["Safe Place", "Container"]))
            else:
                item = self.generate_bilateral_stimulation_log()

            batch.append(item)

        return batch

    def export_data(self, data: list[dict], batch_id: str = "001") -> str:
        """Exports the generated data."""
        output_file = self.emdr_content_path / f"emdr_batch_{batch_id}.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Successfully exported {len(data)} items to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise
