import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
try:
    from ai.common.llm_client import LLMClient
except ImportError:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(project_root))
    from ai.common.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CBTIntegration:
    """
    Generates structured Cognitive Behavioral Therapy (CBT) content.
    Includes Thought Records, Behavioral Activation, and Cognitive Restructuring templates.
    Task 3.1 in Mental Health Datasets Expansion.
    """

    def __init__(self, output_base_path: str = "ai/training_ready/datasets"):
        self.output_base_path = Path(output_base_path)
        self.cbt_content_path = self.output_base_path / "stage2_reasoning" / "cbt_content"
        self._ensure_directories()

        # Initialize GenAI Client
        self.llm = LLMClient(driver="mock")  # Use mock by default for safety

        # Load internal knowledge/templates (simulated for now)
        self.cognitive_distortions = [
            "All-or-Nothing Thinking",
            "Overgeneralization",
            "Mental Filter",
            "Disqualifying the Positive",
            "Jumping to Conclusions",
            "Magnification (Catastrophizing)",
            "Emotional Reasoning",
            "Should Statements",
            "Labeling and Mislabeling",
            "Personalization",
        ]

        logger.info(f"Initialized CBTIntegration. Output path: {self.cbt_content_path}")

    def _ensure_directories(self):
        """Ensure output directories exist."""
        self.cbt_content_path.mkdir(parents=True, exist_ok=True)

    def generate_thought_record(
        self,
        situation: str,
        emotion: str,
        intensity: int,
        negative_thought: str,
        distortion: str | None = None,
    ) -> dict:
        """
        Creates a structured CBT Thought Record.
        """
        if not distortion:
            distortion = random.choice(self.cognitive_distortions)

        return {
            "type": "CBT_Thought_Record",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "situation": situation,
                "emotion": {"label": emotion, "intensity_percent": intensity},
                "automatic_thought": negative_thought,
                "cognitive_distortion": distortion,
                "balanced_thought": {
                    "content": "[Patient to genrate balanced perspective]",
                    "evidence_for": "[List evidence supporting thought]",
                    "evidence_against": "[List evidence contradicting thought]",
                },
                "outcome_emotion": {"label": emotion, "intensity_percent": "[Re-rate intensity]"},
            },
            "validation_status": "template_verified",
        }

    def generate_behavioral_activation(
        self, activity: str, difficulty: int, mastery: int, pleasure: int
    ) -> dict:
        """
        Creates a Behavioral Activation log entry.
        """
        return {
            "type": "CBT_Behavioral_Activation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "activity": activity,
                "metrics": {
                    "difficulty": difficulty,  # 0-10
                    "mastery": mastery,  # 0-10
                    "pleasure": pleasure,  # 0-10
                },
                "notes": "What did you learn from this activity?",
            },
            "validation_status": "template_verified",
        }

    def generate_cognitive_restructuring(self, negative_thought: str) -> dict:
        """
        Creates a Cognitive Restructuring worksheet template.
        """
        return {
            "type": "CBT_Cognitive_Restructuring",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "target_thought": negative_thought,
                "socratic_questions": [
                    "What is the evidence for this thought?",
                    "What is the evidence against this thought?",
                    "Am I basing this on facts or feelings?",
                    "What is the worst that could happen? Could I survive it?",
                    "What would I tell a friend in this situation?",
                ],
                "alternative_perspective": "[Synthesized rational response]",
            },
            "validation_status": "template_verified",
        }

    def get_cbt_session_template(self, phase: str = "middle") -> dict:
        """
        Returns a session structure template for Initial, Middle, or Termination phase.
        """
        structure = []
        if phase.lower() == "initial":
            structure = [
                "Mood Check & Assessment",
                "Set Agenda",
                "Review Presenting Problems",
                "Education about CBT Model",
                "Goal Setting",
                "Assign Homework",
                "Feedback",
            ]
        elif phase.lower() == "termination":
            structure = [
                "Mood Check",
                "Review Progress & Goals",
                "Relapse Prevention Planning",
                "Discuss Future Challenges",
                "Feedback & Closing",
            ]
        else:  # Middle
            structure = [
                "Mood Check",
                "Review Homework",
                "Set Agenda",
                "Work on Specific Items (Techniques)",
                "Summarize & Assign New Homework",
                "Feedback",
            ]

        return {
            "type": "CBT_Session_Template",
            "phase": phase,
            "structure": structure,
            "validation_status": "clinical_standard",
        }

    def generate_batch_content(self, count: int = 10) -> list[dict]:
        """
        Generates a batch of mixed CBT content for datasets.
        """
        logger.info(f"Generating {count} CBT simulated records via LLM...")
        batch = []

        for i in range(count):
            item_type = i % 4

            # Generate dynamic context via LLM
            # We construct a prompt to get a random clinical scenario
            context_prompt = "Generate a short JSON scenario with keys: situation, emotion, intensity (1-100), negative_thought, distortion."
            context = self.llm.generate_structured(context_prompt, schema={})

            # Use defaults if mock/failure
            if not context or "situation" not in context:
                context = {
                    "situation": "Presentation at work",
                    "emotion": "Anxiety",
                    "intensity": 85,
                    "negative_thought": "I will fail",
                    "distortion": "Fortune Telling",
                }

            if item_type == 0:
                item = self.generate_thought_record(
                    context["situation"],
                    context["emotion"],
                    context["intensity"],
                    context["negative_thought"],
                    context.get("distortion"),
                )
            elif item_type == 1:
                item = self.generate_behavioral_activation("Morning Walk", 3, 5, 7)
            elif item_type == 2:
                item = self.generate_cognitive_restructuring(context["negative_thought"])
            else:
                item = self.get_cbt_session_template(
                    phase=random.choice(["initial", "middle", "termination"])
                )

            batch.append(item)

        return batch

    def export_data(self, data: list[dict], batch_id: str = "001") -> str:
        """Exports the generated data to the stage directory."""
        output_file = self.cbt_content_path / f"cbt_batch_{batch_id}.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Successfully exported {len(data)} items to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise
