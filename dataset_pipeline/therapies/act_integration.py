import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class ACTIntegration:
    """
    Generates structured Acceptance and Commitment Therapy (ACT) content.
    Includes Hexaflex exercises, Values Clarification, and Defusion protocols.
    Task 3.4 in Mental Health Datasets Expansion.
    """

    def __init__(self, output_base_path: str = "ai/training_ready/datasets"):
        self.output_base_path = Path(output_base_path)
        self.act_content_path = self.output_base_path / "stage2_reasoning" / "act_content"
        self._ensure_directories()

        self.hexaflex_processes = [
            "Acceptance",
            "Cognitive Defusion",
            "Contact with the Present Moment",
            "Self-as-Context",
            "Values",
            "Committed Action",
        ]

        logger.info(f"Initialized ACTIntegration. Output path: {self.act_content_path}")

    def _ensure_directories(self):
        """Ensure output directories exist."""
        self.act_content_path.mkdir(parents=True, exist_ok=True)

    def generate_hexaflex_exercise(self, process: str | None = None) -> dict:
        """
        Generates an exercise for a specific Hexaflex process.
        """
        if not process:
            process = random.choice(self.hexaflex_processes)

        exercises = {
            "Acceptance": "Focus on a difficult emotion. Breathe into it. Don't try to change it.",
            "Cognitive Defusion": "Repeat the difficult thought out loud until it becomes just sound.",
            "Contact with the Present Moment": "5-4-3-2-1 Grounding Exercise.",
            "Self-as-Context": "Observe the 'You' that is noticing your thoughts (Observer Self).",
            "Values": "Imagine your 80th birthday party. What woud you want people to say about you?",
            "Committed Action": "Identify one small step you can take today towards your value of...",
        }

        return {
            "type": "ACT_Hexaflex_Exercise",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "process": process,
                "exercise_name": f"{process} Practice",
                "instructions": exercises.get(process, "Practice awareness."),
                "reflection_prompt": "What showed up for you during this exercise?",
            },
            "validation_status": "template_verified",
        }

    def generate_values_exercise(self, domain: str = "Relationships") -> dict:
        """
        Creates values clarification prompts.
        """
        return {
            "type": "ACT_Values_Clarification",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "domain": domain,
                "prompts": [
                    f"What sort of person do you want to be in your {domain.lower()}?",
                    "If you were acting completely ideal in this area, what would you be doing?",
                    "What personal strengths do you want to bring to this area?",
                ],
                "bullseye_rating": {
                    "question": "How close are you living to your values in this area?",
                    "scale": "0 (Far away) to 10 (Bullseye)",
                },
            },
            "validation_status": "template_verified",
        }

    def generate_defusion_technique(self, target_thought: str) -> dict:
        """
        Generates a specific cognitive defusion script.
        """
        techniques = [
            ("Leaves on a Stream", "Visualize your thoughts as leaves floating down a stream."),
            ("Voice Shifting", "Say the thought in a silly voice (e.g., Donald Duck)."),
            (
                "I'm having the thought that...",
                "Rephrase: 'I'm having the thought that [thought]' instead of just '[thought]'.",
            ),
        ]
        chosen_tech = random.choice(techniques)

        return {
            "type": "ACT_Defusion_Technique",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "target_thought": target_thought,
                "technique": chosen_tech[0],
                "instructions": chosen_tech[1],
                "goal": "To see the thought AS a thought, not as a truth/command.",
            },
            "validation_status": "template_verified",
        }

    def generate_act_matrix(self) -> dict:
        """
        Creates a template for the ACT Matrix.
        """
        return {
            "type": "ACT_Matrix",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "framework": "Toward vs Away, Internal vs External",
                "quadrants": {
                    "Q1_Bottom_Right": {
                        "label": "Who/What is important? (Values)",
                        "content": "[User Input]",
                    },
                    "Q2_Bottom_Left": {
                        "label": "What shows up and gets in the way? (Internal barriers)",
                        "content": "[Fear, Doubts, Feelings]",
                    },
                    "Q3_Top_Left": {
                        "label": "What do you do to move away from internal stuff? (Away Moves)",
                        "content": "[Avoidance behaviors]",
                    },
                    "Q4_Top_Right": {
                        "label": "What could you do to move toward who/what is important? (Toward Moves)",
                        "content": "[Committed Actions]",
                    },
                },
                "noticing_question": "Can you notice the difference between moving toward and moving away?",
            },
            "validation_status": "clinical_standard",
        }

    def generate_batch_content(self, count: int = 10) -> list[dict]:
        """
        Generates a batch of mixed ACT content.
        """
        logger.info(f"Generating {count} ACT simulated records...")
        batch = []

        for i in range(count):
            item_type = i % 4
            if item_type == 0:
                item = self.generate_hexaflex_exercise()
            elif item_type == 1:
                item = self.generate_values_exercise(
                    random.choice(["Relationships", "Work/Education", "Health", "Leisure"])
                )
            elif item_type == 2:
                item = self.generate_defusion_technique("I am not good enough")
            else:
                item = self.generate_act_matrix()

            batch.append(item)

        return batch

    def export_data(self, data: list[dict], batch_id: str = "001") -> str:
        """Exports the generated data."""
        output_file = self.act_content_path / f"act_batch_{batch_id}.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Successfully exported {len(data)} items to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise
