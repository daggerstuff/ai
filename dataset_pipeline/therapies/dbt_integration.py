import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class DBTIntegration:
    """
    Generates structured Dialectical Behavior Therapy (DBT) content.
    Includes Skills, Diary Cards, and Chain Analysis templates.
    Task 3.2 in Mental Health Datasets Expansion.
    """

    def __init__(self, output_base_path: str | None = "ai/training_ready/datasets"):
        self.output_base_path = Path(output_base_path)
        self.dbt_content_path = self.output_base_path / "stage2_reasoning" / "dbt_content"
        self._ensure_directories()

        self.modules = {
            "Mindfulness": [
                "Wise Mind",
                "Observe",
                "Describe",
                "Participate",
                "Non-judgmental",
                "One-mindfully",
                "Effectively",
            ],
            "Distress Tolerance": [
                "TIP Skills",
                "STOP",
                "Pros and Cons",
                "Radical Acceptance",
                "Distract with ACCEPTS",
                "Self-Soothe",
                "Improve the Moment",
            ],
            "Emotion Regulation": [
                "Check the Facts",
                "Opposite Action",
                "ABC PLEASE",
                "Accumulate Positive Emotions",
                "Build Mastery",
                "Cope Ahead",
            ],
            "Interpersonal Effectiveness": ["DEAR MAN", "GIVE", "FAST", "Validation"],
        }

        logger.info(f"Initialized DBTIntegration. Output path: {self.dbt_content_path}")

    def _ensure_directories(self):
        """Ensure output directories exist."""
        self.dbt_content_path.mkdir(parents=True, exist_ok=True)

    def generate_skill_exercise(self, module: str | None = None, skill_name: str | None = None) -> dict:
        """
        Generates a skills training exercise.
        """
        if not module:
            module = random.choice(list(self.modules.keys()))
        if not skill_name:
            skill_name = random.choice(self.modules[module])

        return {
            "type": "DBT_Skill_Exercise",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "module": module,
                "skill": skill_name,
                "instructions": f"Practice the {skill_name} skill in a low-stress situation.",
                "worksheet_prompt": f"Describe a situation where you used (or could have used) {skill_name}.",
                "reflection_questions": [
                    "What happened?",
                    "How did you apply the skill?",
                    "What was the outcome?",
                    "Rate the effectiveness (0-5)",
                ],
            },
            "validation_status": "template_verified",
        }

    def generate_diary_card_entry(
        self, urges_level: int, emotion_intensity: int, skill_used: str | None = None
    ) -> dict:
        """
        Creates a structured daily diary card entry.
        """
        return {
            "type": "DBT_Diary_Card",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "urges": {
                    "self_harm": urges_level,  # 0-5
                    "suicide": max(0, urges_level - 1),  # Simulated correlation
                    "drug_use": random.randint(0, 5),
                },
                "emotions": {
                    "sadness": emotion_intensity,
                    "anger": random.randint(0, 5),
                    "anxiety": random.randint(0, 5),
                    "joy": random.randint(0, 5),
                },
                "skills_used": skill_used or "None",
                "notes": "Briefly describe the day's events...",
            },
            "validation_status": "template_verified",
        }

    def generate_chain_analysis(self, problem_behavior: str) -> dict:
        """
        Creates a Chain Analysis template.
        """
        return {
            "type": "DBT_Chain_Analysis",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "problem_behavior": problem_behavior,
                "chain": {
                    "vulnerability_factors": "[What made you vulnerable? e.g., tired, hungry, sick]",
                    "prompting_event": "[What started the chain?]",
                    "links": [
                        {"type": "thought", "content": "..."},
                        {"type": "emotion", "content": "..."},
                        {"type": "sensation", "content": "..."},
                    ],
                    "target_behavior": problem_behavior,
                    "consequences": {
                        "immediate": "[Short-term relief?]",
                        "long_term": "[Damage to goals/relationships?]",
                    },
                },
                "solution_analysis": {
                    "skills_to_break_chain": "[Where could you have intervened?]",
                    "repair_action": "[How to fix current damage?]",
                },
            },
            "validation_status": "template_verified",
        }

    def get_dbt_session_template(self, stage: str = "Stage 1") -> dict:
        """
        Returns a session structure template.
        """
        structure = [
            "Review Diary Card",
            "Attention to Target Hierarchy (Life-threatening -> Therapy-interfering -> QoL-interfering)",
            "Chain Analysis of Problem Behavior",
            "Solution Analysis & Skills Coaching",
            "Consultation Team Support (Therapist note)",
        ]

        return {
            "type": "DBT_Session_Template",
            "stage": stage,
            "structure": structure,
            "focus": "Stabilization and Behavioral Control"
            if stage == "Stage 1"
            else "Emotional Processing",
            "validation_status": "clinical_standard",
        }

    def generate_batch_content(self, count: int = 10) -> list[dict]:
        """
        Generates a batch of mixed DBT content.
        """
        logger.info(f"Generating {count} DBT simulated records...")
        batch = []

        for i in range(count):
            item_type = i % 4
            if item_type == 0:
                item = self.generate_skill_exercise()
            elif item_type == 1:
                item = self.generate_diary_card_entry(
                    urges_level=random.randint(0, 5),
                    emotion_intensity=random.randint(0, 5),
                    skill_used="Wise Mind",
                )
            elif item_type == 2:
                item = self.generate_chain_analysis("Self-harm urge")
            else:
                item = self.get_dbt_session_template()

            batch.append(item)

        return batch

    def export_data(self, data: list[dict], batch_id: str = "001") -> str:
        """Exports the generated data."""
        output_file = self.dbt_content_path / f"dbt_batch_{batch_id}.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Successfully exported {len(data)} items to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise
