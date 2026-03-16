import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys

# Adjust import for project structure
try:
    from ai.common.llm_client import LLMClient
except ImportError:
    # Fallback if running as script from diverse locations
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from ai.common.llm_client import LLMClient

logger = logging.getLogger(__name__)


class SessionSimulator:
    """
    Simulates multi-turn therapeutic sessions between a Virtual Patient and Virtual Therapist.
    Task 5.2 in Mental Health Datasets Expansion.
    """

    def __init__(self, output_base_path: str = "ai/training/ready_packages/datasets"):
        self.client = LLMClient(driver="mock")  # Default to mock for safety
        self.output_base_path = Path(output_base_path)
        self.simulation_path = self.output_base_path / "stage2_reasoning" / "simulated_sessions"
        self._ensure_directories()

    def _ensure_directories(self):
        self.simulation_path.mkdir(parents=True, exist_ok=True)

    def simulate_session(self, modality: str, topic: str, turns: int = 5) -> dict:
        """
        Orchestrates a conversation.
        """
        logger.info(f"Starting session simulation: {modality} - {topic}")

        # In a real implementation with valid API keys:
        # system_prompt_therapist = f"You are an expert {modality} therapist..."
        # system_prompt_patient = f"You are a patient struggling with {topic}..."

        transcript = []

        # Initial context
        transcript.append({"role": "system", "content": f"Session Context: {modality} for {topic}"})

        # Mock Loop
        for i in range(turns):
            # 1. Therapist speaks
            therapist_msg = self.client.generate(f"Therapist turn {i + 1} for {topic}")
            transcript.append({"role": "Therapist", "content": therapist_msg})

            # 2. Patient responds
            patient_msg = self.client.generate(f"Patient turn {i + 1} response to {therapist_msg}")
            transcript.append({"role": "Patient", "content": patient_msg})

        return {
            "type": "Simulated_Session",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"modality": modality, "topic": topic, "turn_count": turns},
            "transcript": transcript,
        }

    def export_conversation(self, session_data: dict, format_type: str = "sharegpt") -> dict:
        """
        Formats the session for training.
        """
        if format_type == "sharegpt":
            conversations = []

            # System
            conversations.append(
                {
                    "from": "system",
                    "value": f"The following is a {session_data['metadata']['modality']} therapy session focusing on {session_data['metadata']['topic']}.",
                }
            )

            transcript = session_data.get("transcript", [])
            for turn in transcript:
                role = turn["role"]
                content = turn["content"]

                if role == "system":
                    continue

                conversations.append(
                    {"from": "gpt" if role == "Therapist" else "human", "value": content}
                )

            return {"conversations": conversations}

        return session_data  # Return raw if format unknown

    def generate_journaling_session(self, topic: str) -> Dict[str, Any]:
        """
        Simulates a long-term journaling exercise.
        Patient writes a long entry, Therapist provides a reflective summary.
        """
        prompt = f"Write a raw, unedited journal entry about {topic}. It should be long, meandering, and emotional."
        journal_entry = self.client.generate(prompt)

        reflection_prompt = f"As a compassionate therapist, read this journal entry and provide a summary of the core emotional themes and a single gentling guiding question.\n\nJournal:\n{journal_entry}"
        therapist_reflection = self.client.generate(reflection_prompt)

        return {
            "topic": topic,
            "turns": 2,
            "type": "journaling_exercise",
            "transcript": [
                {"role": "Patient_Journal", "content": journal_entry},
                {"role": "Therapist_Reflection", "content": therapist_reflection}
            ]
        }

    def generate_batch(self, count: int = 5) -> List[Dict[str, Any]]:
        """Generates a batch of simulated sessions (mixed types)."""
        data = []
        topics = ["Anxiety", "Depression", "Work Stress", "Relationship Issues", "Trauma Triggers", "Addiction Cravings"]
        modalities = ["CBT", "ACT", "DBT"] # Kept for simulate_session calls

        for i in range(count):
            import random # Moved inside loop to match original behavior, or could be moved to top of method
            current_topic = topics[i % len(topics)]
            # Mix standard sessions and journaling
            if i % 3 == 0:
                 session = self.generate_journaling_session(current_topic)
            else:
                 session = self.simulate_session(modality=random.choice(modalities), topic=current_topic, turns=5)
            data.append(session)
        logger.info(f"Generated {len(data)} simulations.")

        # Export
        output_file = (
            self.simulation_path / f"sim_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        return data
