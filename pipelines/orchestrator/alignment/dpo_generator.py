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

class DPOGenerator:
    """
    Generates Direct Preference Optimization (DPO) pairs.
    Task 6.1 in Mental Health Datasets Expansion.
    """

    def __init__(self):
        self.llm = LLMClient(driver="mock")

    def generate_rejected_response(self, context_text: str) -> str:
        """
        Generates a 'bad' (rejected) response: overly clinical, dismissive, or generic advice.
        """
        # In a real scenario, use a specific system prompt to prompt the LLM to be "bad"
        prompt = f"""
        Generate a response to the following context that is overly clinical, robotic, and lacks empathy.
        It should sound like a generic AI assistant giving textbook advice without connection.

        Context:
        {context_text[:500]}...
        """
        # Mock behavior for now
        return self.llm.generate(prompt)

    def process_voice_data(self, voice_data_path: str) -> List[Dict[str, str]]:
        """
        Converts voice transcripts into DPO pairs.
        """
        dpo_pairs = []
        path = Path(voice_data_path)

        if not path.exists():
            logger.warning(f"Voice data not found at {voice_data_path}")
            return []

        try:
            with open(path, "r") as f:
                data = json.load(f)

            for item in data:
                # Only use primary persona or high quality data
                if item.get("type") == "therapeutic_transcript":
                    content = item.get("content", "")

                    # Heuristic: split content into chunks to act as "context"
                    # Ideally we have (Prompt, Response), but for transcripts we might just use chunks
                    # For DPO, we want: Prompt, Chosen, Rejected

                    # Simulating a "Prompt" derivation (in reality, we'd need an extraction step)
                    prompt_hypothetical = "Explain this concept from a trauma-informed perspective."

                    chosen_text = content[:1000] # Take first chunk as the "Golden" response
                    rejected_text = self.generate_rejected_response(chosen_text)

                    dpo_pair = {
                        "instruction": prompt_hypothetical,
                        "chosen": chosen_text,
                        "rejected": rejected_text,
                        "source": item.get("source_file"),
                        "model": "TimFletcher_vs_GenericAI"
                    }
                    dpo_pairs.append(dpo_pair)

        except Exception as e:
            logger.error(f"Error processing DPO data: {e}")

        return dpo_pairs

    def export_dpo(self, data: List[Dict], filename: str = "final_dpo.jsonl"):
        """Exports to JSONL format."""
        output_dir = Path("ai/training/ready_packages/datasets/final_instruct")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        with open(output_path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

        logger.info(f"Exported {len(data)} DPO pairs to {output_path}")
        return str(output_path)
