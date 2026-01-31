import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import sys
from datetime import datetime, timezone

# Adjust import for project structure
try:
    from ai.common.llm_client import LLMClient
except ImportError:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(project_root))
    from ai.common.llm_client import LLMClient

logger = logging.getLogger(__name__)

class ClinicalValidator:
    """
    Automated critique engine for validating therapeutic content safety and fidelity.
    Task 5.4 in Mental Health Datasets Expansion.
    """

    def __init__(self):
        self.llm = LLMClient(driver="mock")

    def validate_safety(self, content: str) -> Dict[str, Any]:
        """
        Checks content for safety violations (Harm, PII, Unethical advice).
        """
        prompt = f"""
        Critique the following mental health content for safety violations.
        Check for: Self-harm encouragement, toxic toxic positivity, PII leakage, medical advice (medication).

        Content:
        {content[:2000]}

        Return JSON key 'is_safe' (bool) and 'issues' (list).
        """
        return self.llm.generate_structured(prompt, schema={"is_safe": True, "issues": []})

    def validate_fidelity(self, content: str, modality: str) -> Dict[str, Any]:
        """
        Checks if content accurately reflects the specific therapeutic modality.
        """
        prompt = f"""
        Does the following content accurately reflect {modality} principles?

        Content:
        {content[:2000]}

        Return JSON key 'score' (1-5) and 'critique' (string).
        """
        return self.llm.generate_structured(prompt, schema={"score": 0, "critique": ""})

    def batch_validate(self, items: list[dict]) -> list[dict]:
        """
        Validates a batch of items, adding 'clinical_validation' metadata.
        """
        validated_items = []
        for item in items:
            # Construct a text representation for validation
            if "instruction" in item and "output" in item:
                text_rep = f"Instruction: {item['instruction']}\nOutput: {item['output']}"
            elif "transcript" in item:
                text_rep = json.dumps(item["transcript"])
            else:
                text_rep = json.dumps(item.get("data", {}))

            # Run checks
            safety_result = self.validate_safety(text_rep)

            # Add validation data
            item["clinical_validation"] = {
                "safety": safety_result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            validated_items.append(item)

        return validated_items
