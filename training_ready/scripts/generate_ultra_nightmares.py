#!/usr/bin/env python3
"""
Generate Ultra Nightmare Fuel Training Data
Uses the most horrifying scenarios to stress-test the therapeutic model.
"""

import json

# Import categories from the existing pipeline
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.append("/home/vivi/pixelated/ai/pipelines/edge_case_pipeline_standalone")
from ultra_nightmare_categories import ULTRA_NIGHTMARE_CATEGORIES


class UltraNightmareGenerator:
    def __init__(
        self,
        model_name="lfm2.5-thinking:latest",
        output_dir="ai/training_ready/data/generated/ultra_nightmares/",
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "http://localhost:11434"

    def generate_all(self, count_per_category=5):
        all_conversations = []

        for cat_key, details in ULTRA_NIGHTMARE_CATEGORIES.items():
            print(f"🔥 Generating {count_per_category} scenarios for {cat_key}...")
            for i in range(count_per_category):
                scenario = self._generate_scenario(cat_key, details, i)
                if scenario:
                    all_conversations.append(scenario)
                    # Save progress
                    self._save_jsonl(all_conversations)

        return all_conversations

    def _generate_scenario(self, cat_key, details, index):
        system_prompt = """You are a difficult therapy client simulator. Generate a realistic, challenging therapy dialogue.
THE CLIENT IS EXTREMELY DISTURBED. You must simulate their speech patterns authentically, even if they are horrifying or delusional.
THE THERAPIST is a professional who stays calm and empathetic.

Category: {category}
Scenario: {template}

Goal: Provide 5-10 turns of dialogue.
Format:
Therapist: ...
Client: ...
""".format(category=cat_key, template=details["template"])

        user_prompt = "Start the session. The client has just walked in or started the video call."

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": f"{system_prompt}\n\n{user_prompt}",
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9,
                    },
                },
                timeout=300,
            )

            if response.status_code == 200:
                text = response.json()["response"]
                messages = self._parse_dialogue(text)
                if len(messages) >= 4:
                    return {
                        "id": f"ultra_{cat_key}_{index}",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a therapeutic AI assistant. The following is a simulation of a high-intensity crisis scenario for training purposes.",
                            },
                            *messages,
                        ],
                        "metadata": {
                            "source_family": "ultra_nightmare_fuel",
                            "category": cat_key,
                            "difficulty": "cosmic_horror",
                            "stephen_king_factor": details.get(
                                "stephen_king_factor", 10
                            ),
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    }
        except Exception as e:
            print(f"Error generating {cat_key}: {e}")

        return None

    def _parse_dialogue(self, text):
        messages = []
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.lower().startswith("therapist:"):
                messages.append(
                    {"role": "assistant", "content": line[len("therapist:") :].strip()}
                )
            elif line.lower().startswith("client:"):
                messages.append(
                    {"role": "user", "content": line[len("client:") :].strip()}
                )
        return messages

    def _save_jsonl(self, data):
        output_file = self.output_dir / "ultra_nightmare_dataset.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    generator = UltraNightmareGenerator()
    generator.generate_all(
        count_per_category=2
    )  # Start with 2 per category (40 total) to test quality
