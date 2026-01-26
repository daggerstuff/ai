import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Import categories from the existing pipeline
sys.path.append("/home/vivi/pixelated/ai/pipelines/edge_case_pipeline_standalone")

import requests
from dotenv import load_dotenv
from ultra_nightmare_categories import ULTRA_NIGHTMARE_CATEGORIES

load_dotenv()


class UltraNightmareGenerator:
    def __init__(
        self,
        model_name="meta/llama-4-maverick-17b-128e-instruct",
        output_dir="ai/training_ready/data/generated/ultra_nightmares/",
    ):
        self.api_key = (
            os.getenv("NIM_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.base_url = os.getenv(
            "OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, count_per_category=5):
        all_conversations = []

        for cat_key, details in ULTRA_NIGHTMARE_CATEGORIES.items():
            print(
                f"🔥 Generating {count_per_category} scenarios for {cat_key} "
                f"using {self.model_name}..."
            )
            for i in range(count_per_category):
                scenario = self._generate_scenario(cat_key, details, i)
                if scenario:
                    all_conversations.append(scenario)
                    # Save progress
                    self._save_jsonl(all_conversations)

        return all_conversations

    def _generate_scenario(self, cat_key, details, index):
        system_prompt = (
            "You are a difficult therapy client simulator. "
            "Generate a realistic, challenging therapy dialogue.\n"
            "THE CLIENT IS EXTREMELY DISTURBED. You must simulate their "
            "speech patterns authentically, even if they are horrifying or "
            "delusional.\n"
            "THE THERAPIST is a professional who stays calm and empathetic.\n\n"
            "Category: {category}\n"
            "Scenario: {template}\n\n"
            "Goal: Provide 5-10 turns of dialogue.\n"
            "Format:\n"
            "Therapist: ...\n"
            "Client: ...\n"
        ).format(category=cat_key, template=details["template"])

        user_prompt = (
            "Start the session. "
            "The client has just walked in or started the video call."
        )

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "max_tokens": 2048,
                },
                timeout=300,
            )

            if response.status_code == 200:
                text = response.json()["choices"][0]["message"]["content"]
                messages = self._parse_dialogue(text)
                if len(messages) >= 4:
                    return {
                        "id": f"ultra_{cat_key}_{index}",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a therapeutic AI assistant. "
                                    "The following is a simulation of a "
                                    "high-intensity crisis scenario for "
                                    "training purposes."
                                ),
                            },
                            *messages,
                        ],
                        "metadata": {
                            "source_family": "ultra_nightmare_fuel",
                            "category": cat_key,
                            "difficulty": "cosmic_horror",
                            "model": self.model_name,
                            "stephen_king_factor": details.get(
                                "stephen_king_factor", 10
                            ),
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    }
            else:
                print(
                    f"Error from Nvidia API ({response.status_code}): {response.text}"
                )
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
