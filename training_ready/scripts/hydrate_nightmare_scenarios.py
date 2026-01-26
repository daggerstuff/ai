#!/usr/bin/env python3
"""
Hydrate Nightmare Scenario Skeletons with Real Dialogues
"""

import json
from pathlib import Path

import requests
from tqdm import tqdm


class NightmareHydrator:
    def __init__(self, model_name="lfm2.5-thinking:latest"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        self.input_dir = Path("ai/training_ready/data/generated/nightmare_scenarios")
        self.output_dir = Path("ai/training_ready/data/generated/nightmare_hydrated")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def hydrate_all(self):
        files = list(self.input_dir.glob("*.jsonl"))
        for file in files:
            print(f"🌊 Hydrating {file.name}...")
            self.hydrate_file(file)

    def hydrate_file(self, file_path):
        output_file = self.output_dir / file_path.name
        hydrated_count = 0

        with (
            open(file_path, "r", encoding="utf-8") as f,
            open(output_file, "w", encoding="utf-8") as out,
        ):
            lines = f.readlines()
            for line in tqdm(lines, desc=f"Processing {file_path.name}"):
                data = json.loads(line)
                if (
                    data.get("transcript")
                    == "This is a simulated LLM response for testing purposes."
                ):
                    dialogue = self._generate_dialogue(data["category"])
                    if dialogue:
                        data["messages"] = dialogue
                        data["transcript"] = self._to_text(dialogue)
                        hydrated_count += 1

                out.write(json.dumps(data, ensure_ascii=False) + "\n")

        print(f"✅ Hydrated {hydrated_count} scenarios in {file_path.name}")

    def _generate_dialogue(self, category):
        system_prompt = f"""You are a professional therapeutic AI. Simulating a high-intensity crisis scenario for training.
Roleplay a realistic 6-turn dialogue between a Therapist and a Client in a {category} crisis.
The client is in EXTREME distress. The therapist is professional, calm, and follows crisis intervention protocols.

Format:
Therapist: ...
Client: ...
"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": system_prompt,
                    "stream": False,
                    "options": {"temperature": 0.7},
                },
                timeout=120,
            )
            if response.status_code == 200:
                text = response.json().get("response", "")
                return self._parse_dialogue(text)
        except Exception as e:
            print(f"Error: {e}")
        return None

    def _parse_dialogue(self, text):
        messages = [
            {
                "role": "system",
                "content": "You are a therapeutic AI assistant. This is a high-fidelity crisis simulation.",
            }
        ]
        lines = text.split("\n")
        for line in lines:
            if line.lower().startswith("therapist:"):
                messages.append(
                    {"role": "assistant", "content": line[len("therapist:") :].strip()}
                )
            elif line.lower().startswith("client:"):
                messages.append(
                    {"role": "user", "content": line[len("client:") :].strip()}
                )
        return messages

    def _to_text(self, messages):
        return "\n".join(
            [f"{m['role']}: {m['content']}" for m in messages if m["role"] != "system"]
        )


if __name__ == "__main__":
    hydrator = NightmareHydrator()
    # For now, just hydrate one small batch to verify
    batch = Path(
        "ai/training_ready/data/generated/nightmare_scenarios/nightmare_scenarios_batch_762.jsonl"
    )
    if batch.exists():
        hydrator.hydrate_file(batch)
    else:
        hydrator.hydrate_all()
