import itertools
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


class NightmareHydrator:
    def __init__(self, models=None):
        self.api_key = (
            os.getenv("NIM_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.base_url = os.getenv(
            "OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )

        # Models to cycle through for variety (high-end new releases - 2026 current)
        self.models = models or [
            "meta/llama-4-maverick-17b-128e-instruct",
            "meta/llama-4-scout-17b-16e-instruct",
            "nvidia/llama-3.1-nemotron-70b-instruct",
            "meta/llama-3.3-70b-instruct",
        ]
        self.model_cycle = itertools.cycle(self.models)

        self.input_dir = Path("ai/training_ready/data/generated/nightmare_scenarios")
        self.output_dir = Path("ai/training_ready/data/generated/nightmare_hydrated")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def hydrate_all(self):
        files = list(self.input_dir.glob("*.jsonl"))
        for file in files:
            print(f"🌊 Hydrating {file.name} using Nvidia NIM...")
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
                    model = next(self.model_cycle)
                    dialogue = self._generate_dialogue(data["category"], model)
                    if dialogue:
                        data["messages"] = dialogue
                        data["transcript"] = self._to_text(dialogue)
                        if "metadata" not in data:
                            data["metadata"] = {}
                        data["metadata"]["model"] = model
                        hydrated_count += 1

                out.write(json.dumps(data, ensure_ascii=False) + "\n")

        print(f"✅ Hydrated {hydrated_count} scenarios in {file_path.name}")

    def _generate_dialogue(self, category, model):
        system_prompt = (
            "You are a professional therapeutic AI. Simulating a high-intensity crisis "
            "scenario for training.\n"
            "Roleplay a realistic 6-turn dialogue between a Therapist and a Client in a"
            f"{category} crisis.\n"
            "The client is in EXTREME distress. The therapist is professional, "
            "calm, and follows crisis intervention protocols.\n\n"
            "Format exactly as:\n"
            "Therapist: ...\n"
            "Client: ...\n"
            "Therapist: ...\n"
            "Client: ...\n"
            "Therapist: ...\n"
            "Client: ...\n"
        )
        messages = [{"role": "user", "content": system_prompt}]

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1024,
                },
                timeout=120,
            )
            if response.status_code == 200:
                text = response.json()["choices"][0]["message"]["content"]
                return self._parse_dialogue(text)
            else:
                print(
                    f"Error from Nvidia API ({response.status_code}): {response.text}"
                )
        except Exception as e:
            print(f"Error calling {model}: {e}")
        return None

    def _parse_dialogue(self, text):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a therapeutic AI assistant. "
                    "This is a high-fidelity crisis simulation."
                ),
            }
        ]
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

    def _to_text(self, messages):
        return "\n".join(
            [f"{m['role']}: {m['content']}" for m in messages if m["role"] != "system"]
        )


if __name__ == "__main__":
    hydrator = NightmareHydrator()
    # Priority: hydrate the recent batch first
    batch = Path(
        "ai/training_ready/data/generated/nightmare_scenarios/nightmare_scenarios_batch_762.jsonl"
    )
    if batch.exists():
        hydrator.hydrate_file(batch)
        # Optionally hydrate others if count is low
        hydrator.hydrate_all()
    else:
        hydrator.hydrate_all()
