import json
from typing import Any


def format_persona(meddies_record: dict[str, Any]) -> str:
    """
    Convert a dense Meddies persona dict into a natural language string.
    Example expected input:
    {
        "demographics": {"age": 45, "gender": "female", "location": "Hanoi"},
        "healthcare_behavior": {"health_literacy": "low", "preference": "traditional medicine"}
    }
    """
    demo = meddies_record.get("demographics", {})
    health = meddies_record.get("healthcare_behavior", {})

    age = demo.get("age", "unknown age")
    gender = demo.get("gender", "person")
    location = demo.get("location", "Vietnam")

    health_literacy = health.get("health_literacy", "average")
    preference = health.get("preference", "standard medicine")

    description = (
        f"This patient is a {age}-year-old {gender} from {location} "
        f"with {health_literacy} health literacy who prefers {preference}."
    )
    return description

def process_file(input_path: str, output_path: str) -> None:
    """Process a JSONL file of Meddies records and output formatted PAL strings."""
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            formatted = format_persona(record)
            fout.write(json.dumps({"persona_string": formatted}, ensure_ascii=False) + "\n")
