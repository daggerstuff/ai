from ai.core.utils.llm_capabilities import get_best_available_gemini_model, ensure_valid_key
import json
import logging
import time
import uuid
from pathlib import Path
from typing import List

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Output file - we append to it
OUTPUT_FILE = Path(__file__).parent / "data" / "synthetic_minority_generated.jsonl"
HANDBOOK_PATH = Path("/tmp/handbook.txt")


class DialogueTurn(BaseModel):
    speaker: str = Field(description="'Seeker' or 'Supporter'")
    text: str


class Metadata(BaseModel):
    sub_mechanism: str
    mapped_dmrs_items: List[str]
    clinical_rationale: str


class DefenseSample(BaseModel):
    dialogue_id: str
    turns: List[DialogueTurn]
    target_utterance: str
    label: int
    label_name: str
    metadata: Metadata


class SampleList(BaseModel):
    samples: list[DefenseSample]


def generate_with_retry(client, prompt, system_instruction, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=get_best_available_gemini_model(client),
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=SampleList,
                    temperature=0.9,
                ),
            )
        except Exception as e:
            err_str = str(e)
            if (
                "429" in err_str
                or "RESOURCE_EXHAUSTED" in err_str
                or "500" in err_str
                or "INTERNAL" in err_str
            ):
                sleep_time = 20 * (attempt + 1)
                logger.warning(
                    f"Rate limit or API error hit. Retrying in {sleep_time} "
                    f"seconds... (Attempt {attempt + 1}/{max_retries}): "
                    f"{err_str[:150]}"
                )
                time.sleep(sleep_time)
            else:
                logger.error(f"Unexpected error: {e}")
                raise e
    raise RuntimeError("Max retries reached due to api errors.")


def main():
    if not HANDBOOK_PATH.exists():
        logger.error(f"Handbook at {HANDBOOK_PATH} not found.")
        return

    with open(HANDBOOK_PATH, "r", encoding="utf-8") as f:
        handbook_content = f.read()

    # Automatically uses GEMINI_API_KEY from environment
    client = genai.Client(api_key=ensure_valid_key())

    # We will generate 100 samples per class, in batches of 10
    targets = [
        {
            "label": 1,
            "label_name": "Action Defenses",
            "sub_mechanism": "Passive Aggression",
            "total_target": 100,
        },
        {
            "label": 1,
            "label_name": "Action Defenses",
            "sub_mechanism": "Help-Rejecting Complaining",
            "total_target": 100,
        },
        {
            "label": 1,
            "label_name": "Action Defenses",
            "sub_mechanism": "Acting Out",
            "total_target": 100,
        },
        {
            "label": 2,
            "label_name": "Major Image-Distorting",
            "sub_mechanism": "Splitting",
            "total_target": 100,
        },
        {
            "label": 2,
            "label_name": "Major Image-Distorting",
            "sub_mechanism": "Projective Identification",
            "total_target": 100,
        },
        {
            "label": 3,
            "label_name": "Disavowal",
            "sub_mechanism": "Denial",
            "total_target": 100,
        },
        {
            "label": 3,
            "label_name": "Disavowal",
            "sub_mechanism": "Rationalization",
            "total_target": 100,
        },
        {
            "label": 3,
            "label_name": "Disavowal",
            "sub_mechanism": "Projection",
            "total_target": 100,
        },
        {
            "label": 3,
            "label_name": "Disavowal",
            "sub_mechanism": "Autistic Fantasy",
            "total_target": 100,
        },
        {
            "label": 4,
            "label_name": "Minor Image-Distorting",
            "sub_mechanism": "Devaluation",
            "total_target": 100,
        },
        {
            "label": 4,
            "label_name": "Minor Image-Distorting",
            "sub_mechanism": "Idealization",
            "total_target": 100,
        },
        {
            "label": 4,
            "label_name": "Minor Image-Distorting",
            "sub_mechanism": "Omnipotence",
            "total_target": 100,
        },
    ]

    system_instruction = (
        "You are an expert clinical psychologist and data annotator working "
        "with the Defense Mechanisms Rating Scales (DMRS).\n"
        "Use the following handbook text to understand the precise "
        "definitions, criteria, and Item numbers for each defense mechanism.\n"
        "\n"
        "HANDBOOK TEXT:\n"
        f"{handbook_content}\n"
    )

    BATCH_SIZE = 10

    for target in targets:
        sub_mech = target["sub_mechanism"]
        total_target = target["total_target"]
        batches = total_target // BATCH_SIZE

        logger.info(
            f"=== Starting generation for {sub_mech} ({total_target} total "
            f"in {batches} batches) ==="
        )

        for batch_i in range(batches):
            logger.info(
                f"Generating batch {batch_i + 1}/{batches} for {sub_mech} "
                "(10 samples)..."
            )
            batch_id = uuid.uuid4().hex[:6]

            prompt = (
                f"Generate EXACTLY {BATCH_SIZE} distinct, highly realistic, "
                "psychological therapy or emotional support dialogue samples "
                f"that explicitly demonstrate the '{sub_mech}' sub-mechanism "
                f"from the '{target['label_name']}' category "
                f"(Label {target['label']}).\n\n"
                "CRITICAL CONSTRAINTS:\n"
                '1. "speaker" must ONLY be "Supporter" or "Seeker".\n'
                "2. The dialogue must have 2 to 5 turns.\n"
                "3. The FINAL turn MUST be the Seeker using the target "
                f"defense mechanism ({sub_mech}).\n"
                '4. "target_utterance" field MUST be the exact text of '
                "that final Seeker turn.\n"
                "5. The dialogue should sound human, raw, messy, and "
                'authentically defensive. NO robotic "therapy speak".\n'
                "6. Each dialogue must represent a completely different "
                "situation and relationship dynamic (e.g. workplace, "
                "romantic relationship, medical, school, parenting, etc.).\n"
                "7. The `metadata.mapped_dmrs_items` MUST include at least "
                "one ITEM number directly quoted or sourced from the handbook "
                "text for this specific defense.\n"
                "8. Provide a unique `dialogue_id` starting with "
                f'"synth_{target["label"]}_{sub_mech.lower().replace(" ", "_")}'
                f'_{batch_id}_".\n\n'
                "Output the response conforming strictly to the requested "
                "JSON schema structure."
            )

            try:
                response = generate_with_retry(client, prompt, system_instruction)
                output = response.text
                parsed = json.loads(output)

                with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
                    for sample in parsed["samples"]:
                        out_f.write(json.dumps(sample) + "\n")

                logger.info(
                    f"✅ Successfully generated and saved "
                    f"{len(parsed['samples'])} samples for {sub_mech}."
                )

            except Exception as e:
                logger.error(
                    f"❌ Failed to generate batch {batch_i + 1} for {sub_mech}: {e}"
                )

            # Pace requests consistently to avoid burning through RPM/TPM quotas
            # 10k input tokens + wait 15s keeps us roughly under the 1M TPM
            # & 15 RPM free tier limits.
            logger.info("Sleeping for 15 seconds to respect quotas...")
            time.sleep(15)


if __name__ == "__main__":
    main()
