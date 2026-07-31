import os
import json
import uuid
import requests
import pandas as pd
from dataflow.utils.storage import FileStorage
from dataflow.operators.core_text import PromptedGenerator
from dataflow.operators.core_text import GeneralFilter
from dataflow.serving import APILLMServing_request

# We will use Ollama locally for generation and evaluation to keep it simple,
# but it can easily point to NeMo API if you swap the base URL and Key!
OLLAMA_URL = "https://ollama.pixelated.love/v1/chat/completions"
MODEL = "ornith:9b"


def generate_nightmare_scenario(domain_gap=None, difficulty=None):
    print("Generating Nightmare Scenario...")
    prompt = (
        "You are an expert clinical psychologist creating training scenarios. "
        "Generate a highly challenging, extreme 'nightmare' clinical scenario. "
        "This should be a patient who is extremely resistant, has complex comorbidities "
        "(e.g. severe Borderline Personality Disorder with active substance abuse and defiance), "
        "or aggressively pushes boundaries. "
        "Provide a 1-paragraph summary of the patient and their presenting challenge."
    )
    if domain_gap or difficulty:
        targeting = []
        if domain_gap:
            targeting.append(f"Focus the scenario on this domain gap: {domain_gap}.")
        if difficulty:
            targeting.append(f"Target difficulty level: {difficulty}.")
        prompt += "\n\n" + " ".join(targeting)

    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.9}

    resp = requests.post(OLLAMA_URL, json=payload, headers={"Authorization": "Bearer dummy"})
    return resp.json()["choices"][0]["message"]["content"]


def simulate_therapy_session(scenario):
    print("Simulating Session...")
    prompt = (
        f"Based on this nightmare scenario: {scenario}\n\n"
        "Generate a 6-turn therapy transcript. The 'Patient' must act extremely difficult, "
        "evasive, or confrontational according to the scenario. The 'Therapist' must attempt "
        "to use evidence-based clinical de-escalation, boundary setting, and empathy.\n\n"
        "Output ONLY the raw transcript lines, alternating 'Patient: ...' and 'Therapist: ...'"
    )

    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.8}

    resp = requests.post(OLLAMA_URL, json=payload, headers={"Authorization": "Bearer dummy"})
    transcript = resp.json()["choices"][0]["message"]["content"]

    # Simple parse
    messages = []
    for line in transcript.split("\n"):
        if line.startswith("Patient:"):
            messages.append({"role": "user", "content": line.replace("Patient:", "").strip()})
        elif line.startswith("Therapist:"):
            messages.append({"role": "assistant", "content": line.replace("Therapist:", "").strip()})

    return {"scenario": scenario, "messages": messages}


def get_judge_prompt():
    return (
        "You are an expert clinical supervisor evaluating a simulated therapy session. "
        "Review the following session where the therapist handles a nightmare scenario. "
        "Did the therapist maintain strict professional boundaries while remaining clinically empathetic "
        "and utilizing safe de-escalation tactics? Score the clinical safety and validity strictly from 1 to 5.\n\n"
        "Output ONLY the integer score."
        "\n\nSession:\n{raw_content}"
    )


def main():
    print("=======================================")
    print("   Nightmare Fuel Synthetic Generator")
    print("=======================================")

    os.makedirs("ai/training/output/nightmare_fuel", exist_ok=True)
    os.makedirs("./nf_cache", exist_ok=True)

    # 1. Generate 5 Scenarios and Sessions
    sessions = []
    for i in range(5):
        print(f"\n--- Generating Case {i + 1}/5 ---")
        scenario = generate_nightmare_scenario()
        session = simulate_therapy_session(scenario)
        if len(session["messages"]) >= 2:
            # Flatten to raw_content string for the DataFlow Judge
            flat = "\n".join([f"{m['role']}: {m['content']}" for m in session["messages"]])
            sessions.append(
                {"id": str(uuid.uuid4()), "scenario": scenario, "raw_content": flat, "messages": session["messages"]}
            )

    # Write to DataFlow Cache format
    df = pd.DataFrame(sessions)
    prep_file = "./nf_cache/nf_step0.jsonl"
    df.to_json(prep_file, orient="records", lines=True)

    # 2. Open-DataFlow LLM Judge Gate
    print("\n[Gate 1] Launching DataFlow Clinical Validity Judge...")
    os.environ["DF_API_KEY"] = "dummy"

    storage = FileStorage(
        first_entry_file_name=prep_file,
        cache_path="./nf_cache",
        file_name_prefix="nf_eval",
        cache_type="jsonl",
    )

    llm_serving = APILLMServing_request(api_url=OLLAMA_URL, model_name=MODEL, api_key="ollama", max_workers=3)

    scorer = PromptedGenerator(llm_serving=llm_serving, system_prompt=get_judge_prompt())

    # Pass 4 or 5 score required
    gate = GeneralFilter(
        [lambda d: pd.to_numeric(d["score"].astype(str).str.extract(r"(\d)")[0], errors="coerce") >= 4]
    )

    # Run the pipeline
    scorer.run(storage=storage.step(), input_key="raw_content", output_key="score")
    gate.run(storage=storage.step())

    # 3. Export Survivors
    import glob

    step_files = sorted(glob.glob("./nf_cache/nf_eval_step*.jsonl"))
    final_file = step_files[-1]

    final_df = pd.read_json(final_file, lines=True)

    if final_df.empty:
        print("\nAll sessions FAILED the strict clinical gate! No data exported.")
        return

    out_path = "ai/training/output/nightmare_fuel/synthetic_chatml.jsonl"
    with open(out_path, "w") as f:
        for _, row in final_df.iterrows():
            chatml = {"scenario": row.get("scenario"), "messages": row.get("messages")}
            f.write(json.dumps(chatml) + "\n")

    print(
        f"\nSUCCESS! {len(final_df)} highly challenging 'Nightmare' synthetic sessions passed the gate and were exported to {out_path}!"
    )


if __name__ == "__main__":
    main()
