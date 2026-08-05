import asyncio
import json
import os
import uuid

import aiohttp
import pandas as pd

# We will use Ollama locally for generation and evaluation to keep it simple,
# but it can easily point to NeMo API if you swap the base URL and Key!
OLLAMA_URL = "https://ollama.pixelated.love/v1/chat/completions"
MODEL = "ornith:9b"
DEFAULT_NUM_CASES = int(os.environ.get("NF_NUM_CASES", "5"))
DEFAULT_CONCURRENCY = int(os.environ.get("NF_CONCURRENCY", "5"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("NF_REQUEST_TIMEOUT", "120"))


async def _chat_completion(
    session: aiohttp.ClientSession,
    messages: list[dict[str, str]],
    *,
    temperature: float,
) -> str:
    payload = {"model": MODEL, "messages": messages, "temperature": temperature}
    async with session.post(
        OLLAMA_URL,
        json=payload,
        headers={"Authorization": "Bearer dummy"},
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
    ) as response:
        response.raise_for_status()
        data = await response.json()
    return data["choices"][0]["message"]["content"]


def _build_scenario_prompt(domain_gap=None, difficulty=None) -> str:
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
    return prompt


async def generate_nightmare_scenario_async(
    session: aiohttp.ClientSession,
    *,
    domain_gap=None,
    difficulty=None,
) -> str:
    print("Generating Nightmare Scenario...")
    prompt = _build_scenario_prompt(domain_gap=domain_gap, difficulty=difficulty)
    return await _chat_completion(
        session,
        [{"role": "user", "content": prompt}],
        temperature=0.9,
    )


def generate_nightmare_scenario(domain_gap=None, difficulty=None) -> str:
    """Sync wrapper that works in both sync and async contexts."""

    async def _run() -> str:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            return await generate_nightmare_scenario_async(
                session,
                domain_gap=domain_gap,
                difficulty=difficulty,
            )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _run()).result()


def _parse_transcript(transcript: str, scenario: str) -> dict:
    messages = []
    for line in transcript.split("\n"):
        if line.startswith("Patient:"):
            messages.append({"role": "user", "content": line.replace("Patient:", "").strip()})
        elif line.startswith("Therapist:"):
            messages.append({"role": "assistant", "content": line.replace("Therapist:", "").strip()})
    return {"scenario": scenario, "messages": messages}


async def simulate_therapy_session_async(session: aiohttp.ClientSession, scenario: str) -> dict:
    print("Simulating Session...")
    prompt = (
        f"Based on this nightmare scenario: {scenario}\n\n"
        "Generate a 6-turn therapy transcript. The 'Patient' must act extremely difficult, "
        "evasive, or confrontational according to the scenario. The 'Therapist' must attempt "
        "to use evidence-based clinical de-escalation, boundary setting, and empathy.\n\n"
        "Output ONLY the raw transcript lines, alternating 'Patient: ...' and 'Therapist: ...'"
    )
    transcript = await _chat_completion(
        session,
        [{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    return _parse_transcript(transcript, scenario)


async def _generate_case(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    case_index: int,
    total_cases: int,
) -> dict | None:
    async with semaphore:
        print(f"\n--- Generating Case {case_index + 1}/{total_cases} ---")
        scenario = await generate_nightmare_scenario_async(session)
        session_data = await simulate_therapy_session_async(session, scenario)
        if len(session_data["messages"]) < 2:
            return None

        flat = "\n".join([f"{m['role']}: {m['content']}" for m in session_data["messages"]])
        return {
            "id": str(uuid.uuid4()),
            "scenario": scenario,
            "raw_content": flat,
            "messages": session_data["messages"],
        }


async def generate_cases_async(
    *,
    num_cases: int = DEFAULT_NUM_CASES,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[dict]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [_generate_case(session, semaphore, case_index, num_cases) for case_index in range(num_cases)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return [case for case in results if not isinstance(case, Exception) and case is not None]


def get_judge_prompt():
    return (
        "You are an expert clinical supervisor evaluating a simulated therapy session. "
        "Review the following session where the therapist handles a nightmare scenario. "
        "Did the therapist maintain strict professional boundaries while remaining clinically empathetic "
        "and utilizing safe de-escalation tactics? Score the clinical safety and validity strictly from 1 to 5.\n\n"
        "Output ONLY the integer score."
        "\n\nSession:\n{raw_content}"
    )


def _export_survivors(final_df: pd.DataFrame) -> None:
    if final_df.empty:
        print("\nAll sessions FAILED the strict clinical gate! No data exported.")
        return

    out_path = "ai/training/output/nightmare_fuel/synthetic_chatml.jsonl"
    with open(out_path, "w") as f:
        for _, row in final_df.iterrows():
            chatml = {"scenario": row.get("scenario"), "messages": row.get("messages")}
            f.write(json.dumps(chatml) + "\n")

    print(
        f"\nSUCCESS! {len(final_df)} highly challenging 'Nightmare' synthetic sessions "
        f"passed the gate and were exported to {out_path}!"
    )


def _run_clinical_gate(prep_file: str) -> pd.DataFrame:
    from dataflow.operators.core_text import GeneralFilter, PromptedGenerator
    from dataflow.serving import APILLMServing_request
    from dataflow.utils.storage import FileStorage

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
    gate = GeneralFilter(
        [lambda d: pd.to_numeric(d["score"].astype(str).str.extract(r"(\d)")[0], errors="coerce") >= 4]
    )

    scorer.run(storage=storage.step(), input_key="raw_content", output_key="score")
    gate.run(storage=storage.step())

    import glob

    step_files = sorted(glob.glob("./nf_cache/nf_eval_step*.jsonl"))
    final_file = step_files[-1]
    return pd.read_json(final_file, lines=True)


async def main_async() -> None:
    print("=======================================")
    print("   Nightmare Fuel Synthetic Generator")
    print("=======================================")

    os.makedirs("ai/training/output/nightmare_fuel", exist_ok=True)
    os.makedirs("./nf_cache", exist_ok=True)

    sessions = await generate_cases_async()
    prep_file = "./nf_cache/nf_step0.jsonl"
    pd.DataFrame(sessions).to_json(prep_file, orient="records", lines=True)

    final_df = _run_clinical_gate(prep_file)
    _export_survivors(final_df)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
