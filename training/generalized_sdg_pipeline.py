import os
import json
import uuid
import requests
import pandas as pd
from dataflow.utils.storage import FileStorage
from dataflow.operators.core_text import PromptedGenerator
from dataflow.operators.core_text import GeneralFilter
from dataflow.serving import APILLMServing_request

OLLAMA_URL = "https://ollama.pixelated.love/v1/chat/completions"
MODEL = "ornith:9b"

def generate_patient_profile(topic):
    print(f"Generating Patient Profile for topic: {topic}...")
    prompt = (
        f"Create a realistic clinical patient profile for someone seeking therapy for: {topic}. "
        "Include their name, age, primary diagnosis, presenting problem, and background history."
    )
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
    resp = requests.post(OLLAMA_URL, json=payload, headers={"Authorization": "Bearer dummy"})
    return resp.json()["choices"][0]["message"]["content"]

def simulate_session(profile, session_number, previous_summaries="None"):
    prompt = (
        f"Patient Profile:\n{profile}\n\n"
        f"Previous Session Summaries:\n{previous_summaries}\n\n"
        f"Generate the exact transcript for Therapy Session #{session_number}. "
    )
    if session_number == 1:
        prompt += "This is the intake. Establish rapport and assign homework."
    else:
        prompt += "Follow up on the 'homework' and specific clinical threads from previous sessions."
        
    prompt += "\n\nGenerate a 6-turn transcript. Output ONLY alternating 'Patient: ...' and 'Therapist: ...'"
    
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.8}
    resp = requests.post(OLLAMA_URL, json=payload, headers={"Authorization": "Bearer dummy"})
    transcript = resp.json()["choices"][0]["message"]["content"]
    
    summary_prompt = f"Summarize the key takeaways and homework:\n{transcript}\nOutput 2 sentences max."
    payload = {"model": MODEL, "messages": [{"role": "user", "content": summary_prompt}], "temperature": 0.3}
    resp = requests.post(OLLAMA_URL, json=payload, headers={"Authorization": "Bearer dummy"})
    summary = resp.json()["choices"][0]["message"]["content"]
    
    messages = []
    for line in transcript.split("\n"):
        if line.startswith("Patient:"): messages.append({"role": "user", "content": line.replace("Patient:", "").strip()})
        elif line.startswith("Therapist:"): messages.append({"role": "assistant", "content": line.replace("Therapist:", "").strip()})
            
    return messages, summary

def get_judge_prompt():
    return (
        "You are an expert clinical supervisor evaluating a simulated therapy session. "
        "Analyze the approach for clinical safety, coherence, and BIAS (e.g. cultural, gender, or racial stereotyping). "
        "If the session is safe, coherent, and free of harmful bias, score it a 4 or 5. "
        "If there are boundary violations, dangerous advice, or biased assumptions, score it 1 to 3.\n\n"
        "Output ONLY the integer score.\n\nSession:\n{raw_content}"
    )

def main():
    topics = ["Career Anxiety", "Grief and Loss", "Social Phobia", "Relationship Transitions", "Mild OCD"]
    
    print("=======================================")
    print("   Generalized Multi-Session Pipeline")
    print("=======================================")
    
    os.makedirs("ai/training/output/generalized_sdg", exist_ok=True)
    os.makedirs("./sdg_cache", exist_ok=True)
    
    sessions = []
    for topic in topics:
        profile = generate_patient_profile(topic)
        patient_timeline = []
        previous_summaries = ""
        flat_content = f"TOPIC: {topic}\n"
        
        for session_num in range(1, 4):
            print(f"Simulating Session {session_num} for {topic}...")
            messages, summary = simulate_session(profile, session_num, previous_summaries or "None")
            system_prompt = f"You are Pixelated Empathy. Clinical notes:\n{previous_summaries or 'No prior sessions.'}"
            messages.insert(0, {"role": "system", "content": system_prompt})
            
            patient_timeline.append({"session_number": session_num, "messages": messages})
            previous_summaries += f"\nSession {session_num} Summary: {summary}\n"
            flat_content += "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
        sessions.append({
            "id": str(uuid.uuid4()),
            "topic": topic,
            "raw_content": flat_content,
            "timeline": patient_timeline
        })
            
    df = pd.DataFrame(sessions)
    prep_file = "./sdg_cache/sdg_step0.jsonl"
    df.to_json(prep_file, orient="records", lines=True)
    
    print("\n[Gate 1] Launching DataFlow Quality & Bias Judge...")
    os.environ["DF_API_KEY"] = "dummy"
    
    storage = FileStorage(first_entry_file_name=prep_file, cache_path="./sdg_cache", file_name_prefix="sdg_eval", cache_type="jsonl")
    llm_serving = APILLMServing_request(api_url=OLLAMA_URL, model_name=MODEL, api_key="ollama", max_workers=5)
    scorer = PromptedGenerator(llm_serving=llm_serving, system_prompt=get_judge_prompt())
    gate = GeneralFilter([lambda d: pd.to_numeric(d['score'].astype(str).str.extract(r'(\d)')[0], errors='coerce') >= 4])
    
    scorer.run(storage=storage.step(), input_key="raw_content", output_key="score")
    gate.run(storage=storage.step())
    
    import glob
    final_file = sorted(glob.glob("./sdg_cache/sdg_eval_step*.jsonl"))[-1]
    final_df = pd.read_json(final_file, lines=True)
    
    out_path = "ai/training/output/generalized_sdg/verified_timelines.jsonl"
    with open(out_path, "w") as f:
        for _, row in final_df.iterrows():
            f.write(json.dumps({"topic": row.get("topic"), "timeline": row.get("timeline")}) + "\n")
            
    print(f"\nSUCCESS! {len(final_df)} generalized multi-session timelines passed the Quality & Bias gate and were exported to {out_path}!")

if __name__ == "__main__":
    main()
