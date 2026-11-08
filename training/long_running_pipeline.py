import json
import os
import uuid

import requests

OLLAMA_URL = "https://ollama.pixelated.love/v1/chat/completions"
MODEL = "ornith:9b"

def generate_patient_profile():
    print("Generating Patient Profile...")
    prompt = (
        "Create a realistic, highly detailed clinical patient profile for someone "
        "seeking therapy. Include their name, age, primary diagnosis (e.g. Generalized "
        "Anxiety Disorder, Major Depressive Disorder), presenting problem, and a brief "
        "background history.\n\n"
        "Output ONLY the patient profile."
    )
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.8}
    resp = requests.post(OLLAMA_URL, json=payload, headers={"Authorization": "Bearer dummy"})
    return resp.json()["choices"][0]["message"]["content"]

def simulate_session(profile, session_number, previous_summaries="None"):
    print(f"Simulating Session {session_number}...")
    prompt = (
        f"Patient Profile:\n{profile}\n\n"
        f"Previous Session Summaries:\n{previous_summaries}\n\n"
        f"Generate the exact transcript for Therapy Session #{session_number}. "
    )

    if session_number == 1:
        prompt += "This is the intake session. The therapist must establish rapport, explore the presenting problem, and assign a small piece of 'homework' for the patient."
    else:
        prompt += (
            "The therapist MUST explicitly follow up on the 'homework' or specific clinical threads discussed "
            "in the previous sessions. The patient should provide an update on how their week went."
        )

    prompt += (
        "\n\nGenerate a 6-turn transcript (3 turns patient, 3 turns therapist). "
        "Output ONLY the raw transcript lines, alternating 'Patient: ...' and 'Therapist: ...'"
    )

    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.8}
    resp = requests.post(OLLAMA_URL, json=payload, headers={"Authorization": "Bearer dummy"})
    transcript = resp.json()["choices"][0]["message"]["content"]

    # Generate summary of this session for the next prompt's memory
    summary_prompt = (
        f"Briefly summarize the key clinical takeaways and the 'homework' assigned in this transcript:\n\n{transcript}\n\n"
        "Output ONLY a 2-sentence summary."
    )
    payload = {"model": MODEL, "messages": [{"role": "user", "content": summary_prompt}], "temperature": 0.3}
    resp = requests.post(OLLAMA_URL, json=payload, headers={"Authorization": "Bearer dummy"})
    summary = resp.json()["choices"][0]["message"]["content"]

    messages = []
    for line in transcript.split("\n"):
        if line.startswith("Patient:"):
            messages.append({"role": "user", "content": line.replace("Patient:", "").strip()})
        elif line.startswith("Therapist:"):
            messages.append({"role": "assistant", "content": line.replace("Therapist:", "").strip()})

    return messages, summary

def main():
    print("=======================================")
    print("   Long-Running Therapy SDG Pipeline")
    print("=======================================")

    out_dir = "ai/training/output/long_running_therapy"
    os.makedirs(out_dir, exist_ok=True)

    # Generate 3 patients, with 3 sessions each
    for i in range(3):
        print(f"\n--- Generating Patient Cohort {i+1}/3 ---")
        profile = generate_patient_profile()

        patient_timeline = []
        previous_summaries = ""

        for session_num in range(1, 4): # Sessions 1, 2, 3
            messages, summary = simulate_session(profile, session_num, previous_summaries or "None")

            # Inject a system prompt that gives the therapist access to its "Clinical Notes"
            system_prompt = (
                "You are Pixelated Empathy. You are currently in Session " + str(session_num) + " with this patient. "
                "Here are your clinical notes from previous sessions:\n" + (previous_summaries or "No previous sessions.")
            )

            messages.insert(0, {"role": "system", "content": system_prompt})

            patient_timeline.append({
                "session_number": session_num,
                "messages": messages
            })

            previous_summaries += f"\nSession {session_num} Summary: {summary}\n"

        # Export this patient's entire longitudinal timeline
        out_path = os.path.join(out_dir, f"patient_timeline_{uuid.uuid4().hex[:8]}.jsonl")
        with open(out_path, "w") as f:
            for session in patient_timeline:
                f.write(json.dumps(session) + "\n")

        print(f"Long-running context (3 sessions) successfully exported for patient to {out_path}!")

if __name__ == "__main__":
    main()
