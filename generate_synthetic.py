import json
import re
import os
import requests
import argparse
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm
import time
import sys
import importlib.util

# Set up environment
load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "artifish/llama3.2-uncensored")

def log(msg, log_file=None):
    print(msg)
    if log_file:
        log_file.write(msg + '\n')
        log_file.flush()

def slack_alert(message, args, slack_available, log_file=None):
    if args.slack_webhook and slack_available:
        try:
            from slack_sdk.webhook import WebhookClient
            webhook = WebhookClient(args.slack_webhook)
            webhook.send(text=message)
        except Exception as e:
            log(f"[SLACK ERROR] {e}", log_file)
    elif args.slack_webhook:
        log("[SLACK ERROR] slack_sdk not installed. Run 'pip install slack_sdk'.", log_file)

def extract_prompts(md_path, log_file=None):
    prompts = []
    prompt_ids = []
    if not os.path.exists(md_path):
        log(f"[FATAL ERROR] Prompt file '{md_path}' not found.", log_file)
        sys.exit(1)
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    current_prompt = None
    current_id = None
    collecting = False
    for line in lines:
        if match := re.match(r"Prompt (\d+): (.+)", line):
            if current_prompt is not None and current_id is not None:
                prompts.append(current_prompt.strip())
                prompt_ids.append(current_id)
            current_prompt = match[2].strip()
            current_id = match[1].zfill(2)
            collecting = True
        elif collecting:
            if line.strip() == "---" or line.strip().startswith(tuple("ABCDEF")):
                if current_prompt is not None and current_id is not None:
                    prompts.append(current_prompt.strip())
                    prompt_ids.append(current_id)
                    current_prompt = None
                    current_id = None
                collecting = False
            elif line.strip():
                if current_prompt is None:
                    current_prompt = ""
                current_prompt += f" {line.strip()}"
    if current_prompt is not None and current_id is not None:
        prompts.append(current_prompt.strip())
        prompt_ids.append(current_id)
    return prompt_ids, prompts

def extract_prompts_jsonl(jsonl_path, log_file=None):
    prompts = []
    prompt_ids = []
    if not os.path.exists(jsonl_path):
        log(f"[FATAL ERROR] Prompt file '{jsonl_path}' not found.", log_file)
        sys.exit(1)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # Use 'prompt_id' and 'instructions' fields (or fallback to 'prompt')
                prompt_id = data.get('prompt_id') or data.get('id')
                prompt = data.get('instructions') or data.get('prompt')
                if prompt_id and prompt:
                    prompt_ids.append(prompt_id)
                    prompts.append(prompt)
            except Exception as e:
                log(f"[ERROR] Failed to parse line in {jsonl_path}: {e}", log_file)
    return prompt_ids, prompts

def load_templates(path):
    if not path:
        return [{"id": "default", "template": "{scenario}"}]
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def apply_template(template, scenario):
    return template["template"].replace("{scenario}", scenario)

def load_chain_templates(path, chain_type):
    if not path:
        if chain_type == 'supervisor_critique':
            return [{"id": "supervisor_critique", "template": "As a clinical supervisor, critique the therapist's response to the client's threats in the following dialogue. Be detailed and specific.\n\n{dialogue}"}]
        elif chain_type == 'session_note':
            return [{"id": "session_note", "template": "Write a professional session note based on the following therapy dialogue.\n\n{dialogue}"}]
        else:
            return [{"id": chain_type, "template": f"{chain_type}: {{dialogue}}"}]
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def apply_chain_template(template, dialogue):
    return template["template"].replace("{dialogue}", dialogue)

def generate_with_ollama(prompt, model, max_retries=3, timeout=120, log_file=None):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            log(f"[ERROR] Ollama API call failed (attempt {attempt}/{max_retries}): {e}", log_file)
            if attempt == max_retries:
                return ""
            time.sleep(2 * attempt)

def parse_dialogue(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    pairs = []
    last_speaker = None
    last_line = ""
    for line in lines:
        if therapist_match := re.match(r"(?:\*\*|__)?Therapist(?:\*\*|__)?[:\-\s]*(.+)", line, re.IGNORECASE):
            if last_speaker == "client" and last_line:
                pairs.append({"prompt": last_line, "response": therapist_match[1].strip()})
            last_speaker = "therapist"
            last_line = therapist_match[1].strip()
        elif client_match := re.match(r"(?:\*\*|__)?Client(?:\*\*|__)?[:\-\s]*(.+)", line, re.IGNORECASE):
            if last_speaker == "therapist" and last_line:
                pairs.append({"prompt": last_line, "response": client_match[1].strip()})
            last_speaker = "client"
            last_line = client_match[1].strip()
    return pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic therapy dialogues using Ollama.")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Ollama model name')
    parser.add_argument('--output', type=str, default=None, help='Output file name (default: auto-named)')
    parser.add_argument('--raw_output', type=str, default=None, help='Raw model output file (default: auto-named)')
    parser.add_argument('--max_retries', type=int, default=3, help='Max retries for API calls')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout for API calls (seconds)')
    parser.add_argument('--log', type=str, default=None, help='Log file name (default: auto-named)')
    parser.add_argument('--templates', type=str, default=None, help='Prompt templates JSON file (optional)')
    parser.add_argument('--chain', action='store_true', help='Enable prompt chaining (e.g., supervisor critique)')
    parser.add_argument('--chain_type', type=str, default='supervisor_critique', help='Type of chaining prompt (e.g., supervisor_critique, session_note)')
    parser.add_argument('--chain_templates', type=str, default=None, help='Chaining templates JSON file (optional)')
    parser.add_argument('--slack_webhook', type=str, default=None, help='Slack webhook URL for alerts (optional)')
    args = parser.parse_args()

    # Use importlib to check for slack_sdk.webhook
    slack_available = False
    if args.slack_webhook:
        slack_available = bool(importlib.util.find_spec("slack_sdk.webhook"))
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_safe = args.model.replace('/', '_').replace(':', '_')
    output_path = args.output or f"synthetic_therapy_dialogues_{model_safe}_{now_str}.jsonl"
    raw_output_path = args.raw_output or f"raw_model_outputs_{model_safe}_{now_str}.jsonl"
    log_path = args.log or f"generation_log_{model_safe}_{now_str}.log"
    chained_output_path = f"chained_outputs_{model_safe}_{now_str}.jsonl" if args.chain else None

    # Ensure output directory exists
    for path in [output_path, raw_output_path, log_path, chained_output_path]:
        if path:
            dir_name = os.path.dirname(path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)

    with open(log_path, 'w', encoding='utf-8') as log_file:
        try:
            # Use the new JSONL extraction function
            prompt_ids, prompts = extract_prompts_jsonl("ai/edge_case_prompts_improved.jsonl", log_file)
            with open("ai/edge_case_prompts_improved.jsonl.json", "w", encoding="utf-8") as f:
                json.dump([{ 'id': pid, 'prompt': p } for pid, p in zip(prompt_ids, prompts)], f, ensure_ascii=False, indent=2)

            templates = load_templates(args.templates)
            chain_templates = load_chain_templates(args.chain_templates, args.chain_type) if args.chain else None

            with open(output_path, "w", encoding="utf-8") as outfile, \
                 open(raw_output_path, "w", encoding="utf-8") as rawfile:
                chainfile = None
                if args.chain and chained_output_path:
                    chainfile = open(chained_output_path, "w", encoding="utf-8")
                total_pairs = 0
                total_chained = 0
                for prompt_id, scenario in tqdm(zip(prompt_ids, prompts), total=len(prompts), desc="Generating scenarios"):
                    for template in templates:
                        template_id = template["id"]
                        prompt_text = apply_template(template, scenario)
                        log(f"Generating for scenario {prompt_id} (template {template_id}): {prompt_text[:60]}...", log_file)
                        session_text = generate_with_ollama(prompt_text, args.model, args.max_retries, args.timeout, log_file)
                        rawfile.write(json.dumps({"prompt_id": prompt_id, "template_id": template_id, "scenario": scenario, "prompt_text": prompt_text, "raw_output": session_text}) + "\n")
                        pairs = parse_dialogue(session_text)
                        pairs = [pair for pair in pairs if pair["response"].strip()]
                        if not pairs:
                            log(f"[WARNING] No pairs extracted for scenario {prompt_id} (template {template_id}): {prompt_text[:60]}", log_file)
                        for pair in pairs:
                            pair["scenario_type"] = "edge_case"
                            pair["source"] = "synthetic"
                            pair["prompt_id"] = prompt_id
                            pair["template_id"] = template_id
                            pair["scenario"] = scenario
                            pair["prompt_text"] = prompt_text
                            outfile.write(json.dumps(pair, ensure_ascii=False) + "\n")
                            total_pairs += 1
                        if args.chain and pairs and chainfile and chain_templates:
                            dialogue = "\n".join([f"Therapist: {p['prompt']}\nClient: {p['response']}" for p in pairs])
                            for chain_template in chain_templates:
                                chain_template_id = chain_template["id"]
                                chain_prompt = apply_chain_template(chain_template, dialogue)
                                log(f"Chaining for scenario {prompt_id} (template {template_id}, chain {chain_template_id})...", log_file)
                                chain_output = generate_with_ollama(chain_prompt, args.model, args.max_retries, args.timeout, log_file)
                                chainfile.write(json.dumps({
                                    "prompt_id": prompt_id,
                                    "template_id": template_id,
                                    "chain_template_id": chain_template_id,
                                    "scenario": scenario,
                                    "prompt_text": prompt_text,
                                    "dialogue": dialogue,
                                    "chain_prompt": chain_prompt,
                                    "chain_output": chain_output
                                }) + "\n")
                                total_chained += 1
                log(f"Total Q&A pairs generated: {total_pairs}", log_file)
                if args.chain:
                    log(f"Total chained outputs generated: {total_chained}", log_file)
                if chainfile:
                    chainfile.close()
            print(f"\nDone! Synthetic data saved to {output_path}")
            print(f"Raw model outputs saved to {raw_output_path}")
            if args.chain:
                print(f"Chained outputs saved to {chained_output_path}")
            print(f"Log written to {log_path}")
            print(f"Total Q&A pairs generated: {total_pairs}")
            if args.chain:
                print(f"Total chained outputs generated: {total_chained}")
            if args.slack_webhook:
                msg = f"Synthetic generation complete. {total_pairs} Q&A pairs. Model: {args.model}. Output: {output_path}"
                if args.chain:
                    msg += f" | Chained: {total_chained} ({chained_output_path})"
                slack_alert(msg, args, slack_available, log_file)
            print("\nTo view results, run: streamlit run ai/synthetic_dashboard.py -- --data", output_path, "--log", log_path)
        except Exception as e:
            log(f"[FATAL ERROR] {e}", log_file)
            if args.slack_webhook:
                slack_alert(f"[ERROR] Synthetic generation failed: {e}", args, slack_available, log_file)
            sys.exit(1)