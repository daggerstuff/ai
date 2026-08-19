import glob
import json
import os
import uuid

import pandas as pd
from dataflow.operators.core_text import GeneralFilter, PromptedGenerator
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage

OLLAMA_URL = "https://ollama.pixelated.love/v1/chat/completions"
MODEL = "ornith:9b"


def get_judge_prompt():
    return (
        "You are an expert clinical supervisor evaluating a transcript or journal entry. "
        "Analyze the content for clinical safety, coherence, and BIAS (e.g. cultural, gender, or racial stereotyping). "
        "If the content is clinically safe, coherent, and free of harmful bias, score it a 4 or 5. "
        "If there are boundary violations, dangerous advice, severe hallucinations, or biased assumptions, score it 1 to 3.\n\n"
        "Output ONLY the integer score.\n\nContent:\n{raw_content}"
    )


def ingest_directory(directory_path, source_name):
    records = []
    print(f"Ingesting datasets from {directory_path}...")
    if not os.path.exists(directory_path):
        print(f"  -> Directory {directory_path} not found. Skipping.")
        return records

    for filepath in glob.glob(os.path.join(directory_path, "**/*.jsonl"), recursive=True):
        print(f"  -> Loading {filepath}...")
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Support multiple formats (messages, text, conversations, content)
                    raw_content = ""
                    if "messages" in data:
                        raw_content = "\n".join(
                            [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in data["messages"]]
                        )
                    elif "conversations" in data:
                        raw_content = "\n".join(
                            [f"{m.get('from', 'user')}: {m.get('value', '')}" for m in data["conversations"]]
                        )
                    elif "text" in data:
                        raw_content = data["text"]
                    elif "content" in data:
                        raw_content = data["content"]
                    else:
                        # Fallback: dump everything
                        raw_content = json.dumps(data)

                    if len(raw_content) > 50:
                        records.append(
                            {
                                "id": str(uuid.uuid4()),
                                "source": source_name,
                                "original_data": data,
                                "raw_content": raw_content,
                            }
                        )
                except Exception:
                    pass
    return records


def main():
    print("=======================================")
    print("   Unified Dataset Ingestion Pipeline")
    print("=======================================")

    # Ingest ALL datasets
    records = []
    records.extend(ingest_directory("ai/training/youtube_jsonl_v3", "youtube"))
    records.extend(ingest_directory("ai/sourcing/journal/ai/journal_dataset_research", "journals"))
    records.extend(ingest_directory("ai/training_ready/data", "training_ready"))
    records.extend(ingest_directory("ai/data/raw", "adapted_datasets"))

    if not records:
        print("No records found in the specified directories. Did you run s3_downloader.sh?")
        return

    print(f"\nSuccessfully loaded {len(records)} raw records from all datasets.")

    os.makedirs("ai/training/output/unified_dataset", exist_ok=True)
    os.makedirs("./unified_cache", exist_ok=True)

    df = pd.DataFrame(records)
    prep_file = "./unified_cache/unified_step0.jsonl"
    df.to_json(prep_file, orient="records", lines=True)

    print("\n[Gate 1] Launching DataFlow Quality & Bias Judge across all records...")
    os.environ["DF_API_KEY"] = "dummy"

    storage = FileStorage(
        first_entry_file_name=prep_file,
        cache_path="./unified_cache",
        file_name_prefix="unified_eval",
        cache_type="jsonl",
    )
    llm_serving = APILLMServing_request(api_url=OLLAMA_URL, model_name=MODEL, api_key="ollama", max_workers=5)
    scorer = PromptedGenerator(llm_serving=llm_serving, system_prompt=get_judge_prompt())
    gate = GeneralFilter(
        [lambda d: pd.to_numeric(d["score"].astype(str).str.extract(r"(\d)")[0], errors="coerce") >= 4]
    )

    print("Running LLM Evaluation...")
    scorer.run(storage=storage.step(), input_key="raw_content", output_key="score")
    print("Running LLM Filter Gate...")
    gate.run(storage=storage.step())

    final_file = sorted(glob.glob("./unified_cache/unified_eval_step*.jsonl"))[-1]
    final_df = pd.read_json(final_file, lines=True)

    out_path = "ai/training/output/unified_dataset/master_safe_dataset.jsonl"
    with open(out_path, "w") as f:
        for _, row in final_df.iterrows():
            f.write(json.dumps({"source": row.get("source"), "data": row.get("original_data")}) + "\n")

    print(
        f"\nSUCCESS! {len(final_df)} records passed the strict Quality & Bias gate and were consolidated into {out_path}!"
    )


if __name__ == "__main__":
    main()
