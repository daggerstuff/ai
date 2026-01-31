#!/usr/bin/env python3
"""
Nemotron 3 Nano Evaluation Harness

Reads an evaluation split from S3, sends prompts to a Nemotron 3 Nano
deployment exposed via an OpenAI-compatible /v1/chat/completions API,
and writes the Nemotron 3 responses back to S3 as JSONL.

This script is intentionally narrow:
- It does NOT train or fine-tune any models.
- It treats Nemotron 3 purely as an external teacher/baseline.
- It assumes evaluation datasets are already in the canonical S3 structure.

Usage (from ai/training_ready/):

  python scripts/nemotron3_evaluate.py \\
    --dataset-name clinical_diagnosis_mental_health_eval.jsonl \\
    --category cot_reasoning \\
    --output-name nemotron3_nano_eval_run1.jsonl

Environment:
- S3 credentials: same as other training_ready scripts (.env / env vars)
- Nemotron endpoint (OpenAI-compatible):
    NEMOTRON3_BASE_URL  (default: http://localhost:8000/v1)
    NEMOTRON3_API_KEY   (optional, if your server requires auth)
    NEMOTRON3_MODEL     (e.g. nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Add project root to path
# Script is at: ai/training_ready/scripts/nemotron3_evaluate.py
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]  # scripts -> training_ready -> ai -> project_root
sys.path.insert(0, str(project_root))

from ai.training.ready_packages.utils.s3_dataset_loader import (  # noqa: E402
    S3DatasetLoader,
    load_dataset_from_s3,
)


def _build_openai_payload(
    sample: dict[str, Any], model: str, max_new_tokens: int, temperature: float
) -> dict[str, Any]:
    """
    Build an OpenAI-compatible chat.completions payload from a dataset sample.

    Expected sample schema (one of):
      - {"messages": [...]}  # preferred
      - {"conversations": [...]}  # fallback
      - {"prompt": "..."}  # as last resort, wrapped as a single user message
    """
    messages = sample.get("messages") or sample.get("conversations")

    if not messages:
        prompt = sample.get("prompt") or sample.get("input") or ""
        messages = [{"role": "user", "content": str(prompt)}]

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }


def _call_nemotron3(
    base_url: str,
    api_key: str | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Call Nemotron 3 Nano via an OpenAI-compatible /v1/chat/completions API.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(url, data=data, headers=headers, method="POST")

    try:
        with urlopen(request) as resp:  # nosec B310
            resp_body = resp.read()
            return json.loads(resp_body.decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Nemotron 3 request failed ({e.code}): {body}") from e
    except URLError as e:
        raise RuntimeError(f"Nemotron 3 endpoint unreachable: {e}") from e


def _extract_assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the assistant message from an OpenAI-style response.
    """
    choices = response.get("choices") or []
    if not choices:
        return {"role": "assistant", "content": "[no response]"}

    message = choices[0].get("message") or {}
    role = message.get("role", "assistant")
    content = message.get("content", "")
    return {"role": role, "content": content}


def run_evaluation(
    dataset_name: str,
    category: str | None,
    output_name: str,
    bucket: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """
    Run Nemotron 3 Nano on the given dataset and write results back to S3.

    Returns the S3 path to the created evaluation file.
    """
    base_url = os.getenv("NEMOTRON3_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("NEMOTRON3_API_KEY")
    model = os.getenv("NEMOTRON3_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")

    print("📂 Loading evaluation dataset from S3")
    dataset = load_dataset_from_s3(dataset_name=dataset_name, category=category, bucket=bucket)

    # Support both {"conversations": [...]} and plain list formats
    if isinstance(dataset, dict) and "conversations" in dataset:
        samples = dataset["conversations"]
    elif isinstance(dataset, list):
        samples = dataset
    else:
        raise ValueError(
            "Unsupported dataset format. Expected {'conversations': [...]} or a list of samples."
        )

    print(f"   ✅ Loaded {len(samples)} samples")

    loader = S3DatasetLoader(bucket=bucket)
    out_key = f"nemotron3/eval/{output_name}"
    s3_path = f"s3://{loader.bucket}/{out_key}"

    print("\n🧠 Querying Nemotron 3 Nano")
    print(f"   🌐 Base URL: {base_url}")
    print(f"   🧬 Model: {model}")
    print(f"   📦 Output: {s3_path}")

    lines: list[str] = []
    for idx, sample in enumerate(samples, start=1):
        payload = _build_openai_payload(
            sample, model=model, max_new_tokens=max_new_tokens, temperature=temperature
        )
        response = _call_nemotron3(base_url=base_url, api_key=api_key, payload=payload)
        assistant_message = _extract_assistant_message(response)

        record = {
            "source": "nemotron3_nano",
            "model": model,
            "sample_index": idx - 1,
            "input": sample,
            "nemotron3_response": assistant_message,
            "raw_response": response,
        }
        lines.append(json.dumps(record, ensure_ascii=False))

        if idx % 10 == 0 or idx == len(samples):
            print(f"   ... processed {idx}/{len(samples)} samples", flush=True)

    body = ("\n".join(lines) + "\n").encode("utf-8")

    print("\n☁️ Uploading evaluation results to S3")
    loader.s3_client.put_object(Bucket=loader.bucket, Key=out_key, Body=body)
    print(f"   ✅ Uploaded to {s3_path}")

    return s3_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Nemotron 3 Nano evaluation on an S3 dataset.")
    parser.add_argument(
        "--dataset-name", required=True, help="Dataset file name in S3 (e.g. my_eval.jsonl)"
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Optional dataset category (e.g. cot_reasoning) for canonical S3 path resolution",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Output file name under nemotron3/eval/ (e.g. nemotron3_run1.jsonl)",
    )
    parser.add_argument(
        "--bucket",
        default="pixel-data",
        help="S3 bucket name (default: pixel-data)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate per sample",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for Nemotron 3 Nano",
    )

    args = parser.parse_args()

    try:
        s3_path = run_evaluation(
            dataset_name=args.dataset_name,
            category=args.category,
            output_name=args.output_name,
            bucket=args.bucket,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"\n✅ Nemotron 3 evaluation complete: {s3_path}")
        print(
            "💡 You can now use this JSONL for distillation or comparison against your own model."
        )
        return 0
    except Exception as exc:
        print(f"\n❌ Nemotron 3 evaluation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
