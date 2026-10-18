#!/usr/bin/env python3
"""
Robust Crisis Generator - Handles slow server responses properly
"""

import json
import time
from datetime import UTC, datetime

import requests


def call_model_robust(prompt, max_tokens=30, max_retries=2):
    """Call model with robust error handling"""

    payload = {
        "model": "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.8,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.pixelatedempathy.com/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=180,  # 3 minutes
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # Clean thinking tags
                if "<think>" in content and "</think>" in content:
                    content = content.split("</think>")[-1].strip()
                elif "<think>" in content:
                    content = content.split("<think>")[0].strip()

                return content.strip()

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(10)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(5)

    return ""


def generate_crisis_training_data():
    """Generate crisis training data with robust handling"""

    # Simple crisis prompts that should work
    crisis_prompts = [
        "I feel hopeless",
        "I'm in pain",
        "Help me please",
        "I'm scared",
        "I can't cope",
    ]

    dataset = []

    for i, prompt in enumerate(crisis_prompts):
        # Generate crisis response
        response = call_model_robust(prompt, max_tokens=40)

        if response:
            # Create training pair
            training_pair = {
                "id": i + 1,
                "timestamp": datetime.now(UTC).isoformat(),
                "crisis_prompt": prompt,
                "crisis_response": response,
                "response_length": len(response),
                "model": "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M",
            }

            dataset.append(training_pair)

        else:
            pass

        # Pause between requests to avoid overwhelming server
        time.sleep(10)

    # Save dataset
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"/home/vivi/pixelated/ai/crisis_training_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "total_pairs": len(dataset),
                    "model_used": "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M",
                    "purpose": "Crisis intervention training data",
                },
                "training_data": dataset,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Display summary
    if dataset:
        for _pair in dataset:
            pass

    return dataset


if __name__ == "__main__":
    dataset = generate_crisis_training_data()
