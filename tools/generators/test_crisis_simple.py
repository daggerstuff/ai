#!/usr/bin/env python3
"""
Simple test of crisis conversation generation using the working OpenAI-compatible endpoint
"""

import json
import logging
from datetime import datetime

import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_crisis_generation():
    """Test basic crisis conversation generation"""

    api_url = "https://api.pixelatedempathy.com/v1/chat/completions"
    model_name = "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M"

    # Simple test prompt first
    crisis_prompt = "Generate a short message from someone in crisis seeking help."

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": crisis_prompt}],
        "max_tokens": 50,
        "temperature": 0.7,
    }

    logger.info("Testing crisis conversation generation...")
    logger.info("=" * 50)

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300,  # 5 minutes
        )

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]

                # Clean up thinking tags if present
                if "<think>" in content:
                    content = content.split("</think>")[-1].strip()

                logger.info("GENERATED CRISIS MESSAGE:")
                logger.info("-" * 30)
                logger.info(content)
                logger.info("-" * 30)
                logger.info(f"Response time: {response.elapsed.total_seconds():.2f} seconds")
                logger.info(f"Model used: {result.get('model', 'unknown')}")

                # Save to file
                now = datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                filename = f"crisis_test_{timestamp}.json"

                test_result = {
                    "timestamp": now.isoformat(),
                    "scenario": "Acute Suicidal Ideation Test",
                    "prompt": crisis_prompt,
                    "response": content,
                    "response_time_seconds": response.elapsed.total_seconds(),
                    "model": result.get("model", "unknown"),
                }

                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(test_result, f, indent=2, ensure_ascii=False)

                logger.info(f"Test result saved to: {filename}")
                return True
            else:
                logger.error(f"No choices in response: {result}")
                return False
        else:
            logger.error(f"API call failed: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error during test: {e}")
        return False


if __name__ == "__main__":
    success = test_crisis_generation()
    if success:
        logger.info("\n✅ Crisis generation test successful!")
        logger.info("The abliterated model is working and can generate crisis training data.")
    else:
        logger.error("\n❌ Crisis generation test failed!")
        exit(1)
