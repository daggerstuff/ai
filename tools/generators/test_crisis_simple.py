#!/usr/bin/env python3
"""
Simple test of crisis conversation generation using the working OpenAI-compatible endpoint
"""

import json
import time
import unittest
from datetime import datetime, timezone

import requests


class TestModule(unittest.TestCase):
    def test_crisis_generation(self):
        """Test basic crisis conversation generation"""
        
        api_url = "https://api.pixelatedempathy.com/v1/chat/completions"
        model_name = "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M"
        
        # Simple test prompt first
        crisis_prompt = "Generate a short message from someone in crisis seeking help."
    
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": crisis_prompt}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        print("Testing crisis conversation generation...")
        print("=" * 50)
        
        try:
            response = requests.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300  # 5 minutes
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Success! Response: {result['choices'][0]['message']['content']}")
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Exception: {e}")

if __name__ == "__main__":
    unittest.main()
