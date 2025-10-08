from ai.inference
from ai.pixel
from ai.dataset_pipeline
from .\1 import
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import pytest
#!/usr/bin/env python3
"""
Simple test of crisis conversation generation using the working OpenAI-compatible endpoint
"""

import json
import requests
import time
from .datetime import datetime


class TestModule(unittest.TestCase):
    def test_crisis_generation():
        """Test basic crisis conversation generation"""
        
        api_url = "https://api.pixelatedempathy.tech/v1/chat/completions"
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
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    
                    # Clean up thinking tags if present
                    if "<think>" in content:
                        content = content.split("</think>")[-1].strip()
                    
                    print("GENERATED CRISIS MESSAGE:")
                    print("-" * 30)
                    print(content)
                    print("-" * 30)
                    print(f"Response time: {response.elapsed.total_seconds():.2f} seconds")
                    print(f"Model used: {result.get('model', 'unknown')}")
                    
                    # Save to file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"crisis_test_{timestamp}.json"
                    
                    test_result = {
                        "timestamp": datetime.now().isoformat(),
                        "scenario": "Acute Suicidal Ideation Test",
                        "prompt": crisis_prompt,
                        "response": content,
                        "response_time_seconds": response.elapsed.total_seconds(),
                        "model": result.get('model', 'unknown')
                    }
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(test_result, f, indent=2, ensure_ascii=False)
                    
                    print(f"Test result saved to: {filename}")
                    return True
                else:
                    print(f"No choices in response: {result}")
                    return False
            else:
                print(f"API call failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Error during test: {e}")
            return False
    
if __name__ == "__main__":
    success = test_crisis_generation()
    if success:
        print("\n✅ Crisis generation test successful!")
        print("The abliterated model is working and can generate crisis training data.")
    else:
        print("\n❌ Crisis generation test failed!")
