#!/usr/bin/env python3
"""
Simplified Crisis Generator - Uses minimal prompts to avoid timeouts
"""

import json
import requests
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleCrisisGenerator:
    def __init__(self):
        self.api_url = "https://api.pixelatedempathy.tech/v1/chat/completions"
        self.model_name = "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M"
    
    def call_model(self, prompt: str, max_tokens: int = 80) -> str:
        """Call model with simple prompt"""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.8
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300
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
            else:
                logger.error(f"API failed: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Error: {e}")
            return ""
    
    def generate_crisis_message(self, crisis_type: str) -> str:
        """Generate simple crisis message"""
        prompt = f"Write a short message from someone having a {crisis_type} crisis seeking help."
        return self.call_model(prompt, 60)
    
    def generate_counselor_response(self, client_msg: str) -> str:
        """Generate counselor response"""
        prompt = f"Write a brief professional counselor response to: '{client_msg}'"
        return self.call_model(prompt, 60)
    
    def create_conversation(self, crisis_type: str) -> dict:
        """Create one crisis conversation"""
        logger.info(f"Creating {crisis_type} conversation")
        
        conversation = {
            "crisis_type": crisis_type,
            "timestamp": datetime.now().isoformat(),
            "exchanges": []
        }
        
        for i in range(3):  # 3 exchanges
            logger.info(f"Exchange {i+1}/3")
            
            # Client message
            client_msg = self.generate_crisis_message(crisis_type)
            if not client_msg:
                logger.error(f"Failed client message {i+1}")
                break
            
            # Counselor response  
            counselor_msg = self.generate_counselor_response(client_msg)
            if not counselor_msg:
                logger.error(f"Failed counselor response {i+1}")
                break
            
            exchange = {
                "number": i+1,
                "client": client_msg,
                "counselor": counselor_msg
            }
            
            conversation["exchanges"].append(exchange)
            time.sleep(3)  # Brief pause
        
        return conversation

def main():
    generator = SimpleCrisisGenerator()
    
    # Test with suicidal crisis
    conversation = generator.create_conversation("suicidal")
    
    if conversation["exchanges"]:
        print("\n" + "="*50)
        print("CRISIS CONVERSATION GENERATED")
        print("="*50)
        
        for ex in conversation["exchanges"]:
            print(f"\n--- EXCHANGE {ex['number']} ---")
            print(f"CLIENT: {ex['client']}")
            print(f"COUNSELOR: {ex['counselor']}")
        
        # Save it
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/home/vivi/pixelated/ai/crisis_conv_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(conversation, f, indent=2)
        
        print(f"\nSaved to: {filename}")
        print("✅ SUCCESS!")
        
    else:
        print("❌ FAILED - No exchanges generated")

if __name__ == "__main__":
    main()
