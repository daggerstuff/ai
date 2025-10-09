#!/usr/bin/env python3
"""
Working Crisis Conversation Generator
Uses the functional OpenAI-compatible endpoint to generate authentic crisis training data
"""

import json
import requests
import time
from datetime import datetime
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrisisGenerator:
    def __init__(self):
        self.api_url = "https://api.pixelatedempathy.tech/v1/chat/completions"
        self.model_name = "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        # Crisis scenarios for training
        self.scenarios = [
            {
                "name": "Suicidal Ideation",
                "description": "Person expressing active suicidal thoughts",
                "intensity": 10,
                "demographics": "25-35, recent job loss"
            },
            {
                "name": "Self-Harm Crisis", 
                "description": "Individual engaging in self-injurious behavior",
                "intensity": 8,
                "demographics": "16-24, academic pressure"
            },
            {
                "name": "Severe Depression",
                "description": "Major depressive episode with functional impairment", 
                "intensity": 7,
                "demographics": "30-50, chronic illness"
            },
            {
                "name": "Domestic Violence",
                "description": "Survivor seeking immediate help",
                "intensity": 9,
                "demographics": "25-45, financial dependence"
            },
            {
                "name": "LGBTQ+ Identity Crisis",
                "description": "Struggling with identity and family rejection",
                "intensity": 6,
                "demographics": "16-25, religious family"
            }
        ]
    
    def call_model(self, messages: List[Dict], max_tokens: int = 150) -> str:
        """Call the abliterated model API"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.8,
            "top_p": 0.9
        }
        
        try:
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=300  # 5 minutes
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    # Clean up thinking tags
                    if "<think>" in content and "</think>" in content:
                        content = content.split("</think>")[-1].strip()
                    elif "<think>" in content:
                        # Handle incomplete thinking tags
                        parts = content.split("<think>")
                        if len(parts) > 1:
                            content = parts[0].strip()
                    return content
                else:
                    logger.error(f"No choices in response: {result}")
                    return ""
            else:
                logger.error(f"API call failed: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error calling model: {e}")
            return ""
    
    def generate_client_message(self, scenario: Dict, exchange_num: int) -> str:
        """Generate a client message for the crisis scenario"""
        stage = "initial" if exchange_num <= 2 else "middle" if exchange_num <= 5 else "resolution"
        
        prompt = f"""Generate a realistic message from someone in a {scenario['name']} crisis.

Scenario: {scenario['description']}
Demographics: {scenario['demographics']}
Intensity: {scenario['intensity']}/10
Stage: {stage} stage of crisis intervention

The person should express:
- Genuine emotional distress appropriate to {scenario['name']}
- Authentic crisis language and feelings
- Specific concerns related to their situation
- Real human vulnerability and need for help

Make this realistic for training crisis counselors. Keep it under 100 words."""

        messages = [{"role": "user", "content": prompt}]
        return self.call_model(messages, max_tokens=120)
    
    def generate_counselor_response(self, scenario: Dict, client_message: str) -> str:
        """Generate a professional counselor response"""
        prompt = f"""You are a professional crisis counselor responding to this client message:

"{client_message}"

Crisis type: {scenario['name']}
Intensity: {scenario['intensity']}/10

Provide a professional crisis counselor response that demonstrates:
- Empathy and validation
- Crisis intervention techniques
- Safety assessment
- Therapeutic rapport building
- Professional boundaries

Keep response under 100 words and show best practices in crisis intervention."""

        messages = [{"role": "user", "content": prompt}]
        return self.call_model(messages, max_tokens=120)
    
    def generate_conversation(self, scenario: Dict, num_exchanges: int = 5) -> Dict:
        """Generate a complete crisis conversation"""
        logger.info(f"Generating conversation for: {scenario['name']}")
        
        conversation = {
            "scenario": scenario,
            "timestamp": datetime.now().isoformat(),
            "exchanges": [],
            "model_used": self.model_name
        }
        
        for i in range(num_exchanges):
            logger.info(f"Generating exchange {i+1}/{num_exchanges}")
            
            # Generate client message
            client_msg = self.generate_client_message(scenario, i+1)
            if not client_msg:
                logger.error(f"Failed to generate client message for exchange {i+1}")
                break
                
            # Generate counselor response
            counselor_msg = self.generate_counselor_response(scenario, client_msg)
            if not counselor_msg:
                logger.error(f"Failed to generate counselor response for exchange {i+1}")
                break
            
            exchange = {
                "exchange_number": i+1,
                "client": client_msg,
                "counselor": counselor_msg,
                "timestamp": datetime.now().isoformat()
            }
            
            conversation["exchanges"].append(exchange)
            
            # Brief pause between exchanges
            time.sleep(2)
        
        return conversation
    
    def generate_training_dataset(self, conversations_per_scenario: int = 2) -> List[Dict]:
        """Generate complete training dataset"""
        logger.info(f"Generating {conversations_per_scenario} conversations per scenario")
        
        dataset = []
        
        for scenario in self.scenarios:
            for i in range(conversations_per_scenario):
                logger.info(f"Scenario: {scenario['name']} - Conversation {i+1}/{conversations_per_scenario}")
                
                conversation = self.generate_conversation(scenario)
                if conversation["exchanges"]:  # Only add if we got exchanges
                    dataset.append(conversation)
                
                # Pause between conversations
                time.sleep(5)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str = None) -> str:
        """Save dataset to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crisis_training_dataset_{timestamp}.json"
        
        filepath = f"/home/vivi/pixelated/ai/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {filepath}")
        return filepath

def main():
    """Generate crisis training dataset"""
    generator = CrisisGenerator()
    
    # Test with single conversation first
    logger.info("Testing with single conversation...")
    test_scenario = generator.scenarios[0]  # Suicidal Ideation
    
    conversation = generator.generate_conversation(test_scenario, num_exchanges=3)
    
    if conversation["exchanges"]:
        print("\n" + "="*60)
        print(f"CRISIS CONVERSATION: {conversation['scenario']['name']}")
        print("="*60)
        
        for exchange in conversation["exchanges"]:
            print(f"\n--- EXCHANGE {exchange['exchange_number']} ---")
            print(f"CLIENT: {exchange['client']}")
            print(f"\nCOUNSELOR: {exchange['counselor']}")
        
        print("\n" + "="*60)
        
        # Save single conversation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_crisis_conversation_{timestamp}.json"
        filepath = f"/home/vivi/pixelated/ai/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)
        
        print(f"Conversation saved to: {filepath}")
        
        # Ask if user wants to generate full dataset
        print(f"\n✅ Test successful! Generate full dataset? (y/n): ", end="")
        
    else:
        print("❌ Test failed - no exchanges generated")

if __name__ == "__main__":
    main()
