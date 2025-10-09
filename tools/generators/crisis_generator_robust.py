#!/usr/bin/env python3
"""
Robust Crisis Generator - Handles slow server responses properly
"""

import requests
import json
import time
from datetime import datetime

def call_model_robust(prompt, max_tokens=30, max_retries=2):
    """Call model with robust error handling"""
    
    payload = {
        'model': 'huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': 0.8
    }
    
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}...")
            
            response = requests.post(
                'https://api.pixelatedempathy.tech/v1/chat/completions',
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=180  # 3 minutes
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Clean thinking tags
                if '<think>' in content and '</think>' in content:
                    content = content.split('</think>')[-1].strip()
                elif '<think>' in content:
                    content = content.split('<think>')[0].strip()
                
                return content.strip()
            else:
                print(f"  API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"  Timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                print("  Retrying...")
                time.sleep(10)
        except Exception as e:
            print(f"  Error: {e}")
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
        "I can't cope"
    ]
    
    dataset = []
    
    print("🚀 Starting Crisis Training Data Generation")
    print("=" * 50)
    
    for i, prompt in enumerate(crisis_prompts):
        print(f"\n📝 Generating crisis response {i+1}/5")
        print(f"Prompt: '{prompt}'")
        
        # Generate crisis response
        response = call_model_robust(prompt, max_tokens=40)
        
        if response:
            print(f"✅ Response: {response}")
            
            # Create training pair
            training_pair = {
                "id": i + 1,
                "timestamp": datetime.now().isoformat(),
                "crisis_prompt": prompt,
                "crisis_response": response,
                "response_length": len(response),
                "model": "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M"
            }
            
            dataset.append(training_pair)
            
        else:
            print("❌ Failed to generate response")
        
        # Pause between requests to avoid overwhelming server
        print("⏳ Waiting 10 seconds...")
        time.sleep(10)
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/home/vivi/pixelated/ai/crisis_training_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_pairs": len(dataset),
                "model_used": "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M",
                "purpose": "Crisis intervention training data"
            },
            "training_data": dataset
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 SUCCESS!")
    print(f"📁 Dataset saved to: {filename}")
    print(f"📊 Generated {len(dataset)} training pairs")
    
    # Display summary
    if dataset:
        print(f"\n📋 SUMMARY:")
        for pair in dataset:
            print(f"  {pair['id']}: '{pair['crisis_prompt']}' -> {len(pair['crisis_response'])} chars")
    
    return dataset

if __name__ == "__main__":
    dataset = generate_crisis_training_data()
