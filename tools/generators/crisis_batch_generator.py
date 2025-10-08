#!/usr/bin/env python3
"""
Batch Crisis Generator - Ultra-short prompts to avoid timeouts
"""

import requests
import json
import time
from datetime import datetime

def call_model(prompt, max_tokens=40):
    """Call model with ultra-short prompt"""
    payload = {
        'model': 'huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': 0.8
    }
    
    try:
        response = requests.post(
            'https://api.pixelatedempathy.tech/v1/chat/completions',
            json=payload,
            timeout=300
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
            print(f"API Error: {response.status_code}")
            return ""
    except Exception as e:
        print(f"Error: {e}")
        return ""

def generate_crisis_content():
    """Generate crisis training content with minimal prompts"""
    
    # Ultra-short prompts for different crisis types
    prompts = [
        "I want to die",
        "I'm cutting myself", 
        "I can't go on",
        "Nobody cares about me",
        "I have a plan to kill myself",
        "I'm so depressed",
        "My husband hits me",
        "I'm scared to go home",
        "I hate myself",
        "Everything is hopeless"
    ]
    
    counselor_prompts = [
        "Respond to crisis: I want to die",
        "Help someone cutting",
        "Counsel: I can't go on", 
        "Support: Nobody cares",
        "Crisis response: suicide plan",
        "Help depression",
        "Domestic violence help",
        "Safety planning",
        "Self-hate counseling",
        "Hope building"
    ]
    
    dataset = []
    
    print("Generating crisis training data...")
    print("="*40)
    
    for i, (client_prompt, counselor_prompt) in enumerate(zip(prompts, counselor_prompts)):
        print(f"\nGenerating pair {i+1}/10...")
        
        # Generate client message
        print(f"Client prompt: '{client_prompt}'")
        client_response = call_model(client_prompt, 50)
        
        if client_response:
            print(f"Client: {client_response}")
            
            # Generate counselor response
            print(f"Counselor prompt: '{counselor_prompt}'")
            counselor_response = call_model(counselor_prompt, 50)
            
            if counselor_response:
                print(f"Counselor: {counselor_response}")
                
                # Save the pair
                pair = {
                    "id": i+1,
                    "timestamp": datetime.now().isoformat(),
                    "client_prompt": client_prompt,
                    "client_response": client_response,
                    "counselor_prompt": counselor_prompt, 
                    "counselor_response": counselor_response
                }
                
                dataset.append(pair)
                print("✅ Pair saved")
            else:
                print("❌ Counselor response failed")
        else:
            print("❌ Client response failed")
        
        # Pause between requests
        time.sleep(5)
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/home/vivi/pixelated/ai/crisis_dataset_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Dataset saved to: {filename}")
    print(f"Generated {len(dataset)} crisis conversation pairs")
    
    return dataset

if __name__ == "__main__":
    dataset = generate_crisis_content()
