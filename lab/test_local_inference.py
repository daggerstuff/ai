import requests
import json

def test_inference():
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "messages": [
            {"role": "system", "content": "You are a highly empathetic therapeutic AI known as Wayfarer. You specialize in validating emotions and providing deep psychological insights."},
            {"role": "user", "content": "I've been feeling so overwhelmed lately. It feels like I'm doing everything but accomplishing nothing."}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    print("📡 Sending request to local inference server...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print("\n" + "="*60)
        print("🤖 Wayfarer Response:")
        print(result["choices"][0]["message"]["content"])
        print("="*60)
        print(f"\n📊 Usage: {result['usage']}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the server. Make sure inference_server.py is running on port 8000.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_inference()
