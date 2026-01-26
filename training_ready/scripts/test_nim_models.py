import os

import requests
from dotenv import load_dotenv

load_dotenv()

api_key = (
    os.getenv("NIM_API_KEY")
    or os.getenv("NVIDIA_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
base_url = os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")

models_to_test = [
    "meta/llama-4-maverick-17b-128e-instruct",
    "meta/llama-4-scout-17b-16e-instruct",
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.3-70b-instruct",
]

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

for model in models_to_test:
    print(f"Testing {model}...")
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
            },
            timeout=30,
        )
        if response.status_code == 200:
            print(f"✅ {model} SUCCESS")
        else:
            print(f"❌ {model} FAILED: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ {model} ERROR: {e}")
