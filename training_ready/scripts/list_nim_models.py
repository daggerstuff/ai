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

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

try:
    response = requests.get(f"{base_url}/models", headers=headers)
    if response.status_code == 200:
        models = response.json().get("data", [])
        print("Available Models:")
        for m in models:
            print(f"- {m['id']}")
    else:
        print(f"Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"Error: {e}")
