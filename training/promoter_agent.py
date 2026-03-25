import json

import requests

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "promoter-ai"


def chat_with_promoter():
    print("Promoter AI Agent Initialized.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            payload = {"model": MODEL_NAME, "prompt": user_input, "stream": True}

            print("Promoter AI: ", end="", flush=True)

            response = requests.post(OLLAMA_URL, json=payload, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    body = json.loads(line)
                    print(body.get("response", ""), end="", flush=True)
                    if body.get("done"):
                        print("\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[Error communicating with Ollama: {e}]")
            break


if __name__ == "__main__":
    chat_with_promoter()
