import requests


def test_inference():
    url = "http://localhost:8000/v1/chat/completions"

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a highly empathetic therapeutic AI known as Wayfarer. You specialize in validating emotions and providing deep psychological insights.",
            },
            {
                "role": "user",
                "content": "I've been feeling so overwhelmed lately. It feels like I'm doing everything but accomplishing nothing.",
            },
        ],
        "temperature": 0.7,
        "max_tokens": 150,
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        response.json()

    except requests.exceptions.ConnectionError:
        pass
    except Exception:
        pass


if __name__ == "__main__":
    test_inference()
