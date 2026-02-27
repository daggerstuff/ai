from google import genai
import os

_WORKING_MODEL_CACHE = None


def get_best_available_gemini_model(client: genai.Client) -> str:
    """
    Dynamically interrogates the Gemini API to find the best functioning
    model available for the current API key's tier/region. This prevents
    hardcoded models from throwing 404s if they are restricted.
    """
    global _WORKING_MODEL_CACHE
    if _WORKING_MODEL_CACHE:
        return _WORKING_MODEL_CACHE

    target_models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]

    try:
        available_models = [m.name for m in client.models.list()]
    except Exception as e:
        print(f"Failed to list models: {e}")
        return "gemini-2.5-flash"  # Fallback guess

    for target in target_models:
        for available in available_models:
            if target == available or available.endswith(target):
                # Double check that we can actually invoke it (some show up in list but 404 on invoke due to constraints)
                try:
                    client.models.generate_content(model=target, contents="ping")
                    _WORKING_MODEL_CACHE = target
                    print(f"Dynamically locked to functioning Gemini model: {target}")
                    return target
                except Exception as eval_e:
                    print(f"Model {target} is listed but uninvokeable: {eval_e}")
                    continue

    print(
        f"CRITICAL WARNING: No preferred Gemini models available on this API Key. Falling back to gemini-2.5-flash."
    )
    return "gemini-2.5-flash"


def ensure_valid_key() -> str:
    """Validates that the Gemini API key provided is a REST key, not an OAuth token."""
    key = os.environ.get("GOOGLE_CLOUD_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "Neither GOOGLE_CLOUD_API_KEY nor GEMINI_API_KEY are configured."
        )
    if key.startswith("AQ"):
        raise ValueError(
            "Provided GEMINI_API_KEY is an OAuth token (AQ...). The AI engine requires a Google Cloud REST API key (AIza...). Please update your .env file."
        )
    return key
