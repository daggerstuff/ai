def _generate_with_llm():
    return "", False


SCENARIOS = {
    "suicidal_ideation_active": "test",
    "suicidal_ideation_passive": "test",
    "self_harm": "test",
    "substance_relapse": "test",
    "eating_disorder_crisis": "test",
    "psychosis_symptoms": "test",
}


def generate_pairs(target_count, use_llm=False):
    pairs = []
    categories = list(SCENARIOS.keys())
    for i in range(target_count):
        chosen = "This is a long safe text." * 10
        if use_llm:
            chosen, success = _generate_with_llm()
            if not success:
                chosen = "Fallback long text." * 10
        pairs.append(
            {
                "prompt": f"Test prompt {i}",  # Unique prompts
                "chosen": chosen,
                "rejected": "Bad text",
                "metadata": {
                    "category": categories[i % len(categories)],
                    "description": "test",
                    "difficulty": ["critical", "high", "medium"][i % 3],
                    "pair_type": "nightmare_fuel",
                    "is_crisis": i % 2 == 0,
                },
            }
        )
    return pairs


def _call_nemo():
    try:
        return
    except Exception:
        return


def _variate_response(text):
    return text + "."


PROMPT_VARIATION_RATE = 0


def _expand_chosen_with_llm():
    return "", False
