#!/usr/bin/env python3
"""
Model Comparison Script
"""

import json

from evaluate_model import run_evaluation


def compare_models():
    """Compare baseline vs fine-tuned model"""

    models = {
        "baseline": "LatitudeGames/Wayfarer-2-12B",
        "finetuned": "./wayfarer-finetuned"
    }

    results = {}

    for name, path in models.items():
        try:
            results[name] = run_evaluation(path)
        except Exception:
            results[name] = None

    # Generate comparison report
    if results["baseline"] and results["finetuned"]:
        comparison = {}
        for eval_set in results["baseline"]:
            comparison[eval_set] = {
                "baseline": results["baseline"][eval_set]["overall_score"],
                "finetuned": results["finetuned"][eval_set]["overall_score"],
                "improvement": results["finetuned"][eval_set]["overall_score"] - results["baseline"][eval_set]["overall_score"]
            }

        with open("model_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)


    return results

if __name__ == "__main__":
    compare_models()
