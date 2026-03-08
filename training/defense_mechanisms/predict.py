"""
Defense Mechanism Inference / Ensemble Prediction
Using NVIDIA NIM
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from ai.training.defense_mechanisms.dataset import load_psydefconv
from ai.training.defense_mechanisms.model import DefenseClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def nim_predict(
    test_data_path: str,
    limit: int = -1,
) -> list[dict]:
    """
    Run predictions using NVIDIA NIM model.
    """
    samples = load_psydefconv(test_data_path, has_labels=False)
    if 0 < limit < len(samples):
        samples = samples[:limit]
        logger.info("Limited to %d samples", limit)

    model = DefenseClassifier()
    logger.info(f"Initialized NIM classifier with {model.model_name}")

    predictions = []

    # Process sequentially or in small batches
    for sample in samples:
        # Reconstruct text context similar to format_dialogue
        text_context = ""
        if "dialogue" in sample:
            # Try to grab the last few turns
            turns = sample["dialogue"][-5:]
            for t in turns:
                speaker = t.get("speaker", "Unknown")
                text_content = t.get("text", "")
                text_context += f"{speaker}: {text_content}\n"

        target = sample.get("target_utterance", "")
        if target:
            text_context += f"\nTarget Utterance to analyze: {target}"

        if not text_context.strip():
            # Fallback
            text_context = sample.get("text", str(sample))

        # Call NIM
        try:
            preds = model.predict([text_context])
            if preds and len(preds) > 0:
                p = preds[0]
                predictions.append(
                    {
                        "id": sample.get("id", "unk_id"),
                        "label": p.label,
                        "label_name": p.label_name,
                        "confidence": round(p.confidence, 4),
                        "maturity_score": (
                            float(p.maturity_score)
                            if p.maturity_score is not None
                            else None
                        ),
                        "probabilities": [
                            round(float(pr), 4) for pr in p.probabilities
                        ],
                    }
                )
            else:
                raise ValueError("No prediction returned.")
        except Exception as e:
            logger.error(f"Failed prediction for {sample.get('id')}: {e}")
            predictions.append(
                {
                    "id": sample.get("id", "unk_id"),
                    "label": 0,
                    "label_name": "Neutral",
                    "confidence": 0.0,
                    "maturity_score": None,
                    "probabilities": [1.0] + [0.0] * 8,
                }
            )

    return predictions


def write_submission(predictions: list[dict], output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pred in predictions:
            line = json.dumps(
                {"id": pred["id"], "label": pred["label"]}, ensure_ascii=False
            )
            f.write(line + "\n")
    logger.info("Submission written to %s (%d predictions)", path, len(predictions))


def main():
    parser = argparse.ArgumentParser(description="Defense mechanism NIM prediction")
    parser.add_argument(
        "--test-file", type=str, required=True, help="Path to test.json"
    )
    parser.add_argument(
        "--checkpoints", type=str, nargs="+", required=False, help="Legacy: ignored"
    )
    parser.add_argument(
        "--weights", type=float, nargs="+", default=None, help="Legacy: ignored"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature scaling"
    )
    parser.add_argument(
        "--out", type=str, default="submission.jsonl", help="Output file path"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Legacy: ignored")
    parser.add_argument(
        "--limit", type=int, default=-1, help="Limit number of samples (-1 = all)"
    )
    args = parser.parse_args()

    predictions = nim_predict(
        test_data_path=args.test_file,
        limit=args.limit,
    )

    write_submission(predictions, args.out)

    # Print summary
    label_counts = {}
    for pred in predictions:
        name = pred["label_name"]
        label_counts[name] = label_counts.get(name, 0) + 1

    logger.info("Prediction distribution:")
    for name, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(predictions)
        logger.info("  %s: %d (%.1f%%)", name, count, pct)


if __name__ == "__main__":
    main()
