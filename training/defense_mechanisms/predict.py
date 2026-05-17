"""
Defense Mechanism Inference / Ensemble Prediction

Load one or multiple fold checkpoints, ensemble their predictions
via softmax-average, and output a submission file for CodaBench.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from training.defense_mechanisms.constants import DEFENSE_LABELS, DEFENSE_MATURITY
from training.defense_mechanisms.dataset import (
    DefenseDataset,
    load_psydefconv,
)
from training.defense_mechanisms.model import DefenseClassifier

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def load_checkpoint(
    checkpoint_dir: str,
    device: torch.device,
) -> tuple[DefenseClassifier, dict]:
    """
    Load a trained model from a fold checkpoint directory.

    Args:
        checkpoint_dir: Path to fold directory containing best_model.pt
        device: Target device

    Returns:
        Tuple of (model, checkpoint_metadata)
    """
    ckpt_path = Path(checkpoint_dir)
    model_path = ckpt_path / "best_model.pt"
    if not model_path.exists():
        # Try the directory itself as a .pt file
        if ckpt_path.suffix == ".pt" and ckpt_path.exists():
            model_path = ckpt_path
        else:
            raise FileNotFoundError(f"No best_model.pt found in {ckpt_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    config = checkpoint.get("config", {})

    model = DefenseClassifier(
        model_name=config.get("base_model", "microsoft/deberta-v3-base"),
        num_labels=config.get("num_labels", 9),
        r_drop_enabled=False,  # Disable R-Drop for inference
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Loaded checkpoint from %s (fold %d, epoch %d, macro-F1=%.4f)",
        model_path.name,
        checkpoint.get("fold_index", -1),
        checkpoint.get("epoch", -1),
        checkpoint.get("macro_f1", 0.0),
    )

    return model, checkpoint


@torch.no_grad()
def predict_single_model(
    model: DefenseClassifier,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
) -> tuple[np.ndarray, list[str]]:
    """
    Get softmax probabilities from a single model.

    Args:
        model: Trained classifier
        dataloader: Data to predict on
        device: Compute device
        temperature: Temperature scaling factor

    Returns:
        Tuple of (probabilities array [N, 9], sample_ids list)
    """
    model.eval()
    all_probs = []
    all_ids = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask)
        logits = outputs["logits"]

        if temperature != 1.0:
            logits = logits / temperature

        probs = F.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_ids.extend(batch.get("sample_id", [f"unk_{i}" for i in range(len(probs))]))

    return np.concatenate(all_probs, axis=0), all_ids


def ensemble_predict(
    checkpoint_dirs: list[str],
    test_data_path: str,
    weights: list[float] | None = None,
    temperature: float = 1.0,
    batch_size: int = 16,
    max_length: int = 512,
    max_turns: int = 40,
    limit: int = -1,
) -> list[dict]:
    """
    Ensemble predictions from multiple fold checkpoints.

    Computes weighted softmax-average across models and returns
    final predictions.

    Args:
        checkpoint_dirs: Paths to fold checkpoint directories
        test_data_path: Path to test.json
        weights: Per-checkpoint weights (default: equal)
        temperature: Temperature scaling
        batch_size: Inference batch size
        max_length: Max tokenizer length
        max_turns: Max dialogue turns
        limit: Limit number of samples (-1 = all)

    Returns:
        List of prediction dicts with id, label, label_name,
        confidence, maturity_score
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_psydefconv(test_data_path, has_labels=False)
    if 0 < limit < len(samples):
        samples = samples[:limit]
        logger.info("Limited to %d samples", limit)

    # Use tokenizer from first checkpoint's config
    first_ckpt = torch.load(
        Path(checkpoint_dirs[0]) / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model_name = first_ckpt.get("config", {}).get(
        "base_model", "microsoft/deberta-v3-base"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_ds = DefenseDataset(
        samples, tokenizer, max_length=max_length, max_turns=max_turns
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    if weights is None:
        weights = [1.0] * len(checkpoint_dirs)
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    logger.info(
        "Ensembling %d models with weights %s",
        len(checkpoint_dirs),
        [f"{w:.3f}" for w in weights],
    )

    ensemble_probs = None
    sample_ids = None

    for ckpt_dir, weight in zip(checkpoint_dirs, weights):
        model, _ = load_checkpoint(ckpt_dir, device)
        probs, ids = predict_single_model(model, test_loader, device, temperature)
        if ensemble_probs is None:
            ensemble_probs = probs * weight
            sample_ids = ids
        else:
            ensemble_probs += probs * weight

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = []
    for i in range(len(ensemble_probs)):
        label = int(np.argmax(ensemble_probs[i]))
        confidence = float(ensemble_probs[i][label])
        maturity = DEFENSE_MATURITY.get(label)

        predictions.append(
            {
                "id": sample_ids[i],
                "label": label,
                "label_name": DEFENSE_LABELS.get(label, f"Unknown ({label})"),
                "confidence": round(confidence, 4),
                "maturity_score": float(maturity) if maturity is not None else None,
                "probabilities": [round(float(p), 4) for p in ensemble_probs[i]],
            }
        )

    return predictions


def write_submission(predictions: list[dict], output_path: str):
    """
    Write predictions in CodaBench submission format.

    Format: one JSON object per line with {"id": ..., "label": int}
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for pred in predictions:
            line = json.dumps(
                {"id": pred["id"], "label": pred["label"]},
                ensure_ascii=False,
            )
            f.write(line + "\n")

    logger.info("Submission written to %s (%d predictions)", path, len(predictions))


def main():
    parser = argparse.ArgumentParser(
        description="Defense mechanism ensemble prediction"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test.json",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Paths to fold checkpoint directories",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Per-checkpoint weights (default: equal)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature scaling for logits",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="submission.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit number of samples (-1 = all)",
    )
    args = parser.parse_args()

    predictions = ensemble_predict(
        checkpoint_dirs=args.checkpoints,
        test_data_path=args.test_file,
        weights=args.weights,
        temperature=args.temperature,
        batch_size=args.batch_size,
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
