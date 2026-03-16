"""
PSYDEFCONV Dataset Loader

Loads the PsyDefDetect dataset, formats multi-turn dialogues for
sequence classification, and provides GroupKFold splitting to
prevent dialogue leakage across train/val splits.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from training.defense_mechanisms.constants import NUM_LABELS

logger = logging.getLogger(__name__)


SPEAKER_MAP = {
    "seeker": "Seeker",
    "supporter": "Supporter",
    "help-seeker": "Seeker",
    "helper": "Supporter",
    "patient": "Seeker",
    "therapist": "Supporter",
    "user": "Seeker",
    "system": "Supporter",
}


@dataclass
class DialogueSample:
    """A single annotated dialogue sample from PSYDEFCONV."""

    sample_id: str
    dialogue_id: str
    turns: list[dict] = field(default_factory=list)
    target_text: str = ""
    target_turn_index: int = -1
    label: Optional[int] = None


def normalize_speaker(speaker: str) -> str:
    """Map various speaker labels to Seeker/Supporter."""
    cleaned = speaker.strip().lower().rstrip(":")
    return SPEAKER_MAP.get(cleaned, "Seeker")


def format_dialogue(
    turns: list[dict],
    target_text: str,
    max_turns: int = 40,
    target_turn_index: int = -1,
) -> str:
    """
    Format a multi-turn dialogue for model input.

    Keeps the last `max_turns` turns. Wraps the target utterance
    in <t>...</t> markers for the model to focus on.

    Args:
        turns: List of dicts with 'speaker' and 'text' keys
        target_text: The utterance to classify
        max_turns: Maximum turns to keep (truncates from start)
        target_turn_index: Index of the target turn in the
            dialogue. If -1, matches by text content.

    Returns:
        Formatted dialogue string with target marked
    """
    if len(turns) > max_turns:
        offset = len(turns) - max_turns
        turns = turns[offset:]
        if target_turn_index >= 0:
            target_turn_index -= offset

    formatted_parts = []
    target_normalized = _normalize_text(target_text)

    for i, turn in enumerate(turns):
        speaker = normalize_speaker(turn.get("speaker", "Seeker"))
        text = turn.get("text", turn.get("content", "")).strip()

        is_target = False
        if target_turn_index >= 0:
            is_target = i == target_turn_index
        else:
            is_target = _normalize_text(text) == target_normalized

        if is_target:
            formatted_parts.append(f"{speaker}: <t>{text}</t>")
        else:
            formatted_parts.append(f"{speaker}: {text}")

    return " ".join(formatted_parts)


def _normalize_text(text: str) -> str:
    """Normalize whitespace for comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """
    Compute inverse-square-root frequency class weights.

    For a class with frequency f_i in the dataset, weight = 1/sqrt(f_i).
    Weights are then normalized so they sum to NUM_LABELS.

    Args:
        labels: List of integer labels from the training set

    Returns:
        Tensor of shape (NUM_LABELS,) with class weights
    """
    counts = np.zeros(NUM_LABELS, dtype=np.float64)
    for label in labels:
        if 0 <= label < NUM_LABELS:
            counts[label] += 1

    # Avoid division by zero for classes with no samples
    counts = np.maximum(counts, 1.0)

    weights = 1.0 / np.sqrt(counts)
    weights = weights * NUM_LABELS / weights.sum()

    return torch.tensor(weights, dtype=torch.float32)


def load_psydefconv(
    data_path: str,
    has_labels: bool = True,
) -> list[DialogueSample]:
    """
    Load PSYDEFCONV data from a JSON file.

    Handles multiple possible data formats:
    - List of objects with 'dialogue', 'current_text', 'label'
    - List of objects with 'conversation', 'target', 'defense_level'

    Args:
        data_path: Path to JSON file
        has_labels: Whether the file contains ground-truth labels

    Returns:
        List of DialogueSample objects
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download from "
            "https://www.codabench.org/competitions/12124/"
        )

    with open(path, encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        raw_data = raw_data.get("data", raw_data.get("samples", []))

    samples = []
    for i, item in enumerate(raw_data):
        sample_id = str(item.get("id", item.get("sample_id", f"s_{i}")))
        dialogue_id = str(
            item.get(
                "dialogue_id",
                item.get("conversation_id", sample_id.split("_")[0]),
            )
        )

        # Parse dialogue turns
        raw_dialogue = item.get("dialogue", item.get("conversation", []))
        if isinstance(raw_dialogue, str):
            turns = _parse_dialogue_string(raw_dialogue)
        elif isinstance(raw_dialogue, list):
            turns = []
            for turn in raw_dialogue:
                if isinstance(turn, dict):
                    turns.append(
                        {
                            "speaker": turn.get("speaker", turn.get("role", "Seeker")),
                            "text": turn.get(
                                "text",
                                turn.get("content", turn.get("utterance", "")),
                            ),
                        }
                    )
                elif isinstance(turn, str):
                    turns.append({"speaker": "Seeker", "text": turn})
        else:
            turns = []

        target_text = str(
            item.get("current_text", item.get("target", item.get("text", "")))
        )

        target_turn_index = item.get("target_turn_index", -1)

        label = None
        if has_labels:
            raw_label = item.get("label", item.get("defense_level"))
            if raw_label is not None:
                label = int(raw_label)

        samples.append(
            DialogueSample(
                sample_id=sample_id,
                dialogue_id=dialogue_id,
                turns=turns,
                target_text=target_text,
                target_turn_index=target_turn_index,
                label=label,
            )
        )

    logger.info(
        "Loaded %d samples from %s (labels=%s)",
        len(samples),
        path.name,
        has_labels,
    )
    return samples


def _parse_dialogue_string(dialogue_str: str) -> list[dict]:
    """
    Parse a dialogue stored as a single string into turn dicts.

    Handles formats like:
        "Seeker: text\\nSupporter: text"
    """
    turns = []
    current_speaker = "Seeker"
    current_text_parts: list[str] = []

    for line in dialogue_str.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        speaker_match = re.match(
            r"^(Seeker|Supporter|Helper|Patient|Therapist|User|System)\s*:\s*",
            line,
            re.IGNORECASE,
        )
        if speaker_match:
            if current_text_parts:
                turns.append(
                    {
                        "speaker": normalize_speaker(current_speaker),
                        "text": " ".join(current_text_parts),
                    }
                )
                current_text_parts = []
            current_speaker = speaker_match.group(1)
            remainder = line[speaker_match.end() :].strip()
            if remainder:
                current_text_parts.append(remainder)
        else:
            current_text_parts.append(line)

    if current_text_parts:
        turns.append(
            {
                "speaker": normalize_speaker(current_speaker),
                "text": " ".join(current_text_parts),
            }
        )

    return turns


class DefenseDataset(Dataset):
    """
    PyTorch Dataset for defense mechanism classification.

    Tokenizes formatted dialogues with target markers for
    sequence classification with DeBERTa.
    """

    def __init__(
        self,
        samples: list[DialogueSample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_turns: int = 40,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_turns = max_turns

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        text = format_dialogue(
            turns=sample.turns,
            target_text=sample.target_text,
            max_turns=self.max_turns,
            target_turn_index=sample.target_turn_index,
        )

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sample_id": sample.sample_id,
        }

        if sample.label is not None:
            item["labels"] = torch.tensor(sample.label, dtype=torch.long)

        return item

    def get_labels(self) -> list[int]:
        """Return all labels for class weight computation."""
        return [s.label for s in self.samples if s.label is not None]

    def get_dialogue_ids(self) -> list[str]:
        """Return all dialogue IDs for GroupKFold splitting."""
        return [s.dialogue_id for s in self.samples]


def create_fold_datasets(
    samples: list[DialogueSample],
    tokenizer: PreTrainedTokenizer,
    num_folds: int = 5,
    fold_index: int = 0,
    max_length: int = 512,
    max_turns: int = 40,
) -> tuple["DefenseDataset", "DefenseDataset"]:
    """
    Create train/val datasets for a specific cross-validation fold.

    Uses GroupKFold on dialogue_id to prevent data leakage — all
    utterances from the same dialogue appear in the same fold.

    Args:
        samples: All labeled samples
        tokenizer: Tokenizer for encoding
        num_folds: Number of cross-validation folds
        fold_index: Which fold to use as validation
        max_length: Maximum sequence length for tokenizer
        max_turns: Maximum dialogue turns to include

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    dialogue_ids = np.array([s.dialogue_id for s in samples])
    labels = np.array([s.label if s.label is not None else 0 for s in samples])

    gkf = GroupKFold(n_splits=num_folds)
    splits = list(gkf.split(range(len(samples)), labels, dialogue_ids))

    if fold_index < 0 or fold_index >= len(splits):
        raise ValueError(f"fold_index {fold_index} out of range for {num_folds} folds")

    train_idx, val_idx = splits[fold_index]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    # Verify no dialogue leakage
    train_dialogue_ids = {s.dialogue_id for s in train_samples}
    val_dialogue_ids = {s.dialogue_id for s in val_samples}
    leaked = train_dialogue_ids & val_dialogue_ids
    if leaked:
        raise RuntimeError(
            f"Dialogue leakage detected: {len(leaked)} dialogue IDs "
            f"appear in both train and val splits"
        )

    logger.info(
        "Fold %d/%d: train=%d samples (%d dialogues), val=%d samples (%d dialogues)",
        fold_index + 1,
        num_folds,
        len(train_samples),
        len(train_dialogue_ids),
        len(val_samples),
        len(val_dialogue_ids),
    )

    train_ds = DefenseDataset(train_samples, tokenizer, max_length, max_turns)
    val_ds = DefenseDataset(val_samples, tokenizer, max_length, max_turns)

    return train_ds, val_ds
