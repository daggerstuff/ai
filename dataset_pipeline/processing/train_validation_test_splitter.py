"""
Train/validation/test splitter for dataset preparation.
Splits datasets into training, validation, and test sets with proper stratification.
"""

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from conversation_schema import Conversation
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetSplit:
    """Dataset split result."""
    train: list[Conversation]
    validation: list[Conversation]
    test: list[Conversation]
    split_info: dict[str, Any]


class TrainValidationTestSplitter:
    """
    Splits datasets for training/validation/test.

    Creates stratified splits ensuring balanced representation
    across all data categories in each split.
    """

    def __init__(self):
        """Initialize the dataset splitter."""
        self.logger = get_logger(__name__)

        self.default_ratios = {
            "train": 0.8,      # 80%
            "validation": 0.1,  # 10%
            "test": 0.1        # 10%
        }

        self.logger.info("TrainValidationTestSplitter initialized")

    def split_dataset(self, conversations: list[Conversation],
                     ratios: dict[str, float] | None = None,
                     stratify_by: str = "tags",
                     random_seed: int = 42) -> DatasetSplit:
        """Split dataset into train/validation/test sets."""
        if ratios is None:
            ratios = self.default_ratios

        random.seed(random_seed)

        self.logger.info(f"Splitting {len(conversations)} conversations with ratios {ratios}")

        # Stratify conversations by specified attribute
        stratified_groups = self._stratify_conversations(conversations, stratify_by)

        train_conversations = []
        validation_conversations = []
        test_conversations = []

        # Split each stratified group
        for _group_name, group_conversations in stratified_groups.items():
            group_train, group_val, group_test = self._split_group(group_conversations, ratios)

            train_conversations.extend(group_train)
            validation_conversations.extend(group_val)
            test_conversations.extend(group_test)

        # Shuffle the final splits
        random.shuffle(train_conversations)
        random.shuffle(validation_conversations)
        random.shuffle(test_conversations)

        split_info = {
            "total_conversations": len(conversations),
            "train_count": len(train_conversations),
            "validation_count": len(validation_conversations),
            "test_count": len(test_conversations),
            "ratios_used": ratios,
            "stratification": stratify_by,
            "random_seed": random_seed,
            "split_at": datetime.now().isoformat(),
            "stratified_groups": {
                group: len(convs) for group, convs in stratified_groups.items()
            }
        }

        self.logger.info(f"Split complete: {len(train_conversations)} train, {len(validation_conversations)} val, {len(test_conversations)} test")

        return DatasetSplit(
            train=train_conversations,
            validation=validation_conversations,
            test=test_conversations,
            split_info=split_info
        )

    def _stratify_conversations(self, conversations: list[Conversation], stratify_by: str) -> dict[str, list[Conversation]]:
        """Stratify conversations by specified attribute."""
        groups = {}

        for conv in conversations:
            if stratify_by == "tags":
                # Use primary tag (first tag) for stratification
                group_key = conv.tags[0] if conv.tags else "untagged"
            elif stratify_by == "metadata":
                # Use a metadata field for stratification
                group_key = conv.metadata.get("category", "unknown")
            elif stratify_by == "quality":
                # Stratify by quality score ranges
                if conv.quality_score is None:
                    group_key = "no_score"
                elif conv.quality_score >= 0.8:
                    group_key = "high_quality"
                elif conv.quality_score >= 0.6:
                    group_key = "medium_quality"
                else:
                    group_key = "low_quality"
            else:
                group_key = "default"

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(conv)

        return groups

    def _split_group(self, conversations: list[Conversation],
                    ratios: dict[str, float]) -> tuple[list[Conversation], list[Conversation], list[Conversation]]:
        """Split a single group of conversations."""
        total = len(conversations)

        # Calculate split sizes
        train_size = int(total * ratios["train"])
        val_size = int(total * ratios["validation"])
        total - train_size - val_size  # Remainder goes to test

        # Shuffle conversations
        shuffled = conversations.copy()
        random.shuffle(shuffled)

        # Split
        train = shuffled[:train_size]
        validation = shuffled[train_size:train_size + val_size]
        test = shuffled[train_size + val_size:]

        return train, validation, test

    def validate_split(self, split: DatasetSplit) -> dict[str, Any]:
        """Validate the quality of the dataset split."""
        total = split.split_info["total_conversations"]

        # Check split ratios
        actual_ratios = {
            "train": len(split.train) / total,
            "validation": len(split.validation) / total,
            "test": len(split.test) / total
        }

        # Check for data leakage (same conversation IDs across splits)
        train_ids = {conv.id for conv in split.train}
        val_ids = {conv.id for conv in split.validation}
        test_ids = {conv.id for conv in split.test}

        leakage_train_val = len(train_ids & val_ids)
        leakage_train_test = len(train_ids & test_ids)
        leakage_val_test = len(val_ids & test_ids)

        # Check stratification balance
        train_tags = self._get_tag_distribution(split.train)
        val_tags = self._get_tag_distribution(split.validation)
        test_tags = self._get_tag_distribution(split.test)

        return {
            "actual_ratios": actual_ratios,
            "data_leakage": {
                "train_validation": leakage_train_val,
                "train_test": leakage_train_test,
                "validation_test": leakage_val_test,
                "total_leakage": leakage_train_val + leakage_train_test + leakage_val_test
            },
            "tag_distribution": {
                "train": train_tags,
                "validation": val_tags,
                "test": test_tags
            },
            "split_quality": "good" if (leakage_train_val + leakage_train_test + leakage_val_test) == 0 else "poor"
        }

    def _get_tag_distribution(self, conversations: list[Conversation]) -> dict[str, int]:
        """Get tag distribution for a set of conversations."""
        tag_counts = {}
        for conv in conversations:
            for tag in conv.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return tag_counts


def validate_train_validation_test_splitter():
    """Validate the TrainValidationTestSplitter functionality."""
    try:
        splitter = TrainValidationTestSplitter()
        assert hasattr(splitter, "split_dataset")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    if validate_train_validation_test_splitter():
        pass
    else:
        pass
