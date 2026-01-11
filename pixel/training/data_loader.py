"""
Pixel LLM Training Data Loader
Loads merged_dataset.jsonl and creates train/validation/test splits (70/15/15)
"""

import json
import logging
import os
import random
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class DataLoaderConfig:
    """Configuration for data loading"""
    dataset_path: str
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    max_length: int = 512
    shuffle: bool = True


class TherapeuticConversationDataset(Dataset):
    """PyTorch Dataset for therapeutic conversations"""

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int = 512,
        split: str = "train"
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        logger.info(f"Initialized {split} dataset with {len(records)} records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]

        # Extract conversation text
        text = record.get("text", "")
        if not text:
            # Fallback: construct from messages if available
            messages = record.get("messages", [])
            text = " ".join([msg.get("content", "") for msg in messages])

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(),
            "record_id": record.get("_id", ""),
            "source": record.get("_source", "unknown")
        }


class PixelDataLoader:
    """Main data loader for Pixel LLM training"""

    def __init__(self, config: DataLoaderConfig, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        self.records = []
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        random.seed(config.seed)
        torch.manual_seed(config.seed)

    def load_dataset(self) -> int:
        """Load merged_dataset.jsonl"""
        logger.info(f"Loading dataset from {self.config.dataset_path}")

        if not os.path.exists(self.config.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.config.dataset_path}")

        with open(self.config.dataset_path) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    self.records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")

        logger.info(f"Loaded {len(self.records)} records from dataset")
        return len(self.records)

    def create_splits(self) -> tuple[list[dict], list[dict], list[dict]]:
        """Create train/val/test splits (70/15/15)"""
        if not self.records:
            raise ValueError("No records loaded. Call load_dataset() first.")

        total = len(self.records)
        train_size = int(total * self.config.train_split)
        val_size = int(total * self.config.val_split)
        total - train_size - val_size

        if self.config.shuffle:
            random.shuffle(self.records)

        train_records = self.records[:train_size]
        val_records = self.records[train_size:train_size + val_size]
        test_records = self.records[train_size + val_size:]

        logger.info(f"Split dataset: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

        return train_records, val_records, test_records

    def create_datasets(self) -> tuple[TherapeuticConversationDataset, TherapeuticConversationDataset, TherapeuticConversationDataset]:
        """Create PyTorch datasets"""
        train_records, val_records, test_records = self.create_splits()

        self.train_dataset = TherapeuticConversationDataset(
            train_records, self.tokenizer, self.config.max_length, "train"
        )
        self.val_dataset = TherapeuticConversationDataset(
            val_records, self.tokenizer, self.config.max_length, "validation"
        )
        self.test_dataset = TherapeuticConversationDataset(
            test_records, self.tokenizer, self.config.max_length, "test"
        )

        return self.train_dataset, self.val_dataset, self.test_dataset

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch DataLoaders"""
        if not self.train_dataset:
            self.create_datasets()

        # Only use prefetch_factor and persistent_workers if num_workers > 0
        prefetch_factor = self.config.prefetch_factor if self.config.num_workers > 0 else None
        persistent_workers = self.config.persistent_workers if self.config.num_workers > 0 else False

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        )

        logger.info(f"Created DataLoaders: train={len(self.train_loader)}, val={len(self.val_loader)}, test={len(self.test_loader)}")

        return self.train_loader, self.val_loader, self.test_loader

    def get_dataset_stats(self) -> dict:
        """Get dataset statistics"""
        if not self.records:
            return {}

        stats = {
            "total_records": len(self.records),
            "train_records": len(self.train_dataset) if self.train_dataset else 0,
            "val_records": len(self.val_dataset) if self.val_dataset else 0,
            "test_records": len(self.test_dataset) if self.test_dataset else 0,
            "batch_size": self.config.batch_size,
            "max_length": self.config.max_length,
            "seed": self.config.seed
        }

        return stats


def setup_data_loaders(
    dataset_path: str,
    tokenizer,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Convenience function to set up all data loaders
    
    Args:
        dataset_path: Path to merged_dataset.jsonl
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        **kwargs: Additional config parameters
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, stats)
    """
    config = DataLoaderConfig(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )

    loader = PixelDataLoader(config, tokenizer)
    loader.load_dataset()
    loader.create_datasets()
    train_loader, val_loader, test_loader = loader.create_dataloaders()
    stats = loader.get_dataset_stats()

    return train_loader, val_loader, test_loader, stats

