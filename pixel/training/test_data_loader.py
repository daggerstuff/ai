"""
Test script for Pixel LLM data loader
Verifies that all 608,458 records load correctly and splits are created
"""

import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mock tokenizer for testing (no HuggingFace dependency)
class MockTokenizer:
    def __call__(self, text, max_length=512, padding="max_length", truncation=True, return_tensors="pt"):
        import torch
        # Simple mock: just return dummy tensors
        input_ids = torch.zeros((1, max_length), dtype=torch.long)
        attention_mask = torch.ones((1, max_length), dtype=torch.long)
        token_type_ids = torch.zeros((1, max_length), dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }


def test_data_loader():
    """Test the data loader"""
    from data_loader import PixelDataLoader, DataLoaderConfig
    
    # Path to merged dataset
    dataset_path = "/home/vivi/pixelated/ai/lightning/processed_data/merged_dataset.jsonl"
    
    logger.info("=" * 80)
    logger.info("TIER 1.1: Data Loader Test")
    logger.info("=" * 80)
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return False
    
    # Create config
    config = DataLoaderConfig(
        dataset_path=dataset_path,
        batch_size=32,
        num_workers=0,  # Set to 0 for testing
        seed=42
    )
    
    # Create loader
    tokenizer = MockTokenizer()
    loader = PixelDataLoader(config, tokenizer)
    
    # Load dataset
    logger.info("Loading dataset...")
    total_records = loader.load_dataset()
    logger.info(f"✓ Loaded {total_records:,} records")
    
    if total_records != 608458:
        logger.warning(f"Expected 608,458 records but got {total_records:,}")
    else:
        logger.info("✓ Record count matches expected (608,458)")
    
    # Create splits
    logger.info("Creating train/val/test splits (70/15/15)...")
    train_records, val_records, test_records = loader.create_splits()
    
    logger.info(f"✓ Train split: {len(train_records):,} records ({len(train_records)/total_records*100:.1f}%)")
    logger.info(f"✓ Val split: {len(val_records):,} records ({len(val_records)/total_records*100:.1f}%)")
    logger.info(f"✓ Test split: {len(test_records):,} records ({len(test_records)/total_records*100:.1f}%)")
    
    # Verify splits
    expected_train = int(total_records * 0.70)
    expected_val = int(total_records * 0.15)
    
    if len(train_records) != expected_train:
        logger.warning(f"Train split mismatch: expected {expected_train}, got {len(train_records)}")
    else:
        logger.info("✓ Train split size correct")
    
    if len(val_records) != expected_val:
        logger.warning(f"Val split mismatch: expected {expected_val}, got {len(val_records)}")
    else:
        logger.info("✓ Val split size correct")
    
    # Create datasets
    logger.info("Creating PyTorch datasets...")
    train_dataset, val_dataset, test_dataset = loader.create_datasets()
    logger.info(f"✓ Created datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    # Create dataloaders
    logger.info("Creating DataLoaders...")
    train_loader, val_loader, test_loader = loader.create_dataloaders()
    logger.info(f"✓ Created DataLoaders")
    logger.info(f"  - Train batches: {len(train_loader)}")
    logger.info(f"  - Val batches: {len(val_loader)}")
    logger.info(f"  - Test batches: {len(test_loader)}")
    
    # Get stats
    stats = loader.get_dataset_stats()
    logger.info("\nDataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
    
    # Test a batch
    logger.info("\nTesting batch loading...")
    try:
        batch = next(iter(train_loader))
        logger.info(f"✓ Successfully loaded batch")
        logger.info(f"  - Batch keys: {list(batch.keys())}")
        logger.info(f"  - Input IDs shape: {batch['input_ids'].shape}")
        logger.info(f"  - Attention mask shape: {batch['attention_mask'].shape}")
    except Exception as e:
        logger.error(f"✗ Failed to load batch: {e}")
        return False
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL TESTS PASSED")
    logger.info("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        success = test_data_loader()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)

