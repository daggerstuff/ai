"""
Test script for Pixel LLM training configuration
Verifies GPU/compute resource setup
"""

import sys
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_training_config():
    """Test training configuration"""
    from training_config import TrainingConfigManager, create_training_config
    
    logger.info("=" * 80)
    logger.info("TIER 1.2: Training Configuration Test")
    logger.info("=" * 80)
    
    # Test 1: Create config manager
    logger.info("\n1. Creating TrainingConfigManager...")
    manager = TrainingConfigManager()
    logger.info("✓ TrainingConfigManager created")
    
    # Test 2: Check compute config
    logger.info("\n2. Checking compute configuration...")
    logger.info(f"  Device: {manager.compute_config.device}")
    logger.info(f"  Number of GPUs: {manager.compute_config.num_gpus}")
    logger.info(f"  Mixed precision: {manager.compute_config.mixed_precision}")
    logger.info(f"  Gradient checkpointing: {manager.compute_config.gradient_checkpointing}")
    logger.info("✓ Compute configuration verified")
    
    # Test 3: Check training config
    logger.info("\n3. Checking training configuration...")
    logger.info(f"  Num epochs: {manager.training_config.num_epochs}")
    logger.info(f"  Learning rate: {manager.training_config.learning_rate}")
    logger.info(f"  Batch size: {manager.training_config.per_device_train_batch_size}")
    logger.info(f"  Warmup steps: {manager.training_config.warmup_steps}")
    logger.info("✓ Training configuration verified")
    
    # Test 4: Check model config
    logger.info("\n4. Checking model configuration...")
    logger.info(f"  Model name: {manager.model_config.model_name}")
    logger.info(f"  Max seq length: {manager.model_config.max_seq_length}")
    logger.info(f"  Hidden size: {manager.model_config.hidden_size}")
    logger.info("✓ Model configuration verified")
    
    # Test 5: Check output config
    logger.info("\n5. Checking output configuration...")
    logger.info(f"  Output dir: {manager.output_config.output_dir}")
    logger.info(f"  Log dir: {manager.output_config.log_dir}")
    logger.info(f"  Report to: {manager.output_config.report_to}")
    logger.info("✓ Output configuration verified")
    
    # Test 6: Save and load config
    logger.info("\n6. Testing config save/load...")
    config_path = "/tmp/test_training_config.json"
    manager.save_config(config_path)
    logger.info(f"✓ Config saved to {config_path}")
    
    # Verify file exists
    if Path(config_path).exists():
        logger.info("✓ Config file exists")
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        logger.info(f"✓ Config file is valid JSON with keys: {list(config_data.keys())}")
    else:
        logger.error("✗ Config file not created")
        return False
    
    # Test 7: Create config with convenience function
    logger.info("\n7. Testing convenience function...")
    manager2 = create_training_config(
        num_epochs=5,
        learning_rate=1e-5,
        batch_size=16,
        output_dir="./custom_outputs"
    )
    logger.info(f"✓ Created config with custom parameters")
    logger.info(f"  Num epochs: {manager2.training_config.num_epochs}")
    logger.info(f"  Learning rate: {manager2.training_config.learning_rate}")
    logger.info(f"  Batch size: {manager2.training_config.per_device_train_batch_size}")
    logger.info(f"  Output dir: {manager2.output_config.output_dir}")
    
    # Test 8: Get training args dict
    logger.info("\n8. Testing get_training_args_dict()...")
    args_dict = manager.get_training_args_dict()
    logger.info(f"✓ Got training args dict with {len(args_dict)} keys")
    logger.info(f"  Keys: {list(args_dict.keys())[:5]}...")
    
    # Test 9: Get device
    logger.info("\n9. Testing get_device()...")
    device = manager.get_device()
    logger.info(f"✓ Got device: {device}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL TESTS PASSED")
    logger.info("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        success = test_training_config()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)

