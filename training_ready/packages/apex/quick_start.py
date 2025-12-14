#!/usr/bin/env python3
"""
Quick Start Script for Lightning.ai KAN-28 Training
Run this script to start training on Lightning.ai with all components
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """Check if we're running on Lightning.ai and environment is set up correctly"""
    
    logger.info("🔍 Checking Lightning.ai environment...")
    
    # Check if we're on Lightning.ai
    is_lightning = os.getenv('LIGHTNING_CLOUD_URL') is not None
    if is_lightning:
        logger.info("✅ Running on Lightning.ai")
    else:
        logger.warning("⚠️ Not detected as Lightning.ai environment")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
        else:
            logger.error("❌ No GPU detected!")
            return False
    except ImportError:
        logger.error("❌ PyTorch not available!")
        return False
    
    return True

def install_dependencies():
    """Install required dependencies"""
    
    logger.info("📦 Installing dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True, text=True)
        logger.info("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def validate_data():
    """Validate training data is present and correct"""
    
    logger.info("📊 Validating training data...")
    
    try:
        from scripts.data_preparation import main as prepare_data
        success = prepare_data()
        if success:
            logger.info("✅ Data validation complete")
        else:
            logger.error("❌ Data validation failed")
        return success
    except Exception as e:
        logger.error(f"❌ Data validation error: {e}")
        return False

def start_training():
    """Start the enhanced training process"""
    
    logger.info("🚀 Starting KAN-28 enhanced training...")
    
    try:
        # Change to scripts directory
        os.chdir("scripts")
        
        # Run the enhanced training script
        result = subprocess.run([
            sys.executable, "train_enhanced.py"
        ], check=True, text=True)
        
        logger.info("🎉 Training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Training failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        return False

def display_package_info():
    """Display information about the training package"""
    
    logger.info("📋 KAN-28 Lightning.ai Training Package")
    logger.info("=" * 50)
    
    # Load configuration
    config_path = Path("config/enhanced_training_config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"🤖 Base Model: {config.get('base_model', 'Unknown')}")
        logger.info(f"📊 Total Conversations: {config.get('dataset_config', {}).get('total_conversations', 'Unknown'):,}")
        logger.info(f"🔧 Components Integrated: {len(config.get('kan28_components', {}).get('integrated_components', []))}")
        logger.info(f"👥 Expert Voices: {', '.join(config.get('kan28_components', {}).get('expert_voices', []))}")
        logger.info(f"🧠 Psychology Concepts: {config.get('kan28_components', {}).get('psychology_concepts', 'Unknown'):,}")
    
    logger.info("=" * 50)

def main():
    """Main quick start function"""
    
    print("\n🚀 Pixelated Empathy KAN-28 Training - Lightning.ai Quick Start")
    print("=" * 70)
    
    # Display package information
    display_package_info()
    
    # Step 1: Check environment
    if not check_environment():
        logger.error("❌ Environment check failed!")
        return False
    
    # Step 2: Install dependencies
    if not install_dependencies():
        logger.error("❌ Dependency installation failed!")
        return False
    
    # Step 3: Validate data
    if not validate_data():
        logger.error("❌ Data validation failed!")
        return False
    
    # Step 4: Start training
    logger.info("\n🎯 All checks passed! Starting training...")
    if not start_training():
        logger.error("❌ Training failed!")
        return False
    
    logger.info("\n🎉 KAN-28 Enhanced Training Complete!")
    logger.info("📁 Model saved to: ./scripts/pixelated_empathy_kan28")
    logger.info("🎯 Your therapeutic AI now includes ALL 6 components!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ SUCCESS: Training completed successfully!")
    else:
        print("\n❌ FAILED: Training encountered errors!")
    exit(0 if success else 1)