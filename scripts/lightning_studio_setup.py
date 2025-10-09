#!/usr/bin/env python3
"""
Lightning.ai Studio Setup Script
Automated setup for H100 therapeutic AI training in Lightning.ai Studio environment.
"""

import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LightningStudioSetup:
    """Automated Lightning.ai Studio environment setup"""
    
    def __init__(self):
        self.studio_workspace = Path("/teamspace/studios/this_studio")
        self.project_dir = self.studio_workspace / "therapeutic-ai-training"
        
    def check_lightning_environment(self) -> Dict:
        """Check Lightning.ai Studio environment capabilities"""
        logger.info("🔍 Checking Lightning.ai Studio environment...")
        
        env_info = {
            "python_version": None,
            "gpu_available": False,
            "gpu_type": None,
            "memory_available": None,
            "cuda_version": None,
            "pytorch_available": False,
            "lightning_available": False,
            "studio_ready": False
        }
        
        try:
            # Check Python version
            result = subprocess.run(['python', '--version'], capture_output=True, text=True)
            env_info["python_version"] = result.stdout.strip()
            
            # Check GPU availability
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout:
                    gpu_info = result.stdout.strip().split(', ')
                    env_info["gpu_available"] = True
                    env_info["gpu_type"] = gpu_info[0] if gpu_info else "Unknown"
                    env_info["memory_available"] = gpu_info[1] if len(gpu_info) > 1 else "Unknown"
            except:
                pass
            
            # Check CUDA version
            try:
                result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
                if "release" in result.stdout:
                    env_info["cuda_version"] = result.stdout.split("release ")[1].split(",")[0]
            except:
                pass
            
            # Check PyTorch
            try:
                import torch
                env_info["pytorch_available"] = True
                env_info["pytorch_version"] = torch.__version__
                env_info["cuda_available_pytorch"] = torch.cuda.is_available()
            except:
                pass
            
            # Check Lightning
            try:
                import lightning
                env_info["lightning_available"] = True
                env_info["lightning_version"] = lightning.__version__
            except:
                pass
            
        except Exception as e:
            logger.error(f"Error checking environment: {e}")
        
        # Determine if studio is ready
        env_info["studio_ready"] = (
            env_info["gpu_available"] and
            env_info["pytorch_available"] and
            "H100" in str(env_info["gpu_type"])
        )
        
        # Log environment info
        logger.info(f"   Python: {env_info['python_version']}")
        logger.info(f"   GPU: {env_info['gpu_type']} ({env_info['memory_available']})")
        logger.info(f"   CUDA: {env_info['cuda_version']}")
        logger.info(f"   PyTorch: {'✅' if env_info['pytorch_available'] else '❌'}")
        logger.info(f"   Lightning: {'✅' if env_info['lightning_available'] else '❌'}")
        logger.info(f"   H100 Ready: {'✅' if env_info['studio_ready'] else '❌'}")
        
        return env_info
    
    def install_dependencies(self) -> bool:
        """Install required dependencies for therapeutic AI training"""
        logger.info("📦 Installing dependencies...")
        
        requirements = [
            "torch>=2.0.0",
            "lightning>=2.1.0",
            "transformers>=4.35.0",
            "peft>=0.6.0", 
            "datasets>=2.14.0",
            "accelerate>=0.24.0",
            "bitsandbytes>=0.41.0",
            "wandb>=0.16.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0"
        ]
        
        try:
            for requirement in requirements:
                logger.info(f"   Installing {requirement}...")
                result = subprocess.run(['pip', 'install', requirement], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(f"   Warning installing {requirement}: {result.stderr}")
            
            logger.info("✅ Dependencies installation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error installing dependencies: {e}")
            return False
    
    def setup_project_structure(self) -> bool:
        """Setup project directory structure in Lightning Studio"""
        logger.info("📁 Setting up project structure...")
        
        try:
            # Create main project directory
            self.project_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            subdirs = ["data", "models", "logs", "configs", "scripts", "outputs"]
            for subdir in subdirs:
                (self.project_dir / subdir).mkdir(exist_ok=True)
            
            logger.info(f"✅ Project structure created: {self.project_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up project structure: {e}")
            return False
    
    def configure_wandb(self) -> bool:
        """Configure Weights & Biases for training monitoring"""
        logger.info("📊 Configuring Weights & Biases...")
        
        try:
            # Check if wandb is available
            result = subprocess.run(['wandb', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("⚠️  WandB not available, installing...")
                subprocess.run(['pip', 'install', 'wandb'], check=True)
            
            # Login to wandb (user will need to provide key)
            logger.info("   WandB ready for configuration")
            logger.info("   💡 Run 'wandb login' with your API key when ready")
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  WandB setup warning: {e}")
            return False
    
    def create_training_launcher(self) -> Path:
        """Create training launcher script for Lightning Studio"""
        launcher_script = '''#!/usr/bin/env python3
"""
Lightning.ai Studio Training Launcher
Launch therapeutic AI training with proper GPU setup and monitoring.
"""

import os
import json
import torch
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu_setup():
    """Verify H100 GPU setup"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    logger.info(f"🚀 GPU Ready: {gpu_name} ({gpu_memory:.1f}GB)")
    
    if "H100" not in gpu_name:
        logger.warning("⚠️  Expected H100 GPU, check your Lightning.ai compute settings")

def setup_environment():
    """Setup training environment"""
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    # Set optimal memory settings for H100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def launch_training():
    """Launch the therapeutic AI training"""
    logger.info("🎯 Launching Therapeutic AI Training on H100...")
    
    # Check prerequisites
    check_gpu_setup()
    setup_environment()
    
    # Verify data is available
    if not Path("data/train.json").exists():
        raise FileNotFoundError("Training data not found! Run prepare_data.py first")
    
    # Launch training
    cmd = ["python", "train_therapeutic_ai.py"]
    logger.info(f"   Executing: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        logger.info("🎉 Training completed successfully!")
    else:
        logger.error("❌ Training failed!")
        
    return result.returncode

if __name__ == "__main__":
    launch_training()
'''
        
        launcher_path = self.project_dir / "scripts" / "launch_training.py"
        with open(launcher_path, 'w') as f:
            f.write(launcher_script)
        
        launcher_path.chmod(0o755)
        logger.info(f"✅ Training launcher created: {launcher_path}")
        return launcher_path
    
    def create_studio_readme(self) -> Path:
        """Create README for Lightning Studio setup"""
        readme_content = '''# Therapeutic AI Training - Lightning.ai Studio

## 🎯 Mission
Train a breakthrough therapeutic AI using H100 GPU with the intelligent multi-pattern dataset that solves the "100% generic questions" problem.

## 🚀 Quick Start

### 1. Setup Environment
```bash
python scripts/setup_studio.py
```

### 2. Prepare Data
```bash
python prepare_data.py
```

### 3. Launch Training
```bash
python scripts/launch_training.py
```

## 📊 What You're Training
- **Dataset**: 8,000+ high-quality therapeutic conversations
- **Innovation**: Intelligent agent-processed Q/A pairs (no more generic questions!)
- **Architecture**: 4-Expert MoE LoRA on DialoGPT-medium
- **GPU**: H100 (80GB VRAM) optimized training
- **Training Time**: 6-12 hours

## 🧠 Expert Specialization
- **Expert 0**: Therapeutic conversations
- **Expert 1**: Educational content
- **Expert 2**: Empathetic responses  
- **Expert 3**: Practical advice

## 📈 Expected Results
- **Model Size**: ~1.5GB LoRA adapters
- **Quality**: Contextually appropriate therapeutic responses
- **Innovation**: First AI trained on intelligent pattern-analyzed therapeutic data

## 🔍 Monitoring
- Lightning logs: `./logs/`
- WandB dashboard: Configure with `wandb login`
- Real-time metrics: Training loss, perplexity, expert utilization

## 🎉 Success Criteria
- ✅ Validation loss < 1.5
- ✅ Therapeutically appropriate responses
- ✅ Balanced expert utilization
- ✅ No catastrophic forgetting

This training represents a breakthrough in therapeutic AI - the first system trained on contextually appropriate Q/A pairs instead of generic templates.
'''
        
        readme_path = self.project_dir / "README.md" 
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"✅ Studio README created: {readme_path}")
        return readme_path
    
    def run_full_setup(self) -> Dict:
        """Run complete Lightning Studio setup"""
        logger.info("🚀 Running complete Lightning.ai Studio setup...")
        
        setup_results = {
            "environment_check": False,
            "dependencies_installed": False,
            "project_structure_created": False,
            "wandb_configured": False,
            "launcher_created": False,
            "readme_created": False,
            "setup_complete": False
        }
        
        # Step 1: Check environment
        env_info = self.check_lightning_environment()
        setup_results["environment_check"] = env_info["studio_ready"]
        
        # Step 2: Install dependencies
        setup_results["dependencies_installed"] = self.install_dependencies()
        
        # Step 3: Setup project structure
        setup_results["project_structure_created"] = self.setup_project_structure()
        
        # Step 4: Configure WandB
        setup_results["wandb_configured"] = self.configure_wandb()
        
        # Step 5: Create launcher
        launcher_path = self.create_training_launcher()
        setup_results["launcher_created"] = launcher_path.exists()
        
        # Step 6: Create README
        readme_path = self.create_studio_readme()
        setup_results["readme_created"] = readme_path.exists()
        
        # Overall success
        setup_results["setup_complete"] = all([
            setup_results["dependencies_installed"],
            setup_results["project_structure_created"],
            setup_results["launcher_created"],
            setup_results["readme_created"]
        ])
        
        # Summary
        if setup_results["setup_complete"]:
            logger.info("🎉 Lightning.ai Studio setup complete!")
            logger.info(f"📁 Project directory: {self.project_dir}")
            logger.info("📋 Next steps:")
            logger.info("   1. Upload your dataset to the data/ directory")
            logger.info("   2. Run python prepare_data.py")  
            logger.info("   3. Run python scripts/launch_training.py")
        else:
            logger.error("❌ Setup incomplete. Check errors above.")
        
        return setup_results

def main():
    """Main setup function"""
    setup = LightningStudioSetup()
    results = setup.run_full_setup()
    return results["setup_complete"]

if __name__ == "__main__":
    main()