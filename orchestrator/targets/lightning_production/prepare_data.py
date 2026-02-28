#!/usr/bin/env python3
"""
Prepare unified dataset for Lightning.ai H100 deployment
"""

import json
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_lightning_data():
    """Prepare data for Lightning.ai deployment"""
    from path_utils import get_unified_training_dir, get_lightning_dir
    source_dir = get_unified_training_dir()
    target_dir = get_lightning_dir() / "production/data"
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all dataset files
    required_files = [
        "train.json",
        "validation.json",
        "expert_therapeutic.json", 
        "expert_educational.json",
        "expert_empathetic.json",
        "expert_practical.json",
        "unified_lightning_config.json",
        "comprehensive_processing_report.json"
    ]
    
    for filename in required_files:
        source_file = source_dir / filename
        target_file = target_dir / filename
        
        if source_file.exists():
            shutil.copy2(source_file, target_file)
            logger.info(f"✅ Copied {filename}")
        else:
            logger.warning(f"⚠️  Missing {filename}")
    
    # Create deployment summary
    summary = {
        "preparation_complete": True,
        "files_copied": len([f for f in required_files if (source_dir / f).exists()]),
        "total_files": len(required_files),
        "data_ready_for_lightning": True
    }
    
    with open(target_dir / "deployment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"🚀 Data preparation complete: {target_dir}")
    return target_dir

if __name__ == "__main__":
    prepare_lightning_data()
