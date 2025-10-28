#!/usr/bin/env python3
"""
Monitor Multi-Dataset Processing Progress
Track the intelligent agent processing and prepare for next steps.
"""

import json
import time
from pathlib import Path
import subprocess

def check_processing_status():
    """Check if the multi-dataset pipeline is still running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'multi_dataset_intelligent_pipeline'], 
                              capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False

def check_output_progress():
    """Check progress in output directory"""
    from path_utils import get_unified_training_dir
    output_dir = get_unified_training_dir()
    
    if not output_dir.exists():
        return {"status": "processing", "files": [], "total_conversations": 0}
    
    files = list(output_dir.glob("*.json"))
    total_conversations = 0
    
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    total_conversations += len(data)
        except:
            continue
    
    return {
        "status": "completed" if any(f.name == "unified_lightning_config.json" for f in files) else "processing",
        "files": [f.name for f in files],
        "total_conversations": total_conversations
    }

def main():
    print("🔍 Monitoring Multi-Dataset Intelligent Processing Pipeline...")
    
    while True:
        is_running = check_processing_status()
        progress = check_output_progress()
        
        print(f"\n📊 Status Update:")
        print(f"   Pipeline Running: {'✅ Yes' if is_running else '❌ No'}")
        print(f"   Output Status: {progress['status']}")
        print(f"   Files Created: {len(progress['files'])}")
        print(f"   Total Conversations: {progress['total_conversations']}")
        
        if progress['files']:
            print(f"   Created Files: {', '.join(progress['files'])}")
        
        if not is_running and progress['status'] == 'completed':
            print("\n🎉 Processing Complete!")
            
            # Show final results
            from path_utils import get_unified_training_dir
            config_file = get_unified_training_dir() / "unified_lightning_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                stats = config.get('dataset_stats', {}).get('processing_stats', {})
                print(f"\n📈 Final Results:")
                print(f"   Total Sources: {stats.get('total_sources', 0)}")
                print(f"   Total Files: {stats.get('total_files', 0)}")
                print(f"   Processed Conversations: {stats.get('processed_conversations', 0)}")
                print(f"   High Quality: {stats.get('high_quality', 0)}")
                print(f"   Medium Quality: {stats.get('medium_quality', 0)}")
                print(f"   Extracted Questions: {stats.get('extracted_questions', 0)}")
                print(f"   Contextual Questions: {stats.get('contextual_questions', 0)}")
                print(f"   Duplicates Removed: {stats.get('duplicates_removed', 0)}")
                
                print(f"\n🚀 Ready for Lightning.ai H100 Training!")
                print(f"📁 Dataset Location: /root/pixelated/data/unified_training/")
            
            break
        
        if not is_running:
            print("\n⚠️  Processing stopped but not completed. Check for errors.")
            break
            
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    main()