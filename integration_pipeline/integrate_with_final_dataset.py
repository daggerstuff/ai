#!/usr/bin/env python3
"""
Integrate 6-Component Dataset with 2.5GB Final Dataset - KAN-28 Step 2
The REAL final step - adds our enhanced components to the existing massive dataset
"""

import json
import sys
from pathlib import Path

def integrate_with_final_dataset():
    """Integrate the unified 6-component dataset with the 2.5GB final dataset"""
    
    print("🚀 FINAL INTEGRATION: Adding 6-component data to 2.5GB dataset...")
    
    # Paths
    component_dataset_path = "ai/training_data_consolidated/unified_6_component/unified_6_component_dataset.jsonl"
    final_dataset_path = "../training_data_consolidated/final_datasets/merged_dataset.jsonl"
    output_path = "../training_data_consolidated/final_datasets/ULTIMATE_FINAL_DATASET.jsonl"
    
    # Verify files exist
    if not Path(component_dataset_path).exists():
        print(f"❌ Component dataset not found: {component_dataset_path}")
        return
    
    if not Path(final_dataset_path).exists():
        print(f"❌ Final dataset not found: {final_dataset_path}")
        return
    
    print(f"✅ Found component dataset: {component_dataset_path}")
    print(f"✅ Found final dataset: {final_dataset_path}")
    
    # Load 6-component dataset
    print("\n📥 Loading 6-component dataset...")
    component_conversations = []
    with open(component_dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                component_conversations.append(json.loads(line))
    
    print(f"✅ Loaded {len(component_conversations)} component-enhanced conversations")
    
    # Process final dataset and add component data
    print(f"\n🔄 Processing 2.5GB final dataset and adding component enhancements...")
    
    total_original = 0
    total_enhanced = 0
    
    with open(final_dataset_path, 'r') as input_file, open(output_path, 'w') as output_file:
        
        # First, add all original conversations from final dataset
        for line_num, line in enumerate(input_file):
            if line.strip():
                total_original += 1
                
                # Progress update every 10000 lines
                if total_original % 10000 == 0:
                    print(f"   Processed {total_original} original conversations...")
                
                # Parse original conversation
                original_conv = json.loads(line)
                
                # Add metadata indicating this is from original final dataset
                original_conv["dataset_source"] = "original_final_dataset"
                original_conv["integration_metadata"] = {
                    "component_enhanced": False,
                    "original_final_dataset": True
                }
                
                # Write to output
                output_file.write(json.dumps(original_conv) + '\n')
        
        # Then add all the component-enhanced conversations
        print(f"\n➕ Adding {len(component_conversations)} component-enhanced conversations...")
        
        for enhanced_conv in component_conversations:
            # Mark as component enhanced
            enhanced_conv["dataset_source"] = "component_enhanced"
            enhanced_conv["added_to_final"] = True
            
            output_file.write(json.dumps(enhanced_conv) + '\n')
            total_enhanced += 1
    
    # Create integration summary
    integration_summary = {
        "integration_type": "final_dataset_with_components",
        "original_conversations": total_original,
        "component_enhanced_conversations": total_enhanced,
        "total_conversations": total_original + total_enhanced,
        "components_added": [
            "journaling_system",
            "voice_blending", 
            "edge_case_handling",
            "dual_persona_dynamics",
            "bias_detection",
            "psychology_knowledge_base"
        ],
        "expert_voices_added": ["Tim Ferriss", "Gabor Maté", "Brené Brown"],
        "psychology_concepts_added": 4867,
        "original_dataset_size": "2.5GB",
        "kan_28_complete": True,
        "final_dataset_path": output_path
    }
    
    summary_path = "../training_data_consolidated/final_datasets/ULTIMATE_FINAL_INTEGRATION_SUMMARY.json"
    with open(summary_path, 'w') as f:
        json.dump(integration_summary, f, indent=2)
    
    # Get file sizes
    original_size = Path(final_dataset_path).stat().st_size / (1024**3)  # GB
    new_size = Path(output_path).stat().st_size / (1024**3)  # GB
    
    print(f"\n🎉 ULTIMATE FINAL DATASET CREATED!")
    print(f"📁 Original dataset: {final_dataset_path} ({original_size:.2f}GB)")
    print(f"📁 NEW FINAL DATASET: {output_path} ({new_size:.2f}GB)")
    print(f"📁 Integration summary: {summary_path}")
    print(f"📊 Original conversations: {total_original:,}")
    print(f"📊 Component-enhanced conversations: {total_enhanced:,}")
    print(f"📊 TOTAL conversations: {total_original + total_enhanced:,}")
    print(f"🎯 KAN-28 OBJECTIVE COMPLETE: Final dataset now includes ALL components!")
    
    return output_path, integration_summary

if __name__ == "__main__":
    integrate_with_final_dataset()