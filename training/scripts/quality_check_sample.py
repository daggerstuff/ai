#!/usr/bin/env python3
"""
Quick quality check on dataset samples
"""
import json
from pathlib import Path

def check_jsonl_file(filepath, max_samples=5):
    """Check a JSONL file for common issues"""
    results = {
        "file": str(filepath),
        "total_lines": 0,
        "valid_json": 0,
        "has_messages": 0,
        "has_metadata": 0,
        "languages": set(),
        "sample_issues": []
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                results["total_lines"] += 1
                if i >= max_samples:
                    break
                    
                try:
                    data = json.loads(line)
                    results["valid_json"] += 1
                    
                    if "messages" in data:
                        results["has_messages"] += 1
                        # Check message structure
                        for msg in data.get("messages", []):
                            if "role" not in msg or "content" not in msg:
                                results["sample_issues"].append(f"Line {i}: Missing role/content in message")
                            
                    if "metadata" in data:
                        results["has_metadata"] += 1
                        
                except json.JSONDecodeError as e:
                    results["sample_issues"].append(f"Line {i}: JSON error - {str(e)}")
    except Exception as e:
        results["sample_issues"].append(f"File read error: {str(e)}")
    
    return results

# Check all local datasets
base_path = Path("/home/vivi/pixelated/ai/datasets/training_v3")
all_results = []

for stage in ["stage1_foundation", "stage2_specialist_addiction", "stage2_specialist_personality", "stage4_voice_persona"]:
    stage_path = base_path / stage
    if stage_path.exists():
        for jsonl_file in stage_path.glob("*.jsonl"):
            result = check_jsonl_file(jsonl_file)
            all_results.append(result)
            print(f"\n📄 {result['file']}")
            print(f"   Lines: {result['total_lines']} | Valid JSON: {result['valid_json']} | Has messages: {result['has_messages']} | Has metadata: {result['has_metadata']}")
            if result["sample_issues"]:
                for issue in result["sample_issues"][:3]:
                    print(f"   ⚠️  {issue}")

# Summary
print("\n" + "="*60)
print("QUALITY CHECK SUMMARY")
print("="*60)
total_lines = sum(r["total_lines"] for r in all_results)
valid_json = sum(r["valid_json"] for r in all_results)
print(f"Total lines checked: {total_lines}")
print(f"Valid JSON: {valid_json}/{total_lines} ({100*valid_json/total_lines:.1f}%)")
print(f"Files with issues: {sum(1 for r in all_results if r['sample_issues'])}")
