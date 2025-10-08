
# Dataset Export Integration Guide

## Overview
The export API provides secure, tiered access to validated therapeutic conversations in multiple formats.

## Access Tiers
- **Priority**: Highest quality clinical conversations (99% accuracy)
- **Professional**: Professional therapist conversations (95% accuracy)
- **CoT**: Chain-of-thought reasoning conversations (90% accuracy)
- **Reddit**: Community-sourced conversations (85% accuracy)
- **Research**: Research paper conversations (80% accuracy)
- **Archive**: Historical conversations (75% accuracy)

## Export Formats
- **JSON**: Structured conversation data
- **CSV**: Tabular format for analysis
- **Parquet**: Optimized for big data processing
- **HuggingFace**: Ready for transformer training
- **JSONL**: Line-delimited JSON for streaming

## Integration Example

### 1. Configure Export
```python
export_config = {
    "formats": ["json", "huggingface"],
    "access_tiers": ["priority", "professional"],
    "quality_threshold": 0.8,
    "include_metadata": True,
    "compress_output": True,
    "max_conversations_per_file": 10000
}

filters = {
    "conditions": ["anxiety", "depression"],
    "approaches": ["CBT", "DBT"],
    "date_range": {
        "start": "2025-01-01",
        "end": "2025-08-10"
    }
}
```

### 2. Request Export
```python
response = requests.post(
    "https://api.pixelated-empathy.com/api/v1/export/dataset",
    headers={"Authorization": "Bearer your_api_key"},
    json={
        "export_config": export_config,
        "filters": filters
    }
)
```

### 3. Download Files
```python
if response.status_code == 200:
    export_data = response.json()
    
    for metadata in export_data["export_metadata"]:
        # Download each export file
        file_url = f"https://api.pixelated-empathy.com{metadata['file_path']}"
        file_response = requests.get(file_url, headers=headers)
        
        # Verify checksum
        import hashlib
        actual_checksum = hashlib.sha256(file_response.content).hexdigest()
        expected_checksum = metadata["checksum"].split(":")[1]
        
        if actual_checksum == expected_checksum:
            # Save file
            with open(f"dataset_{metadata['tier']}.{metadata['format']}", 'wb') as f:
                f.write(file_response.content)
```

## Security Considerations
- API keys provide tier-specific access
- All downloads are logged and audited
- Checksums ensure data integrity
- Rate limits prevent abuse
        