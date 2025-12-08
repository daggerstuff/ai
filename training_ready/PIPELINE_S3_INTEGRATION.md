# Pipeline S3 Integration - Complete

## ✅ Integration Complete

All pipeline scripts have been updated to support S3 storage with automatic fallback to local files.

## Updated Components

### 1. Path Resolver (`tools/data_preparation/path_resolver.py`)
**Purpose**: Unified interface for resolving S3 and local paths

**Features**:
- Automatically detects S3 vs local paths
- Checks S3 first, falls back to local
- Loads datasets from either source
- Transparent to calling code

**Usage**:
```python
from ai.training_ready.tools.data_preparation.path_resolver import get_resolver

resolver = get_resolver()
resolved_path, source_type = resolver.resolve_path(path, manifest_entry)

# Load dataset
for record in resolver.load_dataset(resolved_path, source_type):
    # Process record
    pass
```

### 2. Source Datasets (`tools/data_preparation/source_datasets.py`)
**Updates**:
- ✅ Checks S3 paths from manifest first
- ✅ Falls back to local if S3 not available
- ✅ New `source_s3_dataset()` method
- ✅ Automatic S3 path resolution

**Behavior**:
1. Checks manifest for `s3_path` field
2. Verifies file exists in S3
3. Returns S3 path if available
4. Falls back to local path if S3 unavailable

### 3. Process All Datasets (`pipelines/integrated/process_all_datasets.py`)
**Updates**:
- ✅ Detects S3 paths in sourcing results
- ✅ Downloads S3 files to cache for processing
- ✅ Caches downloads to avoid re-downloading
- ✅ Transparent to downstream pipeline

**Behavior**:
1. Detects S3 paths (`s3://` or `source_type == "s3"`)
2. Downloads to `ai/training_ready/datasets/cache/s3/`
3. Uses cached file if already downloaded
4. Processes cached file through pipeline

### 4. Assemble Final Dataset (`pipelines/integrated/assemble_final_dataset.py`)
**Updates**:
- ✅ Loads directly from S3 when available
- ✅ Falls back to local files
- ✅ Updated `load_conversations_from_file()` to support S3

**Behavior**:
1. Checks if path is S3 (`s3://`)
2. Loads directly from S3 using resolver
3. Falls back to local file if not S3

## Data Flow

```
Manifest (with s3_path)
    ↓
Source Datasets
    ├─→ Check S3 first ✅
    └─→ Fallback to local
    ↓
Process All Datasets
    ├─→ S3: Download to cache
    └─→ Local: Use directly
    ↓
Assemble Final Dataset
    ├─→ S3: Load directly
    └─→ Local: Load from file
```

## Benefits

1. **Seamless Integration**: Pipeline works with S3 or local transparently
2. **Automatic Fallback**: Always tries S3 first, falls back gracefully
3. **Caching**: S3 files cached locally to avoid re-downloads
4. **Backward Compatible**: Still works with local-only setups

## Usage

### Running Pipeline with S3

```bash
# Set environment variables
set -a
source ai/.env
set +a

# Run pipeline (automatically uses S3 if available)
uv run python3 ai/training_ready/scripts/prepare_training_data.py --all
```

### Manual S3 Dataset Loading

```python
from ai.training_ready.tools.data_preparation.s3_dataset_loader import S3DatasetLoader

loader = S3DatasetLoader()
for record in loader.load_jsonl("s3://pixel-data/datasets/huggingface/...jsonl"):
    print(record)
```

## Cache Management

S3 files are cached to:
- Location: `ai/training_ready/datasets/cache/s3/`
- Naming: MD5 hash of S3 path + file extension
- Behavior: Reuses cache if file exists

To clear cache:
```bash
rm -rf ai/training_ready/datasets/cache/s3/*
```

## Testing

### Test S3 Integration
```bash
# Test S3 loader
uv run python3 ai/training_ready/tools/data_preparation/s3_dataset_loader.py \
    s3://pixel-data/datasets/huggingface/huggingface/ShreyaR_DepressionDetection.jsonl 10

# Test path resolver
uv run python3 -c "
from ai.training_ready.tools.data_preparation.path_resolver import get_resolver
resolver = get_resolver()
path, source = resolver.resolve_path('some/path.jsonl', {'s3_path': 's3://bucket/key.jsonl'})
print(f'Resolved: {path} ({source})')
"
```

## Files Modified

- ✅ `tools/data_preparation/path_resolver.py` - NEW
- ✅ `tools/data_preparation/source_datasets.py` - UPDATED
- ✅ `pipelines/integrated/process_all_datasets.py` - UPDATED
- ✅ `pipelines/integrated/assemble_final_dataset.py` - UPDATED

## Next Steps

1. **Test End-to-End**: Run full pipeline with S3 datasets
2. **Performance Testing**: Benchmark S3 vs local performance
3. **Error Handling**: Add retry logic for S3 operations
4. **Documentation**: Update user-facing docs

## Notes

- S3 credentials must be in environment (`OVH_S3_*` or `AWS_*`)
- Pipeline automatically handles S3/local switching
- Cache directory created automatically
- All S3 operations are logged for debugging

