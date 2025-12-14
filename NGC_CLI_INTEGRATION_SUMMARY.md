# NGC CLI Integration Summary

## Overview

NGC CLI (NVIDIA GPU Cloud CLI) has been integrated into both `training_ready` and `dataset_pipeline` systems for downloading NeMo resources, training frameworks, and datasets from the NGC catalog.

## What Was Added

### 1. Shared NGC CLI Utility Module

**Location**: `ai/utils/ngc_cli.py`

A comprehensive utility module that provides:
- NGC CLI detection (system, local, or uv-based installation)
- Automatic installation via uv if needed
- Configuration management (API key setup)
- Resource download functionality
- Error handling and retry logic

**Key Classes**:
- `NGCCLI`: Main wrapper class for NGC CLI operations
- `NGCConfig`: Configuration dataclass
- Custom exceptions: `NGCCLINotFoundError`, `NGCCLIAuthError`, `NGCCLIDownloadError`

### 2. Training Ready Integration

**Location**: `ai/training_ready/utils/ngc_resources.py`

Provides:
- `NGCResourceDownloader`: Downloads NeMo resources for training
- Pre-configured resources: NeMo Microservices quickstart, NeMo framework, NeMo Megatron
- Convenience functions for common downloads

**Usage**:
```python
from ai.training_ready.utils.ngc_resources import download_nemo_quickstart

quickstart_path = download_nemo_quickstart(version="25.10")
```

### 3. Dataset Pipeline Integration

**Location**: `ai/dataset_pipeline/sourcing/ngc_ingestor.py`

Provides:
- `NGCIngestor`: Downloads datasets from NGC catalog
- Similar interface to `HuggingFaceIngestor` for consistency
- Integrated into `multi_source_ingestor.py` automatically

**Usage**:
```python
from ai.dataset_pipeline.sourcing.ngc_ingestor import ingest_ngc_datasets

results = ingest_ngc_datasets()
```

### 4. Documentation

- **`ai/training_ready/docs/NGC_CLI_INTEGRATION.md`**: Complete guide for training_ready
- **`ai/dataset_pipeline/docs/NGC_INTEGRATION.md`**: Complete guide for dataset_pipeline

## Installation

**Important**: NGC CLI is not available as a PyPI package. You must download it directly from NVIDIA.

### Quick Setup

```bash
# Download NGC CLI from https://catalog.ngc.nvidia.com
# Extract to ~/ngc-cli/ or add to system PATH

# Get API key from https://catalog.ngc.nvidia.com
export NGC_API_KEY="your-api-key-here"

# Or configure via CLI
ngc config set
```

## Integration Points

### Training Ready

- **Module**: `ai/training_ready/utils/ngc_resources.py`
- **Exported**: `NGCResourceDownloader`, `download_nemo_quickstart`
- **Use Cases**: Download NeMo Microservices, training frameworks, custom resources

### Dataset Pipeline

- **Module**: `ai/dataset_pipeline/sourcing/ngc_ingestor.py`
- **Integrated**: Automatically included in `multi_source_ingestor.py`
- **Use Cases**: Download NGC catalog datasets for training

## Features

1. **Automatic Detection**: Finds NGC CLI in PATH, local installation, or via uv
2. **Auto-Installation**: Can install NGC CLI via uv if not found
3. **Configuration Management**: Handles API key setup and validation
4. **Error Handling**: Graceful degradation if NGC CLI unavailable
5. **Consistent Interface**: Similar API to HuggingFace ingestor for familiarity

## Requirements

Added to:
- `ai/training_ready/packages/velocity/configs/requirements_moe.txt`
- `ai/dataset_pipeline/utils/requirements.txt`

Note: NGC CLI is optional - systems will work without it, but with limited NGC resource access.

## Next Steps

1. **Configure NGC API Key**: Get from https://catalog.ngc.nvidia.com
2. **Test Integration**: Try downloading a NeMo resource
3. **Add More Resources**: Configure additional NGC datasets/resources as needed

## Related Files

- `ai/utils/ngc_cli.py` - Core NGC CLI utility
- `ai/training_ready/utils/ngc_resources.py` - Training resources downloader
- `ai/dataset_pipeline/sourcing/ngc_ingestor.py` - Dataset ingestor
- `ai/dataset_pipeline/sourcing/multi_source_ingestor.py` - Multi-source integration
- `scripts/infrastructure/deploy-nemo-data-designer-remote-server.sh` - Existing NGC usage

## Notes

- NGC CLI is gitignored (`.gitignore` already has `ngc-cli/`)
- Integration is backward compatible - existing code continues to work
- NGC CLI detection follows the same pattern as in deployment scripts
- Supports both system-installed and uv-managed installations
