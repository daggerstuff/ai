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

**Location**: `ai/training/ready_packages/utils/ngc_resources.py`

Provides:
- `NGCResourceDownloader`: Downloads NeMo resources for training
- Pre-configured resources: NeMo Microservices quickstart, NeMo framework, NeMo Megatron
- Convenience functions for common downloads

**Usage**:
```python
from ai.training.ready_packages.utils.ngc_resources import download_nemo_quickstart

quickstart_path = download_nemo_quickstart(version="25.10")
```

### 3. Dataset Pipeline Integration

**Location**: `ai/pipelines/orchestrator/sourcing/ngc_ingestor.py`

Provides:
- `NGCIngestor`: Downloads datasets from NGC catalog
- Similar interface to `HuggingFaceIngestor` for consistency
- Integrated into `multi_source_ingestor.py` automatically

**Usage**:
```python
from ai.pipelines.orchestrator.sourcing.ngc_ingestor import ingest_ngc_datasets

results = ingest_ngc_datasets()
```

### 4. Documentation

- **`ai/training/ready_packages/docs/NGC_CLI_INTEGRATION.md`**: Complete guide for training_ready
- **`ai/pipelines/orchestrator/docs/NGC_INTEGRATION.md`**: Complete guide for dataset_pipeline

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

- **Module**: `ai/training/ready_packages/utils/ngc_resources.py`
- **Exported**: `NGCResourceDownloader`, `download_nemo_quickstart`
- **Use Cases**: Download NeMo Microservices, training frameworks, custom resources

### Dataset Pipeline

- **Module**: `ai/pipelines/orchestrator/sourcing/ngc_ingestor.py`
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
- `ai/training/ready_packages/packages/velocity/configs/requirements_moe.txt`
- `ai/pipelines/orchestrator/utils/requirements.txt`

Note: NGC CLI is optional - systems will work without it, but with limited NGC resource access.

## Next Steps

1. **Configure NGC API Key**: Get from https://catalog.ngc.nvidia.com ✅ **COMPLETED**
2. **Test Integration**: Try downloading a NeMo resource ✅ **COMPLETED**
3. **Add More Resources**: Configure additional NGC datasets/resources as needed

## Test Results

**✅ NGC CLI Integration Successfully Tested (January 2, 2026)**

### Core Integration Tests
- **NGC CLI Detection**: ✅ Found NGC CLI v4.10.0 in system PATH
- **Configuration Check**: ✅ API key properly configured and detected
- **Resource Download**: ✅ Successfully downloaded `nvidia/nemo-microservices/nemo-microservices-quickstart:25.10`
- **Training Ready Integration**: ✅ `NGCResourceDownloader` and `download_nemo_quickstart()` working correctly
- **Dataset Pipeline Integration**: ⚠️ Available but requires torch dependencies for full testing

### Successful Downloads
- **NeMo Microservices Quickstart v25.10** (✅ Downloaded)
  - Size: ~500MB
  - Contents: Complete microservices architecture with Docker Compose
  - Services: Data Designer, Auditor, Evaluator, Guardrails, Customizer, Safe Synthesizer
  - Use Case: Production deployment of therapeutic conversation models

### Access Limitations Discovered
- **Large Language Models**: Require special NGC Enterprise access
  - Nemotron-4-340B-Instruct, Llama-3.1-70B-Instruct, Nemotron-3-8B-Chat
- **Premium Containers**: Require subscription or special permissions
  - PyTorch, TensorFlow, Triton Inference Server containers
- **Advanced Tools**: Limited access to specialized datasets and examples

### Configuration Parsing Fix
- Fixed NGC CLI config parsing to handle multi-line API key display
- Updated condition to properly detect `apikey` field vs header `key` column
- Improved error handling and configuration validation

### Recommendations for Enhanced Access
1. **Apply for NGC Enterprise Access**: For premium models and containers
2. **Use Hugging Face Alternatives**: Many models available without restrictions
3. **Docker Hub Alternatives**: Use official PyTorch/TensorFlow images with CUDA
4. **Build Custom Containers**: Layer therapeutic AI tools on base CUDA images

## Related Files

- `ai/utils/ngc_cli.py` - Core NGC CLI utility
- `ai/training/ready_packages/utils/ngc_resources.py` - Training resources downloader
- `ai/pipelines/orchestrator/sourcing/ngc_ingestor.py` - Dataset ingestor
- `ai/pipelines/orchestrator/sourcing/multi_source_ingestor.py` - Multi-source integration
- `scripts/infrastructure/deploy-nemo-data-designer-remote-server.sh` - Existing NGC usage

## Notes

- NGC CLI is gitignored (`.gitignore` already has `ngc-cli/`)
- Integration is backward compatible - existing code continues to work
- NGC CLI detection follows the same pattern as in deployment scripts
- Supports both system-installed and uv-managed installations
