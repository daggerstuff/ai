# NGC Integration for Dataset Pipeline

## Overview

NGC (NVIDIA GPU Cloud) integration allows the dataset pipeline to download datasets and resources directly from the NGC catalog, similar to how HuggingFace datasets are ingested.

## Installation

**Important**: NGC CLI is not available as a PyPI package. You must download it directly from NVIDIA.

### Download from NGC Catalog

1. Visit https://catalog.ngc.nvidia.com
2. Sign in with your NVIDIA account
3. Search for "NGC CLI" or navigate to CLI resources
4. Download and extract to `~/ngc-cli/` or add to your PATH

### Manual Installation

```bash
# Create directory
mkdir -p ~/ngc-cli
cd ~/ngc-cli

# Download NGC CLI (get latest URL from NGC catalog)
wget https://ngc.nvidia.com/downloads/ngccli_linux.zip
unzip ngccli_linux.zip
chmod +x ngc

# Add to PATH
export PATH=$HOME/ngc-cli:$PATH
```

## Configuration

### Get NGC API Key

1. Visit https://catalog.ngc.nvidia.com
2. Sign in with your NVIDIA account
3. Go to your profile → Setup → Generate API Key
4. Copy the API key

### Set Environment Variable

```bash
export NGC_API_KEY="your-api-key-here"
```

Or configure via NGC CLI:

```bash
ngc config set
# Enter your API key when prompted
```

## Usage

### Basic Usage

```python
from ai.dataset_pipeline.sourcing.ngc_ingestor import NGCIngestor, ingest_ngc_datasets

# Ingest all configured NGC datasets
results = ingest_ngc_datasets()

# Or use the ingestor directly
ingestor = NGCIngestor()
if ingestor.is_available():
    results = ingestor.process_all_configured()
```

### Download Specific Dataset

```python
from ai.dataset_pipeline.sourcing.ngc_ingestor import NGCIngestor

ingestor = NGCIngestor()

# Download a specific dataset
downloaded_path = ingestor.download_dataset(
    resource_path="nvidia/nemo/datasets/therapeutic_conversations",
    version="1.0",
    target_folder="therapeutic_conversations"
)
```

### Process Custom Resource

```python
from ai.dataset_pipeline.sourcing.ngc_ingestor import NGCIngestor

ingestor = NGCIngestor()

result = ingestor.process_dataset(
    resource_path="nvidia/nemo/datasets/my-dataset",
    config={
        'target': 'stage1_foundation',
        'version': '1.0',
        'description': 'My custom dataset'
    }
)
```

### Integration with Multi-Source Ingestor

The NGC ingestor can be integrated with the existing multi-source ingestor:

```python
from ai.dataset_pipeline.sourcing.multi_source_ingestor import MultiSourceIngestor
from ai.dataset_pipeline.sourcing.ngc_ingestor import NGCIngestor

# Add NGC as a source
ingestor = MultiSourceIngestor()
ngc_ingestor = NGCIngestor()

if ngc_ingestor.is_available():
    ngc_results = ngc_ingestor.process_all_configured()
    # Integrate results with other sources
```

## Available Datasets

Currently configured NGC datasets:

- **nvidia/nemo/datasets/therapeutic_conversations**: Therapeutic conversation dataset from NeMo
  - Target: `stage1_foundation`
  - Version: Latest

Add more datasets to `NGC_DATASETS` in `ngc_ingestor.py` as they become available.

## Configuration

### Adding New NGC Datasets

Edit `ai/dataset_pipeline/sourcing/ngc_ingestor.py`:

```python
NGC_DATASETS = {
    'nvidia/nemo/datasets/new-dataset': {
        'target': 'stage1_foundation',
        'version': '1.0',
        'description': 'New dataset description'
    }
}
```

## Output Structure

NGC datasets are downloaded to:

```
ai/dataset_pipeline/data/ngc/
├── therapeutic_conversations/
│   └── [downloaded files]
└── [other datasets]/
```

## Error Handling

The NGC ingestor gracefully handles:

- **NGC CLI not available**: Returns empty results, logs warning
- **Authentication errors**: Raises `NGCCLIAuthError` with helpful message
- **Download failures**: Returns error status in result dictionary

```python
result = ingestor.process_dataset(resource_path, config)
if result['status'] == 'error':
    print(f"Error: {result['error']}")
```

## Troubleshooting

### "NGC CLI not available"

**Solution**: Install NGC CLI

```bash
uv pip install nvidia-pyindex nvidia-nim ngc-python-cli
```

### "Authentication failed"

**Solution**: Set NGC_API_KEY environment variable or configure via CLI

```bash
export NGC_API_KEY="your-api-key"
# Or
ngc config set
```

### "Download failed"

**Possible causes**:
1. Resource path incorrect
2. No access to resource
3. Network issues

**Solution**: Check resource path and access permissions

## Related Documentation

- [Dataset Pipeline Overview](../README.md)
- [HuggingFace Ingestor](./HF_INGESTOR.md)
- [Multi-Source Ingestor](./MULTI_SOURCE.md)
- [Training Ready NGC Integration](../../training_ready/docs/NGC_CLI_INTEGRATION.md)
