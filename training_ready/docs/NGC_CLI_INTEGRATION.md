# NGC CLI Integration for Training Ready

## Overview

NGC CLI (NVIDIA GPU Cloud CLI) is integrated into `training_ready` for downloading NeMo resources, training frameworks, and other assets from the NGC catalog.

## Installation

**Important**: NGC CLI is not available as a PyPI package. You must download it directly from NVIDIA.

### Option 1: Download from NGC Catalog (Recommended)

1. Visit https://catalog.ngc.nvidia.com
2. Sign in with your NVIDIA account
3. Navigate to the NGC CLI resource
4. Download and extract to `~/ngc-cli/` or add to your PATH

### Option 2: Manual Installation

```bash
# Create directory
mkdir -p ~/ngc-cli
cd ~/ngc-cli

# Download NGC CLI (get latest URL from NGC catalog)
# Example (check for latest version):
wget https://ngc.nvidia.com/downloads/ngccli_linux.zip
unzip ngccli_linux.zip
chmod +x ngc

# Add to PATH
export PATH=$HOME/ngc-cli:$PATH
echo 'export PATH=$HOME/ngc-cli:$PATH' >> ~/.bashrc
```

### Option 3: System-Wide Installation

Follow NVIDIA's official installation guide for system-wide installation.

## Configuration

### Get NGC API Key

1. Visit https://catalog.ngc.nvidia.com
2. Sign in with your NVIDIA account
3. Go to your profile → Setup → Generate API Key
4. Copy the API key

**Note**: This is different from `NVIDIA_API_KEY` used for Docker registry login.

### Configure NGC CLI

```bash
# Set API key
ngc config set
# Enter your API key when prompted

# Or set via environment variable
export NGC_API_KEY="your-api-key-here"
```

## Usage

### Download NeMo Microservices Quickstart

```python
from ai.training_ready.utils.ngc_resources import download_nemo_quickstart

# Download latest version (defaults to 25.10)
quickstart_path = download_nemo_quickstart()

# Download specific version
quickstart_path = download_nemo_quickstart(version="25.10")

# Download to specific directory
from pathlib import Path
quickstart_path = download_nemo_quickstart(
    version="25.10",
    output_dir=Path("resources/nemo-microservices")
)
```

### Using NGCResourceDownloader

```python
from ai.training_ready.utils.ngc_resources import NGCResourceDownloader
from pathlib import Path

# Initialize downloader
downloader = NGCResourceDownloader(
    output_base=Path("resources"),
    api_key="your-api-key"  # Optional, can use NGC_API_KEY env var
)

# Download NeMo quickstart
quickstart = downloader.download_nemo_quickstart(version="25.10")

# Download NeMo framework
framework = downloader.download_nemo_framework()

# Download custom resource
custom = downloader.download_custom_resource(
    resource_path="nvidia/nemo/datasets/my-dataset",
    version="1.0",
    output_dir=Path("resources/my-dataset")
)
```

### Direct NGC CLI Usage

```python
from ai.utils.ngc_cli import get_ngc_cli, ensure_ngc_cli_configured

# Get NGC CLI instance
cli = get_ngc_cli()

# Ensure it's configured
cli = ensure_ngc_cli_configured(api_key="your-api-key")

# Download any resource
downloaded = cli.download_resource(
    resource_path="nvidia/nemo-microservices/nemo-microservices-quickstart",
    version="25.10",
    output_dir=Path("resources"),
    extract=True
)

# Check configuration
config = cli.check_config()
print(f"API Key configured: {bool(config.get('API key'))}")
```

## Integration with Training Scripts

NGC resources are automatically downloaded when needed by training scripts. The integration handles:

- Automatic NGC CLI detection (system, local, or uv-based)
- Configuration checking and setup
- Error handling and retry logic
- Resource caching

### Example: Training Script Integration

```python
from ai.training_ready.utils.ngc_resources import NGCResourceDownloader

def setup_training_environment():
    """Set up training environment with required NeMo resources"""
    downloader = NGCResourceDownloader()
    
    try:
        # Download NeMo quickstart if needed
        quickstart = downloader.download_nemo_quickstart()
        print(f"NeMo quickstart available at: {quickstart}")
    except Exception as e:
        print(f"Warning: Could not download NeMo resources: {e}")
        print("Training may continue with local resources")
```

## Available Resources

### NeMo Microservices Quickstart

- **Path**: `nvidia/nemo-microservices/nemo-microservices-quickstart`
- **Default Version**: `25.10`
- **Description**: Complete NeMo Microservices deployment package

### NeMo Framework

- **Path**: `nvidia/nemo/nemo`
- **Description**: NeMo framework for training

### NeMo Megatron

- **Path**: `nvidia/nemo/nemo-megatron`
- **Description**: NeMo Megatron for large-scale training

## Troubleshooting

### "NGC CLI not found"

**Solution**: Install NGC CLI using one of the methods above.

```bash
uv pip install nvidia-pyindex nvidia-nim ngc-python-cli
```

### "Authentication failed"

**Solution**: Configure NGC CLI with your API key.

```bash
ngc config set
# Enter your API key from https://catalog.ngc.nvidia.com
```

### "Download failed"

**Possible causes**:
1. API key not configured
2. No access to the resource
3. Network connectivity issues

**Solution**:
```bash
# Check configuration
ngc config get

# Test connectivity
curl -I https://catalog.ngc.nvidia.com
```

## Related Documentation

- [NeMo Deployment Troubleshooting](../../docs/nemo-deployment-troubleshooting.md)
- [NGC CLI Official Docs](https://docs.nvidia.com/ngc/)
- [Training Ready README](../README.md)
