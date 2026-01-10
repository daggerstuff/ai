# NGC CLI Integration - Next Steps

## ✅ What's Done

- ✅ NGC CLI utility module created (`ai/utils/ngc_cli.py`)
- ✅ Training Ready integration (`ai/training_ready/utils/ngc_resources.py`)
- ✅ Dataset Pipeline integration (`ai/dataset_pipeline/sourcing/ngc_ingestor.py`)
- ✅ Documentation created
- ✅ Integrated into multi-source ingestor

## 🚀 Next Steps

### 1. Install NGC CLI

NGC CLI is **not** a PyPI package - you must download it from NVIDIA:

```bash
# Option 1: Download from NGC Catalog
# Visit: https://catalog.ngc.nvidia.com
# Search for "NGC CLI" and download

# Option 2: Direct download (check for latest version)
mkdir -p ~/ngc-cli
cd ~/ngc-cli
wget https://ngc.nvidia.com/downloads/ngccli_linux.zip
unzip ngccli_linux.zip
chmod +x ngc

# Add to PATH
export PATH=$HOME/ngc-cli:$PATH
echo 'export PATH=$HOME/ngc-cli:$PATH' >> ~/.bashrc
```

### 2. Get NGC API Key

1. Visit https://catalog.ngc.nvidia.com
2. Sign in with your NVIDIA account
3. Go to your profile → Setup → Generate API Key
4. Copy the API key

**Note**: This is different from `NVIDIA_API_KEY` used for Docker registry.

### 3. Configure NGC CLI

```bash
# Set API key
ngc config set
# Enter your API key when prompted

# Or set via environment variable
export NGC_API_KEY="your-api-key-here"
```

### 4. Test the Integration

```bash
# Test NGC CLI availability
cd /home/vivi/pixelated
python ai/utils/test_ngc_cli.py
```

This will verify:
- NGC CLI is found
- Configuration is set up
- Can access NGC catalog

### 5. Try Downloading a Resource

**Training Ready:**
```python
from ai.training_ready.utils.ngc_resources import download_nemo_quickstart

# Download NeMo Microservices quickstart
quickstart_path = download_nemo_quickstart(version="25.10")
print(f"Downloaded to: {quickstart_path}")
```

**Dataset Pipeline:**
```python
from ai.dataset_pipeline.sourcing.ngc_ingestor import ingest_ngc_datasets

# Ingest all configured NGC datasets
results = ingest_ngc_datasets()
print(f"Results: {results}")
```

### 6. Add More NGC Resources

Edit the configuration files to add more resources:

**For Training Ready** (`ai/training_ready/utils/ngc_resources.py`):
```python
NEMO_RESOURCES = {
    'your-resource': {
        'path': 'nvidia/org/resource-name',
        'default_version': '1.0',
        'description': 'Your resource description'
    }
}
```

**For Dataset Pipeline** (`ai/dataset_pipeline/sourcing/ngc_ingestor.py`):
```python
NGC_DATASETS = {
    'nvidia/nemo/datasets/your-dataset': {
        'target': 'stage1_foundation',
        'version': '1.0',
        'description': 'Your dataset description'
    }
}
```

### 7. Integrate into Workflows

**Training Scripts:**
- Add NGC resource downloads to training setup scripts
- Use `NGCResourceDownloader` in training initialization

**Dataset Pipeline:**
- NGC ingestor is already integrated into `multi_source_ingestor.py`
- It will automatically run when you call the multi-source pipeline

### 8. Update Existing Scripts

If you have existing scripts that manually download NGC resources, update them to use the new utilities:

**Before:**
```bash
# Manual download
ngc registry resource download-version "nvidia/nemo-microservices/nemo-microservices-quickstart:25.10"
```

**After:**
```python
from ai.training_ready.utils.ngc_resources import download_nemo_quickstart
quickstart = download_nemo_quickstart(version="25.10")
```

## 📚 Documentation

- **Training Ready**: `ai/training_ready/docs/NGC_CLI_INTEGRATION.md`
- **Dataset Pipeline**: `ai/dataset_pipeline/docs/NGC_INTEGRATION.md`
- **Summary**: `ai/NGC_CLI_INTEGRATION_SUMMARY.md`

## 🔍 Troubleshooting

### "NGC CLI not found"
- Verify NGC CLI is installed: `which ngc` or `ls ~/ngc-cli/ngc`
- Check PATH includes NGC CLI location
- Re-download from https://catalog.ngc.nvidia.com if needed

### "Authentication failed"
- Verify API key: `ngc config get`
- Make sure you're using NGC API key (not NVIDIA_API_KEY)
- Re-configure: `ngc config set`

### "Download failed"
- Check network connectivity
- Verify you have access to the resource
- Check resource path is correct

## 🎯 Quick Verification Checklist

- [ ] NGC CLI installed and in PATH
- [ ] NGC API key obtained and configured
- [ ] Test script runs successfully (`python ai/utils/test_ngc_cli.py`)
- [ ] Can download a test resource
- [ ] Integration works in training_ready
- [ ] Integration works in dataset_pipeline

## 💡 Usage Examples

See the documentation files for complete usage examples:
- Training resources: `ai/training_ready/docs/NGC_CLI_INTEGRATION.md`
- Dataset ingestion: `ai/dataset_pipeline/docs/NGC_INTEGRATION.md`
