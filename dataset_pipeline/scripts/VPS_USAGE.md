# VPS Dataset Acquisition - Usage Guide

The `vps_dataset_acquisition.sh` script is designed to run independently on the VPS without requiring a persistent SSH connection from your local machine.

## Quick Start

### Option 1: One-Line Background Execution (Recommended)

```bash
# First upload script (see First-Time Setup above), then run:
ssh -i ~/.ssh/planet vivi@146.71.78.184 "cd ~/pixelated-datasets && nohup ./vps_dataset_acquisition.sh > download_nohup.out 2>&1 &"

# Or if directory doesn't exist yet, the script will create it:
ssh -i ~/.ssh/planet vivi@146.71.78.184 "mkdir -p ~/pixelated-datasets && cd ~/pixelated-datasets && nohup ./vps_dataset_acquisition.sh > download_nohup.out 2>&1 &"
```

This will:
- Start the script in the background
- Continue running even if you disconnect
- Redirect all output to `download_nohup.out`

### Option 2: Using Screen (Best for Monitoring)

```bash
# Connect to VPS
ssh -i ~/.ssh/planet vivi@146.71.78.184

# Start a screen session
screen -S dataset-download

# Run the script
cd ~/pixelated-datasets
./vps_dataset_acquisition.sh

# Detach: Press Ctrl+A, then D
# Reattach later: screen -r dataset-download
```

### Option 3: Daemon Mode

```bash
ssh -i ~/.ssh/planet vivi@146.71.78.184 "cd ~/pixelated-datasets && ./vps_dataset_acquisition.sh --daemon"
```

## Checking Status

### Check if process is running:

```bash
ssh -i ~/.ssh/planet vivi@146.71.78.184 "./vps_dataset_acquisition.sh --status"
```

### View live log:

```bash
ssh -i ~/.ssh/planet vivi@146.71.78.184 "tail -f ~/pixelated-datasets/download.log"
```

### View recent status updates:

```bash
ssh -i ~/.ssh/planet vivi@146.71.78.184 "tail -20 ~/pixelated-datasets/download.status"
```

### Check for errors:

```bash
ssh -i ~/.ssh/planet vivi@146.71.78.184 "tail -20 ~/pixelated-datasets/error.log"
```

## Files Created

All files are in `~/pixelated-datasets/`:

- `download.log` - Main log file
- `error.log` - Error log (only errors)
- `download.status` - Status updates (last 100)
- `download.pid` - Process ID (exists while running)
- `inventory/VPS_INVENTORY.json` - Dataset inventory manifest
- `raw/` - All downloaded datasets

## First-Time Setup

The script will auto-create directories, but you need to upload it first:

```bash
# From your local machine (in project root)
# Create directory and upload script
ssh -i ~/.ssh/planet vivi@146.71.78.184 "mkdir -p ~/pixelated-datasets && chmod 755 ~/pixelated-datasets"
scp -i ~/.ssh/planet ai/dataset_pipeline/scripts/vps_dataset_acquisition.sh vivi@146.71.78.184:~/pixelated-datasets/
ssh -i ~/.ssh/planet vivi@146.71.78.184 "chmod +x ~/pixelated-datasets/vps_dataset_acquisition.sh"
```

**Or use this one-liner:**

```bash
ssh -i ~/.ssh/planet vivi@146.71.78.184 "mkdir -p ~/pixelated-datasets" && \
scp -i ~/.ssh/planet ai/dataset_pipeline/scripts/vps_dataset_acquisition.sh vivi@146.71.78.184:~/pixelated-datasets/ && \
ssh -i ~/.ssh/planet vivi@146.71.78.184 "chmod +x ~/pixelated-datasets/vps_dataset_acquisition.sh"
```

## Setup Requirements on VPS

The script will auto-install Python packages if needed, but you should have:

1. **Python 3.8+** installed
2. **HuggingFace token** (one of):
   - Set `HF_TOKEN` environment variable
   - Run `huggingface-cli login` on VPS
3. **Kaggle credentials** (optional):
   - Place `kaggle.json` in `~/.kaggle/`
   - Run `chmod 600 ~/.kaggle/kaggle.json`
4. **rclone** (optional, for Google Drive):
   - Install: `curl https://rclone.org/install.sh | sudo bash`
   - Configure: `rclone config`

## What Gets Downloaded

- **HuggingFace datasets**: 10 datasets (~580MB)
- **Kaggle datasets**: TF-IDF archive (~1.5GB)
- **Google Drive datasets**: 7 datasets (~890MB) via rclone

Total: ~3GB of downloads

## Time Estimates

- HuggingFace downloads: 10-30 minutes
- Kaggle downloads: 30-60 minutes
- Google Drive sync: 20-40 minutes
- **Total**: 1-2 hours (depending on connection speed)

## Troubleshooting

### Script stopped running

```bash
# Check if process died
ssh -i ~/.ssh/planet vivi@146.71.78.184 "cat ~/pixelated-datasets/download.status"
```

### Re-run after failure

The script is idempotent - you can run it again and it will skip already-downloaded datasets.

### View full logs

```bash
ssh -i ~/.ssh/planet vivi@146.71.78.184 "cat ~/pixelated-datasets/download.log"
```

## Next Steps

After downloads complete:
1. Check `~/pixelated-datasets/inventory/VPS_INVENTORY.json` for manifest
2. Run transfer script to upload missing local data
3. Run upload script to push everything to OVH S3

