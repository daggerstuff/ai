<!-- markdownlint-disable -->\n\n# Dependencies: Training Ready Pipeline

## Required Dependencies

### Core ML Framework

- **torch** (CPU-only version for local execution)
- **torchvision** (CPU-only)
- **torchaudio** (CPU-only)

### Optional Dependencies

- **datasets** - For HuggingFace dataset sourcing

## Installation

### Quick Install (Recommended)

```bash
cd ai/training_ready
./install_dependencies.sh
```text

### Manual Installation

#### Using uv (Project Standard)

```bash
# CPU-only torch for local execution
uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Optional: HuggingFace datasets
uv pip install --system datasets
```text

#### Using pip

```bash
# CPU-only torch for local execution
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Optional: HuggingFace datasets
pip install datasets
```text

## Verification

After installation, verify torch is working:

```bash
python3 -c "import torch; print(f'Torch {torch.__version__} installed'); print(f'CUDA: {torch.cuda.is_available()}')"
```text

Expected output:

```text
Torch 2.x.x installed
CUDA: False
```text

## Notes

- **CPU-only version**: This installation uses the CPU-only build of PyTorch,
  which is smaller and suitable for local development/testing
- **For GPU training**: Use the standard torch installation (with CUDA support)
  on GPU-enabled systems
- **Project dependencies**: The main `pyproject.toml` includes torch, but this
  CPU-only version is specifically for local data processing pipeline execution

## Troubleshooting

### Torch not found after installation

- Ensure you're using the same Python interpreter that uv/pip installed to
- Check: `which python3` and `uv pip list | grep torch`
- Try: `uv pip install --system torch` to install system-wide

### Import errors

- Verify PYTHONPATH includes project root:
  `export PYTHONPATH=/home/vivi/pixelated:$PYTHONPATH`
- Check that `ai.pipelines.orchestrator` modules are accessible
