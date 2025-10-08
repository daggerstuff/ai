# Pixel Distributed Training Infrastructure

## Overview

This module provides the foundation for distributed training of Pixel models using PyTorch Distributed Data Parallel (DDP) and DeepSpeed, with robust logging for debugging initialization, device mapping, and gradient synchronization. It is designed to run in both multi-GPU and CPU-only environments, and supports both distributed and single-process (debug) modes.

---

## Running the Script

### Single-Process (Debug/Development) Mode

To run locally (no distributed env vars, runs on CPU if no GPU):

```bash
uv run ai/pixel/infrastructure/distributed_training.py --backend nccl
```

- The script will detect missing distributed environment variables and run in single-process mode.
- If CUDA is not available, it will run on CPU and log a warning.

### Distributed Mode (Production/Cluster)

To run in distributed mode, you must set the following environment variables for each process:

- `RANK`
- `WORLD_SIZE`
- `MASTER_ADDR`
- `MASTER_PORT`

Example (for 2 processes on a single node):

```bash
# Terminal 1
export RANK=0
export WORLD_SIZE=2
export MASTER_ADDR=localhost
export MASTER_PORT=12345
uv run ai/pixel/infrastructure/distributed_training.py --backend nccl

# Terminal 2
export RANK=1
export WORLD_SIZE=2
export MASTER_ADDR=localhost
export MASTER_PORT=12345
uv run ai/pixel/infrastructure/distributed_training.py --backend nccl
```

Or use a custom launcher that sets these variables for you.

---

## Log Interpretation

- **Initialization logs**: Show all relevant env vars, backend, and device info.
- **Distributed mode**: Logs rank, world size, and backend. Any missing env vars are logged as a warning.
- **Device logs**: Warns if CUDA is not available and falls back to CPU.
- **Gradient sync**: Logs all-reduce status and checks for NaNs in gradients.
- **Errors**: All exceptions are logged with stack traces for debugging.

---

## Common Errors & Troubleshooting

- **Missing distributed env vars**: Script will run in single-process mode and log a warning.
- **No NVIDIA GPU/driver**: Script will run on CPU and log a warning.
- **Distributed initialization failure**: Check that all required env vars are set and that all processes can communicate.
- **DeepSpeed not installed**: Script will exit with an error if `--deepspeed` is used but DeepSpeed is not available.

---

## Extending the Infrastructure

- Add new distributed training features (e.g., checkpointing, experiment tracking) by building on this foundation.
- All new features should include robust logging for initialization, error handling, and state transitions.

---

## Contact

For further debugging or infrastructure questions, see the main Pixel documentation or contact the infrastructure lead.
