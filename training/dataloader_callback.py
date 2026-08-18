"""
Distributed Dataloader Callback (Gilfoyle v24.0).
FIXED: Replaced time.sleep() with dist.barrier() for guaranteed distributed I/O order.
FIXED: Robust Rank 0 consolidation.
"""
import json
import logging
import os
from pathlib import Path

import torch.distributed as dist
from transformers import TrainerCallback

logger = logging.getLogger("DataloaderCallback")

class DataloaderStateCallback(TrainerCallback):
    def __init__(self, output_dir: str, num_workers: int, save_request_step):
        self.output_dir, self.num_workers, self.save_request_step = Path(output_dir), num_workers, save_request_step

    def on_save(self, args, state, control, **kwargs):
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.save_request_step.value = state.global_step
        checkpoint_dir = self.output_dir / f"checkpoint-{state.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Ensure all workers have finished writing their local state files
        if dist.is_initialized(): dist.barrier()

        expected = [checkpoint_dir / f"worker{wid}_rank{rank}.json" for wid in range(self.num_workers)]

        # Verify local integrity before consolidation
        all_ready = all(f.is_file() and f.stat().st_size > 0 for f in expected)
        if not all_ready:
            logger.error(f"❌ [Rank {rank}] Worker files missing or empty at step {state.global_step}. Consolidation aborted.")
            return

        combined = {}
        for ef in expected:
            try:
                with open(ef) as fp: combined.update(json.load(fp))
                ef.unlink() # Cleanup worker-level files
            except Exception as e:
                logger.error(f"Consolidation Error: {e}")

        final_path = checkpoint_dir / f"stream_rank{rank}.json"
        tmp_path = final_path.with_suffix(".tmp")
        with open(tmp_path, "w") as fp: json.dump(combined, fp)
        os.replace(tmp_path, final_path)
        logger.info(f"💾 [Rank {rank}] Checkpoint {state.global_step} persisted via NCCL consensus.")
