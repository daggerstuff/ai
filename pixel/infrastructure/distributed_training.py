import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist


class DistributedInitError(Exception):
    pass


class DeepSpeedInitError(Exception):
    pass


try:
    import deepspeed
except ImportError:
    deepspeed = None


def setup_logger():
    logger = logging.getLogger("distributed_training")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s][%(process)d][%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger()


def log_env():
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    logger.info(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    logger.info(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    logger.info(f"RANK: {os.environ.get('RANK')}")
    logger.info(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    logger.info(f"NCCL_DEBUG: {os.environ.get('NCCL_DEBUG')}")
    logger.info(f"PyTorch version: {torch.__version__}")
    if deepspeed:
        logger.info(f"DeepSpeed version: {deepspeed.__version__}")


def init_distributed(backend="nccl"):
    log_env()
    # Check for required env vars for distributed launch
    required_envs = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    missing = [v for v in required_envs if os.environ.get(v) is None]
    if missing:
        logger.warning(
            f"Missing distributed env vars: {missing}. Running in single-process (debug) mode."
        )
        return False
    try:
        if not dist.is_initialized():
            logger.info(f"Initializing torch.distributed with backend={backend}")
            dist.init_process_group(backend=backend)
        logger.info(
            f"Distributed initialized: rank={dist.get_rank()}, world_size={dist.get_world_size()}"
        )
        return True
    except Exception as e:
        logger.error(f"Distributed initialization failed: {e}", exc_info=True)
        raise DistributedInitError(f"Distributed initialization failed: {e}") from e


def init_deepspeed(args, model, optimizer=None, model_parameters=None):
    if not deepspeed:
        logger.error("DeepSpeed is not installed.")
        raise DeepSpeedInitError("DeepSpeed is not installed.")
    try:
        logger.info("Initializing DeepSpeed engine...")
        engine, optimizer, _, _ = deepspeed.initialize(
            args=args, model=model, optimizer=optimizer, model_parameters=model_parameters
        )
        logger.info("DeepSpeed engine initialized successfully.")
        return engine, optimizer
    except Exception as e:
        logger.error(f"DeepSpeed initialization failed: {e}", exc_info=True)
        raise DeepSpeedInitError(f"DeepSpeed initialization failed: {e}") from e


def sync_gradients(model):
    try:
        logger.info("Synchronizing gradients across GPUs (all-reduce)...")
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        logger.info("Gradient synchronization complete.")
    except Exception as e:
        logger.error(f"Gradient synchronization failed: {e}", exc_info=True)
        raise e


def check_for_nans(model):
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            logger.warning(f"NaN detected in gradients of parameter: {name}")


def main():
    # Example: parse args for distributed launch
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--deepspeed", action="store_true")
    args, _ = parser.parse_known_args()

    distributed = init_distributed(backend=args.backend)

    # Dummy model for demonstration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("CUDA is not available. Running on CPU.")
    model = torch.nn.Linear(10, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    if args.deepspeed:
        engine, optimizer = init_deepspeed(args, model, optimizer, model.parameters())
        model = engine

    # Example training loop with logging
    for step in range(2):
        optimizer.zero_grad()
        x = torch.randn(4, 10).to(device)
        y = model(x)
        loss = y.sum()
        loss.backward()
        check_for_nans(model)
        if distributed:
            sync_gradients(model)
        optimizer.step()
        logger.info(f"Step {step} complete. Loss: {loss.item()}")

    logger.info("Training script completed successfully (distributed mode: %s).", distributed)


if __name__ == "__main__":
    main()
