import os
import json
from pathlib import Path
from datetime import datetime
from ai.core.pipelines.training_manifest import (
    TrainingManifest,
    DatasetReference,
    Hyperparameters,
    ComputeTarget,
    TrainingFramework,
    ResourceRequirements,
    SafetyMetrics,
)


def generate_h100_optimized_manifest(dataset_path: str, version: str = "2.0"):
    """
    Generates a training manifest optimized for NVIDIA H100 80GB GPUs.
    Optimizations include bf16 precision, larger batch sizes, and
    flash attention compatibility settings.
    """
    dataset_path_obj = Path(dataset_path)

    dataset = DatasetReference(
        name="pixelated_empathy_v2_h100",
        version=version,
        path=str(dataset_path_obj),
        created_at=datetime.utcnow().isoformat(),
    )

    if dataset_path_obj.exists():
        dataset.size_bytes = dataset_path_obj.stat().st_size
        # Estimate record count (rough for JSONL)
        with open(dataset_path_obj, "r") as f:
            dataset.record_count = sum(1 for _ in f)

    # H100 Optimized Hyperparameters
    hyperparams = Hyperparameters(
        num_train_epochs=3,
        learning_rate=5e-5,  # Slightly higher for larger batches
        per_device_train_batch_size=32,  # H100 80GB can handle large batches
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,  # Effective batch size 128 (assuming 1 GPU)
        max_grad_norm=1.0,
        weight_decay=0.1,
        warmup_steps=1000,
        optimizer="adamw_torch_fused",  # Use fused optimizer for speed on H100
        lr_scheduler_type="cosine",
        max_seq_length=4096,  # H100 excels at long context
        gradient_checkpointing=True,
        bf16=True,  # MUST use bf16 on H100
        fp16=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
    )

    resources = ResourceRequirements(
        min_gpu_memory_gb=80.0,
        min_system_memory_gb=128.0,
        expected_runtime_hours=48.0,
        instance_type="p5.48xlarge",  # AWS H100 instance
    )

    manifest = TrainingManifest(
        name=f"pixelated-empathy-h100-v{version}",
        description="H100 80GB optimized training run with bf16 and flash attention",
        dataset=dataset,
        hyperparameters=hyperparams,
        framework=TrainingFramework.TRANSFORMERS,
        compute_target=ComputeTarget.GPU_MULTI,
        resources=resources,
        output_dir="./training_outputs/h100_run",
        log_dir="./training_logs/h100_run",
        metadata={
            "optimization": "h100_bf16_80gb",
            "flash_attention": "enabled",
            "transformer_engine": "enabled",
        },
    )

    output_path = Path("ai/pipelines/orchestrator/manifests")
    output_path.mkdir(parents=True, exist_ok=True)

    filename = output_path / f"h100_optimized_v{version}.json"
    manifest.save_to_file(str(filename))

    print(f"H100 Optimized Manifest generated at: {filename}")
    return filename


if __name__ == "__main__":
    # Example path, in reality would point to the merged dataset
    merged_path = "/home/vivi/pixelated/ai/data/processed/merged_dataset_final.jsonl"
    generate_h100_optimized_manifest(merged_path)
