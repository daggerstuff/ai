import asyncio
import os

import art
import wandb
from art.utils.sft import train_sft_from_file


def required_config(run, key):
    value = run.config.get(key)
    if value is None:
        raise ValueError(f"Missing required W&B Launch config: {key}")
    return value


async def main():
    with wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wutang"),
        project=os.environ.get("WANDB_PROJECT", "pixelated-empathy-kan28"),
        job_type="serverless-sft",
        tags=["arc-corpus", "serverless-sft", "launch"],
        config={
            "base_model": "Qwen/Qwen3.6-35B-A3B",
            "model_name": "arc-corpus-qwen36-launch-sft",
            "training_type": "sft",
            "epochs": 1,
            "batch_size": 2,
            "peak_lr": 0.0001,
            "schedule_type": "cosine",
            "warmup_ratio": 0.1,
            "assistant_turns": "all",
            "dataset_artifact": "arc-corpus-sft-data:latest",
        },
    ) as run:
        entity = run.entity
        project = run.project
        dataset_ref = required_config(run, "dataset_artifact")
        dataset = run.use_artifact(f"{entity}/{project}/{dataset_ref}", type="dataset")
        dataset_dir = dataset.download()
        train_path = os.path.join(dataset_dir, "train.jsonl")
        if not os.path.isfile(train_path):
            raise FileNotFoundError(f"Dataset artifact has no train.jsonl: {dataset_ref}")

        backend = art.ServerlessBackend()
        model = art.TrainableModel(
            name=required_config(run, "model_name"),
            run_name=run.id,
            project=project,
            entity=entity,
            base_model=required_config(run, "base_model"),
        )
        await model.register(backend)

        with open(train_path, encoding="utf-8") as train_file:
            train_rows = sum(1 for _ in train_file)
        run.log({"dataset/train_rows": train_rows})
        await train_sft_from_file(
            model,
            train_path,
            epochs=int(required_config(run, "epochs")),
            batch_size=int(required_config(run, "batch_size")),
            peak_lr=float(required_config(run, "peak_lr")),
            schedule_type=required_config(run, "schedule_type"),
            warmup_ratio=float(required_config(run, "warmup_ratio")),
            assistant_turns=required_config(run, "assistant_turns"),
            verbose=True,
        )
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
