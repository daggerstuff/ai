import json
import os
from pathlib import Path

import modal

# ============================================================================
# Configuration
# ============================================================================
APP_NAME = "pixel-eval"
MODEL_VOLUME_NAME = "pixel-merged-models"
MODEL_MOUNT_PATH = "/models"
MAX_TOKENS = 512  # Shorter responses to avoid degeneration
CHECKPOINT_EVERY = 1  # Save checkpoint after every batch

# Initialize Modal app
app = modal.App(APP_NAME)

# Create a volume for our models
volume = modal.Volume.from_name(MODEL_VOLUME_NAME)

# Build image with vLLM and dependencies
dependencies = [
    "vllm>=0.16.0",
    "sentencepiece",
    "transformers",
    "peft",
    "accelerate",
]
image = modal.Image.debian_slim(python_version="3.11").pip_install(*dependencies)


@app.cls(
    gpu="A100",
    image=image,
    volumes={MODEL_MOUNT_PATH: volume},
    timeout=1200,
    scaledown_window=300,
)
class EvaluationRunner:
    @modal.enter()
    def load_model(self):
        print("🚀 Loading model into vLLM...")
        from vllm import LLM, SamplingParams

        model_path = os.path.join(MODEL_MOUNT_PATH, "merged-v2")

        # Check if the path exists
        if not os.path.exists(model_path):
            print(f"❌ Model path not found: {model_path}")
            # List directory to help debug
            print(f"Contents of {MODEL_MOUNT_PATH}:")
            from contextlib import suppress

            with suppress(Exception):
                print(os.listdir(MODEL_MOUNT_PATH))

        self.tokenizer_name = "LatitudeGames/Wayfarer-2-12B"
        self.llm = LLM(
            model=model_path,
            tokenizer=self.tokenizer_name,
            tensor_parallel_size=1,
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            enforce_eager=True,
            dtype="float16",
        )

        self.sampling = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=MAX_TOKENS,
            repetition_penalty=1.15,  # Moderate penalty
            stop=[
                "<|im_end|>",
                "</s>",
                "\n\n\n",
            ],
        )

    @modal.exit()
    def stop_engine(self):
        print("🧹 Cleaning up vLLM engine to prevent shutdown errors...")
        if hasattr(self, "llm"):
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()
            del self.llm
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()

    @modal.method()
    def evaluate_batch(self, prompts: list):
        """Generates responses for a batch of prompts."""
        tokenizer = self.llm.get_tokenizer()
        messages_list = [
            [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful and empathetic therapist assistant. "
                        "Provide thoughtful, validating, and constructive answers. "
                        "Be thorough but concise-aim for 400-600 words unless the "
                        "situation requires more depth. End naturally when you've "
                        "addressed the core concern. IMPORTANT: Never repeat phrases "
                        "or sentences. Each sentence should add new information or "
                        "insight. If you find yourself about to repeat something, "
                        "stop and conclude instead."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]

        formatted_prompts = [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_list
        ]

        outputs = self.llm.generate(formatted_prompts, self.sampling)

        return [
            {
                "prompt": output.prompt,
                "response": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason,
            }
            for output in outputs
        ]


# ============================================================================
# Local Entrypoint (Run from your machine)
# ============================================================================
@app.local_entrypoint()
def main(
    prompts_file: str = "ai/lab/evals/golden_questions.json",
    output_file: str = "ai/lab/evals/evaluation_results_modal.json",
    resume: bool = True,
):
    """
    Run evaluation on Modal GPU.

    Args:
        prompts_file: Path to JSON file with prompts
            (list of strings or list of dicts with 'prompt' key)
        output_file: Path to save results
        resume: If True, resume from last checkpoint
    """
    print("================================")
    print("Modal Evaluation Runner")
    print("================================")

    # Determine project root for relative paths
    script_dir = Path(__file__).resolve().parent.parent.parent  # ai/ -> pixelated/

    prompts_path = Path(prompts_file)
    if not prompts_path.is_absolute():
        prompts_path = script_dir / prompts_file

    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = script_dir / output_file

    # Load prompts
    print(f"📖 Loading prompts from {prompts_path}...")

    try:
        with open(prompts_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                # Support both ["prompt", ...] and [{"prompt": "...", ...}, ...] formats
                if isinstance(data[0], dict):
                    prompts = [p["prompt"] for p in data]
                    prompt_ids = [p.get("id", f"P{i}") for i, p in enumerate(data)]
                else:
                    prompts = data
                    prompt_ids = [f"P{i}" for i in range(len(prompts))]
            else:
                print("❌ Invalid prompts format (expected list)")
                return
    except Exception as e:
        print(f"❌ Error loading prompts: {e}")
        return

    print(f"✅ Loaded {len(prompts)} prompts")

    # Checkpoint/resume logic
    existing_results = []
    start_idx = 0

    if resume and output_path.exists():
        try:
            with open(output_path, "r") as f:
                existing_results = json.load(f)
            start_idx = len(existing_results)
            remaining = len(prompts) - start_idx
            print(
                "🔄 Resuming from checkpoint: "
                f"{start_idx} prompts already done, {remaining} remaining"
            )
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}, starting fresh")
            existing_results = []

    if start_idx >= len(prompts):
        print("✅ All prompts already processed!")
        return

    # Process in batches
    batch_size = 8  # Smaller batches for better checkpointing
    remaining_prompts = prompts[start_idx:]
    remaining_ids = prompt_ids[start_idx:]

    print(f"⚙️  Batch size: {batch_size}")
    print("🎯 GPU: A100")
    print(f"📊 Processing {len(remaining_prompts)} remaining prompts...")

    all_results = existing_results.copy()
    has_errors = False
    runner = EvaluationRunner()

    total_batches = (len(remaining_prompts) - 1) // batch_size + 1

    for batch_num, i in enumerate(range(0, len(remaining_prompts), batch_size), 1):
        batch = remaining_prompts[i : i + batch_size]
        batch_ids = remaining_ids[i : i + batch_size]
        global_idx = start_idx + i

        batch_start = global_idx + 1
        batch_end = global_idx + len(batch)
        print(
            f"\n[Batch {batch_num}/{total_batches}] Processing prompts "
            f"{batch_start}-{batch_end}..."
        )

        try:
            results = runner.evaluate_batch.remote(batch)
            # Add prompt IDs to results
            for j, r in enumerate(results):
                r["id"] = batch_ids[j]
            all_results.extend(results)

            # Checkpoint after each batch
            if batch_num % CHECKPOINT_EVERY == 0 or batch_num == total_batches:
                with open(output_path, "w") as f:
                    json.dump(all_results, f, indent=2)
                print(f"💾 Checkpoint saved: {len(all_results)} results")

        except Exception as e:
            print(f"❌ Batch {batch_num} failed: {e}")
            has_errors = True
            # Save what we have so far
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"💾 Saved {len(all_results)} results before error")

    print("\n================================")
    if has_errors:
        print("⚠️ Evaluation completed with ERRORS!")
    else:
        print("✅ Evaluation complete!")
    print(f"📊 Total results: {len(all_results)}/{len(prompts)}")
    print(f"📂 Saved to: {output_path}")
    print("================================")

    if has_errors:
        import sys

        sys.exit(1)
