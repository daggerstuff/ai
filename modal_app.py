import modal
import os
import json
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================
APP_NAME = "pixel-eval"
MODEL_VOLUME_NAME = "pixel-merged-models"
MODEL_MOUNT_PATH = "/models"
MAX_TOKENS = 512

# Initialize Modal app
app = modal.App(APP_NAME)

# Create a volume for our models
volume = modal.Volume.from_name(MODEL_VOLUME_NAME)

# Build image with vLLM and dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.7.0",
        "sentencepiece",
        "transformers",
        "peft",
        "accelerate"
    )
)

@app.cls(
    gpu="A100",
    image=image,
    volumes={MODEL_MOUNT_PATH: volume},
    timeout=1200,
    scaledown_window=300
)
class EvaluationRunner:
    @modal.enter()
    def load_model(self):
        print("🚀 Loading model into vLLM...")
        from vllm import LLM, SamplingParams
        
        model_path = os.path.join(MODEL_MOUNT_PATH, "merged-pixel-merged")
        
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
            tokenizer_kwargs={"fix_mistral_regex": True},
            tensor_parallel_size=1,
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            enforce_eager=True,
            dtype="float16"
        )
        
        self.sampling = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=MAX_TOKENS,
            stop=["<|im_end|>", "</s>"]
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
            [{"role": "system", "content": "You are a helpful and empathetic therapist assistant. Provide thoughtful, validating, and constructive answers to the user."},
             {"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        
        formatted_prompts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
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
def main():
    print("================================")
    print("Modal Evaluation Runner")
    print("================================")
    
    # Load prompts
    prompts_file = "evaluation_prompts.json"
    print(f"📖 Loading prompts from {prompts_file}...")
    
    try:
        with open(prompts_file, "r") as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                prompts = (
                    [p["prompt"] for p in data] if isinstance(data[0], dict) else data
                )
            else:
                print("❌ Invalid prompts format (expected list)")
                return
    except Exception as e:
        print(f"❌ Error loading prompts: {e}")
        return

    print(f"✅ Loaded {len(prompts)} prompts")
    
    # Process in batches
    batch_size = 32
    print(f"⚙️  Batch size: {batch_size}")
    print("🎯 GPU: A100")
    
    all_results = []
    has_errors = False
    runner = EvaluationRunner()
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        print(f"\n[Batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}] Processing {len(batch)} prompts...")
        
        try:
            results = runner.evaluate_batch.remote(batch)
            all_results.extend(results)
        except Exception as e:
            print(f"❌ Batch failed: {e}")
            has_errors = True

    # Save results if we got any
    output_file = "evaluation_results_modal.json"
    if all_results:
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
    
    print("\n================================")
    if has_errors:
        print("⚠️ Evaluation completed with ERRORS!")
    else:
        print("✅ Evaluation complete!")
    print(f"📊 Total results: {len(all_results)}")
    if all_results:
        print(f"📂 Saved to: {output_file}")
    print("================================")
    
    if has_errors:
        import sys
        sys.exit(1)
