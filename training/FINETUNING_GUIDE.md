# Fine-Tuning Guide for Pixelated Empathy AI

This guide documents the complete fine-tuning process for the memory-augmented model.

## Overview

The fine-tuning process uses the memory-augmented dataset prepared in PIX-531 to train a base model to better:
- Recall relevant memories in context
- Generate contextually appropriate responses
- Connect related memories
- Produce self-reflection and insights

## Files

```
ai/training/
├── finetune_model.py                  # Main fine-tuning trainer
├── evaluate_finetuned_model.py        # Evaluation suite
└── FINETUNING_GUIDE.md                # This documentation
```

## Quick Start

```bash
# 1. Prepare dataset (PIX-531)
python -m ai.scripts.prepare_finetuning_dataset \
    --input-dir ./data/transcripts \
    --output-dir ./data/finetuning \
    --memory-source ./data/memories.json

# 2. Fine-tune model
python -m ai.training.finetune_model \
    --dataset-dir ./data/finetuning \
    --output-dir ./models/fine-tuned \
    --base-model meta-llama/Llama-2-7b-hf \
    --epochs 3 \
    --batch-size 8

# 3. Evaluate
python -m ai.training.evaluate_finetuned_model \
    --model-path ./models/fine-tuned \
    --test-data ./data/finetuning/finetuning_test.jsonl
```

## Training Architecture

### Memory-Augmented Training

The fine-tuning process implements memory-augmented training through:

1. **Memory Context Formatting**: Memories are prepended to conversation context
2. **Memory-Aware Loss**: Additional weight on memory-related examples
3. **Example Type Balancing**: Ensures diverse training across:
   - Standard conversation
   - Memory retrieval
   - Memory synthesis
   - Temporal patterns
   - Emotional context

### LoRA Configuration

Default LoRA settings for efficient fine-tuning:

| Parameter | Value | Description |
| ----------- | ------- | ------------- |
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA alpha scaling |
| `lora_dropout` | 0.1 | Dropout rate |
| `target_modules` | q_proj, k_proj, v_proj, o_proj | Attention layers |

### Training Hyperparameters

| Parameter | Default | Description |
| ----------- | --------- | ------------- |
| `epochs` | 3 | Training epochs |
| `batch_size` | 8 | Per-device batch size |
| `learning_rate` | 2e-5 | Peak learning rate |
| `warmup_ratio` | 0.1 | Warmup fraction |
| `gradient_accumulation` | 4 | Gradient accumulation steps |
| `max_seq_length` | 2048 | Maximum sequence length |

## Training Process

### Step 1: Dataset Preparation

Ensure you have completed PIX-531 and have:
- `finetuning_train.jsonl`
- `finetuning_validation.jsonl`
- `finetuning_test.jsonl`

### Step 2: Environment Setup

```bash
# Install dependencies
pip install transformers>=4.36.0 peft>=0.7.0 accelerate>=0.25.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install wandb  # Optional, for logging
```

### Step 3: Configure Training

Create a training configuration:

```python
from ai.training.finetune_model import FineTuningConfig, train

config = FineTuningConfig(
    base_model="meta-llama/Llama-2-7b-hf",
    output_dir="./models/fine-tuned",
    epochs=3,
    batch_size=8,
    learning_rate=2e-5,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    use_memory_augmentation=True,
    memory_loss_weight=0.3,
)
```

### Step 4: Run Training

```bash
python -m ai.training.finetune_model \
    --dataset-dir ./data/finetuning \
    --output-dir ./models/fine-tuned \
    --base-model meta-llama/Llama-2-7b-hf \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 2e-5 \
    --use-lora \
    --use-memory-augmentation
```

### Step 5: Monitor Training

Training metrics are logged to:
- **Console**: Real-time loss and metrics
- **WandB** (optional): Interactive dashboards
- **Checkpoint directory**: `output_dir/checkpoint-*/training_state.json`

### Step 6: Evaluate Results

```bash
python -m ai.training.evaluate_finetuned_model \
    --model-path ./models/fine-tuned \
    --test-data ./data/finetuning/finetuning_test.jsonl \
    --output ./evaluation_report.json
```

## Evaluation Metrics

### Memory Metrics

| Metric | Description | Target |
| -------- | ------------- | -------- |
| **Memory Recall Precision** | Fraction of retrieved memories that are relevant | >0.75 |
| **Memory Recall Recall** | Fraction of relevant memories retrieved | >0.70 |
| **Memory Relevance** | Overall relevance of memories to context | >0.80 |

### Quality Metrics

| Metric | Description | Target |
| -------- | ------------- | -------- |
| **Generation Quality** | Coherence and fluency of generated text | >0.75 |
| **Context Relevance** | Alignment with conversation context | >0.80 |
| **Reflection Quality** | Presence of reflective/insightful content | >0.65 |

### Overall Score

Weighted combination of metrics:
```
overall = 0.25*precision + 0.25*recall + 0.25*generation + 0.25*relevance
```

## Advanced Configuration

### Memory Loss Weighting

Adjust the weight for memory-augmented examples:

```python
config.memory_loss_weight = 0.3  # Default
config.memory_loss_weight = 0.5  # Emphasize memory tasks
config.memory_loss_weight = 0.1  # De-emphasize memory tasks
```

### Custom Target Modules

For different model architectures:

```python
# Mistral/Mixtral
config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Llama-3
config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Distributed Training

For multi-GPU training:

```bash
accelerate launch -m ai.training.finetune_model \
    --dataset-dir ./data/finetuning \
    --output-dir ./models/fine-tuned \
    --base-model meta-llama/Llama-2-7b-hf
```

## Troubleshooting

### CUDA Out of Memory

**Solution 1**: Reduce batch size
```bash
--batch-size 4
```

**Solution 2**: Enable gradient checkpointing
```bash
--gradient-checkpointing
```

**Solution 3**: Use smaller sequences
```bash
--max-seq-length 1024
```

### Poor Memory Recall

**Solution**: Increase memory loss weight
```python
config.memory_loss_weight = 0.5
```

### Overfitting

**Signs**:
- Training loss << validation loss
- Poor generalization to test data

**Solutions**:
1. Reduce training epochs
2. Increase regularization (dropout)
3. Add more training data
4. Reduce model complexity (lower LoRA rank)

### Underfitting

**Signs**:
- High training loss
- Poor performance on both train and validation

**Solutions**:
1. Increase training epochs
2. Increase learning rate
3. Reduce regularization
4. Use larger model

## Output Structure

```
./models/fine-tuned/
├── adapter_config.json       # LoRA configuration
├── adapter_model.safetensors # LoRA weights
├── tokenizer.json            # Tokenizer
├── tokenizer_config.json     # Tokenizer config
├── training_metrics.json     # Training metrics
├── checkpoint-*/             # Intermediate checkpoints
└── runs/                     # WandB logs (if enabled)
```

## Deployment

### Export for Production

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./models/fine-tuned")

# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./models/merged")
```

### Rollback Plan

If issues arise in production:

1. **Revert to previous model**: Keep base model accessible
2. **A/B testing**: Deploy to subset of users first
3. **Monitor metrics**: Track memory recall and user satisfaction

## Version History

| Version | Date | Model | Notes |
| --------- | ------ | ------- | ------- |
| 1.0.0 | 2026-05-10 | Llama-2-7B | Initial fine-tuning |

## References

- PIX-531: Prepare Fine-Tuning Dataset
- PIX-506: Workstream D: Define training-readiness validation gates
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
