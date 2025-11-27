# Mental Health Resources Integration Guide

> Comprehensive guide for using the newly integrated mental health datasets, synthetic data generation, empathy scoring, DPO training, and safety alignment features.

## Quick Start

```bash
cd /home/vivi/pixelated
source .venv/bin/activate  # or: uv venv && source .venv/bin/activate

# Install required dependencies
uv pip install datasets transformers torch aiohttp requests
```

---

## 1. HuggingFace Mental Health Datasets

### Available Datasets

| Dataset | Size | Purpose | HuggingFace ID |
|---------|------|---------|----------------|
| Mental Health SNLI | 43.9k pairs | NLI inference for MH statements | `iqrakiran/customized-mental-health-snli2` |
| MentalHealth Preprocessed | 3k entries | Conversational agent training | `typosonlr/MentalHealthPreProcessed` |
| Depression Detection | Variable | Depression indicator detection | `ShreyaR/DepressionDetection` |

### Usage

```python
from ai.dataset_pipeline.ingestion.tier_loaders import (
    HuggingFaceMentalHealthLoader,
    HUGGINGFACE_MENTAL_HEALTH_DATASETS,
)

# Initialize loader
loader = HuggingFaceMentalHealthLoader(
    quality_threshold=0.85,
    datasets_to_load=["mental_health_snli", "mental_health_preprocessed"]
)

# Load all configured datasets
datasets = loader.load_datasets()

# Check what's available
print(f"Loaded {len(datasets)} datasets:")
for name, conversations in datasets.items():
    print(f"  - {name}: {len(conversations)} conversations")

# Load specific dataset
snli_conversations = loader.load_specific_dataset("mental_health_snli")
print(f"SNLI: {len(snli_conversations)} conversations")
```

### Run as Script

```bash
cd ai/dataset_pipeline/ingestion/tier_loaders
python huggingface_mental_health_loader.py
```

---

## 2. Synthetic Data Distillation

### Prerequisites

```bash
# Option A: Use Ollama (local, free)
ollama serve &
ollama pull llama3.2

# Option B: Use OpenAI (requires API key)
export OPENAI_API_KEY="your-key-here"
```

### Available Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `direct_distillation` | Generate directly from teacher model | Quick dataset creation |
| `self_improvement` | Iteratively refine generated samples | Quality improvement |
| `multi_step_prompting` | Context → Outline → Conversation | Structured generation |
| `targen_style` | Progressive query evolution | Dataset diversity |
| `cultural_augmentation` | Generate across 7 cultures | Cultural diversity |
| `edge_case_expansion` | Generate challenging scenarios | Robustness training |

### Usage

```python
from ai.dataset_pipeline.generation.synthetic_data_distillation import (
    SyntheticDataDistillationPipeline,
    SyntheticGenerationConfig,
    DistillationStrategy,
    create_distillation_pipeline,
)

# Quick start with factory function
pipeline = create_distillation_pipeline(
    strategy="multi_step_prompting",
    provider="ollama",  # or "openai"
    model="llama3.2",   # or "gpt-4"
    quality_threshold=0.8,
)

# Generate synthetic data
result = pipeline.generate_synthetic_data(
    num_samples=50,
    seed_topics=["anxiety management", "depression treatment", "trauma processing"],
)

print(f"Generated: {result.total_generated}")
print(f"Quality passed: {result.quality_passed}")
print(f"Average quality: {result.average_quality:.2f}")

# Convert to standard Conversation format
conversations = pipeline.convert_to_conversations(result)

# Save results
from pathlib import Path
pipeline.save_results(result, Path("ai/datasets/synthetic/therapeutic_conversations.json"))
```

### Generate Edge Cases (Nightmare Fuel)

```python
result = pipeline.generate_synthetic_data(
    num_samples=100,
    strategy_override=DistillationStrategy.EDGE_CASE_EXPANSION,
    seed_topics=["crisis intervention", "trauma processing", "severe depression"],
)
```

### Run as Script

```bash
cd ai/dataset_pipeline/generation
python synthetic_data_distillation.py
```

---

## 3. Empathy Scoring Framework

### Empathy Dimensions (EMNLP 2020)

| Dimension | Code | Description | Weight |
|-----------|------|-------------|--------|
| Emotional Reactions | ER | Expressing warmth, compassion, concern | 30% |
| Interpretations | IP | Understanding seeker's feelings/situation | 30% |
| Explorations | EX | Probing to improve understanding | 25% |
| Validation | VAL | Acknowledging emotions as valid | 15% |

### Usage

```python
from ai.dataset_pipeline.quality.empathy_mental_health_validator import (
    EmpathyMentalHealthValidator,
    EmpathyLevel,
)
from ai.dataset_pipeline.schemas.conversation_schema import Conversation, Message

# Initialize validator
validator = EmpathyMentalHealthValidator(
    min_empathy_threshold=0.5,
    enable_pattern_matching=True,
)

# Create or load conversation
conversation = Conversation(
    conversation_id="test_001",
    source="test",
    messages=[
        Message(role="user", content="I've been feeling really anxious lately."),
        Message(role="assistant", content="I'm sorry to hear you're struggling with anxiety. "
                "That sounds really overwhelming. Can you tell me more about what's been triggering these feelings?"),
    ],
)

# Validate empathy
assessment = validator.validate_conversation(conversation)

print(f"Overall Empathy Score: {assessment.overall_empathy_score}")
print(f"Empathy Level: {assessment.empathy_level.name}")
print(f"Emotional Reactions: {assessment.average_emotional_reactions}")
print(f"Interpretations: {assessment.average_interpretations}")
print(f"Explorations: {assessment.average_explorations}")
print(f"Validation: {assessment.average_validation}")
print(f"Strengths: {assessment.strengths}")
print(f"Issues: {assessment.issues}")
print(f"Recommendations: {assessment.recommendations}")
```

### Batch Processing

```python
# Load your conversations
conversations = [...]  # List of Conversation objects

# Batch validate
assessments = validator.batch_validate(conversations)

# Get statistics
stats = validator.get_empathy_statistics(assessments)
print(f"Average empathy: {stats['average_scores']['overall']:.2f}")
print(f"Level distribution: {stats['level_distribution']}")

# Filter high-empathy conversations
high_empathy = validator.filter_by_empathy(
    conversations,
    min_score=0.7,
    min_level=EmpathyLevel.STRONG_EMPATHY,
)
print(f"High empathy conversations: {len(high_empathy)}")
```

### Run as Script

```bash
cd ai/dataset_pipeline/quality
python empathy_mental_health_validator.py
```

---

## 4. DPO (Direct Preference Optimization) Training

### Available DPO Datasets

| Dataset | Type | Use Case | HuggingFace ID |
|---------|------|----------|----------------|
| Human-Like DPO | HUMAN_LIKE | Natural conversation style | `mlx-community/Human-Like-DPO` |
| Character Roleplay DPO | ROLEPLAY | Persona consistency | `flammenai/character-roleplay-DPO` |
| Toxic Safety DPO | SAFETY | Safety alignment | `PJMixers/unalignment_toxic-dpo-v0.2-ShareGPT` |

### Loading DPO Datasets

```python
from ai.dataset_pipeline.ingestion.tier_loaders import (
    DPODatasetLoader,
    DPODatasetType,
    DPO_DATASETS,
)

# Initialize loader
loader = DPODatasetLoader(
    quality_threshold=0.90,
    datasets_to_load=["human_like_dpo", "toxic_safety_dpo"],
)

# Load as DPO samples (with chosen/rejected pairs)
dpo_samples = loader.load_dpo_samples()

for dataset_name, samples in dpo_samples.items():
    print(f"{dataset_name}: {len(samples)} samples")
    if samples:
        sample = samples[0]
        print(f"  Prompt: {sample.prompt[:100]}...")
        print(f"  Chosen: {sample.chosen_response[:100]}...")
        print(f"  Rejected: {sample.rejected_response[:100]}...")

# Load for TRL DPO training
training_data = loader.load_for_training(
    include_safety=True,
    include_roleplay=True,
)
print(f"Total training samples: {len(training_data)}")
```

### Using DPO Training Style

```python
from ai.dataset_pipeline.training_styles import (
    TrainingStyleManager,
    TrainingStyle,
    DPOConfig,
)

# Create DPO configuration
manager = TrainingStyleManager()
dpo_config = manager.create_config(
    TrainingStyle.DPO,
    name="therapeutic_dpo",
    beta=0.1,
    loss_type="sigmoid",
    max_prompt_length=512,
    max_completion_length=512,
)

print(f"DPO Config: {dpo_config.name}")
print(f"Beta: {dpo_config.beta}")
print(f"Loss Type: {dpo_config.loss_type}")
```

### Training with TRL (Example)

```python
# Requires: uv pip install trl

from trl import DPOTrainer, DPOConfig as TRLDPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Prepare data
loader = DPODatasetLoader()
train_data = loader.load_for_training()

# Configure TRL DPO trainer
trl_config = TRLDPOConfig(
    beta=0.1,
    learning_rate=1e-6,
    per_device_train_batch_size=4,
)

# Train (simplified example)
# trainer = DPOTrainer(model, train_dataset=train_data, ...)
# trainer.train()
```

### Run as Script

```bash
cd ai/dataset_pipeline/ingestion/tier_loaders
python dpo_dataset_loader.py
```

---

## 5. MEMO Counseling Summarization

### Access Request (Required First!)

**The MEMO dataset requires academic agreement. Follow these steps:**

1. Visit: https://github.com/LCS2-IIITD/MEMO
2. Click on the Data Access Agreement link
3. Fill out the agreement form
4. Email to authors (see repo for contact)
5. Wait 1-2 weeks for approval
6. Download and place in: `ai/datasets/memo/`

### Check Access Status

```python
from ai.dataset_pipeline.ingestion.memo_counseling_dataset import (
    MEMODatasetLoader,
    CounselingSummarizer,
    get_access_request_template,
)

# Check if dataset is available
loader = MEMODatasetLoader()
access = loader.check_access()

print(f"Has Access: {access['has_access']}")
print(f"Available Files: {access['available_files']}")
print(f"Missing Files: {access['missing_files']}")

if not access['has_access']:
    print("\n" + access['access_instructions'])
    print("\nAccess Request Template:")
    print(get_access_request_template())
```

### Using the Summarizer (Works Without MEMO Access)

```python
from ai.dataset_pipeline.ingestion.memo_counseling_dataset import CounselingSummarizer
from ai.dataset_pipeline.schemas.conversation_schema import Conversation, Message

# Initialize summarizer
summarizer = CounselingSummarizer()

# Create a conversation
conversation = Conversation(
    conversation_id="session_001",
    source="therapy_session",
    messages=[
        Message(role="user", content="I've been having trouble sleeping for weeks now."),
        Message(role="assistant", content="That sounds exhausting. What thoughts come up when you're trying to fall asleep?"),
        Message(role="user", content="I keep worrying about work and whether I'll get fired."),
        Message(role="assistant", content="Those anxious thoughts about job security can really interfere with rest. Have you noticed when this worry started?"),
    ],
)

# Generate different types of summaries
abstractive = summarizer.summarize_conversation(conversation)
print(f"Abstractive Summary:\n{abstractive['summary']}\n")

clinical = summarizer._clinical_summary(conversation)
print(f"Clinical Note:\n{clinical['summary'][:500]}...")
```

### Run as Script

```bash
cd ai/dataset_pipeline/ingestion
python memo_counseling_dataset.py
```

---

## 6. Safety Alignment Validation

### Safety Rules Included

| Category | Rules | Severity |
|----------|-------|----------|
| Crisis Content | Suicide, self-harm, violence | CRITICAL |
| Toxic Language | Slurs, stigma, discrimination | HIGH |
| Harmful Advice | Medical misinformation, dismissive | MEDIUM-HIGH |
| Professional Boundaries | Romantic, personal info | MEDIUM-CRITICAL |
| PII Exposure | SSN, phone, email | HIGH-CRITICAL |

### Usage

```python
from ai.dataset_pipeline.quality.safety_alignment_validator import (
    SafetyAlignmentValidator,
    SafetySeverity,
    create_safety_dpo_dataset,
)
from ai.dataset_pipeline.schemas.conversation_schema import Conversation, Message

# Initialize validator
validator = SafetyAlignmentValidator(
    safety_threshold=0.85,
    enable_toxic_detection=True,
    enable_crisis_detection=True,
    enable_pii_detection=True,
)

# Validate a conversation
conversation = Conversation(
    conversation_id="test_001",
    source="test",
    messages=[
        Message(role="user", content="I don't know if I can go on anymore."),
        Message(role="assistant", content="I'm really concerned about what you just shared. "
                "Your feelings are valid, and you deserve support. If you're having thoughts of "
                "harming yourself, please reach out to 988 (Suicide & Crisis Lifeline). "
                "Can we talk about what's been going on?"),
    ],
)

assessment = validator.validate_conversation(conversation)

print(f"Is Safe: {assessment.is_safe}")
print(f"Safety Score: {assessment.safety_score}")
print(f"Severity: {assessment.overall_severity.name}")
print(f"Requires Human Review: {assessment.requires_human_review}")

if assessment.violations:
    print(f"\nViolations ({len(assessment.violations)}):")
    for v in assessment.violations:
        print(f"  - {v.rule_id}: {v.violation_type.value} ({v.severity.name})")
        print(f"    Matched: '{v.matched_content}'")
        print(f"    Action: {v.recommended_action}")

print(f"\nRecommended Actions: {assessment.recommended_actions}")
```

### Filter Unsafe Conversations

```python
# Load your conversations
conversations = [...]  # List of Conversation objects

# Split into safe/unsafe
safe_convs, unsafe_convs = validator.filter_safe_conversations(conversations)

print(f"Safe: {len(safe_convs)}")
print(f"Unsafe: {len(unsafe_convs)}")

# Get statistics
assessments = validator.batch_validate(conversations)
stats = validator.get_safety_statistics(assessments)
print(f"Safety Rate: {stats['safety_rate']:.1%}")
print(f"Violation Types: {stats['violation_type_distribution']}")
```

### Create DPO Training Data from Violations

```python
# Generate DPO pairs from unsafe conversations
dpo_pairs = create_safety_dpo_dataset(unsafe_convs, validator)

print(f"Generated {len(dpo_pairs)} DPO training pairs")
for pair in dpo_pairs[:3]:
    print(f"\nPrompt: {pair['prompt'][:100]}...")
    print(f"Rejected: {pair['rejected'][:100]}...")
    print(f"Chosen: {pair['chosen'][:100]}...")
```

### Run as Script

```bash
cd ai/dataset_pipeline/quality
python safety_alignment_validator.py
```

---

## 7. Integrated Pipeline Example

Here's how to use everything together:

```python
#!/usr/bin/env python3
"""Complete pipeline example for mental health resources integration."""

from pathlib import Path

# 1. Load HuggingFace datasets
from ai.dataset_pipeline.ingestion.tier_loaders import HuggingFaceMentalHealthLoader

hf_loader = HuggingFaceMentalHealthLoader()
hf_datasets = hf_loader.load_datasets()
print(f"Loaded {sum(len(v) for v in hf_datasets.values())} conversations from HuggingFace")

# 2. Generate synthetic data
from ai.dataset_pipeline.generation.synthetic_data_distillation import create_distillation_pipeline

pipeline = create_distillation_pipeline(strategy="multi_step_prompting", provider="ollama")
# synthetic_result = pipeline.generate_synthetic_data(num_samples=20)
# synthetic_convs = pipeline.convert_to_conversations(synthetic_result)

# 3. Validate empathy in all conversations
from ai.dataset_pipeline.quality.empathy_mental_health_validator import EmpathyMentalHealthValidator

empathy_validator = EmpathyMentalHealthValidator()

all_conversations = []
for dataset_convs in hf_datasets.values():
    all_conversations.extend(dataset_convs)

empathy_assessments = empathy_validator.batch_validate(all_conversations)
high_empathy = [c for c, a in zip(all_conversations, empathy_assessments) 
                if a.overall_empathy_score >= 0.6]
print(f"High empathy conversations: {len(high_empathy)}/{len(all_conversations)}")

# 4. Safety validation
from ai.dataset_pipeline.quality.safety_alignment_validator import SafetyAlignmentValidator

safety_validator = SafetyAlignmentValidator()
safe_convs, unsafe_convs = safety_validator.filter_safe_conversations(high_empathy)
print(f"Safe and empathetic: {len(safe_convs)}")

# 5. Load DPO data for preference training
from ai.dataset_pipeline.ingestion.tier_loaders import DPODatasetLoader

dpo_loader = DPODatasetLoader()
dpo_training_data = dpo_loader.load_for_training()
print(f"DPO training samples: {len(dpo_training_data)}")

# 6. Generate session summaries
from ai.dataset_pipeline.ingestion.memo_counseling_dataset import CounselingSummarizer

summarizer = CounselingSummarizer()
summaries = summarizer.batch_summarize(safe_convs[:10])
print(f"Generated {len(summaries)} summaries")

print("\n✅ Pipeline complete!")
```

---

## 8. Next Steps & Action Items

### Immediate Actions

1. **Run HuggingFace Loader Test**
   ```bash
   cd ai/dataset_pipeline/ingestion/tier_loaders
   python huggingface_mental_health_loader.py
   ```

2. **Test Synthetic Generation** (requires Ollama)
   ```bash
   ollama serve &
   ollama pull llama3.2
   cd ai/dataset_pipeline/generation
   python synthetic_data_distillation.py
   ```

3. **Request MEMO Dataset Access**
   - Go to: https://github.com/LCS2-IIITD/MEMO
   - Submit data access agreement

### Integration Tasks

1. **Integrate with Existing Pipeline**
   - Add HuggingFace datasets to `tier5_research_loader.py`
   - Add DPO to training orchestrator
   - Add empathy scoring to quality validation pipeline

2. **Run Full Quality Pipeline**
   ```python
   from ai.dataset_pipeline.quality.empathy_mental_health_validator import EmpathyMentalHealthValidator
   from ai.dataset_pipeline.quality.safety_alignment_validator import SafetyAlignmentValidator
   
   # Combine validators
   empathy_v = EmpathyMentalHealthValidator()
   safety_v = SafetyAlignmentValidator()
   
   # Validate your dataset
   conversations = [...]
   empathy_results = empathy_v.batch_validate(conversations)
   safety_results = safety_v.batch_validate(conversations)
   ```

3. **Start DPO Training**
   ```bash
   # Install TRL
   uv pip install trl
   
   # Prepare DPO dataset
   python -c "
   from ai.dataset_pipeline.ingestion.tier_loaders import DPODatasetLoader
   loader = DPODatasetLoader()
   data = loader.load_for_training()
   print(f'Ready to train with {len(data)} samples')
   "
   ```

### Dataset Acquisition Checklist

- [x] `iqrakiran/customized-mental-health-snli2` - Auto-download via HuggingFace
- [x] `typosonlr/MentalHealthPreProcessed` - Auto-download via HuggingFace
- [x] `ShreyaR/DepressionDetection` - Auto-download via HuggingFace
- [x] `mlx-community/Human-Like-DPO` - Auto-download via HuggingFace
- [x] `flammenai/character-roleplay-DPO` - Auto-download via HuggingFace
- [x] `PJMixers/unalignment_toxic-dpo-v0.2-ShareGPT` - Auto-download via HuggingFace
- [ ] `LCS2-IIITD/MEMO` - **Requires manual access request**

---

## File Reference

| Component | File Path |
|-----------|-----------|
| HuggingFace MH Loader | `ai/dataset_pipeline/ingestion/tier_loaders/huggingface_mental_health_loader.py` |
| DPO Dataset Loader | `ai/dataset_pipeline/ingestion/tier_loaders/dpo_dataset_loader.py` |
| Synthetic Distillation | `ai/dataset_pipeline/generation/synthetic_data_distillation.py` |
| Empathy Validator | `ai/dataset_pipeline/quality/empathy_mental_health_validator.py` |
| Safety Validator | `ai/dataset_pipeline/quality/safety_alignment_validator.py` |
| MEMO Integration | `ai/dataset_pipeline/ingestion/memo_counseling_dataset.py` |
| DPO Config | `ai/dataset_pipeline/training_styles.py` (DPOConfig class) |
| Config | `ai/dataset_pipeline/configs/config.py` |

---

## Troubleshooting

### "datasets not installed"
```bash
uv pip install datasets
```

### "Ollama connection refused"
```bash
ollama serve  # Start Ollama server
ollama list   # Check available models
ollama pull llama3.2  # Download model
```

### "MEMO dataset not found"
The MEMO dataset requires academic access. See Section 5 for access instructions.

### "transformers not installed" (for summarization)
```bash
uv pip install transformers torch
```

---

*Generated as part of the Mental Health Resources Investigation implementation.*

