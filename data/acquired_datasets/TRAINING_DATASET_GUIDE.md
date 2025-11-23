# Training Dataset Guide for Lightning.ai H100

## Dataset Inventory

### Successfully Acquired (3,812 total conversations)

#### 1. **CoT Reasoning Dataset** (300 conversations)
- **Source**: Local (`ai/training_data_consolidated/datasets/cot_reasoning_filtered.json`)
- **Style**: Chain of Thought reasoning - mental health focused
- **Format**: Structured therapeutic conversations with reasoning patterns
- **Quality Level**: High (filtered)
- **File**: `ai/data/acquired_datasets/cot_reasoning.json` (188KB)
- **Characteristics**:
  - Demonstrates explicit reasoning process
  - Shows therapeutic thought patterns
  - Mental health domain specific
  - May include step-by-step clinical reasoning

#### 2. **Mental Health Counseling** (3,512 conversations)
- **Source**: HuggingFace `Amod/mental_health_counseling_conversations`
- **Style**: Real therapeutic conversations
- **Format**: Multi-turn client-therapist dialogues
- **File**: `ai/data/acquired_datasets/mental_health_counseling.json` (9.9MB)
- **Characteristics**:
  - Real-world conversation patterns
  - Natural therapeutic flow
  - Various mental health topics
  - Client-therapist interaction dynamics

---

## Dataset Style Differences

### CoT Reasoning vs Standard Conversations

**CoT (Chain of Thought) Reasoning**:
- Demonstrates *how* the therapist thinks through problems
- Makes reasoning explicit and structured
- Better for teaching the model clinical reasoning patterns
- May include more pedagogical elements
- Often shows step-by-step problem decomposition

**Standard Therapeutic Conversations**:
- Focus on natural dialogue flow
- Emphasis on empathy and rapport building
- Real-world interaction patterns
- More conversational, less structured

---

## Training Strategy Options

### Option 1: **Staged Training** (RECOMMENDED)
Train the model in distinct phases with different datasets:

**Phase 1: Foundation**
- Dataset: Standard therapeutic conversations (mental_health_counseling)
- Purpose: Learn natural therapeutic dialogue patterns
- Epochs: 2-3

**Phase 2: Reasoning Enhancement**
- Dataset: CoT reasoning datasets
- Purpose: Learn explicit clinical reasoning patterns
- Epochs: 1-2
- Note: Fine-tune on top of Phase 1 checkpoint

**Phase 3: Voice Injection**
- Dataset: Tim Fletcher style-extracted synthetic conversations
- Purpose: Adopt Tim's teaching style and personality
- Epochs: 1-2

**Advantages**:
- Clear separation of concerns
- Can evaluate each phase independently
- Prevents style conflicts
- Better control over model behavior

---

### Option 2: **Mixed Training**
Combine all datasets with weighted sampling:

**Dataset Weights**:
- Standard conversations: 60%
- CoT reasoning: 20%
- Tim Fletcher voice: 20%

**Advantages**:
- Single training run
- Model learns to blend styles naturally
- Potentially more flexible outputs

**Disadvantages**:
- Risk of style confusion
- Harder to debug issues
- Less control over specific behaviors

---

### Option 3: **Curriculum Learning** (ADVANCED)
Progressive difficulty increase:

1. Start with CoT reasoning (explicit patterns)
2. Move to standard conversations (natural flow)
3. Finish with Tim Fletcher voice (personality injection)

---

## Recommended Approach for Your Use Case

### **Staged Training (Option 1) - Here's Why:**

**Your Goals**:
1. ✅ Natural therapeutic conversation ability
2. ✅ Clinical reasoning skills
3. ✅ Tim Fletcher's voice/teaching style

**Training Sequence**:

```
Stage 1: Base Therapeutic Skills (3,512 conversations)
├── mental_health_counseling
└── Goal: Natural dialogue, empathy, therapeutic flow

Stage 2: Clinical Reasoning (300 conversations)
├── cot_reasoning
└── Goal: Explicit reasoning patterns, clinical thinking

Stage 3: Voice & Style (TBD - synthetic from Tim Fletcher)
├── Tim Fletcher extracted conversations
└── Goal: Teaching style, personality, flow
```

**Why This Works**:
- Foundation first (largest dataset)
- Reasoning enhancement second (specialized skill)
- Voice/personality last (style overlay)
- Each stage builds on previous
- Can checkpoint and evaluate between stages

---

## Lightning.ai Configuration Notes

### For Staged Training:

**Stage 1 Config** (`moe_training_config.json`):
```json
{
  "training": {
    "num_epochs": 3,
    "learning_rate": 2e-4,
    "train_data_path": "ai/data/acquired_datasets/mental_health_counseling.json"
  }
}
```

**Stage 2 Config** (continue from Stage 1 checkpoint):
```json
{
  "training": {
    "num_epochs": 2,
    "learning_rate": 1e-4,
    "train_data_path": "ai/data/acquired_datasets/cot_reasoning.json",
    "resume_from_checkpoint": "path/to/stage1_checkpoint"
  }
}
```

**Stage 3 Config** (continue from Stage 2):
```json
{
  "training": {
    "num_epochs": 2,
    "learning_rate": 5e-5,
    "train_data_path": "ai/data/tim_fletcher_voice/synthetic_conversations.json",
    "resume_from_checkpoint": "path/to/stage2_checkpoint"
  }
}
```

---

## Next Steps

1. ✅ **Acquire datasets** - DONE
   - CoT reasoning: 300 conversations
   - Mental health counseling: 3,512 conversations

2. ⏳ **Extract Tim Fletcher voice**
   - Analyze 913 YouTube transcripts
   - Extract teaching style, flow, personality
   - Generate synthetic conversations with Tim's voice

3. ⏳ **Quality validation**
   - Sample and review dataset quality
   - Ensure conversation coherence
   - Check for formatting issues

4. ⏳ **Lightning.ai deployment**
   - Update training configs for staged approach
   - Set up checkpointing between stages
   - Configure WandB logging for each stage

---

## Dataset Format

All datasets use standardized conversation format:

```json
{
  "conversation": [
    {"role": "client", "content": "..."},
    {"role": "therapist", "content": "..."}
  ],
  "metadata": {
    "source": "dataset_name",
    "quality_level": "high",
    ...
  }
}
```

---

## Questions to Consider

1. **For CoT datasets**: Do you want the reasoning to be explicit in outputs, or just inform the model's internal processing?

2. **For mixing styles**: Should Tim Fletcher's voice be dominant, or should it blend with clinical patterns?

3. **For training duration**: Typical LoRA fine-tuning: 2-5 epochs per stage. How aggressive do you want adaptation?

4. **For evaluation**: What metrics matter most - empathy, clinical accuracy, or style matching?
