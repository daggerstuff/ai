# Integrated Training Pipeline Guide

## 🎯 Overview

This guide explains how to use the **Integrated Training Pipeline** that combines ALL data sources into a unified training dataset:

1. **Edge Case Pipeline** (25 nightmare fuel categories)
2. **Pixel Voice Pipeline** (YouTube transcripts with personality)
3. **Psychology Knowledge Base** (4,867+ concepts)
4. **Dual Persona Training** (Multi-persona interactions)
5. **Standard Therapeutic Conversations** (Base conversations)

## 🚀 Quick Start

### Step 1: Generate Edge Case Data

```bash
cd ai/pipelines/edge_case/
python generate_training_data.py
```

This will:
- Generate 500 prompts across 25 challenging categories
- Create difficult client conversations
- Output to `output/edge_cases_training_format.jsonl`

**Time**: ~30-60 minutes depending on your system

### Step 2: Run Integrated Pipeline

```bash
cd ai/pipelines/orchestrator/orchestration/
python integrated_training_pipeline.py
```

This will:
- Load edge case data
- Load Pixel Voice data
- Load psychology knowledge
- Load dual persona data
- Load standard therapeutic conversations
- Balance into four stages (40% Stage 1 foundation, 25% Stage 2 reasoning, 20% Stage 3 edge stress tests, 15% Stage 4 voice/persona)
- Run bias detection
- Run quality validation
- Output unified dataset to `ai/lightning/training_dataset.json`
- Emit per-stage exports + manifest under `ai/training_data_consolidated/final/`

### Step 3: Train Your Model

```bash
cd ai/lightning/
python train_optimized.py
```

The model will now train on the complete integrated dataset!

## 📊 Data Source Breakdown

### 1. Edge Case Pipeline (25% of dataset)

**Categories** (25 total):
- **Very High Difficulty**: Suicidality, homicidal ideation, psychotic episodes, child abuse reporting, severe dissociation
- **High Difficulty**: Substance abuse crisis, trauma flashbacks, borderline crisis, domestic violence, eating disorders
- **Moderate Difficulty**: Paranoid accusations, medication refusal, family conflicts, adolescent defiance, couples betrayal, and more

**Output Format**:
```json
{
  "prompt": "Therapist statement",
  "response": "Challenging client response",
  "purpose": "difficult_client",
  "category": "suicidality",
  "difficulty_level": "very_high",
  "expected_challenges": ["crisis_intervention", "safety_assessment"],
  "source": "edge_case_generation"
}
```

**Location**: `ai/pipelines/edge_case/output/edge_cases_training_format.jsonl`

### 2. Pixel Voice Pipeline (20% of dataset)

**Features**:
- YouTube transcript processing
- Personality marker extraction
- Emotional pattern analysis
- Dialogue naturalness validation
- Personality consistency checks

**Location**: `ai/pipelines/voice/`

### 3. Psychology Knowledge Base (15% of dataset)

**Content**:
- 4,867+ psychology concepts
- DSM-5 clinical definitions
- Therapeutic techniques (CBT, DBT, ACT, EMDR, etc.)
- Tim Fletcher complex trauma series
- Expert clinical transcripts

**Location**: `ai/training_data_consolidated/`

### 4. Dual Persona Training (10% of dataset)

**Features**:
- Multi-persona therapeutic interactions
- Persona switching scenarios
- Consistency validation

**Location**: `ai/pipelines/dual_persona/`

### 5. Standard Therapeutic (30% of dataset)

**Content**:
- Base therapeutic conversations
- General support dialogues
- Standard therapy interactions

**Location**: `ai/pipelines/orchestrator/pixelated-training/training_dataset.json`

### Stage-balanced Outputs (Pipeline Result)

- **Manifest:** `ai/training_data_consolidated/final/MASTER_STAGE_MANIFEST.json`
- **Files:** `MASTER_stage1_foundation.jsonl`, `MASTER_stage2_therapeutic_expertise.jsonl`, `MASTER_stage3_edge_stress_test.jsonl`, `MASTER_stage4_voice_persona.jsonl`
- **Targets:** 40% Stage 1, 25% Stage 2, 20% Stage 3, 15% Stage 4
- **Usage:** Feed individual stage files into Lightning configs or audits to verify balance before recombining.

## 🔧 Configuration

### Custom Pipeline Configuration

```python
from integrated_training_pipeline import IntegratedPipelineConfig, run_integrated_pipeline

config = IntegratedPipelineConfig(
    # Adjust target percentages
    edge_cases=DataSourceConfig(
        enabled=True,
        target_percentage=0.30,  # Increase edge cases to 30%
    ),
    pixel_voice=DataSourceConfig(
        enabled=True,
        target_percentage=0.25,  # Increase voice to 25%
    ),
    
    # Adjust total samples
    target_total_samples=10000,  # Increase to 10k samples
    
    # Stage distribution overrides (must sum to 1.0)
    stage_distribution={
        "stage1_foundation": 0.35,
        "stage2_therapeutic_expertise": 0.25,
        "stage3_edge_stress_test": 0.25,
        "stage4_voice_persona": 0.15,
    },

    # Quality settings
    enable_bias_detection=True,
    enable_quality_validation=True,
    min_quality_score=0.8,  # Stricter quality threshold
)

result = run_integrated_pipeline(config)
```

### Edge Case Generation Options

```python
from edge_case_generator import EdgeCaseGenerator

# Use OpenAI for higher quality
generator = EdgeCaseGenerator(
    api_provider="openai",
    api_key="your_api_key",
    model_name="gpt-4",
    output_dir="output"
)

# Generate more scenarios per category
prompts = generator.generate_prompts(scenarios_per_category=50)  # 1,250 total
conversations = generator.generate_conversations(prompts, max_conversations=1000)
```

## 📈 Monitoring Integration

### Check Integration Status

```python
from edge_case_jsonl_loader import EdgeCaseJSONLLoader

loader = EdgeCaseJSONLLoader()

# Check if edge case data exists
if loader.check_pipeline_output_exists():
    stats = loader.get_statistics()
    print(f"Edge cases: {stats['total_examples']}")
    print(f"Categories: {stats['categories']}")
else:
    print("Run edge case pipeline first!")
```

### View Integration Report

After running the integrated pipeline, check the report:

```python
import json

with open('ai/lightning/training_dataset.json', 'r') as f:
    data = json.load(f)
    
metadata = data['metadata']
print(f"Total conversations: {metadata['total_conversations']}")
print(f"Sources: {metadata['sources']}")
print(f"Integration stats: {metadata['integration_stats']}")
print(f"Stage metrics: {metadata.get('stage_metrics', {})}")

with open('ai/training_data_consolidated/final/MASTER_STAGE_MANIFEST.json', 'r') as f:
    stage_manifest = json.load(f)
    print("Stage exports:", stage_manifest['stages'])
```

## 🐛 Troubleshooting

### Edge Case Data Not Found

**Problem**: `Edge case training data not found!`

**Solution**:
```bash
cd ai/pipelines/edge_case/
python generate_training_data.py
```

### Ollama Not Running

**Problem**: `Connection refused` when generating edge cases

**Solution**:
```bash
# Start Ollama
ollama serve

# Pull the model
ollama pull artifish/llama3.2-uncensored
```

### Pixel Voice Data Missing

**Problem**: `Pixel Voice directory not found`

**Solution**:
Check if the Pixel Voice pipeline has been run and data exists in `ai/pipelines/voice/`

### Imbalanced Dataset

**Problem**: Some sources have fewer samples than target

**Solution**:
- Generate more edge case data (increase `scenarios_per_category`)
- Adjust target percentages in config
- Accept warnings and use available data

## 🎯 Best Practices

### 1. Generate Edge Cases First

Always run the edge case pipeline before the integrated pipeline:
```bash
cd ai/pipelines/edge_case/
python generate_training_data.py
```

### 2. Review Generated Data

Check the quality of generated edge cases:
```bash
cd ai/pipelines/edge_case/output/
head -n 5 edge_cases_training_format.jsonl
```

### 3. Monitor Integration

Watch for warnings during integration:
- Missing data sources
- Imbalanced datasets
- Bias detection flags
- Quality validation failures

### 4. Validate Final Dataset

Before training, validate the integrated dataset:
```python
import json

with open('ai/lightning/training_dataset.json', 'r') as f:
    data = json.load(f)

conversations = data['conversations']
print(f"Total: {len(conversations)}")

# Check source distribution
sources = {}
for conv in conversations:
    source = conv.get('metadata', {}).get('source', 'unknown')
    sources[source] = sources.get(source, 0) + 1

for source, count in sources.items():
    percentage = (count / len(conversations)) * 100
    print(f"{source}: {count} ({percentage:.1f}%)")
```

## 📚 Additional Resources

- **Edge Case Pipeline**: `ai/pipelines/edge_case/README.md`
- **Pixel Voice Pipeline**: `ai/pipelines/voice/DEPLOYMENT.md`
- **Training Guide**: `ai/lightning/TRAINING_PROCEDURES.md`
- **MoE Architecture**: `ai/lightning/MODEL_ARCHITECTURE_PERFORMANCE.md`

## 🎉 Success Checklist

Before training, ensure:
- [ ] Edge case data generated (`output/edge_cases_training_format.jsonl` exists)
- [ ] Integrated pipeline run successfully
- [ ] Training dataset created (`ai/lightning/training_dataset.json` exists)
- [ ] Dataset has expected number of samples (~8,000)
- [ ] Source distribution matches targets (25% edge, 20% voice, etc.)
- [ ] No critical errors in integration report
- [ ] Bias detection completed
- [ ] Quality validation passed

Now you're ready to train a truly comprehensive therapeutic AI with nightmare fuel edge cases! 🔥

---

**Questions?** Check the troubleshooting section or review the source code in `ai/pipelines/orchestrator/orchestration/integrated_training_pipeline.py`
