# Fine-Tuning Dataset Preparation Guide

This guide documents the fine-tuning dataset preparation pipeline for Pixelated Empathy AI. This pipeline collects transcripts, memories, and reflections to create training data that teaches the model to utilize memory context effectively.

## Overview

The fine-tuning dataset preparation process involves:

1. **Data Collection**: Gather anonymized conversation transcripts, memory entries, and reflections
2. **Data Processing**: Clean, anonymize, and structure into training examples
3. **Dataset Organization**: Split into training, validation, and test sets
4. **Memory-Specific Features**: Include examples testing memory retrieval, filtering, synthesis, and temporal patterns
5. **Quality Assurance**: Automated checks for data quality, consistency, and bias detection
6. **Privacy & Compliance**: Strict anonymization protocols and secure handling

## Files

```
ai/scripts/
├── prepare_finetuning_dataset.py    # Main preparation pipeline
├── finetuning_dataset_qa.py         # Quality assurance and bias detection
└── prepare_finetuning_dataset_guide.md  # This documentation
```

## Quick Start

```bash
# Prepare dataset
python -m ai.scripts.prepare_finetuning_dataset \
    --input-dir ./data/transcripts \
    --output-dir ./data/finetuning \
    --memory-source ./data/memories.json \
    --reflection-source ./data/reflections.json \
    --seed 42

# Run quality assurance
python -m ai.scripts.finetuning_dataset_qa \
    ./data/finetuning/finetuning_train.jsonl \
    --output ./data/finetuning/quality_report.json \
    --verbose
```

## Training Example Schema

Each training example follows this schema:

```json
{
  "id": "unique_example_id",
  "example_type": "standard|memory_retrieval|memory_filtering|memory_synthesis|temporal_pattern|emotional_context",
  "input": "Conversation context + relevant memories",
  "target": "Expected model response or next action",
  "conversation_id": "parent_conversation_id",
  "relevant_memories": [...],
  "memory_retrieval_query": "Optional query for memory retrieval",
  "split": "train|validation|test",
  "conversation_type": "crisis|reflection|therapeutic|onboarding|general",
  "emotional_tone": "Optional emotional tone",
  "therapeutic_modality": "Optional modality",
  "skill_tags": ["tag1", "tag2"],
  "memory_usage_correct": true,
  "emotional_appropriateness": 0.95,
  "skill_application_accuracy": 0.90,
  "source_file": "source_transcript.txt",
  "created_at": "2026-05-10T15:00:00Z"
}
```

## Example Types

### Standard
Basic conversation examples with context and response pairs.

### Memory Retrieval
Tests the model's ability to retrieve relevant memories given conversation context.

### Memory Filtering
Tests filtering memories by category, emotional tag, or importance score.

### Memory Synthesis
Requires synthesizing insights across multiple memory entries.

### Temporal Pattern
Tests understanding of temporal relationships between memories.

### Emotional Context
Focuses on emotional appropriateness and therapeutic skill application.

## Anonymization

The pipeline implements multi-layer anonymization:

- **Emails**: Replaced with `<EMAIL_hash>`
- **Phone numbers**: Replaced with `<PHONE_hash>`
- **SSN**: Replaced with `<SSN_hash>`
- **Credit cards**: Replaced with `<CREDIT_CARD_hash>`
- **IP addresses**: Replaced with `<IP_ADDRESS_hash>`
- **URLs**: Replaced with `<URL_hash>`
- **Names**: Detected and replaced with `<NAME_hash>`

### Disabling Anonymization

For development purposes, you can disable anonymization:

```bash
python -m ai.scripts.prepare_finetuning_dataset \
    --input-dir ./data/transcripts \
    --output-dir ./data/finetuning \
    --no-anonymize
```

**Warning**: Never use `--no-anonymize` with real user data. Only use with synthetic or already-anonymized test data.

## Configuration Options

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | (required) | Directory containing transcript files |
| `--output-dir` | `./data/finetuning` | Output directory for prepared dataset |
| `--memory-source` | `null` | Path to memory data (JSON file) |
| `--reflection-source` | `null` | Path to reflection data (JSON file) |
| `--train-ratio` | `0.8` | Training split ratio |
| `--validation-ratio` | `0.1` | Validation split ratio |
| `--test-ratio` | `0.1` | Test split ratio |
| `--seed` | `42` | Random seed for reproducibility |
| `--no-anonymize` | `false` | Disable PII anonymization |
| `--verbose` | `false` | Enable verbose logging |

### Supported Input Formats

- **JSON**: Single conversation per file
- **JSONL**: Multiple conversations (one per line)
- **TXT**: Plain text with speaker labels

## Quality Assurance

The QA module checks for:

### Format Validity
- Required fields present
- Valid example types
- Correct data types

### Completeness
- Non-empty input/target fields
- Minimum content length

### Consistency
- Unique example IDs
- Consistent split assignments

### Bias Detection
- Demographic balance
- Topical distribution
- Linguistic patterns
- Statistical outliers

### Quality Scores

| Score | Description | Weight |
|-------|-------------|--------|
| Format Validity | Schema compliance | 25% |
| Completeness | Data presence | 25% |
| Consistency | Internal consistency | 25% |
| Bias | Fairness metrics | 25% |

## Output Structure

```
./data/finetuning/
├── finetuning_train.jsonl       # Training examples (80%)
├── finetuning_validation.jsonl  # Validation examples (10%)
├── finetuning_test.jsonl        # Test examples (10%)
├── dataset_statistics.json      # Dataset statistics
└── dataset_metadata.json        # Metadata and configuration
```

## Memory-Specific Features

### Memory Retrieval Examples

```python
{
    "example_type": "memory_retrieval",
    "input": "Context: Client discusses anxiety about job interview\n\nQuery: What memories are relevant?",
    "target": "[{\"content\": \"Client has history of interview anxiety\", \"category\": \"therapeutic_insight\"}]",
    "memory_retrieval_query": "Relevant memories for job interview discussion"
}
```

### Memory Synthesis Examples

```python
{
    "example_type": "memory_synthesis",
    "input": "Context: Session 5 progress review\nMemories: [5 memory entries]",
    "target": "Synthesized insight: Client shows improvement in coping strategies but still struggles with performance anxiety"
}
```

## Privacy & Compliance

### Required Protocols

1. **Consent Verification**: Ensure all data has appropriate consent for training use
2. **PHI Scanning**: Run PHI detection before including any health-related data
3. **Secure Storage**: Store datasets in secure, access-controlled locations
4. **Audit Trail**: Maintain logs of data access and usage

### Compliance Checklist

- [ ] All PII anonymized
- [ ] PHI scan passed
- [ ] Consent verified for all entries
- [ ] Data encrypted at rest
- [ ] Access logged and audited

## Troubleshooting

### Common Issues

**Issue**: "Missing required fields"
- **Cause**: Input data doesn't match expected schema
- **Solution**: Check transcript format and required fields

**Issue**: "Low quality score"
- **Cause**: Multiple format or completeness issues
- **Solution**: Run QA with `--verbose` to see detailed issues

**Issue**: "Bias detected"
- **Cause**: Imbalanced representation in dataset
- **Solution**: Review data sources and consider rebalancing

## Next Steps

After preparing the dataset:

1. **Review Quality Report**: Check `dataset_statistics.json` and QA report
2. **Manual Review**: Sample check of examples for quality
3. **Training**: Use with your fine-tuning pipeline
4. **Evaluation**: Test on held-out test set
5. **Iteration**: Update based on model performance

## API Reference

### FineTuningDatasetPreparer

```python
from ai.scripts.prepare_finetuning_dataset import FineTuningDatasetPreparer

preparer = FineTuningDatasetPreparer(
    output_dir="./data/finetuning",
    train_ratio=0.8,
    validation_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    anonymize=True,
)

# Full pipeline
output_files = preparer.prepare(
    transcript_dir="./data/transcripts",
    memory_source="./data/memories.json",
    reflection_source="./data/reflections.json",
)

# Or step by step
transcripts = preparer.load_transcripts("./data/transcripts")
memories = preparer.load_memories("./data/memories.json")
examples = preparer.create_training_examples(transcripts, memories)
examples = preparer.anonymize_examples(examples)
splits = preparer.split_dataset(examples)
output_files = preparer.save_dataset(splits)
```

### QualityAssurance

```python
from ai.scripts.finetuning_dataset_qa import QualityAssurance

qa = QualityAssurance()
report = qa.run_full_check("./data/finetuning/finetuning_train.jsonl")

print(f"Quality Score: {report.overall_quality_score:.2f}")
print(f"Issues: {len(report.issues)}")

for rec in report.recommendations:
    print(f"- {rec}")
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-05-10 | Initial implementation |
