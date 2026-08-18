# Evaluation-to-Data Feedback Loop Guide

This guide documents the feedback loop system that connects evaluation findings to upstream dataset pipeline actions.

## Overview

The feedback loop ensures that evaluation failures and model-quality signals don't stay isolated in downstream reporting. Instead, they become concrete changes to:
- Source prioritization (acquisition)
- Curation rules
- Review focus areas
- Validation thresholds
- Privacy gates

## Files

```
ai/training/
├── feedback_loop.py              # Main feedback loop implementation
└── FEEDBACK_LOOP_GUIDE.md       # This documentation
```

## Quick Start

```bash
# 1. Run evaluation
python -m ai.training.evaluate_finetuned_model \
    --model-path ./models/fine-tuned \
    --test-data ./data/finetuning/finetuning_test.jsonl \
    --output ./evaluation_report.json

# 2. Generate feedback actions
python -m ai.training.feedback_loop \
    --evaluation-report ./evaluation_report.json \
    --output-dir ./feedback/actions

# 3. Review generated Linear issues
cat ./feedback/actions/linear_issues/*.md
```

## Architecture

### Components

1. **EvaluationParser**: Parses evaluation results and identifies failure patterns
2. **UpstreamCauseMapper**: Maps failures to likely upstream causes
3. **InterventionGenerator**: Generates concrete interventions
4. **FeedbackLoop**: Orchestrates the complete pipeline

### Failure Pattern Detection

The system detects these failure patterns:

| Pattern | Metric | Threshold | Type |
| --------- | -------- | ----------- | ------ |
| memory_recall_low | memory_recall_recall | < 0.6 | memory_deficiency |
| memory_irrelevant | memory_recall_precision | < 0.5 | memory_noise |
| context_drift | context_relevance | < 0.6 | context_alignment |
| reflection_absent | reflection_quality | < 0.5 | reflection_quality |
| generation_incoherent | generation_quality | < 0.6 | generation_quality |

### Upstream Domain Mapping

| Failure Type | Upstream Domain | Related Issue |
| -------------- | ----------------- | --------------- |
| memory_deficiency | Acquisition | PIX-188 |
| memory_noise | Curation | PIX-247 |
| context_alignment | Curation | PIX-247 |
| reflection_quality | Review | PIX-250 |
| generation_quality | Acquisition | PIX-188 |
| privacy_concern | Privacy | PIX-248 |

## Intervention Types

### Rule Change
Modify curation or validation rules to address patterns.

```json
{
  "type": "rule_change",
  "title": "Update curation rules",
  "description": "Add curation rule for memory_noise patterns",
  "upstream_domain": "curation"
}
```

### Threshold Change
Adjust quality thresholds for dataset inclusion.

```json
{
  "type": "threshold_change",
  "title": "Update quality thresholds",
  "description": "Raise context_relevance threshold from 0.55 to 0.75",
  "upstream_domain": "acquisition"
}
```

### Priority Change
Change source data priorities based on quality signals.

```json
{
  "type": "priority_change",
  "title": "Adjust source data priorities",
  "description": "Increase priority of high-quality memory-context pairs",
  "upstream_domain": "acquisition"
}
```

### Review Focus
Add human review focus area for borderline cases.

```json
{
  "type": "review_focus",
  "title": "Add human review focus area",
  "description": "Focus human review on reflection_quality patterns",
  "upstream_domain": "review"
}
```

### Validation Gate
Add new validation check to pipeline.

```json
{
  "type": "validation_gate",
  "title": "Add privacy validation gate",
  "description": "Add gate for privacy_concern detection",
  "upstream_domain": "privacy"
}
```

### Dataset Filter
Filter specific patterns from dataset.

```json
{
  "type": "dataset_filter",
  "title": "Add dataset filter",
  "description": "Filter memory_noise patterns from dataset",
  "upstream_domain": "curation"
}
```

## Usage

### Basic Flow

```python
from ai.training.feedback_loop import FeedbackLoop

# Initialize
loop = FeedbackLoop()

# Run complete feedback loop
report = loop.run(
    evaluation_report_path="./evaluation_report.json",
    output_dir="./feedback/actions"
)

# Access results
print(f"Failure patterns: {len(report.failure_patterns)}")
print(f"Interventions: {len(report.interventions)}")

# Save report
with open("./feedback_report.json", "w") as f:
    json.dump(report.to_dict(), f, indent=2)
```

### Advanced: Custom Pattern Detection

```python
from ai.training.feedback_loop import EvaluationParser

parser = EvaluationParser()

# Add custom pattern template
parser.PATTERN_TEMPLATES["custom_pattern"] = {
    "type": "custom_type",
    "description": "Custom failure pattern description",
    "threshold": 0.7,
    "metric": "custom_metric",
}

# Parse with custom pattern
patterns = parser.parse(evaluation_results)
```

### Advanced: Custom Mapping Rules

```python
from ai.training.feedback_loop import UpstreamCauseMapper

mapper = UpstreamCauseMapper()

# Add custom mapping rule
mapper.MAPPING_RULES["custom_type"] = {
    "domain": UpstreamDomain.CURATION,
    "hypothesis": "Custom upstream cause hypothesis",
    "evidence_sources": ["custom_evidence"],
}

# Map with custom rules
mappings = mapper.map(failure_patterns)
```

## Output Structure

```
./feedback/actions/
├── feedback_report.json          # Complete feedback report
└── linear_issues/                # Linear issue templates
    ├── intervention_pattern_1.md
    ├── intervention_pattern_2.md
    └── ...
```

### Feedback Report Format

```json
{
  "evaluation_source": "./evaluation_report.json",
  "generated_at": "2026-05-10T16:00:00Z",
  "total_evaluated": 1000,
  "overall_score": 0.72,
  "failure_patterns": [
    {
      "pattern_id": "pattern_memory_recall_low",
      "pattern_type": "memory_deficiency",
      "description": "Model fails to recall relevant memories in context",
      "severity": "high",
      "frequency": 0.35,
      "metrics_impacted": ["memory_recall_recall"]
    }
  ],
  "upstream_mappings": [
    {
      "failure_pattern": {...},
      "upstream_domain": "acquisition",
      "confidence": 0.85,
      "root_cause_hypothesis": "Source data lacks high-quality memory-context pairs"
    }
  ],
  "interventions": [
    {
      "intervention_id": "intervention_pattern_memory_recall_low",
      "intervention_type": "priority_change",
      "title": "Adjust source data priorities",
      "description": "Increase priority of high-quality memory-context pairs",
      "upstream_domain": "acquisition",
      "priority": "high",
      "expected_impact": "Improve memory_recall_recall by 10-20%"
    }
  ]
}
```

## Integration with Upstream Pipelines

### PIX-188 (Acquisition)
- Receives priority change recommendations
- Adjusts source data acquisition priorities
- Updates quality threshold requirements

### PIX-247 (Curation)
- Receives rule change recommendations
- Updates normalization and deduplication rules
- Implements new filtering criteria

### PIX-248 (Privacy)
- Receives validation gate recommendations
- Adds new privacy checks
- Updates compliance requirements

### PIX-250 (Review)
- Receives review focus recommendations
- Adds borderline case handling
- Updates review guidelines

## Monitoring

### Key Metrics

| Metric | Description | Target |
| -------- | ------------- | -------- |
| Feedback loop latency | Time from evaluation to action | < 24h |
| Intervention success rate | % of interventions that improve metrics | > 70% |
| Pattern detection accuracy | % of patterns validated by humans | > 80% |
| Upstream action completion | % of recommended actions completed | > 60% |

### Logging

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("feedback_loop")

# Feedback loop logs:
# - Pattern detection results
# - Upstream mapping confidence
# - Intervention generation
# - Linear issue creation
```

## Troubleshooting

### No Patterns Detected

**Cause**: Evaluation metrics all above thresholds

**Solution**: Lower thresholds or add more sensitive patterns

### Low Confidence Mappings

**Cause**: Insufficient evidence or ambiguous patterns

**Solution**: Add more evidence sources or refine mapping rules

### Interventions Not Improving Metrics

**Cause**: Root cause hypothesis incorrect

**Solution**: Review validation criteria, adjust hypothesis, iterate

## Best Practices

1. **Run feedback loop after every evaluation cycle**
2. **Prioritize critical and high severity patterns first**
3. **Validate intervention effectiveness before next cycle**
4. **Keep pattern definitions and mapping rules up to date**
5. **Document intervention outcomes for learning**

## Version History

| Version | Date | Changes |
| --------- | ------ | --------- |
| 1.0.0 | 2026-05-10 | Initial implementation |
