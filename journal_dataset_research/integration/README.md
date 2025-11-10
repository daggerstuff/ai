# Integration Planning Engine

The Integration Planning Engine assesses integration feasibility and creates preprocessing plans for acquired datasets from journal research sources.

## Overview

This module implements Task 5 of the journal dataset research system, providing comprehensive tools to:

1. **Analyze Dataset Structure** - Parse and understand different dataset formats
2. **Map Schemas** - Create mappings from dataset fields to training pipeline format
3. **Generate Transformation Specs** - Define required data transformations
4. **Estimate Complexity** - Calculate integration effort and complexity
5. **Generate Preprocessing Scripts** - Create Python scripts for data transformation

## Components

### IntegrationPlanningEngine

Main engine class that orchestrates all integration planning functionality.

**Key Methods:**

- `analyze_dataset_structure(dataset: AcquiredDataset) -> DatasetStructure`
  - Analyzes dataset format, schema, field types, and quality issues
  - Supports CSV, JSON, JSONL, Parquet, XML, and custom formats

- `create_schema_mapping(structure: DatasetStructure, target_format: str) -> List[SchemaMapping]`
  - Maps dataset fields to training pipeline schema
  - Supports ChatML and ConversationRecord target formats

- `create_transformation_specs(structure: DatasetStructure, mappings: List[SchemaMapping]) -> List[TransformationSpec]`
  - Generates transformation specifications for format conversion, field mapping, cleaning, and validation

- `estimate_complexity(structure, mappings, specs) -> Tuple[str, int]`
  - Estimates integration complexity (low/medium/high) and effort in hours

- `create_integration_plan(dataset: AcquiredDataset, target_format: str) -> IntegrationPlan`
  - Creates a complete integration plan combining all analysis

- `generate_preprocessing_script(plan: IntegrationPlan, output_path: str) -> str`
  - Generates a Python preprocessing script template

- `validate_integration_feasibility(plan: IntegrationPlan) -> bool`
  - Validates whether an integration plan is feasible

### DatasetStructure

Data class representing analyzed dataset structure:

- `format`: Dataset format (csv, json, jsonl, parquet, xml, custom)
- `schema`: Field names and types
- `field_types`: Type information for each field
- `field_distributions`: Statistical distributions for fields
- `quality_issues`: List of detected data quality problems
- `nested_structures`: List of nested field paths
- `sample_size`: Number of records analyzed
- `total_records`: Total records in dataset (if available)

### SchemaMapping

Data class representing field mappings:

- `dataset_field`: Source field name
- `pipeline_field`: Target pipeline field name
- `transformation_type`: Type of transformation (direct, transform, combine, extract)
- `transformation_logic`: Description of transformation logic
- `required`: Whether field is required
- `default_value`: Default value if field is missing

## Supported Formats

### CSV
- Automatic schema detection
- Field type inference
- Missing value detection
- Distribution analysis for numeric fields

### JSON/JSONL
- Nested structure analysis
- Field extraction from complex objects
- Type consistency checking
- Support for arrays and nested objects

### Parquet
- Schema extraction from Parquet metadata
- Efficient sampling for large files
- Type information from Arrow schema

### XML
- Basic format identification
- Note: Requires format-specific parsing (manual analysis recommended)

### Custom
- Placeholder for unknown formats
- Requires manual structure analysis

## Training Pipeline Schema

The engine supports two target formats:

### ChatML Format
- Required fields: `messages`, `id`, `source`
- Message structure: `{"role": "user|assistant|system", "content": "text"}`
- Optional fields: `timestamp`, `quality_score`, `tags`, `mental_health_condition`

### ConversationRecord Format
- Required fields: `id`, `title`, `turns`, `source_type`, `source_id`
- Turn structure: `{"speaker_id": str, "content": str, "timestamp": optional, "metadata": optional}`

## Complexity Estimation

Complexity is calculated based on:

1. **Format Complexity** (1-5 points)
   - CSV: 1 point
   - JSON/JSONL: 2 points
   - Parquet: 2 points
   - XML: 4 points
   - Custom: 5 points

2. **Schema Complexity** (0+ points)
   - Missing required fields: 2 points each
   - Nested structures: 2 points each

3. **Transformation Complexity** (0+ points)
   - Number of transformations: 1 point each

4. **Quality Issues** (0+ points)
   - Each quality issue: 1 point

**Complexity Levels:**
- **Low**: ≤3 points → 2-4 hours
- **Medium**: 4-7 points → 4-11 hours
- **High**: >7 points → 8+ hours

## Usage Example

```python
from ai.journal_dataset_research.integration import IntegrationPlanningEngine
from ai.journal_dataset_research.models.dataset_models import AcquiredDataset

# Initialize engine
engine = IntegrationPlanningEngine()

# Create integration plan
plan = engine.create_integration_plan(
    dataset=acquired_dataset,
    target_format="chatml"
)

# Check feasibility
if engine.validate_integration_feasibility(plan):
    # Generate preprocessing script
    script_path = engine.generate_preprocessing_script(
        plan,
        output_path="scripts/preprocess_dataset.py"
    )
    print(f"Integration plan created: {plan.complexity} complexity, {plan.estimated_effort_hours}h")
else:
    print("Integration not feasible")
```

## Testing

Comprehensive test suite in `tests/test_integration_planning_engine.py`:

- Dataset structure analysis tests
- Schema mapping tests
- Transformation spec generation tests
- Complexity estimation tests
- Integration plan creation tests
- Preprocessing script generation tests
- Feasibility validation tests

Run tests with:
```bash
uv run pytest ai/journal_dataset_research/tests/test_integration_planning_engine.py
```

## Dependencies

- `pandas`: CSV and data analysis
- `pyarrow`: Parquet file support
- Standard library: `json`, `logging`, `os`, `pathlib`, `dataclasses`

## Related Modules

- `models.dataset_models`: Data model definitions (AcquiredDataset, IntegrationPlan, etc.)
- `acquisition.acquisition_manager`: Dataset acquisition functionality
- `evaluation.evaluation_engine`: Dataset quality evaluation

## Implementation Status

✅ **Task 5.1**: Dataset structure analysis - Complete
✅ **Task 5.2**: Schema mapping - Complete
✅ **Task 5.3**: Transformation specification generator - Complete
✅ **Task 5.4**: Complexity estimation - Complete
✅ **Task 5.5**: Preprocessing script generation - Complete

All subtasks for Task 5 (Integration Planning Engine) are complete and tested.

