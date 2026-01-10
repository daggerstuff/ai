# NVIDIA NeMo Data Designer Integration

This module provides integration with [NVIDIA NeMo Data Designer](https://build.nvidia.com/nemo/data-designer) for generating high-quality, domain-specific synthetic datasets for training, fine-tuning, and evaluating AI models in the Pixelated Empathy platform.

## Overview

NVIDIA NeMo Data Designer is a powerful tool for generating realistic synthetic datasets using large language models (LLMs). It allows you to:

- **Create realistic datasets** with various data types (categorical, integer, float, text)
- **Define custom column types** with constraints and relationships
- **Leverage LLM intelligence** for intelligent data generation
- **Scale to large datasets** via batch processing
- **Access via RESTful API** for programmatic integration

## Features

### Therapeutic Dataset Generation
Generate synthetic therapeutic datasets with:
- Demographic information (age, gender, ethnicity)
- Mental health symptoms and diagnoses
- Treatment types and frequencies
- Treatment outcomes and satisfaction scores

### Bias Detection Dataset Generation
Create datasets specifically designed for bias detection analysis:
- Protected attributes (gender, ethnicity, age groups)
- Outcome variables (treatment response, attendance, ratings)
- Fairness analysis ready format

### Custom Dataset Generation
Build completely custom datasets with:
- User-defined column configurations
- Multiple data types (categories, integers, floats, text)
- Custom constraints and value ranges

## Installation

### Prerequisites

1. **Python 3.11+** with `uv` package manager
2. **NVIDIA API Key** from [NVIDIA Build](https://build.nvidia.com)
3. **Docker and Docker Compose** (for local deployment)
4. **NeMo Data Designer service deployed** (see deployment section below)

### Important: Deployment Required

**NeMo Data Designer must be deployed locally or on a cluster** - it is not a cloud API service. You need to deploy the service before using this integration.

### Setup

1. **Install the package** (already included in `pyproject.toml`):
   ```bash
   uv pip install 'nemo-microservices[data-designer]'
   ```

2. **Deploy NeMo Data Designer**:
   
   **Option A: Docker Compose (Recommended for local development)**
   ```bash
   ./scripts/deploy-nemo-data-designer.sh
   ```
   
   Or manually:
   ```bash
   docker-compose -f docker-compose.nemo-data-designer.yml up -d
   ```
   
   **Option B: Kubernetes/Helm** (for production)

   For Kubernetes deployment, use the new deployment script:
   ```bash
   ./scripts/infrastructure/deploy-nemo-data-designer-k8s.sh
   ```

   Or manually deploy using the provided Kubernetes manifest:
   ```bash
    kubectl apply -f ai/deployment/nemo-data-designer-k8s.yaml
   ```

   See the [official deployment guide](https://docs.nvidia.com/nemo/microservices/latest/set-up/deploy-as-microservices/data-designer/parent-chart.html) for more details.

3. **Set environment variables**:
   ```bash
   export NVIDIA_API_KEY="your-api-key-here"
   export NEMO_DATA_DESIGNER_BASE_URL="http://localhost:8000"  # For local Docker Compose
   export NEMO_DATA_DESIGNER_TIMEOUT="300"  # Optional, default: 300 seconds
   export NEMO_DATA_DESIGNER_MAX_RETRIES="3"  # Optional, default: 3
   export NEMO_DATA_DESIGNER_BATCH_SIZE="1000"  # Optional, default: 1000
   ```

   Or create a `.env` file:
   ```env
   NVIDIA_API_KEY=your-api-key-here
   # For local Docker Compose deployment
   NEMO_DATA_DESIGNER_BASE_URL=http://localhost:8000
   # For Kubernetes deployment, use your cluster ingress URL
   # NEMO_DATA_DESIGNER_BASE_URL=https://nemo-data-designer.your-cluster-domain.com
   NEMO_DATA_DESIGNER_TIMEOUT=300
   NEMO_DATA_DESIGNER_MAX_RETRIES=3
   NEMO_DATA_DESIGNER_BATCH_SIZE=1000
   ```
   
   **Note**: The `NEMO_DATA_DESIGNER_BASE_URL` depends on your deployment:
   - `http://localhost:8000` for local Docker Compose
   - Your cluster ingress URL for Kubernetes (e.g., `https://nemo-data-designer.your-cluster-domain.com`)

## Quick Start

### Basic Usage

```python
from ai.data_designer import NeMoDataDesignerService

# Initialize service (loads config from environment)
service = NeMoDataDesignerService()

# Generate therapeutic dataset
result = service.generate_therapeutic_dataset(num_samples=1000)

print(f"Generated {result['num_samples']} samples")
print(f"Columns: {result['columns']}")
print(f"Data: {result['data']}")
```

### Generate Therapeutic Dataset

```python
from ai.data_designer import NeMoDataDesignerService

service = NeMoDataDesignerService()

result = service.generate_therapeutic_dataset(
    num_samples=1000,
    include_demographics=True,
    include_symptoms=True,
    include_treatments=True,
    include_outcomes=True,
)

# Access the generated data
data = result['data']
```

### Generate Bias Detection Dataset

```python
from ai.data_designer import NeMoDataDesignerService

service = NeMoDataDesignerService()

result = service.generate_bias_detection_dataset(
    num_samples=500,
    protected_attributes=["gender", "ethnicity", "age_group"],
)

# Use for bias analysis
data = result['data']
protected_attrs = result['protected_attributes']
```

### Generate Custom Dataset

```python
from ai.data_designer import NeMoDataDesignerService
from nemo_microservices.data_designer.essentials import (
    SamplerColumnConfig,
    SamplerType,
    CategorySamplerParams,
    IntegerSamplerParams,
)

service = NeMoDataDesignerService()

# Define custom columns
column_configs = [
    SamplerColumnConfig(
        name="patient_id",
        sampler_type=SamplerType.INTEGER,
        params=IntegerSamplerParams(min_value=1, max_value=10000),
    ),
    SamplerColumnConfig(
        name="therapy_type",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["Individual", "Group", "Couples", "Family"],
        ),
    ),
]

result = service.generate_custom_dataset(
    column_configs=column_configs,
    num_samples=200,
)
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NVIDIA_API_KEY` | Your NVIDIA API key (required) | - |
| `NEMO_DATA_DESIGNER_BASE_URL` | Base URL for the Data Designer service | `http://localhost:8000` |
| `NEMO_DATA_DESIGNER_TIMEOUT` | Request timeout in seconds | `300` |
| `NEMO_DATA_DESIGNER_MAX_RETRIES` | Maximum retry attempts | `3` |
| `NEMO_DATA_DESIGNER_BATCH_SIZE` | Batch size for processing | `1000` |

### Custom Configuration

```python
from ai.data_designer import NeMoDataDesignerService, DataDesignerConfig

config = DataDesignerConfig(
    base_url="http://localhost:8000",
    api_key="your-api-key",
    timeout=600,
    max_retries=5,
    batch_size=500,
)

service = NeMoDataDesignerService(config=config)
```

## Integration with Existing Systems

### Bias Detection Integration

The generated datasets can be directly used with the existing bias detection system:

```python
from ai.data_designer import NeMoDataDesignerService
from src.lib.ai.bias_detection.python_service.bias_detection_service import BiasDetectionService

# Generate dataset
designer_service = NeMoDataDesignerService()
dataset = designer_service.generate_bias_detection_dataset(num_samples=1000)

# Use with bias detection
bias_service = BiasDetectionService()
analysis = bias_service.analyze_session_bias(dataset['data'])
```

### Dataset Pipeline Integration

Integrate with the existing dataset pipeline:

```python
from ai.data_designer import NeMoDataDesignerService
from ai.dataset_pipeline.main_orchestrator import DatasetOrchestrator

# Generate synthetic data
designer_service = NeMoDataDesignerService()
synthetic_data = designer_service.generate_therapeutic_dataset(num_samples=5000)

# Process through pipeline
orchestrator = DatasetOrchestrator()
processed = orchestrator.process_dataset(synthetic_data['data'])
```

## Examples

See `ai/data_designer/examples.py` for complete working examples:

```bash
# Run examples
uv run python ai/data_designer/examples.py
```

## API Reference

### `NeMoDataDesignerService`

Main service class for generating synthetic datasets.

#### Methods

- `generate_therapeutic_dataset(num_samples, include_demographics, include_symptoms, include_treatments, include_outcomes)` - Generate therapeutic dataset
- `generate_bias_detection_dataset(num_samples, protected_attributes)` - Generate bias detection dataset
- `generate_custom_dataset(column_configs, num_samples)` - Generate custom dataset

### `DataDesignerConfig`

Configuration class for the service.

#### Parameters

- `base_url` - API base URL
- `api_key` - NVIDIA API key
- `timeout` - Request timeout in seconds
- `max_retries` - Maximum retry attempts
- `batch_size` - Batch size for processing

## Deployment Options

NeMo Data Designer is deployed on your infrastructure (local Docker Compose or Kubernetes/Helm) and then accessed via its REST API.
See the [NVIDIA Documentation](https://docs.nvidia.com/nemo/microservices/latest/set-up/deploy-as-microservices/data-designer/parent-chart.html) for deployment details.

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Ensure `NVIDIA_API_KEY` is set in your environment
   - Get your API key from https://build.nvidia.com/nemo/data-designer

2. **Import Error**
   - Install the package: `uv pip install 'nemo-microservices[data-designer]'`
   - Ensure you're in the correct Python environment

3. **Timeout Errors**
   - Increase `NEMO_DATA_DESIGNER_TIMEOUT` for large datasets
   - Reduce `num_samples` and process in batches

4. **Rate Limiting**
   - The API may have rate limits
   - Implement retry logic with exponential backoff
   - Consider batch processing for large datasets

## Resources

- [NVIDIA NeMo Data Designer Documentation](https://docs.nvidia.com/nemo/microservices/latest/set-up/deploy-as-microservices/data-designer/parent-chart.html)
- [NVIDIA Build Platform](https://build.nvidia.com/nemo/data-designer)
- [NeMo Microservices SDK](https://github.com/NVIDIA/NeMo-Microservices)

## License

This integration is part of the Pixelated Empathy platform and follows the project's licensing terms.

