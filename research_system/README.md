# Journal Dataset Research System

A systematic approach for researching, evaluating, and acquiring therapeutic journal datasets from open access sources to enhance the Pixelated Empathy training data pipeline.

## Overview

This system implements a structured research methodology combining automated search tools, manual evaluation processes, and integration planning to discover and acquire high-quality academic therapeutic content.

## Project Structure

```
ai/research_system/
├── __init__.py                 # Package initialization
├── config.py                   # Configuration management
├── config.yaml                 # Configuration file
├── models.py                   # Core data models
├── .env.example                # Environment variables template
├── README.md                   # This file
│
├── source_discovery/           # Source Discovery Engine
│   └── __init__.py
│
├── evaluation/                 # Dataset Evaluation Engine
│   └── __init__.py
│
├── acquisition/                # Access & Acquisition Manager
│   └── __init__.py
│
├── integration/                # Integration Planning Engine
│   └── __init__.py
│
├── orchestration/              # Research Orchestrator
│   └── __init__.py
│
├── documentation/              # Documentation & Tracking System
│   └── __init__.py
│
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
│
├── logs/                       # Research activity logs
├── acquired_datasets/          # Downloaded datasets
└── reports/                    # Generated reports
```

## Core Data Models

### DatasetSource
Represents a potential therapeutic dataset source from academic literature.

### DatasetEvaluation
Quality assessment and scoring for a dataset source across four dimensions:
- Therapeutic Relevance (1-10)
- Data Structure Quality (1-10)
- Training Integration Potential (1-10)
- Ethical Accessibility (1-10)

### AccessRequest
Tracks dataset access requests and their status.

### AcquiredDataset
Represents a successfully acquired dataset with storage metadata.

### IntegrationPlan
Plan for integrating a dataset into the training pipeline.

### ResearchSession
Represents a research session with targets and current state.

### ResearchProgress
Tracks progress metrics for the research workflow.

## Configuration

### Setup

1. Copy the environment variables template:
   ```bash
   cp .env.example .env
   ```

2. Fill in your API credentials in `.env`:
   - NCBI_API_KEY and NCBI_EMAIL for PubMed Central
   - Optional: DOAJ_API_KEY, DRYAD_API_KEY, ZENODO_ACCESS_TOKEN

3. Customize `config.yaml` as needed for your research targets

### Configuration Structure

The `config.yaml` file contains:
- **API Endpoints**: URLs and credentials for data sources
- **Search Keywords**: Organized by dataset type
- **MeSH Terms**: For PubMed searches
- **Storage Paths**: Where to store datasets, logs, and reports
- **Research Targets**: Weekly goals and metrics
- **Evaluation Weights**: Scoring dimension weights
- **Rate Limits**: API request throttling
- **Security Settings**: Encryption and audit configuration

## Usage

### Loading Configuration

```python
from research_system.config import get_config

# Get configuration instance
config = get_config()

# Access configuration values
pubmed_config = config.get_api_endpoint("pubmed")
keywords = config.get_search_keywords("therapy_transcripts")
storage_path = config.get_storage_path("acquired_datasets")
```

### Working with Data Models

```python
from datetime import datetime
from research_system.models import DatasetSource, DatasetEvaluation

# Create a dataset source
source = DatasetSource(
    source_id="PMC12345",
    title="Therapeutic Dialogue Dataset",
    authors=["Smith, J.", "Doe, A."],
    publication_date=datetime(2024, 1, 15),
    source_type="journal",
    url="https://example.com/dataset",
    doi="10.1234/example",
    abstract="A dataset of therapy transcripts...",
    keywords=["therapy", "transcripts", "mental health"],
    open_access=True,
    data_availability="available",
    discovery_date=datetime.now(),
    discovery_method="pubmed_search"
)

# Validate the source
is_valid, errors = source.validate()
if not is_valid:
    print(f"Validation errors: {errors}")

# Create an evaluation
evaluation = DatasetEvaluation(
    source_id="PMC12345",
    therapeutic_relevance=9,
    therapeutic_relevance_notes="High-quality therapy transcripts",
    data_structure_quality=8,
    data_structure_notes="Well-organized CSV format",
    training_integration=7,
    integration_notes="Requires field mapping",
    ethical_accessibility=9,
    ethical_notes="CC-BY license, fully anonymized",
    overall_score=0.0,  # Will be calculated
    priority_tier="high",
    evaluation_date=datetime.now(),
    evaluator="researcher_name",
    competitive_advantages=["Unique crisis intervention data"]
)

# Calculate overall score
evaluation.overall_score = evaluation.calculate_overall_score()
print(f"Overall score: {evaluation.overall_score}")
```

## Development

### Running Tests

```bash
# Run all tests
pytest ai/research_system/tests/

# Run unit tests only
pytest ai/research_system/tests/unit/

# Run integration tests only
pytest ai/research_system/tests/integration/
```

### Adding New Components

1. Create module in appropriate directory (e.g., `source_discovery/pubmed.py`)
2. Implement functionality following existing patterns
3. Add unit tests in `tests/unit/`
4. Update this README with usage examples

## Next Steps

The following components need to be implemented:

1. **Source Discovery Engine** - PubMed, DOAJ, repository searches
2. **Dataset Evaluation Engine** - Automated and manual evaluation
3. **Access & Acquisition Manager** - Download and storage management
4. **Integration Planning Engine** - Preprocessing and pipeline integration
5. **Research Orchestrator** - Workflow coordination
6. **Documentation & Tracking System** - Logging and reporting
7. **CLI Interface** - Command-line tools for research operations

See `tasks.md` in the spec directory for detailed implementation plan.

## License

Part of the Pixelated Empathy AI project.
