# Journal Dataset Research System - User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [CLI Commands](#cli-commands)
5. [Workflow Examples](#workflow-examples)
6. [Best Practices](#best-practices)
7. [Tips and Tricks](#tips-and-tricks)

## Getting Started

### Overview

The Journal Dataset Research System is an automated platform for discovering, evaluating, acquiring, and integrating therapeutic datasets from academic sources. The system operates through a coordinated workflow that spans four main phases:

1. **Discovery**: Discover dataset sources from academic repositories
2. **Evaluation**: Evaluate discovered datasets across multiple dimensions
3. **Acquisition**: Acquire datasets from identified sources
4. **Integration**: Create integration plans for acquired datasets

### Key Features

- **Automated Discovery**: Search across multiple academic repositories (PubMed, DOAJ, Dryad, Zenodo, ClinicalTrials.gov)
- **Multi-Dimensional Evaluation**: Evaluate datasets across therapeutic relevance, data structure quality, training integration potential, and ethical accessibility
- **Compliance Checking**: Comprehensive compliance checking including license compatibility, privacy verification, and HIPAA validation
- **Secure Storage**: Organized storage with optional encryption for sensitive data
- **Integration Planning**: Automated integration planning with preprocessing script generation
- **Progress Tracking**: Real-time progress tracking and reporting
- **Session Management**: Session-based workflow with checkpointing and resume capability

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager
- Internet connection for API access

### Installation Steps

1. **Clone the repository** (if not already available):
```bash
git clone <repository-url>
cd pixelated
```

2. **Install dependencies**:
```bash
cd ai
uv install
```

Or with pip:
```bash
cd ai
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -m ai.sourcing.journal.cli.cli --help
```

### Optional Dependencies

For YAML configuration support:
```bash
uv add pyyaml
```

For enhanced reporting and visualization:
```bash
uv add matplotlib pandas
```

## Configuration

### Configuration File

The system uses a configuration file located at `~/.journal_research/config.yaml` (or `config.json` if YAML is not available).

### Default Configuration

The default configuration includes:

```yaml
orchestrator:
  max_retries: 3
  retry_delay_seconds: 1.0
  progress_history_limit: 100
  parallel_evaluation: false
  parallel_integration_planning: false
  max_workers: 4
  session_storage_path: null
  visualization_max_points: 100
  fallback_on_failure: true

discovery:
  pubmed:
    api_key: null
    base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    search_limit: 100
  doaj:
    base_url: "https://doaj.org/api/v2"
  repositories:
    dryad:
      base_url: "https://datadryad.org/api/v2"
    zenodo:
      base_url: "https://zenodo.org/api"
    clinical_trials:
      base_url: "https://clinicaltrials.gov/api/v2"

evaluation:
  therapeutic_relevance_weight: 0.35
  data_structure_quality_weight: 0.25
  training_integration_weight: 0.20
  ethical_accessibility_weight: 0.20
  high_priority_threshold: 7.5
  medium_priority_threshold: 5.0

acquisition:
  storage_base_path: "data/acquired_datasets"
  encryption_enabled: false
  download_timeout: 3600
  max_retries: 3
  chunk_size: 8192
  resume_downloads: true

integration:
  target_format: "chatml"
  default_complexity: "medium"

logging:
  level: "INFO"
  file: null
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Environment Variables

You can override configuration values using environment variables:

- `JOURNAL_RESEARCH_PUBMED_API_KEY`: PubMed API key
- `JOURNAL_RESEARCH_STORAGE_PATH`: Storage path for acquired datasets
- `JOURNAL_RESEARCH_LOG_LEVEL`: Logging level
- `JOURNAL_RESEARCH_MAX_RETRIES`: Maximum retries
- `JOURNAL_RESEARCH_MAX_WORKERS`: Maximum workers for parallel processing

Example:
```bash
export JOURNAL_RESEARCH_PUBMED_API_KEY="your-api-key"
export JOURNAL_RESEARCH_STORAGE_PATH="/path/to/datasets"
export JOURNAL_RESEARCH_LOG_LEVEL="DEBUG"
```

### Configuration Management

View configuration:
```bash
python -m ai.sourcing.journal.cli.cli config show
```

Get specific configuration value:
```bash
python -m ai.sourcing.journal.cli.cli config get "orchestrator.max_retries"
```

Set configuration value:
```bash
python -m ai.sourcing.journal.cli.cli config set "orchestrator.max_retries" "5"
```

## CLI Commands

### Search Command

Search for dataset sources from academic repositories.

```bash
python -m ai.sourcing.journal.cli.cli search \
    --keywords "therapy" "counseling" "psychotherapy" \
    --sources "pubmed" "doaj" \
    --session-id "my_session" \
    --interactive
```

**Options**:
- `--keywords, -k`: Search keywords (multiple allowed)
- `--sources, -s`: Target sources (pubmed, doaj, etc.) (multiple allowed)
- `--session-id`: Session ID (optional, auto-generated if not provided)
- `--interactive, -i`: Interactive mode for manual oversight

### Evaluate Command

Evaluate discovered datasets across multiple dimensions.

```bash
python -m ai.sourcing.journal.cli.cli evaluate \
    --session-id "my_session" \
    --interactive
```

**Options**:
- `--session-id`: Session ID (required)
- `--interactive, -i`: Interactive mode for manual evaluation overrides

### Acquire Command

Acquire datasets from identified sources.

```bash
python -m ai.sourcing.journal.cli.cli acquire \
    --session-id "my_session" \
    --interactive
```

**Options**:
- `--session-id`: Session ID (required)
- `--interactive, -i`: Interactive mode for manual acquisition approval

### Integrate Command

Create integration plans for acquired datasets.

```bash
python -m ai.sourcing.journal.cli.cli integrate \
    --session-id "my_session" \
    --target-format "chatml" \
    --interactive
```

**Options**:
- `--session-id`: Session ID (required)
- `--target-format`: Target format (chatml, conversation_record) (default: chatml)
- `--interactive, -i`: Interactive mode for manual integration approval

### Status Command

Check the status of a research session.

```bash
# Check specific session
python -m ai.sourcing.journal.cli.cli status --session-id "my_session"

# List all sessions
python -m ai.sourcing.journal.cli.cli status
```

**Options**:
- `--session-id`: Session ID (optional, lists all sessions if not provided)

### Report Command

Generate reports for a research session.

```bash
python -m ai.sourcing.journal.cli.cli report \
    --session-id "my_session" \
    --output "report.json" \
    --format "json"
```

**Options**:
- `--session-id`: Session ID (required)
- `--output`: Output file path (default: stdout)
- `--format`: Report format (json, markdown) (default: json)

### Config Command

Manage configuration.

```bash
# Show configuration
python -m ai.sourcing.journal.cli.cli config show

# Get specific config value
python -m ai.sourcing.journal.cli.cli config get "orchestrator.max_retries"

# Set config value
python -m ai.sourcing.journal.cli.cli config set "orchestrator.max_retries" "5"
```

### Global Options

All commands support these global options:

- `--config PATH`: Path to configuration file
- `--dry-run`: Run in dry-run mode (no actual changes)
- `--verbose, -v`: Enable verbose logging
- `--log-file PATH`: Log file path

## Workflow Examples

### Example 1: Complete Automated Workflow

Run the complete workflow automatically using the main execution script:

```bash
python ai/sourcing/journal/main.py \
    --target-sources "pubmed" "doaj" \
    --keywords "therapy" "counseling" "psychotherapy" \
    --interactive
```

This will:
1. Create a new research session
2. Discover dataset sources from PubMed and DOAJ
3. Evaluate discovered datasets
4. Acquire high-priority datasets
5. Create integration plans for acquired datasets
6. Generate progress reports

### Example 2: Step-by-Step Workflow

Run the workflow step-by-step using CLI commands:

```bash
# 1. Search for sources
python -m ai.sourcing.journal.cli.cli search \
    --keywords "therapy" "counseling" \
    --sources "pubmed" "doaj" \
    --session-id "my_session"

# 2. Evaluate sources
python -m ai.sourcing.journal.cli.cli evaluate \
    --session-id "my_session" \
    --interactive

# 3. Acquire datasets
python -m ai.sourcing.journal.cli.cli acquire \
    --session-id "my_session" \
    --interactive

# 4. Create integration plans
python -m ai.sourcing.journal.cli.cli integrate \
    --session-id "my_session" \
    --target-format "chatml"

# 5. Generate report
python -m ai.sourcing.journal.cli.cli report \
    --session-id "my_session" \
    --output "report.json"
```

### Example 3: Resume Interrupted Workflow

Resume an interrupted workflow from the last checkpoint:

```bash
python ai/sourcing/journal/main.py \
    --session-id "my_session" \
    --resume
```

### Example 4: Dry-Run Mode

Test the workflow without making actual changes:

```bash
python ai/sourcing/journal/main.py \
    --target-sources "pubmed" \
    --keywords "therapy" \
    --dry-run
```

### Example 5: Interactive Mode

Run the workflow with manual oversight at each phase:

```bash
python ai/sourcing/journal/main.py \
    --target-sources "pubmed" "doaj" \
    --keywords "therapy" "counseling" \
    --interactive
```

In interactive mode, you can:
- Review datasets before evaluation
- Approve acquisition requests
- Review integration plans
- Override evaluation scores

### Example 6: Custom Configuration

Use a custom configuration file:

```bash
python -m ai.sourcing.journal.cli.cli search \
    --config /path/to/custom/config.yaml \
    --keywords "therapy" \
    --sources "pubmed"
```

### Example 7: Parallel Processing

Enable parallel processing for evaluation and integration planning:

```bash
# Update configuration
python -m ai.sourcing.journal.cli.cli config set "orchestrator.parallel_evaluation" "true"
python -m ai.sourcing.journal.cli.cli config set "orchestrator.parallel_integration_planning" "true"
python -m ai.sourcing.journal.cli.cli config set "orchestrator.max_workers" "8"

# Run workflow
python ai/sourcing/journal/main.py \
    --target-sources "pubmed" "doaj" \
    --keywords "therapy" "counseling"
```

### Example 8: Weekly Targets

Set weekly targets for research progress:

```bash
python ai/sourcing/journal/main.py \
    --target-sources "pubmed" "doaj" \
    --keywords "therapy" "counseling" \
    --weekly-targets '{"sources_identified": 10, "datasets_evaluated": 5, "datasets_acquired": 3}'
```

## Best Practices

### 1. Session Management

- **Use unique session IDs**: Always use unique session IDs for different research efforts
- **Save session state regularly**: The system automatically saves session state, but you can also save manually
- **Resume interrupted workflows**: Use the `--resume` flag to resume interrupted workflows from the last checkpoint

### 2. Configuration

- **Use environment variables**: Use environment variables for sensitive configuration like API keys
- **Validate configuration**: Always validate configuration before running workflows
- **Use default values**: The system provides sensible default values for all configuration options

### 3. Error Handling

- **Enable logging**: Enable verbose logging for debugging and monitoring
- **Handle exceptions**: Always handle exceptions and implement retry logic
- **Use dry-run mode**: Use dry-run mode to test workflows without making actual changes

### 4. Compliance

- **Perform compliance checks**: Always perform compliance checks before acquiring datasets
- **Review compliance reports**: Review compliance reports to ensure datasets meet requirements
- **Document compliance decisions**: Document compliance decisions for audit purposes

### 5. Progress Tracking

- **Track progress regularly**: Track progress regularly and generate reports
- **Set weekly targets**: Set weekly targets to track progress against goals
- **Monitor metrics**: Monitor metrics like sources identified, datasets evaluated, and datasets acquired

### 6. Security

- **Use encryption**: Use encryption for sensitive data and secure credential management
- **Secure storage**: Store datasets in secure locations with appropriate access controls
- **Audit logging**: Enable audit logging for all data access and modifications

### 7. Performance

- **Use parallel processing**: Use parallel processing for evaluation and integration planning when possible
- **Configure timeouts**: Configure appropriate timeouts for network operations
- **Monitor resources**: Monitor system resources and adjust configuration as needed

### 8. Documentation

- **Document workflows**: Document workflows and decisions for future reference
- **Generate reports**: Generate reports regularly for stakeholders
- **Maintain catalogs**: Maintain catalogs of discovered and acquired datasets

## Tips and Tricks

### 1. Efficient Searching

- **Use specific keywords**: Use specific keywords to narrow down search results
- **Combine keywords**: Combine multiple keywords to find relevant datasets
- **Use MeSH terms**: Use Medical Subject Headings (MeSH) terms for PubMed searches
- **Filter by open access**: Filter by open access status to find freely available datasets

### 2. Evaluation Optimization

- **Prioritize high-score datasets**: Prioritize datasets with high overall scores
- **Review evaluation notes**: Review evaluation notes to understand scoring decisions
- **Use interactive mode**: Use interactive mode to override evaluation scores when needed
- **Check compliance status**: Always check compliance status before acquiring datasets

### 3. Acquisition Efficiency

- **Use direct downloads**: Prefer direct downloads over API-based retrieval when possible
- **Resume interrupted downloads**: Use resume capability for interrupted downloads
- **Verify integrity**: Always verify download integrity using checksums
- **Organize storage**: Organize storage by source type and acquisition date

### 4. Integration Planning

- **Review integration plans**: Review integration plans before executing preprocessing scripts
- **Estimate complexity**: Use complexity estimates to plan integration efforts
- **Test preprocessing scripts**: Test preprocessing scripts on sample data before full integration
- **Document transformations**: Document all transformations for future reference

### 5. Troubleshooting

- **Check logs**: Check logs for error messages and debugging information
- **Use verbose mode**: Use verbose mode to get detailed output
- **Test with dry-run**: Test workflows with dry-run mode before actual execution
- **Review configuration**: Review configuration for incorrect settings

### 6. Performance Optimization

- **Use parallel processing**: Enable parallel processing for large datasets
- **Configure workers**: Configure the number of workers based on system resources
- **Use caching**: Use caching for frequently accessed data
- **Monitor resources**: Monitor system resources and adjust configuration as needed

### 7. Reporting

- **Generate regular reports**: Generate reports regularly for stakeholders
- **Use multiple formats**: Use multiple formats (JSON, Markdown) for different audiences
- **Include visualizations**: Include visualizations in reports for better understanding
- **Document findings**: Document key findings and challenges in reports

### 8. Collaboration

- **Share session IDs**: Share session IDs with team members for collaboration
- **Use version control**: Use version control for configuration and scripts
- **Document decisions**: Document decisions and rationale for future reference
- **Review reports**: Review reports with team members for feedback

## Advanced Usage

### Custom Discovery Services

Implement custom discovery services by extending the `DiscoveryServiceProtocol`:

```python
from ai.sourcing.journal.orchestrator.types import DiscoveryServiceProtocol
from ai.sourcing.journal.models.dataset_models import DatasetSource, ResearchSession

class CustomDiscoveryService(DiscoveryServiceProtocol):
    def discover_sources(self, session: ResearchSession) -> List[DatasetSource]:
        # Implement custom discovery logic
        return sources
```

### Custom Evaluation Criteria

Extend the evaluation engine to add custom evaluation criteria:

```python
from ai.sourcing.journal.evaluation.evaluation_engine import EvaluationEngine

class CustomEvaluationEngine(EvaluationEngine):
    def assess_custom_criteria(self, source: DatasetSource) -> Tuple[int, str]:
        # Implement custom evaluation logic
        return score, notes
```

### Custom Acquisition Methods

Extend the acquisition manager to add custom acquisition methods:

```python
from ai.sourcing.journal.acquisition.acquisition_manager import AcquisitionManager

class CustomAcquisitionManager(AcquisitionManager):
    def custom_acquisition_method(self, source: DatasetSource) -> AcquiredDataset:
        # Implement custom acquisition logic
        return acquired_dataset
```

### Custom Integration Transformations

Extend the integration planning engine to add custom transformation logic:

```python
from ai.sourcing.journal.integration.integration_planning_engine import IntegrationPlanningEngine

class CustomIntegrationEngine(IntegrationPlanningEngine):
    def custom_transformation(self, dataset: AcquiredDataset) -> IntegrationPlan:
        # Implement custom transformation logic
        return integration_plan
```

## Getting Help

### Documentation

- **Architecture Documentation**: See `docs/ARCHITECTURE.md` for system architecture details
- **API Reference**: See `docs/API_REFERENCE.md` for API documentation
- **Troubleshooting Guide**: See `docs/TROUBLESHOOTING.md` for troubleshooting tips

### Support

- **Logs**: Check logs for error messages and debugging information
- **Configuration**: Review configuration for incorrect settings
- **Examples**: Review workflow examples for usage patterns

### Community

- **Issues**: Report issues and bugs to the issue tracker
- **Contributions**: Contribute improvements and enhancements
- **Feedback**: Provide feedback and suggestions for improvements

## Conclusion

The Journal Dataset Research System provides a comprehensive, automated platform for discovering, evaluating, acquiring, and integrating therapeutic datasets from academic sources. By following this user guide, you can effectively use the system to streamline your research workflow and ensure compliance with ethical and legal requirements.

For more information, see the architecture documentation, API reference, and troubleshooting guide.

