# Journal Dataset Research System - Test Suite

This directory contains comprehensive tests for the journal dataset research system, covering unit tests, integration tests, and end-to-end tests.

## Test Structure

### Unit Tests

- **`test_dataset_models.py`**: Tests for all data models (DatasetSource, DatasetEvaluation, AccessRequest, etc.)
- **`test_evaluation_engine.py`**: Tests for dataset evaluation engine
- **`test_acquisition_manager.py`**: Tests for access and acquisition manager
- **`test_compliance.py`**: Tests for compliance module (license checker, privacy verifier, HIPAA validator, etc.)
- **`test_integration_planning_engine.py`**: Tests for integration planning engine
- **`test_pipeline_integration_service.py`**: Tests for pipeline integration service
- **`test_documentation.py`**: Tests for documentation module (research logger, report generator, dataset catalog, tracking updater, progress visualization)
- **`test_cli.py`**: Tests for CLI interface and commands
- **`test_research_orchestrator.py`**: Tests for research orchestrator

### Integration Tests

- **`test_integration.py`**: Tests for component communication, workflow state transitions, error handling, and progress tracking

### End-to-End Tests

- **`test_e2e.py`**: Tests for complete research workflow, report generation, dataset acquisition, and pipeline integration

### Test Fixtures

- **`conftest.py`**: Shared fixtures, mocks, and test utilities for all test modules

## Running Tests

### Run All Tests

```bash
# From project root
cd ai
pytest journal_dataset_research/tests/ -v
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest journal_dataset_research/tests/test_*.py -v -m "not integration and not e2e"

# Integration tests only
pytest journal_dataset_research/tests/test_integration.py -v -m integration

# End-to-end tests only
pytest journal_dataset_research/tests/test_e2e.py -v -m e2e
```

### Run Specific Test Files

```bash
# Test data models
pytest journal_dataset_research/tests/test_dataset_models.py -v

# Test evaluation engine
pytest journal_dataset_research/tests/test_evaluation_engine.py -v

# Test compliance
pytest journal_dataset_research/tests/test_compliance.py -v
```

### Run with Coverage

```bash
pytest journal_dataset_research/tests/ -v --cov=journal_dataset_research --cov-report=html --cov-report=term-missing
```

## Test Coverage

The test suite covers:

1. **Data Models**: All validation methods and data integrity checks
2. **Discovery Engine**: Source discovery and metadata extraction
3. **Evaluation Engine**: Dataset evaluation across all dimensions
4. **Acquisition Manager**: Access requests and dataset downloads
5. **Integration Planning**: Schema mapping and transformation planning
6. **Pipeline Integration**: Format conversion, validation, merging, and quality checks
7. **Compliance**: License checking, privacy verification, HIPAA validation
8. **Documentation**: Report generation, logging, catalog export, progress visualization
9. **CLI**: Command-line interface and interactive mode
10. **Orchestrator**: Workflow coordination and state management

## Test Fixtures

The `conftest.py` file provides:

- **Dataset Sources**: Sample, high-quality, and low-quality dataset sources
- **Evaluations**: Sample, high-score, and low-score evaluations
- **Access Requests**: Sample and approved access requests
- **Acquired Datasets**: Sample acquired datasets with test files
- **Integration Plans**: Sample integration plans
- **Research Sessions**: Sample research sessions
- **Test Datasets**: CSV, JSONL, and JSON test datasets
- **Mock API Responses**: Mock responses for PubMed, Zenodo, Dryad, ClinicalTrials.gov
- **Mock Services**: Mock discovery, evaluation, acquisition, and integration services
- **Temporary Directories**: Temporary directories for logs, reports, and storage

## Mock Services

Tests use mock services to avoid external API calls:

- **Mock Discovery Service**: Returns predefined dataset sources
- **Mock Evaluation Engine**: Returns deterministic evaluation results
- **Mock Acquisition Manager**: Tracks requests and returns stub datasets
- **Mock Integration Engine**: Creates simple integration plans

## Test Data

Test datasets are created dynamically using fixtures:

- **CSV Datasets**: Sample CSV files with conversation data
- **JSONL Datasets**: Sample JSONL files with conversation records
- **JSON Datasets**: Sample JSON files with nested conversation structures

## Best Practices

1. **Isolation**: Each test should be independent and not rely on other tests
2. **Fixtures**: Use fixtures from `conftest.py` for common test data
3. **Mocks**: Use mocks for external services and API calls
4. **Temporary Files**: Use `tmp_path` fixture for temporary file operations
5. **Assertions**: Use descriptive assertions with clear error messages
6. **Coverage**: Aim for high test coverage (80%+)

## Troubleshooting

### Import Errors

If you encounter import errors, make sure you're running tests from the correct directory:

```bash
cd ai
pytest journal_dataset_research/tests/ -v
```

### Missing Dependencies

Install test dependencies:

```bash
pip install pytest pytest-cov pytest-mock
```

### Test Failures

Run tests with verbose output to see detailed error messages:

```bash
pytest journal_dataset_research/tests/ -v -s
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines. They:

- Use temporary directories for file operations
- Mock external API calls
- Clean up after themselves
- Run in parallel when possible

## Contributing

When adding new tests:

1. Follow the existing test structure
2. Use fixtures from `conftest.py`
3. Add appropriate test markers (unit, integration, e2e)
4. Update this README if adding new test categories
5. Ensure tests pass before committing

