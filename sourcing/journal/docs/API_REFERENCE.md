# Journal Dataset Research System - API Reference

## Table of Contents

1. [Data Models](#data-models)
2. [Research Orchestrator](#research-orchestrator)
3. [Evaluation Engine](#evaluation-engine)
4. [Acquisition Manager](#acquisition-manager)
5. [Integration Planning Engine](#integration-planning-engine)
6. [Compliance Module](#compliance-module)
7. [Documentation & Tracking](#documentation--tracking)
8. [CLI Interface](#cli-interface)
9. [Configuration Management](#configuration-management)

## Data Models

### DatasetSource

Represents a discovered dataset source from academic repositories.

**Location**: `models/dataset_models.py`

```python
from ai.sourcing.journal.models.dataset_models import DatasetSource
from datetime import datetime

source = DatasetSource(
    source_id="pubmed_12345",
    title="Therapeutic Conversation Dataset",
    authors=["Smith, J.", "Doe, A."],
    publication_date=datetime(2024, 1, 15),
    source_type="journal",
    url="https://example.com/dataset",
    doi="10.1234/example",
    abstract="A dataset of therapeutic conversations...",
    keywords=["therapy", "counseling", "mental health"],
    open_access=True,
    data_availability="available",
    discovery_method="pubmed_search"
)

# Validate the source
errors = source.validate()
if errors:
    print(f"Validation errors: {errors}")
```

**Fields**:
- `source_id` (str): Unique identifier for the source
- `title` (str): Title of the dataset
- `authors` (List[str]): List of authors
- `publication_date` (datetime): Publication date
- `source_type` (str): Type of source (journal, repository, clinical_trial, training_material)
- `url` (str): URL to the source
- `doi` (Optional[str]): Digital Object Identifier
- `abstract` (str): Abstract text
- `keywords` (List[str]): List of keywords
- `open_access` (bool): Open access status
- `data_availability` (str): Availability status (available, upon_request, restricted, unknown)
- `discovery_date` (datetime): Date of discovery
- `discovery_method` (str): Discovery method (pubmed_search, doaj_manual, repository_api, citation)

**Methods**:
- `validate() -> List[str]`: Validates the source and returns list of errors

### DatasetEvaluation

Represents the evaluation of a dataset across multiple dimensions.

**Location**: `models/dataset_models.py`

```python
from ai.sourcing.journal.models.dataset_models import DatasetEvaluation

evaluation = DatasetEvaluation(
    source_id="pubmed_12345",
    therapeutic_relevance=8,
    data_structure_quality=7,
    training_integration=9,
    ethical_accessibility=8,
    therapeutic_relevance_notes="High relevance to therapeutic applications",
    data_structure_notes="Well-structured conversation data",
    integration_notes="Easy to integrate into training pipeline",
    ethical_notes="Fully compliant with ethical requirements",
    overall_score=8.0,
    priority_tier="high",
    evaluator="system",
    compliance_checked=True,
    compliance_status="compliant",
    compliance_score=0.95,
    license_compatible=True,
    privacy_compliant=True,
    hipaa_compliant=True
)

# Validate the evaluation
errors = evaluation.validate()
```

**Fields**:
- `source_id` (str): Source identifier
- `therapeutic_relevance` (int): Relevance score (1-10)
- `data_structure_quality` (int): Structure quality score (1-10)
- `training_integration` (int): Integration potential score (1-10)
- `ethical_accessibility` (int): Ethical accessibility score (1-10)
- `therapeutic_relevance_notes` (str): Notes on therapeutic relevance
- `data_structure_notes` (str): Notes on data structure
- `integration_notes` (str): Notes on integration
- `ethical_notes` (str): Notes on ethical accessibility
- `overall_score` (float): Weighted overall score
- `priority_tier` (str): Priority tier (high, medium, low)
- `evaluation_date` (datetime): Evaluation date
- `evaluator` (str): Evaluator identifier
- `competitive_advantages` (List[str]): List of competitive advantages
- `compliance_checked` (bool): Compliance check status
- `compliance_status` (str): Compliance status (compliant, partially_compliant, non_compliant)
- `compliance_score` (float): Compliance score (0.0-1.0)
- `license_compatible` (bool): License compatibility status
- `privacy_compliant` (bool): Privacy compliance status
- `hipaa_compliant` (bool): HIPAA compliance status

**Methods**:
- `validate() -> List[str]`: Validates the evaluation and returns list of errors

### AccessRequest

Represents a request to access a dataset.

**Location**: `models/dataset_models.py`

```python
from ai.sourcing.journal.models.dataset_models import AccessRequest

access_request = AccessRequest(
    source_id="pubmed_12345",
    access_method="direct",
    status="pending",
    access_url="https://example.com/download",
    credentials_required=False,
    institutional_affiliation_required=False
)

# Validate the access request
errors = access_request.validate()
```

**Fields**:
- `source_id` (str): Source identifier
- `access_method` (str): Access method (direct, api, request_form, collaboration, registration)
- `request_date` (datetime): Request date
- `status` (str): Request status (pending, approved, denied, downloaded, error)
- `access_url` (str): Access URL
- `credentials_required` (bool): Credentials requirement flag
- `institutional_affiliation_required` (bool): Institutional affiliation requirement flag
- `estimated_access_date` (Optional[datetime]): Estimated access date
- `notes` (str): Additional notes

**Methods**:
- `validate() -> List[str]`: Validates the access request and returns list of errors

### AcquiredDataset

Represents an acquired dataset with storage information.

**Location**: `models/dataset_models.py`

```python
from ai.sourcing.journal.models.dataset_models import AcquiredDataset

acquired_dataset = AcquiredDataset(
    source_id="pubmed_12345",
    storage_path="data/acquired_datasets/pubmed_12345/dataset.json",
    file_format="json",
    file_size_mb=125.5,
    license="CC BY 4.0",
    usage_restrictions=[],
    attribution_required=True,
    checksum="abc123def456",
    encrypted=False,
    compliance_status="compliant",
    compliance_score=0.95,
    hipaa_compliant=True,
    privacy_assessed=True
)

# Validate the acquired dataset
errors = acquired_dataset.validate()
```

**Fields**:
- `source_id` (str): Source identifier
- `acquisition_date` (datetime): Acquisition date
- `storage_path` (str): Storage path
- `file_format` (str): File format
- `file_size_mb` (float): File size in MB
- `license` (str): License information
- `usage_restrictions` (List[str]): List of usage restrictions
- `attribution_required` (bool): Attribution requirement flag
- `checksum` (str): File checksum
- `encrypted` (bool): Encryption status
- `compliance_status` (str): Compliance status
- `compliance_score` (float): Compliance score
- `hipaa_compliant` (bool): HIPAA compliance status
- `privacy_assessed` (bool): Privacy assessment status

**Methods**:
- `validate() -> List[str]`: Validates the acquired dataset and returns list of errors

### IntegrationPlan

Represents a plan for integrating a dataset into the training pipeline.

**Location**: `models/dataset_models.py`

```python
from ai.sourcing.journal.models.dataset_models import IntegrationPlan

integration_plan = IntegrationPlan(
    source_id="pubmed_12345",
    dataset_format="json",
    schema_mapping={
        "conversation": "messages",
        "participant": "role",
        "text": "content"
    },
    required_transformations=["format_conversion", "field_mapping"],
    preprocessing_steps=["clean_text", "normalize_timestamps"],
    complexity="medium",
    estimated_effort_hours=8,
    dependencies=[],
    integration_priority=1
)

# Validate the integration plan
errors = integration_plan.validate()
```

**Fields**:
- `source_id` (str): Source identifier
- `dataset_format` (str): Dataset format (csv, json, xml, parquet, custom)
- `schema_mapping` (Dict[str, str]): Field mapping (dataset_field -> pipeline_field)
- `required_transformations` (List[str]): List of required transformations
- `preprocessing_steps` (List[str]): List of preprocessing steps
- `complexity` (str): Integration complexity (low, medium, high)
- `estimated_effort_hours` (int): Estimated effort in hours
- `dependencies` (List[str]): List of dependencies
- `integration_priority` (int): Integration priority
- `created_date` (datetime): Creation date

**Methods**:
- `validate() -> List[str]`: Validates the integration plan and returns list of errors

### ResearchSession

Represents a research session with targets and progress tracking.

**Location**: `models/dataset_models.py`

```python
from ai.sourcing.journal.models.dataset_models import ResearchSession

session = ResearchSession(
    session_id="session_abc123",
    target_sources=["pubmed", "doaj"],
    search_keywords={
        "therapeutic": ["therapy", "counseling", "psychotherapy"],
        "dataset": ["dataset", "conversation", "transcript"]
    },
    weekly_targets={
        "sources_identified": 10,
        "datasets_evaluated": 5,
        "datasets_acquired": 3
    },
    current_phase="discovery",
    progress_metrics={
        "sources_identified": 0,
        "datasets_evaluated": 0,
        "datasets_acquired": 0
    }
)

# Validate the session
errors = session.validate()
```

**Fields**:
- `session_id` (str): Unique session identifier
- `start_date` (datetime): Session start date
- `target_sources` (List[str]): List of target sources
- `search_keywords` (Dict[str, List[str]]): Search keywords dictionary
- `weekly_targets` (Dict[str, int]): Weekly targets dictionary
- `current_phase` (str): Current phase (discovery, evaluation, acquisition, integration)
- `progress_metrics` (Dict[str, int]): Progress metrics dictionary

**Methods**:
- `validate() -> List[str]`: Validates the session and returns list of errors

## Research Orchestrator

### ResearchOrchestrator

Coordinates the journal dataset research workflow.

**Location**: `orchestrator/research_orchestrator.py`

```python
from ai.sourcing.journal.orchestrator.research_orchestrator import ResearchOrchestrator
from ai.sourcing.journal.orchestrator.types import OrchestratorConfig

# Initialize orchestrator
config = OrchestratorConfig(
    max_retries=3,
    retry_delay_seconds=1.0,
    progress_history_limit=100,
    parallel_evaluation=False,
    parallel_integration_planning=False,
    max_workers=4,
    session_storage_path="checkpoints",
    fallback_on_failure=True
)

orchestrator = ResearchOrchestrator(
    discovery_service=None,  # Optional discovery service
    evaluation_engine=None,  # Optional evaluation engine
    acquisition_manager=None,  # Optional acquisition manager
    integration_engine=None,  # Optional integration engine
    config=config
)
```

#### Methods

##### start_research_session

Initialize a new research session with provided targets and keywords.

```python
session = orchestrator.start_research_session(
    target_sources=["pubmed", "doaj"],
    search_keywords={
        "therapeutic": ["therapy", "counseling"],
        "dataset": ["dataset", "conversation"]
    },
    weekly_targets={
        "sources_identified": 10,
        "datasets_evaluated": 5
    },
    session_id="session_abc123"  # Optional, auto-generated if not provided
)
```

**Parameters**:
- `target_sources` (List[str]): List of target sources
- `search_keywords` (Dict[str, List[str]]): Search keywords dictionary
- `weekly_targets` (Optional[Dict[str, int]]): Weekly targets dictionary
- `session_id` (Optional[str]): Session ID (auto-generated if not provided)

**Returns**: `ResearchSession` instance

##### advance_phase

Advance the research session to the next phase in the workflow sequence.

```python
next_phase = orchestrator.advance_phase("session_abc123")
```

**Parameters**:
- `session_id` (str): Session ID

**Returns**: `str` - Next phase name

##### update_progress

Update progress metrics for a session and record a snapshot.

```python
orchestrator.update_progress(
    "session_abc123",
    {"sources_identified": 5, "datasets_evaluated": 3}
)
```

**Parameters**:
- `session_id` (str): Session ID
- `metrics` (Dict[str, int]): Progress metrics dictionary

##### run_session

Execute the research workflow for the given session.

```python
state = orchestrator.run_session(
    "session_abc123",
    evaluator="system",
    auto_acquire=True,
    target_format="chatml"
)
```

**Parameters**:
- `session_id` (str): Session ID
- `evaluator` (str): Evaluator identifier (default: "system")
- `auto_acquire` (bool): Automatically acquire datasets (default: True)
- `target_format` (str): Target format for integration (default: "chatml")

**Returns**: `SessionState` instance

##### get_session_state

Get the session state for a given session.

```python
state = orchestrator.get_session_state("session_abc123")
```

**Parameters**:
- `session_id` (str): Session ID

**Returns**: `SessionState` instance

##### save_session_state

Save the session state to disk.

```python
checkpoint_path = orchestrator.save_session_state(
    "session_abc123",
    directory=Path("checkpoints")
)
```

**Parameters**:
- `session_id` (str): Session ID
- `directory` (Optional[Path]): Checkpoint directory (default: configured storage path)

**Returns**: `Path` - Checkpoint file path

##### load_session_state

Load the session state from disk.

```python
orchestrator.load_session_state("session_abc123", directory=Path("checkpoints"))
```

**Parameters**:
- `session_id` (str): Session ID
- `directory` (Optional[Path]): Checkpoint directory (default: configured storage path)

##### log_activity

Log an activity for a session.

```python
orchestrator.log_activity(
    "session_abc123",
    activity_type="search",
    description="Searching for sources",
    outcome="5 sources found",
    duration_minutes=2
)
```

**Parameters**:
- `session_id` (str): Session ID
- `activity_type` (str): Activity type
- `description` (str): Activity description
- `outcome` (str): Activity outcome
- `duration_minutes` (int): Activity duration in minutes

##### get_progress_report

Get a progress report for a session.

```python
report = orchestrator.get_progress_report("session_abc123")
```

**Parameters**:
- `session_id` (str): Session ID

**Returns**: `Dict[str, Any]` - Progress report dictionary

##### generate_weekly_report

Generate a weekly progress report for a session.

```python
weekly_report = orchestrator.generate_weekly_report("session_abc123", week_number=1)
```

**Parameters**:
- `session_id` (str): Session ID
- `week_number` (int): Week number

**Returns**: `WeeklyReport` instance

## Evaluation Engine

### EvaluationEngine

Performs systematic quality assessment of identified datasets.

**Location**: `evaluation/evaluation_engine.py`

```python
from ai.sourcing.journal.evaluation.evaluation_engine import EvaluationEngine, EvaluationConfig

# Initialize evaluation engine
config = EvaluationConfig(
    therapeutic_relevance_weight=0.35,
    data_structure_quality_weight=0.25,
    training_integration_weight=0.20,
    ethical_accessibility_weight=0.20,
    high_priority_threshold=7.5,
    medium_priority_threshold=5.0
)

evaluation_engine = EvaluationEngine(config=config)
```

#### Methods

##### evaluate_dataset

Evaluate a dataset across multiple dimensions.

```python
evaluation = evaluation_engine.evaluate_dataset(
    source=dataset_source,
    evaluator="system",
    include_compliance=True
)
```

**Parameters**:
- `source` (DatasetSource): Dataset source to evaluate
- `evaluator` (str): Evaluator identifier (default: "system")
- `include_compliance` (bool): Include compliance checks (default: True)

**Returns**: `DatasetEvaluation` instance

##### assess_therapeutic_relevance

Assess the therapeutic relevance of a dataset.

```python
relevance_score, notes = evaluation_engine.assess_therapeutic_relevance(source)
```

**Parameters**:
- `source` (DatasetSource): Dataset source to assess

**Returns**: `Tuple[int, str]` - Relevance score (1-10) and notes

##### assess_data_structure_quality

Assess the data structure quality of a dataset.

```python
quality_score, notes = evaluation_engine.assess_data_structure_quality(source)
```

**Parameters**:
- `source` (DatasetSource): Dataset source to assess

**Returns**: `Tuple[int, str]` - Quality score (1-10) and notes

##### assess_training_integration

Assess the training integration potential of a dataset.

```python
integration_score, notes = evaluation_engine.assess_training_integration(source)
```

**Parameters**:
- `source` (DatasetSource): Dataset source to assess

**Returns**: `Tuple[int, str]` - Integration score (1-10) and notes

##### assess_ethical_accessibility

Assess the ethical accessibility of a dataset.

```python
ethical_score, notes = evaluation_engine.assess_ethical_accessibility(source)
```

**Parameters**:
- `source` (DatasetSource): Dataset source to assess

**Returns**: `Tuple[int, str]` - Ethical score (1-10) and notes

##### calculate_overall_score

Calculate the overall score from individual dimension scores.

```python
overall_score = evaluation_engine.calculate_overall_score(
    therapeutic_relevance=8,
    data_structure_quality=7,
    training_integration=9,
    ethical_accessibility=8
)
```

**Parameters**:
- `therapeutic_relevance` (int): Therapeutic relevance score
- `data_structure_quality` (int): Data structure quality score
- `training_integration` (int): Training integration score
- `ethical_accessibility` (int): Ethical accessibility score

**Returns**: `float` - Overall score

##### determine_priority_tier

Determine the priority tier based on overall score.

```python
priority_tier = evaluation_engine.determine_priority_tier(overall_score=8.0)
```

**Parameters**:
- `overall_score` (float): Overall score

**Returns**: `str` - Priority tier (high, medium, low)

## Acquisition Manager

### AcquisitionManager

Handles dataset access requests, downloads, and secure storage.

**Location**: `acquisition/acquisition_manager.py`

```python
from ai.sourcing.journal.acquisition.acquisition_manager import AcquisitionManager, AcquisitionConfig

# Initialize acquisition manager
config = AcquisitionConfig(
    storage_base_path="data/acquired_datasets",
    encryption_enabled=False,
    download_timeout=3600,
    max_retries=3,
    chunk_size=8192,
    resume_downloads=True
)

acquisition_manager = AcquisitionManager(config=config)
```

#### Methods

##### determine_access_method

Determine the access method for a dataset source.

```python
access_method = acquisition_manager.determine_access_method(source)
```

**Parameters**:
- `source` (DatasetSource): Dataset source

**Returns**: `str` - Access method (direct, api, request_form, collaboration, registration)

##### submit_access_request

Submit an access request for a dataset.

```python
access_request = acquisition_manager.submit_access_request(source)
```

**Parameters**:
- `source` (DatasetSource): Dataset source

**Returns**: `AccessRequest` instance

##### download_dataset

Download a dataset from a source.

```python
acquired_dataset = acquisition_manager.download_dataset(
    source=source,
    access_request=access_request
)
```

**Parameters**:
- `source` (DatasetSource): Dataset source
- `access_request` (AccessRequest): Access request

**Returns**: `AcquiredDataset` instance

##### get_download_progress

Get the download progress for a dataset.

```python
progress = acquisition_manager.get_download_progress("source_id")
```

**Parameters**:
- `source_id` (str): Source ID

**Returns**: `DownloadProgress` instance

##### verify_download_integrity

Verify the integrity of a downloaded dataset.

```python
is_valid = acquisition_manager.verify_download_integrity(
    file_path="data/acquired_datasets/source_id/dataset.json",
    expected_checksum="abc123def456"
)
```

**Parameters**:
- `file_path` (str): File path
- `expected_checksum` (str): Expected checksum

**Returns**: `bool` - True if integrity is valid

## Integration Planning Engine

### IntegrationPlanningEngine

Assesses integration feasibility and creates preprocessing plans.

**Location**: `integration/integration_planning_engine.py`

```python
from ai.sourcing.journal.integration.integration_planning_engine import IntegrationPlanningEngine

# Initialize integration planning engine
integration_engine = IntegrationPlanningEngine()
```

#### Methods

##### analyze_dataset_structure

Analyze the structure of a dataset.

```python
structure = integration_engine.analyze_dataset_structure(
    dataset_path="data/acquired_datasets/source_id/dataset.json"
)
```

**Parameters**:
- `dataset_path` (str): Path to the dataset file

**Returns**: `DatasetStructure` instance

##### create_schema_mapping

Create a schema mapping from dataset fields to pipeline fields.

```python
schema_mapping = integration_engine.create_schema_mapping(
    dataset_structure=structure,
    target_format="chatml"
)
```

**Parameters**:
- `dataset_structure` (DatasetStructure): Dataset structure
- `target_format` (str): Target format (chatml, conversation_record)

**Returns**: `List[SchemaMapping]` - List of schema mappings

##### create_integration_plan

Create an integration plan for a dataset.

```python
integration_plan = integration_engine.create_integration_plan(
    acquired_dataset=acquired_dataset,
    target_format="chatml"
)
```

**Parameters**:
- `acquired_dataset` (AcquiredDataset): Acquired dataset
- `target_format` (str): Target format (chatml, conversation_record)

**Returns**: `IntegrationPlan` instance

##### estimate_complexity

Estimate the integration complexity for a dataset.

```python
complexity, effort_hours = integration_engine.estimate_complexity(
    integration_plan=integration_plan
)
```

**Parameters**:
- `integration_plan` (IntegrationPlan): Integration plan

**Returns**: `Tuple[str, int]` - Complexity (low, medium, high) and effort in hours

##### generate_preprocessing_script

Generate a preprocessing script for a dataset.

```python
script_path = integration_engine.generate_preprocessing_script(
    integration_plan=integration_plan,
    output_path="data/preprocessing_scripts/source_id/preprocess.py"
)
```

**Parameters**:
- `integration_plan` (IntegrationPlan): Integration plan
- `output_path` (str): Output path for the script

**Returns**: `str` - Path to the generated script

## Compliance Module

### ComplianceChecker

Orchestrates all compliance checks.

**Location**: `compliance/compliance_checker.py`

```python
from ai.sourcing.journal.compliance.compliance_checker import ComplianceChecker

# Initialize compliance checker
compliance_checker = ComplianceChecker()
```

#### Methods

##### check_compliance

Perform comprehensive compliance check for a dataset.

```python
compliance_result = compliance_checker.check_compliance(
    source=dataset_source,
    dataset_sample="Sample dataset content...",
    dataset_path="data/acquired_datasets/source_id/dataset.json",
    license_text="MIT License",
    metadata={"key": "value"}
)
```

**Parameters**:
- `source` (DatasetSource): Dataset source
- `dataset_sample` (Optional[str]): Sample dataset content
- `dataset_path` (Optional[str]): Path to the dataset file
- `license_text` (Optional[str]): License text
- `metadata` (Optional[Dict]): Metadata dictionary

**Returns**: `ComplianceResult` instance

### LicenseChecker

Checks license compatibility.

**Location**: `compliance/license_checker.py`

```python
from ai.sourcing.journal.compliance.license_checker import LicenseChecker

license_checker = LicenseChecker()

# Check license compatibility
license_result = license_checker.check_license(
    license_text="MIT License",
    source=dataset_source
)
```

### PrivacyVerifier

Verifies privacy and anonymization.

**Location**: `compliance/privacy_verifier.py`

```python
from ai.sourcing.journal.compliance.privacy_verifier import PrivacyVerifier

privacy_verifier = PrivacyVerifier()

# Verify privacy
privacy_assessment = privacy_verifier.verify_privacy(
    dataset_sample="Sample dataset content...",
    dataset_path="data/acquired_datasets/source_id/dataset.json"
)
```

### HIPAAValidator

Validates HIPAA compliance.

**Location**: `compliance/hipaa_validator.py`

```python
from ai.sourcing.journal.compliance.hipaa_validator import HIPAAValidator

hipaa_validator = HIPAAValidator()

# Validate HIPAA compliance
hipaa_result = hipaa_validator.validate_hipaa_compliance(
    source=dataset_source,
    dataset_path="data/acquired_datasets/source_id/dataset.json"
)
```

## Documentation & Tracking

### ResearchLogger

Logs research activities.

**Location**: `documentation/research_logger.py`

```python
from ai.sourcing.journal.documentation.research_logger import ResearchLogger

research_logger = ResearchLogger(log_file="logs/research.log")

# Log an activity
research_logger.log_activity(
    session_id="session_abc123",
    activity_type="search",
    description="Searching for sources",
    outcome="5 sources found",
    duration_minutes=2
)
```

### ReportGenerator

Generates reports.

**Location**: `documentation/report_generator.py`

```python
from ai.sourcing.journal.documentation.report_generator import ReportGenerator

report_generator = ReportGenerator()

# Generate evaluation report
report_path = report_generator.generate_evaluation_report(
    evaluation=dataset_evaluation,
    output_path="reports/evaluation_report.md"
)

# Generate weekly report
weekly_report_path = report_generator.generate_weekly_report(
    session_id="session_abc123",
    week_number=1,
    output_path="reports/weekly_report.md"
)
```

### DatasetCatalog

Maintains dataset catalog.

**Location**: `documentation/dataset_catalog.py`

```python
from ai.sourcing.journal.documentation.dataset_catalog import DatasetCatalog

dataset_catalog = DatasetCatalog()

# Add a source to the catalog
dataset_catalog.add_source(dataset_source)

# Export catalog to markdown
catalog_path = dataset_catalog.export_to_markdown(
    output_path="catalog/dataset_catalog.md"
)

# Export catalog to CSV
csv_path = dataset_catalog.export_to_csv(
    output_path="catalog/dataset_catalog.csv"
)
```

## CLI Interface

### CommandHandler

Handles CLI commands.

**Location**: `cli/commands.py`

```python
from ai.sourcing.journal.cli.commands import CommandHandler
from ai.sourcing.journal.cli.config import load_config

# Initialize command handler
config = load_config()
command_handler = CommandHandler(config=config, dry_run=False)

# Search for sources
result = command_handler.search(
    keywords=["therapy", "counseling"],
    sources=["pubmed", "doaj"],
    session_id="session_abc123",
    interactive=False
)

# Evaluate datasets
result = command_handler.evaluate(
    session_id="session_abc123",
    interactive=False
)

# Acquire datasets
result = command_handler.acquire(
    session_id="session_abc123",
    interactive=False
)

# Create integration plans
result = command_handler.integrate(
    session_id="session_abc123",
    target_format="chatml"
)

# Check status
status = command_handler.status(session_id="session_abc123")

# Generate report
report_path = command_handler.report(
    session_id="session_abc123",
    output_path="reports/report.json",
    format="json"
)
```

## Configuration Management

### ConfigManager

Manages configuration.

**Location**: `cli/config.py`

```python
from ai.sourcing.journal.cli.config import ConfigManager, load_config, save_config, get_config_value

# Load configuration
config = load_config()

# Get configuration value
max_retries = get_config_value("orchestrator.max_retries", default=3)

# Save configuration
save_config(config)

# Initialize config manager
config_manager = ConfigManager(config_path=Path("config.yaml"))

# Get configuration value
value = config_manager.get("orchestrator.max_retries", default=3)

# Set configuration value
config_manager.set("orchestrator.max_retries", 5)

# Load configuration with environment overrides
config = config_manager.load()
config = config_manager.apply_env_overrides(config)
```

## Usage Examples

### Complete Workflow

```python
from ai.sourcing.journal.orchestrator.research_orchestrator import ResearchOrchestrator
from ai.sourcing.journal.orchestrator.types import OrchestratorConfig

# Initialize orchestrator
config = OrchestratorConfig()
orchestrator = ResearchOrchestrator(config=config)

# Start research session
session = orchestrator.start_research_session(
    target_sources=["pubmed", "doaj"],
    search_keywords={
        "therapeutic": ["therapy", "counseling"],
        "dataset": ["dataset", "conversation"]
    },
    weekly_targets={
        "sources_identified": 10,
        "datasets_evaluated": 5,
        "datasets_acquired": 3
    }
)

# Run session
state = orchestrator.run_session(
    session.session_id,
    evaluator="system",
    auto_acquire=True,
    target_format="chatml"
)

# Get progress report
report = orchestrator.get_progress_report(session.session_id)
print(f"Progress: {report}")

# Generate weekly report
weekly_report = orchestrator.generate_weekly_report(session.session_id, week_number=1)
print(f"Weekly Report: {weekly_report}")
```

### Evaluation Only

```python
from ai.sourcing.journal.evaluation.evaluation_engine import EvaluationEngine, EvaluationConfig
from ai.sourcing.journal.models.dataset_models import DatasetSource

# Initialize evaluation engine
config = EvaluationConfig()
evaluation_engine = EvaluationEngine(config=config)

# Evaluate dataset
evaluation = evaluation_engine.evaluate_dataset(
    source=dataset_source,
    evaluator="system",
    include_compliance=True
)

print(f"Overall Score: {evaluation.overall_score}")
print(f"Priority Tier: {evaluation.priority_tier}")
print(f"Compliance Status: {evaluation.compliance_status}")
```

### Acquisition Only

```python
from ai.sourcing.journal.acquisition.acquisition_manager import AcquisitionManager, AcquisitionConfig

# Initialize acquisition manager
config = AcquisitionConfig()
acquisition_manager = AcquisitionManager(config=config)

# Determine access method
access_method = acquisition_manager.determine_access_method(dataset_source)
print(f"Access Method: {access_method}")

# Submit access request
access_request = acquisition_manager.submit_access_request(dataset_source)

# Download dataset
acquired_dataset = acquisition_manager.download_dataset(
    source=dataset_source,
    access_request=access_request
)

print(f"Acquired Dataset: {acquired_dataset.storage_path}")
```

### Integration Planning Only

```python
from ai.sourcing.journal.integration.integration_planning_engine import IntegrationPlanningEngine

# Initialize integration planning engine
integration_engine = IntegrationPlanningEngine()

# Create integration plan
integration_plan = integration_engine.create_integration_plan(
    acquired_dataset=acquired_dataset,
    target_format="chatml"
)

print(f"Complexity: {integration_plan.complexity}")
print(f"Estimated Effort: {integration_plan.estimated_effort_hours} hours")

# Generate preprocessing script
script_path = integration_engine.generate_preprocessing_script(
    integration_plan=integration_plan,
    output_path="data/preprocessing_scripts/preprocess.py"
)

print(f"Preprocessing Script: {script_path}")
```

## Error Handling

All methods may raise exceptions. Common exceptions include:

- `ValueError`: Invalid input parameters
- `FileNotFoundError`: File or session not found
- `PermissionError`: Permission denied
- `ConnectionError`: Network connection error
- `TimeoutError`: Operation timeout

Example error handling:

```python
try:
    evaluation = evaluation_engine.evaluate_dataset(source=dataset_source)
except ValueError as e:
    print(f"Validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Thread Safety

The Research Orchestrator is thread-safe and can be used in multi-threaded environments. Session state management uses locks to ensure thread safety.

## Performance Considerations

- Use parallel processing for evaluation and integration planning when possible
- Configure appropriate timeouts for network operations
- Use checkpointing for long-running sessions
- Monitor progress and adjust configuration as needed

## Best Practices

1. **Session Management**: Always use unique session IDs and save session state regularly
2. **Error Handling**: Always handle exceptions and implement retry logic
3. **Configuration**: Use environment variables for sensitive configuration
4. **Logging**: Enable logging for debugging and monitoring
5. **Compliance**: Always perform compliance checks before acquiring datasets
6. **Validation**: Validate all data models before use
7. **Progress Tracking**: Track progress regularly and generate reports
8. **Security**: Use encryption for sensitive data and secure credential management

