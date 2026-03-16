# Journal Dataset Research System - Architecture Documentation

## Overview

The Journal Dataset Research System is an automated research automation platform designed to discover, evaluate, acquire, and integrate therapeutic datasets from academic sources. The system operates through a coordinated workflow that spans four main phases: discovery, evaluation, acquisition, and integration planning.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLI Interface & Main Script                  │
│  (cli/cli.py, main.py)                                          │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Research Orchestrator                          │
│  (orchestrator/research_orchestrator.py)                        │
│  - Session Management                                           │
│  - Workflow Coordination                                        │
│  - Progress Tracking                                            │
│  - Error Recovery                                               │
└───────┬───────────────┬───────────────┬───────────────┬─────────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Discovery   │ │  Evaluation  │ │  Acquisition │ │ Integration  │
│   Service    │ │    Engine    │ │   Manager    │ │    Engine    │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Compliance Module                            │
│  - License Checker                                              │
│  - Privacy Verifier                                             │
│  - HIPAA Validator                                              │
│  - Audit Logger                                                 │
│  - Encryption Manager                                           │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│              Documentation & Tracking System                    │
│  - Research Logger                                              │
│  - Report Generator                                             │
│  - Progress Visualization                                       │
│  - Dataset Catalog                                              │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Research Orchestrator

**Location**: `orchestrator/research_orchestrator.py`

The Research Orchestrator is the central coordination component that manages the entire research workflow. It provides:

- **Session Management**: Creates and manages research sessions with unique session IDs
- **Workflow Coordination**: Orchestrates the four-phase workflow (discovery → evaluation → acquisition → integration)
- **Progress Tracking**: Tracks progress metrics and generates progress snapshots
- **Error Recovery**: Implements retry logic with exponential backoff and fallback strategies
- **State Persistence**: Saves and loads session state for checkpointing and resume capability

**Key Classes**:
- `ResearchOrchestrator`: Main orchestrator class
- `OrchestratorConfig`: Configuration dataclass
- `SessionState`: Session state container

**Mixins**:
- `WorkflowMixin`: Workflow coordination logic
- `ProgressReportingMixin`: Progress tracking and reporting
- `RetryMixin`: Retry logic with exponential backoff

### 2. Source Discovery Engine

**Location**: `discovery/` (structure for future implementation)

The Discovery Engine is responsible for discovering dataset sources from various academic repositories. While the current implementation provides the structure, the discovery service integrates with:

- **PubMed Central**: NCBI E-utilities API for medical literature search
- **DOAJ**: Directory of Open Access Journals API
- **Repository APIs**: Dryad, Zenodo, ClinicalTrials.gov
- **Metadata Extraction**: Parsing and extracting dataset metadata from various formats

**Key Responsibilities**:
- Search across multiple academic repositories
- Extract and parse metadata
- Deduplicate sources based on DOI, title, and author similarity
- Filter by open access status and data availability

### 3. Dataset Evaluation Engine

**Location**: `evaluation/evaluation_engine.py`

The Evaluation Engine performs systematic quality assessment of identified datasets across four dimensions:

1. **Therapeutic Relevance** (35% weight): Assesses how relevant the dataset is to therapeutic applications
2. **Data Structure Quality** (25% weight): Evaluates the organization and completeness of data
3. **Training Integration Potential** (20% weight): Assesses how easily the dataset can be integrated into the training pipeline
4. **Ethical Accessibility** (20% weight): Evaluates license compatibility, privacy compliance, and ethical considerations

**Key Classes**:
- `EvaluationEngine`: Main evaluation engine
- `EvaluationConfig`: Configuration for evaluation weights and thresholds
- `DatasetEvaluation`: Evaluation result dataclass

**Scoring Algorithm**:
- Each dimension is scored 1-10
- Weighted average calculates overall score
- Priority tiers: High (≥7.5), Medium (≥5.0), Low (<5.0)

### 4. Access & Acquisition Manager

**Location**: `acquisition/acquisition_manager.py`

The Acquisition Manager handles dataset access requests, downloads, and secure storage:

- **Access Method Determination**: Analyzes source to determine access method (direct, API, request form, collaboration, registration)
- **Direct Downloads**: HTTP download with resume capability and progress tracking
- **API-Based Retrieval**: Repository API integration with authentication and rate limiting
- **Access Request Tracking**: Tracks access requests with status updates and follow-up reminders
- **Secure Storage**: Organized directory structure with encryption support

**Key Classes**:
- `AcquisitionManager`: Main acquisition manager
- `AcquisitionConfig`: Configuration for storage and download settings
- `DownloadProgress`: Download progress tracker
- `AccessRequest`: Access request dataclass
- `AcquiredDataset`: Acquired dataset dataclass

### 5. Integration Planning Engine

**Location**: `integration/integration_planning_engine.py`

The Integration Planning Engine assesses integration feasibility and creates preprocessing plans:

- **Dataset Structure Analysis**: Parses various formats (CSV, JSON, XML, Parquet) and extracts schema
- **Schema Mapping**: Maps dataset fields to training pipeline schema (ChatML or ConversationRecord formats)
- **Transformation Specification**: Generates transformation specs for format conversions and field mappings
- **Complexity Estimation**: Estimates integration complexity (low, medium, high) with effort estimates
- **Preprocessing Script Generation**: Generates Python scripts for data transformation

**Key Classes**:
- `IntegrationPlanningEngine`: Main integration planning engine
- `DatasetStructure`: Analyzed dataset structure
- `SchemaMapping`: Field mapping specification
- `IntegrationPlan`: Integration plan dataclass
- `TransformationSpec`: Transformation specification dataclass

### 6. Compliance Module

**Location**: `compliance/`

The Compliance Module ensures datasets meet legal and ethical requirements:

#### License Checker (`license_checker.py`)
- Parses and classifies licenses (MIT, Apache, BSD, CC, GPL)
- Checks AI training permissions
- Verifies commercial use compatibility
- Flags incompatible licenses for review

#### Privacy Verifier (`privacy_verifier.py`)
- Detects PII (email, phone, SSN, medical IDs)
- Verifies anonymization quality
- Assesses re-identification risks
- Generates privacy assessment reports

#### HIPAA Validator (`hipaa_validator.py`)
- Validates HIPAA compliance requirements
- Checks encryption requirements
- Verifies access control implementation
- Validates audit logging completeness

#### Audit Logger (`audit_logger.py`)
- Comprehensive audit log system
- Tamper-proof log storage with hash chaining
- Tracks all dataset access and modifications
- Logs user actions with timestamps

#### Encryption Manager (`encryption_manager.py`)
- Symmetric encryption (Fernet) for datasets at rest
- Asymmetric encryption (RSA) for key management
- Encrypts sensitive configuration data
- Supports encryption for data in transit

**Key Classes**:
- `ComplianceChecker`: Main compliance orchestrator
- `LicenseChecker`: License compatibility checker
- `PrivacyVerifier`: Privacy and anonymization verifier
- `HIPAAValidator`: HIPAA compliance validator
- `AuditLogger`: Audit logging system
- `EncryptionManager`: Encryption management

### 7. Documentation & Tracking System

**Location**: `documentation/`

The Documentation & Tracking System provides comprehensive logging and reporting:

#### Research Logger (`research_logger.py`)
- Logs all component activities with timestamps
- Tracks activity outcomes and durations
- Implements log rotation and archival
- Creates `ResearchLog` entries

#### Report Generator (`report_generator.py`)
- Generates evaluation reports for each dataset
- Creates weekly progress reports
- Generates final research summary reports
- Uses markdown templates

#### Progress Visualization (`progress_visualization.py`)
- Creates progress metrics charts
- Generates timeline visualizations for research phases
- Creates quality score distributions
- Exports visualizations as images or HTML

#### Dataset Catalog (`dataset_catalog.py`)
- Maintains catalog of all sources
- Generates markdown catalog
- Exports to CSV/JSON for analysis
- Creates catalog statistics and summaries

#### Tracking Updater (`tracking_updater.py`)
- Automatically updates tracking documents
- Updates progress sections with current metrics
- Marks completed tasks with timestamps
- Generates status summaries

### 8. CLI Interface

**Location**: `cli/`

The CLI provides a command-line interface for research operations:

#### Main CLI (`cli.py`)
- Search for dataset sources
- Evaluate datasets
- Acquire datasets
- Create integration plans
- Check status
- Generate reports
- Configuration management

#### Commands (`commands.py`)
- `CommandHandler`: Handles CLI commands
- Integrates with orchestrator and services
- Provides interactive mode for manual oversight

#### Interactive Mode (`interactive.py`)
- Interactive prompts for manual decisions
- Dataset review and approval workflow
- Manual evaluation override capability
- Interactive progress monitoring

#### Configuration (`config.py`)
- Configuration management with YAML/JSON support
- Environment variable overrides
- Configuration validation
- Default configuration values

#### Main Execution Script (`main.py`)
- Automated workflow execution
- Phase-by-phase execution with checkpoints
- Resume capability for interrupted workflows
- Dry-run mode for testing
- Interactive mode for manual approvals

## Data Models

### Core Data Models

**Location**: `models/dataset_models.py`

#### DatasetSource
Represents a discovered dataset source from academic repositories.

**Fields**:
- `source_id`: Unique identifier
- `title`: Dataset title
- `authors`: List of authors
- `publication_date`: Publication date
- `source_type`: Type (journal, repository, clinical_trial, training_material)
- `url`: Source URL
- `doi`: Digital Object Identifier (optional)
- `abstract`: Abstract text
- `keywords`: List of keywords
- `open_access`: Open access status
- `data_availability`: Availability status (available, upon_request, restricted, unknown)
- `discovery_date`: Discovery date
- `discovery_method`: Discovery method (pubmed_search, doaj_manual, repository_api, citation)

#### DatasetEvaluation
Represents the evaluation of a dataset across multiple dimensions.

**Fields**:
- `source_id`: Source identifier
- `therapeutic_relevance`: Relevance score (1-10)
- `data_structure_quality`: Structure quality score (1-10)
- `training_integration`: Integration potential score (1-10)
- `ethical_accessibility`: Ethical accessibility score (1-10)
- `overall_score`: Weighted overall score
- `priority_tier`: Priority tier (high, medium, low)
- `evaluation_date`: Evaluation date
- `evaluator`: Evaluator identifier
- `competitive_advantages`: List of competitive advantages
- `compliance_checked`: Compliance check status
- `compliance_status`: Compliance status (compliant, partially_compliant, non_compliant)
- `compliance_score`: Compliance score (0.0-1.0)
- `license_compatible`: License compatibility status
- `privacy_compliant`: Privacy compliance status
- `hipaa_compliant`: HIPAA compliance status

#### AccessRequest
Represents a request to access a dataset.

**Fields**:
- `source_id`: Source identifier
- `access_method`: Access method (direct, api, request_form, collaboration, registration)
- `request_date`: Request date
- `status`: Request status (pending, approved, denied, downloaded, error)
- `access_url`: Access URL
- `credentials_required`: Credentials requirement flag
- `institutional_affiliation_required`: Institutional affiliation requirement flag
- `estimated_access_date`: Estimated access date (optional)
- `notes`: Additional notes

#### AcquiredDataset
Represents an acquired dataset with storage information.

**Fields**:
- `source_id`: Source identifier
- `acquisition_date`: Acquisition date
- `storage_path`: Storage path
- `file_format`: File format
- `file_size_mb`: File size in MB
- `license`: License information
- `usage_restrictions`: List of usage restrictions
- `attribution_required`: Attribution requirement flag
- `checksum`: File checksum
- `encrypted`: Encryption status
- `compliance_status`: Compliance status
- `compliance_score`: Compliance score
- `hipaa_compliant`: HIPAA compliance status
- `privacy_assessed`: Privacy assessment status

#### IntegrationPlan
Represents a plan for integrating a dataset into the training pipeline.

**Fields**:
- `source_id`: Source identifier
- `dataset_format`: Dataset format (csv, json, xml, parquet, custom)
- `schema_mapping`: Field mapping (dataset_field -> pipeline_field)
- `required_transformations`: List of required transformations
- `preprocessing_steps`: List of preprocessing steps
- `complexity`: Integration complexity (low, medium, high)
- `estimated_effort_hours`: Estimated effort in hours
- `dependencies`: List of dependencies
- `integration_priority`: Integration priority
- `created_date`: Creation date

#### ResearchSession
Represents a research session with targets and progress tracking.

**Fields**:
- `session_id`: Unique session identifier
- `start_date`: Session start date
- `target_sources`: List of target sources
- `search_keywords`: Search keywords dictionary
- `weekly_targets`: Weekly targets dictionary
- `current_phase`: Current phase (discovery, evaluation, acquisition, integration)
- `progress_metrics`: Progress metrics dictionary

#### ResearchProgress
Tracks overall research progress metrics.

**Fields**:
- `sources_identified`: Number of sources identified
- `datasets_evaluated`: Number of datasets evaluated
- `access_established`: Number of access requests established
- `datasets_acquired`: Number of datasets acquired
- `integration_plans_created`: Number of integration plans created
- `last_updated`: Last update timestamp

#### ResearchLog
Represents a log entry for research activities.

**Fields**:
- `timestamp`: Log timestamp
- `activity_type`: Activity type (session_start, phase_transition, search, evaluation, acquisition, integration, access_request, download, manual_intervention)
- `source_id`: Source identifier (optional)
- `description`: Activity description
- `outcome`: Activity outcome
- `duration_minutes`: Activity duration in minutes

#### WeeklyReport
Represents a weekly progress report.

**Fields**:
- `week_number`: Week number
- `start_date`: Week start date
- `end_date`: Week end date
- `sources_identified`: Number of sources identified
- `datasets_evaluated`: Number of datasets evaluated
- `access_established`: Number of access requests established
- `datasets_acquired`: Number of datasets acquired
- `integration_plans_created`: Number of integration plans created
- `key_findings`: List of key findings
- `challenges`: List of challenges
- `next_week_priorities`: List of next week priorities

## Workflow Phases

### 1. Discovery Phase

**Objective**: Discover dataset sources from academic repositories

**Process**:
1. Search across configured sources (PubMed, DOAJ, repositories)
2. Extract and parse metadata
3. Deduplicate sources
4. Filter by open access and data availability
5. Store discovered sources in session state

**Output**: List of `DatasetSource` objects

### 2. Evaluation Phase

**Objective**: Evaluate discovered datasets across multiple dimensions

**Process**:
1. For each source, perform evaluation across four dimensions:
   - Therapeutic relevance
   - Data structure quality
   - Training integration potential
   - Ethical accessibility
2. Calculate overall score and priority tier
3. Perform compliance checks (license, privacy, HIPAA)
4. Generate evaluation notes and competitive advantages
5. Store evaluations in session state

**Output**: List of `DatasetEvaluation` objects

### 3. Acquisition Phase

**Objective**: Acquire datasets from identified sources

**Process**:
1. Determine access method for each source
2. Submit access requests if required
3. Download datasets (direct or API-based)
4. Verify download integrity (checksum)
5. Store datasets securely with encryption if enabled
6. Track access requests and acquisition status
7. Store acquired datasets in session state

**Output**: List of `AcquiredDataset` objects

### 4. Integration Phase

**Objective**: Create integration plans for acquired datasets

**Process**:
1. Analyze dataset structure and schema
2. Map dataset fields to training pipeline schema
3. Generate transformation specifications
4. Estimate integration complexity and effort
5. Generate preprocessing scripts
6. Store integration plans in session state

**Output**: List of `IntegrationPlan` objects

## Design Decisions

### 1. Session-Based Architecture

**Rationale**: Sessions provide isolation between different research workflows, allowing multiple concurrent research efforts and easy state management.

**Benefits**:
- State persistence and checkpointing
- Resume capability for interrupted workflows
- Progress tracking per session
- Isolation between research efforts

### 2. Phase-Based Workflow

**Rationale**: The four-phase workflow (discovery → evaluation → acquisition → integration) provides clear separation of concerns and allows for checkpointing between phases.

**Benefits**:
- Clear workflow progression
- Checkpointing between phases
- Easy resume from any phase
- Parallel processing within phases

### 3. Protocol-Based Service Architecture

**Rationale**: Services are defined by protocols (interfaces) rather than concrete implementations, allowing for easy testing and extensibility.

**Benefits**:
- Easy mocking for testing
- Flexible service implementations
- Clear service contracts
- Dependency injection support

### 4. Compliance-First Approach

**Rationale**: Compliance checks are integrated into the evaluation phase to ensure only compliant datasets proceed to acquisition.

**Benefits**:
- Early compliance detection
- Reduced risk of non-compliant data
- Comprehensive compliance reporting
- Audit trail for all compliance checks

### 5. Configuration Management

**Rationale**: Centralized configuration management with environment variable overrides provides flexibility and security.

**Benefits**:
- Environment-specific configurations
- Secure credential management
- Easy configuration updates
- Default values for all settings

### 6. Error Recovery and Retry Logic

**Rationale**: Robust error recovery with exponential backoff and fallback strategies ensures system resilience.

**Benefits**:
- Resilience to transient failures
- Automatic retry for failed operations
- Fallback strategies for component failures
- Comprehensive error logging

## Data Flow

### Discovery → Evaluation Flow

```
Discovery Service
    │
    ▼
DatasetSource objects
    │
    ▼
Evaluation Engine
    │
    ▼
DatasetEvaluation objects (with compliance checks)
    │
    ▼
Session State (evaluations list)
```

### Evaluation → Acquisition Flow

```
DatasetEvaluation objects (priority tier: high/medium/low)
    │
    ▼
Acquisition Manager
    │
    ▼
AccessRequest objects
    │
    ▼
Download/API Retrieval
    │
    ▼
AcquiredDataset objects
    │
    ▼
Session State (acquired_datasets list)
```

### Acquisition → Integration Flow

```
AcquiredDataset objects
    │
    ▼
Integration Planning Engine
    │
    ▼
Dataset Structure Analysis
    │
    ▼
Schema Mapping
    │
    ▼
Transformation Specifications
    │
    ▼
IntegrationPlan objects
    │
    ▼
Session State (integration_plans list)
```

## Storage and Persistence

### Session State Storage

Sessions are stored in the configured session storage path (default: `checkpoints/`). Each session includes:

- Session metadata (JSON)
- Discovered sources (JSON)
- Evaluations (JSON)
- Access requests (JSON)
- Acquired datasets (JSON)
- Integration plans (JSON)
- Progress metrics (JSON)
- Activity logs (JSON)

### Dataset Storage

Acquired datasets are stored in the configured storage base path (default: `data/acquired_datasets/`). Each dataset is stored in a directory structure:

```
data/acquired_datasets/
    └── {source_id}/
        ├── dataset.{format}
        ├── metadata.json
        ├── checksum.txt
        └── access_log.json
```

### Encryption

If encryption is enabled, datasets are encrypted at rest using Fernet symmetric encryption. Encryption keys are managed by the `EncryptionManager`.

## Security Considerations

### 1. Credential Management

- API keys and credentials are stored in configuration files or environment variables
- Never hardcoded in source code
- Environment variable overrides for secure deployment

### 2. Data Encryption

- Optional encryption for datasets at rest
- Symmetric encryption (Fernet) for datasets
- Asymmetric encryption (RSA) for key management
- Encryption for sensitive configuration data

### 3. Audit Logging

- Comprehensive audit logging for all data access
- Tamper-proof log storage with hash chaining
- Tracks all dataset access and modifications
- Logs user actions with timestamps

### 4. Privacy Compliance

- PII detection and verification
- Anonymization quality assessment
- Re-identification risk assessment
- HIPAA compliance validation

### 5. License Compliance

- License compatibility checking
- AI training permission verification
- Commercial use compatibility verification
- Incompatible license flagging

## Performance Considerations

### 1. Parallel Processing

- Optional parallel processing for evaluation and integration planning
- Configurable maximum workers
- Thread-safe session state management

### 2. Caching

- Session state caching in memory
- Progress snapshot caching
- Evaluation result caching

### 3. Rate Limiting

- API rate limiting for repository APIs
- Configurable rate limit delays
- Exponential backoff for retries

### 4. Progress Tracking

- Efficient progress tracking with snapshots
- Configurable progress history limit
- Real-time progress updates

## Extension Points

### 1. Custom Discovery Services

Implement `DiscoveryServiceProtocol` to add custom discovery sources.

### 2. Custom Evaluation Criteria

Extend `EvaluationEngine` to add custom evaluation criteria.

### 3. Custom Acquisition Methods

Extend `AcquisitionManager` to add custom acquisition methods.

### 4. Custom Integration Transformations

Extend `IntegrationPlanningEngine` to add custom transformation logic.

### 5. Custom Compliance Checks

Extend `ComplianceChecker` to add custom compliance checks.

## Testing Strategy

### 1. Unit Tests

- Test individual components in isolation
- Mock external dependencies
- Test error handling and edge cases

### 2. Integration Tests

- Test component interactions
- Test workflow state transitions
- Test error recovery and retry logic

### 3. End-to-End Tests

- Test complete workflow with sample data
- Test report generation and documentation
- Test dataset acquisition and storage
- Test integration with training pipeline

### 4. Test Fixtures

- Mock API responses for external services
- Sample datasets for testing
- Test database and storage
- Test configuration files

## Future Enhancements

### 1. Discovery Service Implementation

- Implement PubMed Central search integration
- Implement DOAJ journal search
- Implement repository search modules
- Implement metadata extraction and deduplication

### 2. Enhanced Evaluation

- Machine learning-based relevance scoring
- Automated quality assessment
- Competitive advantage detection
- Advanced compliance checking

### 3. Enhanced Acquisition

- Automated access request submission
- Institutional collaboration workflow
- Batch download optimization
- Advanced storage organization

### 4. Enhanced Integration

- Automated preprocessing script execution
- Integration testing automation
- Pipeline integration automation
- Quality assurance automation

### 5. Enhanced Reporting

- Real-time dashboards
- Advanced visualizations
- Automated report generation
- Export to various formats

## Conclusion

The Journal Dataset Research System provides a comprehensive, automated platform for discovering, evaluating, acquiring, and integrating therapeutic datasets from academic sources. The architecture is designed for extensibility, reliability, and compliance, with robust error recovery and comprehensive logging and reporting capabilities.

