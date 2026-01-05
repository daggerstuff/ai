# Mental Health Datasets Expansion - Release 0 Implementation

This directory contains the complete implementation of the Mental Health Datasets Expansion project as defined in the GitHub issues tracking document.

## Overview

The implementation provides an end-to-end S3-first dataset pipeline with strict privacy, compliance, and quality gates for mental health training datasets.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    S3 Canonical Storage                 │
│  s3://pixel-data/exports/releases/vYYYY-MM-DD/         │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                Release Pipeline                         │
│  Coverage → Manifest → Gates → QA → Training Test      │
└─────────────────────────────────────────────────────────┘
```

## Scripts

### Core Implementation Scripts

| Script | Issue | Purpose |
|--------|-------|---------|
| `build_coverage_matrix.py` | Issue 2 | Build coverage matrix from S3 inventory |
| `build_release_manifest.py` | Issue 3 | Generate versioned manifest + compiled ChatML export |
| `privacy_provenance_gates.py` | Issue 4 | Enforce privacy and provenance gates (fail closed) |
| `dedup_leakage_gates.py` | Issue 5 | Run dedup and cross-split leakage gates |
| `distribution_gate.py` | Issue 6 | Record distribution stats by family and split |
| `human_qa_signoff.py` | Issue 7 | Clinician QA + bias/cultural review signoff |
| `training_consumption_test.py` | Issue 8 | Smoke test training consumes S3 release artifacts |

### Orchestration

| Script | Purpose |
|--------|---------|
| `release_orchestrator.py` | Master orchestrator that runs all issues in sequence |

## Quick Start

### Prerequisites

1. **S3 Configuration**: Set up OVH S3 credentials
   ```bash
   export DATASET_STORAGE_BACKEND=s3
   export OVH_S3_BUCKET=pixel-data
   export OVH_S3_ACCESS_KEY=<your-access-key>
   export OVH_S3_SECRET_KEY=<your-secret-key>
   export OVH_S3_ENDPOINT=<your-s3-endpoint>
   ```

2. **Python Dependencies**: Install required packages
   ```bash
   cd ai/dataset_pipeline
   uv pip install boto3 pathlib
   ```

### Run Complete Release Process

```bash
# Run complete Release 0 process
python scripts/release_orchestrator.py

# Run with custom release version
python scripts/release_orchestrator.py --release-version v2025-01-02

# Dry run to see execution plan
python scripts/release_orchestrator.py --dry-run

# Continue on non-blocking failures
python scripts/release_orchestrator.py --no-fail-fast
```

### Run Individual Issues

```bash
# Issue 2: Coverage Matrix
python scripts/build_coverage_matrix.py

# Issue 3: Manifest + Export
python scripts/build_release_manifest.py v2025-01-02

# Issue 4: Privacy + Provenance Gates
python scripts/privacy_provenance_gates.py v2025-01-02

# Issue 5: Dedup + Leakage Gates
python scripts/dedup_leakage_gates.py v2025-01-02

# Issue 6: Distribution Gate
python scripts/distribution_gate.py v2025-01-02

# Issue 7: Human QA Signoff
python scripts/human_qa_signoff.py v2025-01-02

# Issue 8: Training Consumption Test
python scripts/training_consumption_test.py v2025-01-02
```

## Release Artifacts

A successful release creates the following S3 structure:

```
s3://pixel-data/exports/releases/vYYYY-MM-DD/
├── manifest.json                    # Dataset manifest with provenance
├── compiled_export.jsonl           # ChatML export (placeholder)
├── routing_config.json             # Training curriculum configuration
├── gates/                          # Gate reports
│   ├── pii_detection_report.json
│   ├── provenance_validation_report.json
│   ├── deduplication_report.json
│   ├── cross_split_leakage_report.json
│   ├── distribution_report.json
│   └── combined_gate_report.json
├── qa/                             # QA artifacts
│   ├── qa_samples.json
│   ├── review_template.json
│   └── signoff_record.json
├── tests/                          # Test reports
│   └── training_consumption_test.json
├── release_notes.json              # Human-readable release notes
└── release_summary.json            # Complete orchestration summary
```

## Gate System

The implementation uses a **fail-closed** gate system:

### Blocking Gates (Release Blockers)
- **Privacy Gate**: No PII detected in final exports
- **Provenance Gate**: All files have complete provenance metadata
- **Deduplication Gate**: No exact duplicates in priority families
- **Leakage Gate**: No cross-split leakage in holdout families
- **Human QA Gate**: Clinical review approval required

### Non-Blocking Gates (Warnings)
- **Distribution Gate**: Statistics within expected ranges
- **Training Consumption Gate**: End-to-end training compatibility

## Dataset Families

Release 0 supports these dataset families:

### Stage 1: Foundation
- `professional_therapeutic`: High-quality therapeutic conversations
- `priority_datasets`: Curated priority therapeutic content

### Stage 2: Therapeutic Expertise  
- `cot_reasoning`: Clinical reasoning and chain-of-thought patterns

### Stage 3: Edge/Crisis
- `edge_cases`: Crisis scenarios and edge case handling

### Stage 4: Voice/Persona
- `voice_persona`: Voice and persona training data

## Configuration

### Storage Configuration

The system uses `ai/dataset_pipeline/storage_config.py` for S3 configuration:

```python
from storage_config import get_storage_config

config = get_storage_config()
# Automatically loads from environment variables
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DATASET_STORAGE_BACKEND` | Storage backend | `s3` |
| `OVH_S3_BUCKET` | S3 bucket name | `pixel-data` |
| `OVH_S3_ACCESS_KEY` | S3 access key | `your-access-key` |
| `OVH_S3_SECRET_KEY` | S3 secret key | `your-secret-key` |
| `OVH_S3_ENDPOINT` | S3 endpoint URL | `https://s3.region.cloud.ovh.us` |
| `OVH_S3_REGION` | S3 region | `us-east-1` |

## Privacy & Compliance

### PII Detection
- Automated PII scanning with pattern matching
- Email, phone, SSN, credit card detection
- Name pattern recognition
- Fail-closed on high-confidence PII detection

### Provenance Tracking
- Complete source tracking for all files
- Registry integration for metadata validation
- Audit trail for all transformations

### HIPAA Compliance
- End-to-end encryption for sensitive operations
- Audit logging for all data access
- Minimal data collection principles
- Zero-knowledge architecture where possible

## Quality Gates

### Deduplication
- Exact duplicate detection via content hashing
- Near-duplicate detection via text similarity
- Cross-family deduplication analysis

### Cross-Split Leakage
- Prevents data leakage between train/val/test splits
- Special enforcement for holdout-sensitive families
- Similarity-based leakage detection

### Distribution Analysis
- Token/turn/length statistics by family and split
- Expected range validation
- Regression detection for dataset changes

## Human QA Process

### Review Types
1. **Foundation Review**: Core therapeutic datasets
2. **Edge Review**: Crisis and edge case datasets  
3. **Bias/Cultural Review**: Cultural competency and bias detection

### Review Criteria
- Therapeutic accuracy and clinical appropriateness
- Crisis handling and safety prioritization
- Cultural sensitivity and bias avoidance
- Empathy, boundaries, and accessibility awareness

### Mock Implementation
The current implementation includes mock QA signoff for demonstration. In production, this would integrate with actual human reviewers.

## Training Integration

### S3-First Architecture
- All training scripts read directly from S3
- No local file dependencies required
- Streaming data loading for large datasets

### Curriculum Configuration
- Multi-stage training curriculum (Foundation → Expertise → Edge → Voice)
- Configurable weights and epoch recommendations
- Family-to-stage mapping

### Smoke Testing
- End-to-end training pipeline validation
- S3 connectivity and access testing
- Environment prerequisite validation

## Error Handling

### Fail-Closed Security
- All gates fail closed on errors
- Release blocked if any critical gate fails
- Comprehensive error logging and reporting

### Graceful Degradation
- Non-blocking gates produce warnings but don't block release
- Partial failures are clearly documented
- Recovery guidance provided for common issues

## Monitoring & Observability

### Gate Reports
- Detailed JSON reports for each gate
- Combined gate summaries
- Historical tracking capability

### Release Summaries
- Complete orchestration logs
- Success/failure tracking
- Performance metrics

### Audit Trails
- All operations logged with timestamps
- Provenance tracking for data lineage
- Compliance reporting

## Development

### Adding New Gates
1. Create new script in `scripts/` directory
2. Follow existing gate pattern (load manifest, analyze, report)
3. Add to orchestrator execution sequence
4. Update documentation

### Extending Dataset Families
1. Update `ai/data/dataset_registry.json`
2. Add family to Release 0 requirements in coverage matrix
3. Update routing configuration
4. Add family-specific validation rules

### Testing
Each script includes comprehensive error handling and can be run independently for testing and development.

## Troubleshooting

### Common Issues

1. **S3 Connection Failures**
   - Verify credentials and endpoint configuration
   - Check network connectivity
   - Validate bucket permissions

2. **Gate Failures**
   - Review gate reports for specific issues
   - Check data quality and provenance
   - Verify expected ranges and thresholds

3. **Missing Dataset Families**
   - Run coverage matrix to identify gaps
   - Check S3 sync status
   - Verify dataset registry entries

### Debug Mode
Run individual scripts with verbose output to diagnose issues:

```bash
python scripts/build_coverage_matrix.py 2>&1 | tee debug.log
```

## Future Enhancements

### Planned Improvements
1. **Real ChatML Compilation**: Replace placeholder with actual ChatML export
2. **Advanced PII Detection**: Integration with NLP-based PII detection
3. **Real-time QA Integration**: Web interface for human reviewers
4. **Automated Remediation**: Self-healing for common gate failures
5. **Performance Optimization**: Parallel processing for large datasets

### Integration Opportunities
1. **Training Pipeline Integration**: Direct integration with training scripts
2. **Monitoring Dashboard**: Real-time release status monitoring
3. **Automated Scheduling**: Scheduled release generation
4. **Multi-Environment Support**: Dev/staging/prod release pipelines

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review gate reports for specific error details
3. Examine the release summary for overall status
4. Check S3 connectivity and permissions

This implementation provides a robust, compliant, and scalable foundation for mental health dataset management with strict privacy and quality controls.