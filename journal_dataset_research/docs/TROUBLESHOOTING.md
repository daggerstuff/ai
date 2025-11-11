# Journal Dataset Research System - Troubleshooting Guide

## Table of Contents

1. [Common Issues](#common-issues)
2. [Error Messages](#error-messages)
3. [Debugging Tips](#debugging-tips)
4. [FAQ](#faq)
5. [Performance Issues](#performance-issues)
6. [Configuration Issues](#configuration-issues)
7. [Network Issues](#network-issues)
8. [Data Issues](#data-issues)

## Common Issues

### Issue: Configuration Not Loading

**Symptoms**:
- Configuration file not found errors
- Default configuration values not being used
- Environment variable overrides not working

**Solutions**:

1. **Check configuration file path**:
```bash
# Default path: ~/.journal_research/config.yaml
ls -la ~/.journal_research/config.yaml
```

2. **Create configuration file if missing**:
```bash
mkdir -p ~/.journal_research
touch ~/.journal_research/config.yaml
```

3. **Check file permissions**:
```bash
chmod 644 ~/.journal_research/config.yaml
```

4. **Validate YAML syntax**:
```bash
python -c "import yaml; yaml.safe_load(open('~/.journal_research/config.yaml'))"
```

5. **Use environment variables**:
```bash
export JOURNAL_RESEARCH_PUBMED_API_KEY="your-api-key"
export JOURNAL_RESEARCH_STORAGE_PATH="/path/to/datasets"
```

### Issue: Session Not Found

**Symptoms**:
- "Session not found" errors
- Cannot resume interrupted workflows
- Session state not persisting

**Solutions**:

1. **Check session storage path**:
```bash
# Default path: checkpoints/
ls -la checkpoints/
```

2. **Verify session ID**:
```bash
python -m ai.journal_dataset_research.cli.cli status
```

3. **Check session file exists**:
```bash
ls -la checkpoints/session_*.json
```

4. **Verify file permissions**:
```bash
chmod 644 checkpoints/session_*.json
```

5. **Recreate session if needed**:
```bash
python -m ai.journal_dataset_research.cli.cli search \
    --session-id "new_session_id" \
    --keywords "therapy" \
    --sources "pubmed"
```

### Issue: Discovery Service Not Initialized

**Symptoms**:
- "No discovery service configured" warnings
- No sources found during discovery
- Discovery phase skipped

**Solutions**:

1. **Check discovery service configuration**:
```bash
python -m ai.journal_dataset_research.cli.cli config get "discovery.pubmed.api_key"
```

2. **Set API keys if required**:
```bash
python -m ai.journal_dataset_research.cli.cli config set "discovery.pubmed.api_key" "your-api-key"
```

3. **Verify API endpoints**:
```bash
python -m ai.journal_dataset_research.cli.cli config get "discovery.pubmed.base_url"
```

4. **Check network connectivity**:
```bash
curl -I https://eutils.ncbi.nlm.nih.gov/entrez/eutils
```

5. **Use environment variables**:
```bash
export JOURNAL_RESEARCH_PUBMED_API_KEY="your-api-key"
```

### Issue: Evaluation Engine Not Working

**Symptoms**:
- Evaluation scores are all zeros
- Evaluation phase skipped
- "No evaluation engine configured" warnings

**Solutions**:

1. **Check evaluation engine configuration**:
```bash
python -m ai.journal_dataset_research.cli.cli config get "evaluation.therapeutic_relevance_weight"
```

2. **Verify evaluation weights**:
```bash
python -m ai.journal_dataset_research.cli.cli config show | grep evaluation
```

3. **Check compliance module availability**:
```bash
python -c "from ai.journal_dataset_research.compliance.compliance_checker import ComplianceChecker; print('OK')"
```

4. **Enable verbose logging**:
```bash
python -m ai.journal_dataset_research.cli.cli evaluate \
    --session-id "my_session" \
    --verbose
```

### Issue: Acquisition Manager Failing

**Symptoms**:
- Download failures
- "No acquisition manager configured" warnings
- Access request failures

**Solutions**:

1. **Check acquisition configuration**:
```bash
python -m ai.journal_dataset_research.cli.cli config get "acquisition.storage_base_path"
```

2. **Verify storage path exists**:
```bash
mkdir -p data/acquired_datasets
chmod 755 data/acquired_datasets
```

3. **Check download timeout**:
```bash
python -m ai.journal_dataset_research.cli.cli config get "acquisition.download_timeout"
```

4. **Verify network connectivity**:
```bash
curl -I https://example.com/dataset
```

5. **Check file permissions**:
```bash
ls -la data/acquired_datasets/
```

### Issue: Integration Planning Engine Not Working

**Symptoms**:
- Integration plans not generated
- "No integration engine configured" warnings
- Preprocessing scripts not generated

**Solutions**:

1. **Check integration configuration**:
```bash
python -m ai.journal_dataset_research.cli.cli config get "integration.target_format"
```

2. **Verify target format**:
```bash
python -m ai.journal_dataset_research.cli.cli config set "integration.target_format" "chatml"
```

3. **Check dataset format support**:
```bash
# Supported formats: csv, json, xml, parquet
```

4. **Verify dataset structure**:
```bash
python -c "import json; print(json.load(open('data/acquired_datasets/source_id/dataset.json'))[:1])"
```

### Issue: Compliance Checks Failing

**Symptoms**:
- Compliance checks returning errors
- "Compliance module not available" warnings
- License checks failing

**Solutions**:

1. **Check compliance module availability**:
```bash
python -c "from ai.journal_dataset_research.compliance.compliance_checker import ComplianceChecker; print('OK')"
```

2. **Verify license text format**:
```bash
# License text should be in plain text format
```

3. **Check privacy verifier configuration**:
```bash
python -c "from ai.journal_dataset_research.compliance.privacy_verifier import PrivacyVerifier; print('OK')"
```

4. **Verify HIPAA validator configuration**:
```bash
python -c "from ai.journal_dataset_research.compliance.hipaa_validator import HIPAAValidator; print('OK')"
```

### Issue: Progress Tracking Not Working

**Symptoms**:
- Progress metrics not updating
- Progress reports not generated
- Progress history not saving

**Solutions**:

1. **Check progress history limit**:
```bash
python -m ai.journal_dataset_research.cli.cli config get "orchestrator.progress_history_limit"
```

2. **Verify session state**:
```bash
python -m ai.journal_dataset_research.cli.cli status --session-id "my_session"
```

3. **Check progress storage**:
```bash
ls -la checkpoints/session_*.json
```

4. **Enable progress logging**:
```bash
python -m ai.journal_dataset_research.cli.cli search \
    --session-id "my_session" \
    --verbose
```

## Error Messages

### Error: "Invalid research session"

**Cause**: Session validation failed

**Solution**:
```bash
# Check session validation errors
python -c "from ai.journal_dataset_research.models.dataset_models import ResearchSession; session = ResearchSession(session_id='test'); errors = session.validate(); print(errors)"
```

### Error: "Session not found"

**Cause**: Session file not found or session ID incorrect

**Solution**:
```bash
# List all sessions
python -m ai.journal_dataset_research.cli.cli status

# Verify session ID
ls -la checkpoints/session_*.json
```

### Error: "Configuration file not found"

**Cause**: Configuration file missing or path incorrect

**Solution**:
```bash
# Create configuration file
mkdir -p ~/.journal_research
touch ~/.journal_research/config.yaml

# Or use environment variables
export JOURNAL_RESEARCH_PUBMED_API_KEY="your-api-key"
```

### Error: "API key not found"

**Cause**: API key not configured

**Solution**:
```bash
# Set API key in configuration
python -m ai.journal_dataset_research.cli.cli config set "discovery.pubmed.api_key" "your-api-key"

# Or use environment variable
export JOURNAL_RESEARCH_PUBMED_API_KEY="your-api-key"
```

### Error: "Download failed"

**Cause**: Network error or file not available

**Solution**:
```bash
# Check network connectivity
curl -I https://example.com/dataset

# Check download timeout
python -m ai.journal_dataset_research.cli.cli config get "acquisition.download_timeout"

# Increase timeout if needed
python -m ai.journal_dataset_research.cli.cli config set "acquisition.download_timeout" "7200"
```

### Error: "Validation failed"

**Cause**: Data validation error

**Solution**:
```bash
# Check validation errors
python -c "from ai.journal_dataset_research.models.dataset_models import DatasetSource; source = DatasetSource(...); errors = source.validate(); print(errors)"
```

### Error: "Compliance check failed"

**Cause**: Compliance check error

**Solution**:
```bash
# Check compliance module availability
python -c "from ai.journal_dataset_research.compliance.compliance_checker import ComplianceChecker; print('OK')"

# Enable verbose logging
python -m ai.journal_dataset_research.cli.cli evaluate \
    --session-id "my_session" \
    --verbose
```

## Debugging Tips

### 1. Enable Verbose Logging

Enable verbose logging to get detailed output:

```bash
python -m ai.journal_dataset_research.cli.cli search \
    --session-id "my_session" \
    --keywords "therapy" \
    --sources "pubmed" \
    --verbose
```

### 2. Use Dry-Run Mode

Test workflows without making actual changes:

```bash
python ai/journal_dataset_research/main.py \
    --target-sources "pubmed" \
    --keywords "therapy" \
    --dry-run
```

### 3. Check Logs

Check log files for error messages:

```bash
# Check log file if configured
tail -f logs/research.log

# Or check stdout/stderr
python -m ai.journal_dataset_research.cli.cli search \
    --session-id "my_session" \
    --keywords "therapy" \
    --sources "pubmed" \
    2>&1 | tee debug.log
```

### 4. Verify Configuration

Verify configuration values:

```bash
# Show all configuration
python -m ai.journal_dataset_research.cli.cli config show

# Get specific configuration value
python -m ai.journal_dataset_research.cli.cli config get "orchestrator.max_retries"
```

### 5. Test Individual Components

Test individual components in isolation:

```python
# Test evaluation engine
from ai.journal_dataset_research.evaluation.evaluation_engine import EvaluationEngine
from ai.journal_dataset_research.models.dataset_models import DatasetSource

evaluation_engine = EvaluationEngine()
source = DatasetSource(...)
evaluation = evaluation_engine.evaluate_dataset(source)
print(evaluation)
```

### 6. Check Session State

Check session state for issues:

```bash
# Check session status
python -m ai.journal_dataset_research.cli.cli status --session-id "my_session"

# View session file
cat checkpoints/session_*.json | jq .
```

### 7. Verify Data Models

Verify data models are valid:

```python
from ai.journal_dataset_research.models.dataset_models import DatasetSource

source = DatasetSource(...)
errors = source.validate()
if errors:
    print(f"Validation errors: {errors}")
```

### 8. Test Network Connectivity

Test network connectivity to APIs:

```bash
# Test PubMed API
curl -I https://eutils.ncbi.nlm.nih.gov/entrez/eutils

# Test DOAJ API
curl -I https://doaj.org/api/v2

# Test repository APIs
curl -I https://datadryad.org/api/v2
curl -I https://zenodo.org/api
```

## FAQ

### Q: How do I resume an interrupted workflow?

**A**: Use the `--resume` flag with the session ID:

```bash
python ai/journal_dataset_research/main.py \
    --session-id "my_session" \
    --resume
```

### Q: How do I configure API keys?

**A**: Use environment variables or configuration file:

```bash
# Environment variable
export JOURNAL_RESEARCH_PUBMED_API_KEY="your-api-key"

# Configuration file
python -m ai.journal_dataset_research.cli.cli config set "discovery.pubmed.api_key" "your-api-key"
```

### Q: How do I enable encryption?

**A**: Configure encryption in the acquisition settings:

```bash
python -m ai.journal_dataset_research.cli.cli config set "acquisition.encryption_enabled" "true"
python -m ai.journal_dataset_research.cli.cli config set "acquisition.encryption_key" "your-encryption-key"
```

### Q: How do I change the storage path?

**A**: Use environment variable or configuration file:

```bash
# Environment variable
export JOURNAL_RESEARCH_STORAGE_PATH="/path/to/datasets"

# Configuration file
python -m ai.journal_dataset_research.cli.cli config set "acquisition.storage_base_path" "/path/to/datasets"
```

### Q: How do I enable parallel processing?

**A**: Configure parallel processing in orchestrator settings:

```bash
python -m ai.journal_dataset_research.cli.cli config set "orchestrator.parallel_evaluation" "true"
python -m ai.journal_dataset_research.cli.cli config set "orchestrator.parallel_integration_planning" "true"
python -m ai.journal_dataset_research.cli.cli config set "orchestrator.max_workers" "8"
```

### Q: How do I generate reports?

**A**: Use the report command:

```bash
python -m ai.journal_dataset_research.cli.cli report \
    --session-id "my_session" \
    --output "report.json" \
    --format "json"
```

### Q: How do I check progress?

**A**: Use the status command:

```bash
python -m ai.journal_dataset_research.cli.cli status --session-id "my_session"
```

### Q: How do I debug evaluation issues?

**A**: Enable verbose logging and check evaluation notes:

```bash
python -m ai.journal_dataset_research.cli.cli evaluate \
    --session-id "my_session" \
    --verbose
```

### Q: How do I test with sample data?

**A**: Use dry-run mode or create test sessions:

```bash
python ai/journal_dataset_research/main.py \
    --target-sources "pubmed" \
    --keywords "therapy" \
    --dry-run
```

### Q: How do I handle network timeouts?

**A**: Increase timeout settings:

```bash
python -m ai.journal_dataset_research.cli.cli config set "acquisition.download_timeout" "7200"
python -m ai.journal_dataset_research.cli.cli config set "discovery.pubmed.api_timeout" "60"
```

## Performance Issues

### Issue: Slow Discovery

**Symptoms**: Discovery phase taking too long

**Solutions**:

1. **Reduce search limit**:
```bash
python -m ai.journal_dataset_research.cli.cli config set "discovery.pubmed.search_limit" "50"
```

2. **Use specific keywords**:
```bash
python -m ai.journal_dataset_research.cli.cli search \
    --keywords "therapy" "counseling" \
    --sources "pubmed"
```

3. **Filter by open access**:
```bash
# Filter by open access in discovery service
```

### Issue: Slow Evaluation

**Symptoms**: Evaluation phase taking too long

**Solutions**:

1. **Enable parallel processing**:
```bash
python -m ai.journal_dataset_research.cli.cli config set "orchestrator.parallel_evaluation" "true"
python -m ai.journal_dataset_research.cli.cli config set "orchestrator.max_workers" "8"
```

2. **Skip compliance checks if not needed**:
```bash
# Set include_compliance=False in evaluation
```

3. **Reduce evaluation history**:
```bash
python -m ai.journal_dataset_research.cli.cli config set "orchestrator.progress_history_limit" "50"
```

### Issue: Slow Downloads

**Symptoms**: Download phase taking too long

**Solutions**:

1. **Increase chunk size**:
```bash
python -m ai.journal_dataset_research.cli.cli config set "acquisition.chunk_size" "16384"
```

2. **Enable resume downloads**:
```bash
python -m ai.journal_dataset_research.cli.cli config set "acquisition.resume_downloads" "true"
```

3. **Increase timeout**:
```bash
python -m ai.journal_dataset_research.cli.cli config set "acquisition.download_timeout" "7200"
```

### Issue: High Memory Usage

**Symptoms**: High memory usage during processing

**Solutions**:

1. **Reduce parallel workers**:
```bash
python -m ai.journal_dataset_research.cli.cli config set "orchestrator.max_workers" "2"
```

2. **Process in batches**:
```bash
# Process datasets in smaller batches
```

3. **Clear progress history**:
```bash
python -m ai.journal_dataset_research.cli.cli config set "orchestrator.progress_history_limit" "10"
```

## Configuration Issues

### Issue: Configuration Not Persisting

**Symptoms**: Configuration changes not saving

**Solutions**:

1. **Check file permissions**:
```bash
chmod 644 ~/.journal_research/config.yaml
```

2. **Verify configuration path**:
```bash
python -m ai.journal_dataset_research.cli.cli config show
```

3. **Check YAML syntax**:
```bash
python -c "import yaml; yaml.safe_load(open('~/.journal_research/config.yaml'))"
```

### Issue: Environment Variables Not Working

**Symptoms**: Environment variable overrides not applied

**Solutions**:

1. **Verify environment variable names**:
```bash
echo $JOURNAL_RESEARCH_PUBMED_API_KEY
```

2. **Check environment variable format**:
```bash
# Format: JOURNAL_RESEARCH_<CONFIG_PATH>
export JOURNAL_RESEARCH_PUBMED_API_KEY="your-api-key"
```

3. **Reload configuration**:
```bash
# Restart the application to reload environment variables
```

## Network Issues

### Issue: API Connection Timeouts

**Symptoms**: API requests timing out

**Solutions**:

1. **Increase timeout**:
```bash
python -m ai.journal_dataset_research.cli.cli config set "discovery.pubmed.api_timeout" "60"
```

2. **Check network connectivity**:
```bash
curl -I https://eutils.ncbi.nlm.nih.gov/entrez/eutils
```

3. **Use retry logic**:
```bash
python -m ai.journal_dataset_research.cli.cli config set "orchestrator.max_retries" "5"
```

### Issue: Rate Limiting

**Symptoms**: API rate limit errors

**Solutions**:

1. **Increase rate limit delay**:
```bash
python -m ai.journal_dataset_research.cli.cli config set "acquisition.rate_limit_delay" "2.0"
```

2. **Reduce concurrent requests**:
```bash
python -m ai.journal_dataset_research.cli.cli config set "orchestrator.max_workers" "2"
```

3. **Use API keys**:
```bash
python -m ai.journal_dataset_research.cli.cli config set "discovery.pubmed.api_key" "your-api-key"
```

## Data Issues

### Issue: Invalid Data Format

**Symptoms**: Data validation errors

**Solutions**:

1. **Check data format**:
```bash
file data/acquired_datasets/source_id/dataset.json
```

2. **Validate JSON syntax**:
```bash
python -c "import json; json.load(open('data/acquired_datasets/source_id/dataset.json'))"
```

3. **Check data model validation**:
```python
from ai.journal_dataset_research.models.dataset_models import DatasetSource

source = DatasetSource(...)
errors = source.validate()
if errors:
    print(f"Validation errors: {errors}")
```

### Issue: Missing Required Fields

**Symptoms**: Missing field errors

**Solutions**:

1. **Check required fields**:
```bash
# Check data model requirements
python -c "from ai.journal_dataset_research.models.dataset_models import DatasetSource; import inspect; print(inspect.signature(DatasetSource.__init__))"
```

2. **Validate data before processing**:
```python
source = DatasetSource(...)
errors = source.validate()
if errors:
    print(f"Validation errors: {errors}")
```

### Issue: Data Corruption

**Symptoms**: Data integrity errors

**Solutions**:

1. **Verify checksums**:
```bash
# Check file checksum
python -c "import hashlib; print(hashlib.md5(open('data/acquired_datasets/source_id/dataset.json', 'rb').read()).hexdigest())"
```

2. **Re-download if needed**:
```bash
python -m ai.journal_dataset_research.cli.cli acquire \
    --session-id "my_session" \
    --interactive
```

3. **Check file permissions**:
```bash
ls -la data/acquired_datasets/source_id/dataset.json
```

## Getting Help

### Documentation

- **Architecture Documentation**: See `docs/ARCHITECTURE.md`
- **API Reference**: See `docs/API_REFERENCE.md`
- **User Guide**: See `docs/USER_GUIDE.md`

### Logs

Check logs for detailed error messages:

```bash
# Check log file if configured
tail -f logs/research.log

# Or check stdout/stderr
python -m ai.journal_dataset_research.cli.cli search \
    --session-id "my_session" \
    --keywords "therapy" \
    --sources "pubmed" \
    2>&1 | tee debug.log
```

### Support

- **Issues**: Report issues to the issue tracker
- **Community**: Join the community for help and support
- **Feedback**: Provide feedback and suggestions

## Conclusion

This troubleshooting guide provides solutions for common issues, error messages, and performance problems. If you encounter issues not covered in this guide, please refer to the documentation or contact support.

For more information, see the architecture documentation, API reference, and user guide.

