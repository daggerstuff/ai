# Journal Dataset Research CLI

Command-line interface for the journal dataset research system.

## Installation

The CLI is part of the journal dataset research package. Make sure you have all dependencies installed:

```bash
cd ai/
uv install
```

## Usage

### CLI Commands

The main CLI provides several commands for research operations:

#### Search for Dataset Sources

```bash
python -m ai.sourcing.journal.cli.cli search \
    --keywords "therapy" "counseling" "psychotherapy" \
    --sources "pubmed" "doaj" \
    --interactive
```

#### Evaluate Datasets

```bash
python -m ai.sourcing.journal.cli.cli evaluate \
    --session-id "session_abc123" \
    --interactive
```

#### Acquire Datasets

```bash
python -m ai.sourcing.journal.cli.cli acquire \
    --session-id "session_abc123" \
    --interactive
```

#### Create Integration Plans

```bash
python -m ai.sourcing.journal.cli.cli integrate \
    --session-id "session_abc123" \
    --target-format "chatml" \
    --interactive
```

#### Check Status

```bash
# Check specific session
python -m ai.sourcing.journal.cli.cli status --session-id "session_abc123"

# List all sessions
python -m ai.sourcing.journal.cli.cli status
```

#### Generate Report

```bash
python -m ai.sourcing.journal.cli.cli report \
    --session-id "session_abc123" \
    --output "report.json" \
    --format "json"
```

#### Configuration Management

```bash
# Show configuration
python -m ai.sourcing.journal.cli.cli config show

# Get specific config value
python -m ai.sourcing.journal.cli.cli config get "orchestrator.max_retries"

# Set config value
python -m ai.sourcing.journal.cli.cli config set "orchestrator.max_retries" "5"
```

### Main Execution Script

The main execution script provides automated workflow execution with checkpointing:

```bash
# Run full workflow
python ai/sourcing/journal/main.py \
    --target-sources "pubmed" "doaj" \
    --keywords "therapy" "counseling" \
    --interactive

# Resume interrupted workflow
python ai/sourcing/journal/main.py \
    --session-id "session_abc123" \
    --resume

# Dry-run mode
python ai/sourcing/journal/main.py \
    --dry-run \
    --target-sources "pubmed"
```

## Options

### Global Options

- `--config PATH`: Path to configuration file
- `--dry-run`: Run in dry-run mode (no actual changes)
- `--verbose, -v`: Enable verbose logging
- `--log-file PATH`: Log file path

### Interactive Mode

Many commands support `--interactive` or `-i` flag for manual oversight:
- Review datasets before evaluation
- Approve acquisition requests
- Review integration plans
- Manual evaluation overrides

## Configuration

Configuration is stored in `~/.journal_research/config.yaml` (or JSON if YAML is not available).

You can override configuration using environment variables:
- `JOURNAL_RESEARCH_PUBMED_API_KEY`: PubMed API key
- `JOURNAL_RESEARCH_STORAGE_PATH`: Storage path for acquired datasets
- `JOURNAL_RESEARCH_LOG_LEVEL`: Logging level
- `JOURNAL_RESEARCH_MAX_RETRIES`: Maximum retries
- `JOURNAL_RESEARCH_MAX_WORKERS`: Maximum workers for parallel processing

## Workflow Execution

The main execution script (`main.py`) provides:

1. **Phase-by-phase execution**: Runs discovery → evaluation → acquisition → integration
2. **Checkpointing**: Saves progress after each phase
3. **Resume capability**: Resume from last checkpoint
4. **Dry-run mode**: Test workflows without making changes
5. **Interactive mode**: Manual approvals at each phase

## Examples

### Complete Research Workflow

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

### Automated Workflow

```bash
# Run complete workflow automatically
python ai/sourcing/journal/main.py \
    --target-sources "pubmed" "doaj" \
    --keywords "therapy" "counseling" "psychotherapy" \
    --interactive
```

## Session Management

Sessions are stored in the checkpoint directory (default: `checkpoints/`). Each session includes:
- Session metadata
- Discovered sources
- Evaluations
- Access requests
- Acquired datasets
- Integration plans
- Progress metrics
- Activity logs

## Error Handling

The CLI includes robust error handling:
- Automatic retries with exponential backoff
- Fallback strategies for component failures
- Error logging and notification
- Manual intervention points for critical errors

## Troubleshooting

### Configuration Issues

If configuration doesn't load:
- Check that the config file exists and is valid YAML/JSON
- Verify file permissions
- Check environment variable overrides

### Service Initialization

If services are not initialized:
- Check that discovery/evaluation/acquisition/integration services are configured
- Verify API keys and credentials
- Check service dependencies

### Session Not Found

If a session is not found:
- Verify the session ID is correct
- Check the checkpoint directory
- List all sessions with `status` command

