#!/bin/bash
# Integration guide for dataset validation in edge case pipeline
# This document explains how to integrate the validation module

## Quick Integration Steps

### 1. Update edge_case_generator.py Constructor
Add after line 57 (after `self.output_dir.mkdir(exist_ok=True)`):

```python
# Initialize validator if available
self.validator = DatasetValidator(strict_mode=False) if DatasetValidator else None
self.validation_failures = []
```

### 2. Update Imports at Top of File
Replace the import section (lines 1-30) to include:

```python
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ai.safety.dataset_validation import DatasetValidator, ValidationResult
except ImportError:
    DatasetValidator = None
    ValidationResult = None

logger = logging.getLogger(__name__)
```

### 3. Update _write_jsonl_file Method
Replace the method at line 253 with:

```python
def _write_jsonl_file(self, filepath: Path, items: list[dict]):
    """Write a list of dicts to a JSONL file with validation."""
    # Validate items before writing if validator is available
    if self.validator:
        validation_summary = self.validator.validate_batch(items)
        
        if validation_summary.get("invalid", 0) > 0:
            # Log failures but don't prevent write (for graceful degradation)
            error_msg = (
                f"Dataset validation found {validation_summary['invalid']} invalid items "
                f"({validation_summary['pass_rate']:.1%} pass rate). "
                f"Bias indicators: {validation_summary.get('bias_summary', {})}"
            )
            logger.warning(error_msg)
            self.validation_failures.append({
                "file": str(filepath),
                "invalid_count": validation_summary["invalid"],
                "total_count": validation_summary["total"],
                "bias_indicators": validation_summary.get("bias_summary", {}),
                "failed_cases": validation_summary.get("failed_cases", [])
            })
            
            # Write validation report
            report_path = filepath.parent / f"{filepath.stem}_validation_report.json"
            with open(report_path, "w") as f:
                json.dump(validation_summary, f, indent=2)
            logger.info(f"Validation report saved to {report_path}")
    
    # Write the file
    with open(filepath, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Wrote {len(items)} items to {filepath}")
```

### 4. Add Validation Report Method
Add this new method at the end of the EdgeCaseGenerator class:

```python
def get_validation_report(self) -> dict:
    """Get validation report for all written datasets."""
    return {
        "total_files_written": len(self.validation_failures),
        "files_with_issues": self.validation_failures,
        "has_failures": len(self.validation_failures) > 0
    }
```

## Using Pre-Sync Validation

Before running `sync-datasets.sh`, run the validation:

```bash
# Make the script executable
chmod +x ai/training_ready/platforms/ovh/validate-before-sync.sh

# Run validation
./ai/training_ready/platforms/ovh/validate-before-sync.sh ai/pipelines/edge_case_pipeline_standalone

# Check results
cat pre_sync_validation_report.json | jq '.'
```

If validation fails (any invalid cases detected), the script exits with code 1 and won't proceed to sync.

## Integration with CI/CD

Add to your CI pipeline before the sync-datasets step:

```bash
# Validate datasets
python3 ai/safety/dataset_validation.py

# Or run the bash script
./ai/training_ready/platforms/ovh/validate-before-sync.sh
VALIDATION_RESULT=$?

if [ $VALIDATION_RESULT -ne 0 ]; then
    echo "❌ Dataset validation failed - aborting sync to S3"
    exit 1
fi

# Only if validation passes, proceed with sync
./ai/training_ready/platforms/ovh/sync-datasets.sh upload
```

## Files Created/Modified

1. **ai/safety/dataset_validation.py** - Bias detection and validation module
2. **ai/training_ready/platforms/ovh/validate-before-sync.sh** - Pre-sync validation script
3. **ai/pipelines/edge_case_pipeline_standalone/edge_case_generator.py** - Updated with validation

## Validation Rules

The validator checks for:
- **Stereotypes** - Cultural, racial, gender stereotypes
- **Offensive Generalizations** - Broad harmful statements
- **Problematic Responses** - Dismissive, blame-shifting therapy responses
- **Privacy Violations** - Identifiable information (names, addresses)
- **Harmful Content** - Severe abuse, exploitation

## Failure Modes

- **Strict Mode** - Warnings become errors, prevents upload
- **Graceful Mode** (Default) - Logs warnings, generates report, still writes file
- Files with validation issues get a companion `*_validation_report.json` file

All validation reports are saved alongside generated datasets for audit trail compliance.
