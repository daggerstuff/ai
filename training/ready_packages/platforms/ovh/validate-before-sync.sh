#!/bin/bash
# Pre-Sync Dataset Validation Script
# Validates all edge case datasets before syncing to S3
# Ensures no biased or inappropriate content is uploaded
#
# Usage: ./validate-before-sync.sh [output_dir]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
OUTPUT_DIR="${1:-.}"
AI_DIR="/home/vivi/pixelated/ai"
VALIDATION_REPORT="${OUTPUT_DIR}/pre_sync_validation_report.json"
FAILED_FILES="${OUTPUT_DIR}/validation_failed_files.txt"

# Ensure Python validation module exists
if [ ! -f "$AI_DIR/safety/dataset_validation.py" ]; then
    log_error "Dataset validation module not found at $AI_DIR/safety/dataset_validation.py"
    exit 1
fi

log_info "Starting pre-sync dataset validation..."
log_info "Output directory: $OUTPUT_DIR"
echo ""

# Find all JSONL files that would be synced
JSONL_FILES=()
TOTAL_FILES=0

# Check edge case pipeline output
if [ -d "$AI_DIR/pipelines/edge_case_pipeline_standalone" ]; then
    log_info "Scanning edge case pipeline output..."
    while IFS= read -r -d '' file; do
        JSONL_FILES+=("$file")
        ((TOTAL_FILES++))
    done < <(find "$AI_DIR/pipelines/edge_case_pipeline_standalone" -name "*.jsonl" -print0 2>/dev/null)
fi

if [ $TOTAL_FILES -eq 0 ]; then
    log_warn "No JSONL files found to validate"
    echo "[]" > "$VALIDATION_REPORT"
    exit 0
fi

log_info "Found $TOTAL_FILES JSONL files to validate"
echo ""

# Create Python validation script
VALIDATION_SCRIPT=$(mktemp)
cat > "$VALIDATION_SCRIPT" << 'PYTHON_SCRIPT'
import sys
import json
from pathlib import Path

# Add AI module to path
sys.path.insert(0, '/home/vivi/pixelated')

from ai.safety.dataset_validation import DatasetValidator, validate_jsonl_file

def main():
    files = sys.argv[1:]
    output_dir = sys.argv[-1] if files else "."
    
    # Remove output_dir from files list
    files = files[:-1] if files else []
    
    all_results = {
        "total_files": len(files),
        "files_validated": [],
        "total_valid": 0,
        "total_invalid": 0,
        "critical_failures": [],
        "validation_timestamp": None,
        "overall_status": "PASS"
    }
    
    from datetime import datetime
    all_results["validation_timestamp"] = datetime.utcnow().isoformat()
    
    for filepath in files:
        if not Path(filepath).exists():
            print(f"[WARN] File not found: {filepath}", file=sys.stderr)
            continue
        
        try:
            result = validate_jsonl_file(filepath, strict_mode=False)
            
            if "error" in result:
                print(f"[ERROR] Failed to validate {filepath}: {result['error']}", file=sys.stderr)
                all_results["critical_failures"].append({
                    "file": filepath,
                    "error": result["error"]
                })
                all_results["overall_status"] = "FAIL"
                continue
            
            all_results["total_valid"] += result.get("valid", 0)
            all_results["total_invalid"] += result.get("invalid", 0)
            
            file_result = {
                "file": filepath,
                "valid": result.get("valid", 0),
                "invalid": result.get("invalid", 0),
                "warnings": result.get("warnings", 0),
                "pass_rate": result.get("pass_rate", 0),
                "bias_summary": result.get("bias_summary", {}),
                "failed_cases": result.get("failed_cases", [])
            }
            
            all_results["files_validated"].append(file_result)
            
            # If any file has invalid cases, mark overall as FAIL
            if result.get("invalid", 0) > 0:
                all_results["overall_status"] = "FAIL"
                
        except Exception as e:
            print(f"[ERROR] Exception validating {filepath}: {e}", file=sys.stderr)
            all_results["critical_failures"].append({
                "file": filepath,
                "error": str(e)
            })
            all_results["overall_status"] = "FAIL"
    
    # Output results as JSON
    print(json.dumps(all_results, indent=2))

if __name__ == "__main__":
    main()
PYTHON_SCRIPT

# Run validation
log_info "Running validation checks..."
VALIDATION_OUTPUT=$(python3 "$VALIDATION_SCRIPT" "${JSONL_FILES[@]}" "$OUTPUT_DIR" 2>&1)

# Save results to file
echo "$VALIDATION_OUTPUT" > "$VALIDATION_REPORT"

# Parse results
OVERALL_STATUS=$(echo "$VALIDATION_OUTPUT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('overall_status', 'UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
TOTAL_VALID=$(echo "$VALIDATION_OUTPUT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total_valid', 0))" 2>/dev/null || echo "0")
TOTAL_INVALID=$(echo "$VALIDATION_OUTPUT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total_invalid', 0))" 2>/dev/null || echo "0")
CRITICAL_FAILURES=$(echo "$VALIDATION_OUTPUT" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('critical_failures', [])))" 2>/dev/null || echo "0")

# Report results
echo ""
log_info "=== Validation Summary ==="
log_info "Files validated: $TOTAL_FILES"
log_info "Valid cases: $TOTAL_VALID"
log_info "Invalid cases: $TOTAL_INVALID"
log_info "Critical failures: $CRITICAL_FAILURES"
echo ""

if [ "$OVERALL_STATUS" = "PASS" ] && [ "$TOTAL_INVALID" -eq 0 ]; then
    log_success "✅ All datasets passed validation!"
    log_info "Report saved to: $VALIDATION_REPORT"
    rm -f "$VALIDATION_SCRIPT"
    exit 0
else
    log_error "❌ Validation FAILED"
    log_error "Total invalid cases: $TOTAL_INVALID"
    log_error "Critical failures: $CRITICAL_FAILURES"
    echo ""
    log_error "Detailed report:"
    cat "$VALIDATION_REPORT" | python3 -m json.tool 2>/dev/null || cat "$VALIDATION_REPORT"
    echo ""
    log_error "Failed case details saved to: $VALIDATION_REPORT"
    
    # List files with failures
    if [ -f "$VALIDATION_REPORT" ]; then
        echo "$VALIDATION_OUTPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
failed = [f['file'] for f in data.get('files_validated', []) if f.get('invalid', 0) > 0]
if failed:
    print('\n❌ Files with invalid cases:')
    for f in failed:
        print(f'  - {f}')
" 2>/dev/null || true
    fi
    
    rm -f "$VALIDATION_SCRIPT"
    exit 1
fi
