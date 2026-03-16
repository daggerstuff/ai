#!/bin/bash

# Configuration Management Script for Pixelated Empathy AI
# Provides unified interface for configuration validation, tracking, and rollback

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT="${ENVIRONMENT:-development}"
CONFIG_DIR="${CONFIG_DIR:-$SCRIPT_DIR}"
AUTO_TRACK="${AUTO_TRACK:-true}"
AUTO_SNAPSHOT="${AUTO_SNAPSHOT:-false}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage
show_usage() {
    cat << EOF
Configuration Management for Pixelated Empathy AI

Usage: $0 <command> [options]

Commands:
  validate                    Validate all configuration
  track <file> <type> [desc]  Track configuration change
  snapshot [description]      Create configuration snapshot
  rollback <id>              Rollback to change or snapshot
  history [file]             Show change history
  snapshots                  List snapshots
  compare <id1> <id2>        Compare snapshots
  cleanup [days]             Clean up old backups
  export <file>              Export tracking data
  import <file>              Import tracking data
  edit <file>                Edit configuration with tracking
  diff <id1> [id2]           Show differences
  status                     Show configuration status

Options:
  --environment <env>        Set environment (default: development)
  --config-dir <dir>         Set configuration directory
  --auto-track <true|false>  Enable/disable automatic tracking
  --auto-snapshot <true|false> Enable/disable automatic snapshots
  --help                     Show this help message

Examples:
  $0 validate                           # Validate configuration
  $0 track database.yaml update "Updated connection pool"
  $0 snapshot "Pre-deployment backup"
  $0 rollback change_20231201_123456_abc123
  $0 edit security.yaml                 # Edit with automatic tracking
  $0 history database.yaml              # Show history for specific file
  $0 compare snapshot_123 snapshot_456  # Compare two snapshots

Environment Variables:
  ENVIRONMENT     Target environment (development, staging, production)
  CONFIG_DIR      Configuration directory path
  AUTO_TRACK      Automatically track changes (true/false)
  AUTO_SNAPSHOT   Automatically create snapshots (true/false)
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --config-dir)
                CONFIG_DIR="$2"
                shift 2
                ;;
            --auto-track)
                AUTO_TRACK="$2"
                shift 2
                ;;
            --auto-snapshot)
                AUTO_SNAPSHOT="$2"
                shift 2
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                break
                ;;
        esac
    done
    
    COMMAND="$1"
    shift || true
    ARGS=("$@")
}

# Validate configuration
validate_config() {
    log_info "Validating configuration..."
    
    local validator_script="$SCRIPT_DIR/validate_config.sh"
    
    if [[ ! -f "$validator_script" ]]; then
        log_error "Validator script not found: $validator_script"
        return 1
    fi
    
    # Set environment variables for validator
    export ENVIRONMENT="$ENVIRONMENT"
    export CONFIG_DIR="$CONFIG_DIR"
    
    if "$validator_script"; then
        log_success "Configuration validation passed"
        return 0
    else
        log_error "Configuration validation failed"
        return 1
    fi
}

# Track configuration change
track_change() {
    local file_path="$1"
    local change_type="$2"
    local description="${3:-Configuration change}"
    
    if [[ -z "$file_path" || -z "$change_type" ]]; then
        log_error "Usage: track <file_path> <change_type> [description]"
        return 1
    fi
    
    log_info "Tracking configuration change..."
    
    python3 "$SCRIPT_DIR/config_tracker.py" \
        --config-dir "$CONFIG_DIR" \
        track "$file_path" "$change_type" \
        --description "$description" \
        --environment "$ENVIRONMENT"
}

# Create snapshot
create_snapshot() {
    local description="${1:-Configuration snapshot}"
    
    log_info "Creating configuration snapshot..."
    
    python3 "$SCRIPT_DIR/config_tracker.py" \
        --config-dir "$CONFIG_DIR" \
        snapshot \
        --description "$description" \
        --environment "$ENVIRONMENT"
}

# Rollback configuration
rollback_config() {
    local target_id="$1"
    
    if [[ -z "$target_id" ]]; then
        log_error "Usage: rollback <change_id_or_snapshot_id>"
        return 1
    fi
    
    log_info "Rolling back configuration..."
    
    # Determine if it's a change ID or snapshot ID
    if [[ "$target_id" == change_* ]]; then
        python3 "$SCRIPT_DIR/config_tracker.py" \
            --config-dir "$CONFIG_DIR" \
            rollback --change-id "$target_id"
    elif [[ "$target_id" == snapshot_* ]]; then
        python3 "$SCRIPT_DIR/config_tracker.py" \
            --config-dir "$CONFIG_DIR" \
            rollback --snapshot-id "$target_id"
    else
        log_error "Invalid ID format. Expected change_* or snapshot_*"
        return 1
    fi
}

# Show change history
show_history() {
    local file_path="$1"
    local limit="${2:-20}"
    
    log_info "Showing configuration history..."
    
    local args=(
        "--config-dir" "$CONFIG_DIR"
        "history"
        "--limit" "$limit"
    )
    
    if [[ -n "$file_path" ]]; then
        args+=("--file-path" "$file_path")
    fi
    
    python3 "$SCRIPT_DIR/config_tracker.py" "${args[@]}" | jq '.'
}

# List snapshots
list_snapshots() {
    local limit="${1:-10}"
    
    log_info "Listing configuration snapshots..."
    
    python3 "$SCRIPT_DIR/config_tracker.py" \
        --config-dir "$CONFIG_DIR" \
        snapshots \
        --limit "$limit" | jq '.'
}

# Compare snapshots
compare_snapshots() {
    local snapshot1="$1"
    local snapshot2="$2"
    
    if [[ -z "$snapshot1" || -z "$snapshot2" ]]; then
        log_error "Usage: compare <snapshot1_id> <snapshot2_id>"
        return 1
    fi
    
    log_info "Comparing snapshots..."
    
    python3 "$SCRIPT_DIR/config_tracker.py" \
        --config-dir "$CONFIG_DIR" \
        compare "$snapshot1" "$snapshot2" | jq '.'
}

# Clean up old backups
cleanup_backups() {
    local days="${1:-30}"
    
    log_info "Cleaning up old backups (older than $days days)..."
    
    python3 "$SCRIPT_DIR/config_tracker.py" \
        --config-dir "$CONFIG_DIR" \
        cleanup --days "$days"
}

# Export tracking data
export_tracking() {
    local output_file="$1"
    
    if [[ -z "$output_file" ]]; then
        log_error "Usage: export <output_file>"
        return 1
    fi
    
    log_info "Exporting tracking data..."
    
    python3 "$SCRIPT_DIR/config_tracker.py" \
        --config-dir "$CONFIG_DIR" \
        export "$output_file"
}

# Import tracking data
import_tracking() {
    local input_file="$1"
    
    if [[ -z "$input_file" ]]; then
        log_error "Usage: import <input_file>"
        return 1
    fi
    
    if [[ ! -f "$input_file" ]]; then
        log_error "Input file not found: $input_file"
        return 1
    fi
    
    log_info "Importing tracking data..."
    
    python3 "$SCRIPT_DIR/config_tracker.py" \
        --config-dir "$CONFIG_DIR" \
        import "$input_file"
}

# Edit configuration file with tracking
edit_config() {
    local file_path="$1"
    
    if [[ -z "$file_path" ]]; then
        log_error "Usage: edit <file_path>"
        return 1
    fi
    
    # Convert to absolute path
    if [[ ! "$file_path" = /* ]]; then
        file_path="$CONFIG_DIR/$file_path"
    fi
    
    # Check if file exists
    local change_type="update"
    if [[ ! -f "$file_path" ]]; then
        change_type="create"
        log_info "Creating new configuration file: $file_path"
    else
        log_info "Editing configuration file: $file_path"
    fi
    
    # Create pre-edit snapshot if auto-snapshot is enabled
    local pre_snapshot_id=""
    if [[ "$AUTO_SNAPSHOT" == "true" ]]; then
        pre_snapshot_id=$(create_snapshot "Pre-edit snapshot for $(basename "$file_path")")
        log_info "Created pre-edit snapshot: $pre_snapshot_id"
    fi
    
    # Get file hash before editing (if file exists)
    local old_hash=""
    if [[ -f "$file_path" ]]; then
        old_hash=$(sha256sum "$file_path" | cut -d' ' -f1)
    fi
    
    # Open editor
    local editor="${EDITOR:-nano}"
    "$editor" "$file_path"
    
    # Check if file was actually changed
    local new_hash=""
    if [[ -f "$file_path" ]]; then
        new_hash=$(sha256sum "$file_path" | cut -d' ' -f1)
    fi
    
    if [[ "$old_hash" != "$new_hash" ]]; then
        # File was changed, track it
        if [[ "$AUTO_TRACK" == "true" ]]; then
            local description="Edited $(basename "$file_path") using $editor"
            track_change "$file_path" "$change_type" "$description"
            log_success "Configuration change tracked automatically"
        else
            log_info "File was modified. Use 'track' command to record the change."
        fi
        
        # Validate configuration after edit
        log_info "Validating configuration after edit..."
        if ! validate_config; then
            log_warning "Configuration validation failed after edit"
            
            # Offer to rollback
            read -p "Do you want to rollback the changes? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                if [[ -n "$pre_snapshot_id" ]]; then
                    rollback_config "$pre_snapshot_id"
                    log_info "Rolled back to pre-edit snapshot"
                else
                    log_warning "No pre-edit snapshot available for rollback"
                fi
            fi
        else
            log_success "Configuration validation passed"
        fi
    else
        log_info "No changes detected in file"
    fi
}

# Show configuration differences
show_diff() {
    local id1="$1"
    local id2="${2:-current}"
    
    if [[ -z "$id1" ]]; then
        log_error "Usage: diff <id1> [id2]"
        return 1
    fi
    
    log_info "Showing configuration differences..."
    
    if [[ "$id2" == "current" ]]; then
        # Compare with current state
        local current_snapshot=$(create_snapshot "Temporary snapshot for diff")
        compare_snapshots "$id1" "$current_snapshot"
    else
        compare_snapshots "$id1" "$id2"
    fi
}

# Show configuration status
show_status() {
    log_info "Configuration Status"
    echo "===================="
    echo "Environment: $ENVIRONMENT"
    echo "Config Directory: $CONFIG_DIR"
    echo "Auto Track: $AUTO_TRACK"
    echo "Auto Snapshot: $AUTO_SNAPSHOT"
    echo ""
    
    # Show recent changes
    log_info "Recent Changes (last 5):"
    show_history "" 5
    
    echo ""
    
    # Show recent snapshots
    log_info "Recent Snapshots (last 3):"
    list_snapshots 3
    
    echo ""
    
    # Show validation status
    log_info "Validation Status:"
    if validate_config >/dev/null 2>&1; then
        log_success "Configuration is valid"
    else
        log_error "Configuration has validation errors"
    fi
}

# Main execution
main() {
    # Parse arguments
    parse_args "$@"
    
    # Check if command is provided
    if [[ -z "$COMMAND" ]]; then
        show_usage
        exit 1
    fi
    
    # Check prerequisites
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        log_warning "jq is not installed - JSON output will not be formatted"
    fi
    
    # Execute command
    case "$COMMAND" in
        validate)
            validate_config
            ;;
        track)
            track_change "${ARGS[@]}"
            ;;
        snapshot)
            create_snapshot "${ARGS[0]}"
            ;;
        rollback)
            rollback_config "${ARGS[0]}"
            ;;
        history)
            show_history "${ARGS[@]}"
            ;;
        snapshots)
            list_snapshots "${ARGS[0]}"
            ;;
        compare)
            compare_snapshots "${ARGS[@]}"
            ;;
        cleanup)
            cleanup_backups "${ARGS[0]}"
            ;;
        export)
            export_tracking "${ARGS[0]}"
            ;;
        import)
            import_tracking "${ARGS[0]}"
            ;;
        edit)
            edit_config "${ARGS[0]}"
            ;;
        diff)
            show_diff "${ARGS[@]}"
            ;;
        status)
            show_status
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
