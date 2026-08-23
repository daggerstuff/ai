#!/bin/bash

# Alert Escalation Management Script for Pixelated Empathy AI
# Manages alert escalation procedures and notification channels

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ESCALATION_MANAGER="$SCRIPT_DIR/alert_escalation.py"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/escalation_config.json}"
DB_FILE="${DB_FILE:-$SCRIPT_DIR/alert_escalation.db}"

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check if escalation manager exists
    if [[ ! -f "$ESCALATION_MANAGER" ]]; then
        log_error "Escalation manager not found: $ESCALATION_MANAGER"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create a new alert
create_alert() {
    local title="$1"
    local description="$2"
    local severity="$3"
    local source="$4"
    
    if [[ -z "$title" || -z "$description" || -z "$severity" || -z "$source" ]]; then
        log_error "Usage: create_alert <title> <description> <severity> <source>"
        log_info "Severity levels: critical, high, medium, low, info"
        return 1
    fi
    
    log_info "Creating alert: $title ($severity)"
    
    python3 "$ESCALATION_MANAGER" \
        --db-path "$DB_FILE" \
        --config-path "$CONFIG_FILE" \
        create "$title" "$description" "$severity" "$source"
}

# Acknowledge an alert
acknowledge_alert() {
    local alert_id="$1"
    local user="${2:-$(whoami)}"
    
    if [[ -z "$alert_id" ]]; then
        log_error "Usage: acknowledge_alert <alert_id> [user]"
        return 1
    fi
    
    log_info "Acknowledging alert: $alert_id"
    
    python3 "$ESCALATION_MANAGER" \
        --db-path "$DB_FILE" \
        --config-path "$CONFIG_FILE" \
        acknowledge "$alert_id" "$user"
}

# Resolve an alert
resolve_alert() {
    local alert_id="$1"
    local user="${2:-$(whoami)}"
    
    if [[ -z "$alert_id" ]]; then
        log_error "Usage: resolve_alert <alert_id> [user]"
        return 1
    fi
    
    log_info "Resolving alert: $alert_id"
    
    python3 "$ESCALATION_MANAGER" \
        --db-path "$DB_FILE" \
        --config-path "$CONFIG_FILE" \
        resolve "$alert_id" "$user"
}

# List active alerts
list_alerts() {
    log_info "Listing active alerts..."
    
    python3 "$ESCALATION_MANAGER" \
        --db-path "$DB_FILE" \
        --config-path "$CONFIG_FILE" \
        list | jq '.'
}

# Show escalation statistics
show_statistics() {
    log_info "Escalation Statistics:"
    
    python3 "$ESCALATION_MANAGER" \
        --db-path "$DB_FILE" \
        --config-path "$CONFIG_FILE" \
        stats | jq '.'
}

# Test escalation configuration
test_escalation() {
    log_info "Testing escalation configuration..."
    
    # Create a test alert
    local test_alert_id=$(python3 "$ESCALATION_MANAGER" \
        --db-path "$DB_FILE" \
        --config-path "$CONFIG_FILE" \
        create "Test Alert" "This is a test alert for configuration validation" "low" "test_system" | grep -o 'alert_[^"]*')
    
    if [[ -n "$test_alert_id" ]]; then
        log_success "Test alert created: $test_alert_id"
        
        # Wait a moment for escalation to trigger
        sleep 2
        
        # Resolve the test alert
        python3 "$ESCALATION_MANAGER" \
            --db-path "$DB_FILE" \
            --config-path "$CONFIG_FILE" \
            resolve "$test_alert_id" "test_user" > /dev/null
        
        log_success "Test alert resolved: $test_alert_id"
        log_success "Escalation configuration test completed"
    else
        log_error "Failed to create test alert"
        return 1
    fi
}

# Configure notification channels
configure_notifications() {
    log_info "Configuring notification channels..."
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_info "Creating default configuration file..."
        python3 "$ESCALATION_MANAGER" \
            --db-path "$DB_FILE" \
            --config-path "$CONFIG_FILE" \
            stats > /dev/null 2>&1
    fi
    
    log_info "Configuration file: $CONFIG_FILE"
    log_info "Please edit the configuration file to set up notification channels:"
    echo ""
    echo "1. Email Configuration:"
    echo "   - Set SMTP server details"
    echo "   - Configure authentication credentials"
    echo "   - Set from address"
    echo ""
    echo "2. Slack Configuration:"
    echo "   - Set webhook URL"
    echo "   - Configure channel"
    echo ""
    echo "3. Escalation Rules:"
    echo "   - Adjust delay times for each severity level"
    echo "   - Configure recipient lists"
    echo "   - Set notification channels per escalation level"
    
    # Offer to open editor
    read -p "Do you want to edit the configuration now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        local editor="${EDITOR:-nano}"
        "$editor" "$CONFIG_FILE"
        log_success "Configuration updated"
    fi
}

# Monitor alerts in real-time
monitor_alerts() {
    log_info "Monitoring alerts in real-time (press Ctrl+C to stop)..."
    
    while true; do
        clear
        echo "Pixelated Empathy AI - Alert Monitor"
        echo "===================================="
        echo "Time: $(date)"
        echo ""
        
        # Show active alerts
        echo "Active Alerts:"
        echo "=============="
        python3 "$ESCALATION_MANAGER" \
            --db-path "$DB_FILE" \
            --config-path "$CONFIG_FILE" \
            list 2>/dev/null | jq -r '.[] | "\(.alert_id) | \(.severity) | \(.title) | \(.status)"' || echo "No active alerts"
        
        echo ""
        
        # Show statistics
        echo "Statistics:"
        echo "==========="
        python3 "$ESCALATION_MANAGER" \
            --db-path "$DB_FILE" \
            --config-path "$CONFIG_FILE" \
            stats 2>/dev/null | jq -r 'to_entries[] | "\(.key): \(.value)"' || echo "No statistics available"
        
        sleep 10
    done
}

# Validate configuration
validate_config() {
    log_info "Validating escalation configuration..."
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        log_info "Run 'configure_notifications' to create default configuration"
        return 1
    fi
    
    # Check JSON syntax
    if ! jq '.' "$CONFIG_FILE" > /dev/null 2>&1; then
        log_error "Invalid JSON syntax in configuration file"
        return 1
    fi
    
    # Check required sections
    local required_sections=("escalation_rules" "notification_channels")
    for section in "${required_sections[@]}"; do
        if ! jq -e ".$section" "$CONFIG_FILE" > /dev/null 2>&1; then
            log_error "Missing required section: $section"
            return 1
        fi
    done
    
    # Check escalation rules
    local severities=("critical" "high" "medium" "low")
    for severity in "${severities[@]}"; do
        if ! jq -e ".escalation_rules.$severity" "$CONFIG_FILE" > /dev/null 2>&1; then
            log_warning "No escalation rules defined for severity: $severity"
        fi
    done
    
    # Check notification channels
    local channels=$(jq -r '.notification_channels[].name' "$CONFIG_FILE" 2>/dev/null)
    if [[ -z "$channels" ]]; then
        log_warning "No notification channels configured"
    else
        log_info "Configured notification channels: $(echo "$channels" | tr '\n' ' ')"
    fi
    
    log_success "Configuration validation completed"
}

# Generate escalation report
generate_report() {
    local output_file="${1:-escalation_report_$(date +%Y%m%d_%H%M%S).json}"
    
    log_info "Generating escalation report..."
    
    # Get statistics
    local stats=$(python3 "$ESCALATION_MANAGER" \
        --db-path "$DB_FILE" \
        --config-path "$CONFIG_FILE" \
        stats 2>/dev/null)
    
    # Get active alerts
    local alerts=$(python3 "$ESCALATION_MANAGER" \
        --db-path "$DB_FILE" \
        --config-path "$CONFIG_FILE" \
        list 2>/dev/null)
    
    # Create report
    cat > "$output_file" << EOF
{
  "report_generated": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "statistics": $stats,
  "active_alerts": $alerts,
  "configuration_file": "$CONFIG_FILE",
  "database_file": "$DB_FILE"
}
EOF
    
    log_success "Report generated: $output_file"
}

# Show usage
show_usage() {
    cat << EOF
Alert Escalation Management for Pixelated Empathy AI

Usage: $0 <command> [options]

Commands:
  create <title> <description> <severity> <source>  Create a new alert
  acknowledge <alert_id> [user]                     Acknowledge an alert
  resolve <alert_id> [user]                        Resolve an alert
  list                                              List active alerts
  stats                                             Show escalation statistics
  test                                              Test escalation configuration
  configure                                         Configure notification channels
  monitor                                           Monitor alerts in real-time
  validate                                          Validate configuration
  report [output_file]                             Generate escalation report
  help                                              Show this help message

Examples:
  $0 create "Database Down" "Primary database is unreachable" critical "monitoring_system"
  $0 acknowledge alert_1234567890_abcd1234 john.doe
  $0 resolve alert_1234567890_abcd1234 jane.smith
  $0 list
  $0 monitor

Environment Variables:
  CONFIG_FILE    Path to escalation configuration file
  DB_FILE        Path to alert database file
  EDITOR         Text editor for configuration editing

Severity Levels:
  critical       Immediate attention required, escalates quickly
  high           Important issues, escalates with moderate delay
  medium         Standard issues, basic escalation
  low            Minor issues, minimal escalation
  info           Informational alerts, no escalation
EOF
}

# Main execution
main() {
    case "${1:-help}" in
        create)
            check_prerequisites
            create_alert "$2" "$3" "$4" "$5"
            ;;
        acknowledge|ack)
            check_prerequisites
            acknowledge_alert "$2" "$3"
            ;;
        resolve)
            check_prerequisites
            resolve_alert "$2" "$3"
            ;;
        list)
            check_prerequisites
            list_alerts
            ;;
        stats|statistics)
            check_prerequisites
            show_statistics
            ;;
        test)
            check_prerequisites
            test_escalation
            ;;
        configure|config)
            check_prerequisites
            configure_notifications
            ;;
        monitor)
            check_prerequisites
            monitor_alerts
            ;;
        validate)
            check_prerequisites
            validate_config
            ;;
        report)
            check_prerequisites
            generate_report "$2"
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            log_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
