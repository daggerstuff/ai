#!/bin/bash

# Configuration Validation Script for Pixelated Empathy AI
# Validates all configuration aspects with error handling and recovery

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
FAIL_ON_WARNINGS="${FAIL_ON_WARNINGS:-false}"
GENERATE_DEFAULTS="${GENERATE_DEFAULTS:-true}"

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
    
    # Check if required Python packages are available
    python3 -c "import yaml, json" 2>/dev/null || {
        log_error "Required Python packages not found. Installing..."
        pip3 install pyyaml || {
            log_error "Failed to install required packages"
            exit 1
        }
    }
    
    log_success "Prerequisites check passed"
}

# Validate environment variables
validate_environment_variables() {
    log_info "Validating environment variables..."
    
    local required_vars=(
        "DATABASE_URL"
        "REDIS_URL"
        "JWT_SECRET"
        "ENCRYPTION_KEY"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        
        if [[ "$GENERATE_DEFAULTS" == "true" ]]; then
            log_info "Generating default values..."
            generate_default_env_vars "${missing_vars[@]}"
        else
            log_error "Set missing environment variables or enable GENERATE_DEFAULTS=true"
            exit 1
        fi
    else
        log_success "All required environment variables are set"
    fi
}

# Generate default environment variables
generate_default_env_vars() {
    local vars=("$@")
    local env_file="$PROJECT_ROOT/.env.generated"
    
    log_info "Generating default environment variables in $env_file"
    
    cat > "$env_file" << EOF
# Generated environment variables for Pixelated Empathy AI
# WARNING: These are default values - CHANGE THEM FOR PRODUCTION!
# Generated on: $(date)

EOF
    
    for var in "${vars[@]}"; do
        case "$var" in
            "DATABASE_URL")
                echo "DATABASE_URL=postgresql://postgres:changeme@localhost:5432/pixelated_empathy" >> "$env_file"
                ;;
            "REDIS_URL")
                echo "REDIS_URL=redis://localhost:6379/0" >> "$env_file"
                ;;
            "JWT_SECRET")
                local jwt_secret=$(openssl rand -base64 64 | tr -d '\n')
                echo "JWT_SECRET=$jwt_secret" >> "$env_file"
                ;;
            "ENCRYPTION_KEY")
                local enc_key=$(openssl rand -base64 32 | tr -d '\n')
                echo "ENCRYPTION_KEY=$enc_key" >> "$env_file"
                ;;
        esac
    done
    
    cat >> "$env_file" << EOF

# Additional configuration
LOG_LEVEL=INFO
ENVIRONMENT=$ENVIRONMENT
MAX_WORKERS=4
BATCH_SIZE=100
DEBUG=false
EOF
    
    log_warning "Generated default environment variables in $env_file"
    log_warning "IMPORTANT: Review and customize these values before production use!"
    log_info "To use these variables: source $env_file"
}

# Validate configuration files
validate_config_files() {
    log_info "Validating configuration files..."
    
    local config_files=(
        "database.yaml"
        "redis.yaml"
        "security.yaml"
        "monitoring.yaml"
        "backup.yaml"
    )
    
    local missing_files=()
    local invalid_files=()
    
    for file in "${config_files[@]}"; do
        local file_path="$CONFIG_DIR/$file"
        
        if [[ ! -f "$file_path" ]]; then
            missing_files+=("$file")
        else
            # Validate YAML syntax
            if ! python3 -c "import yaml; yaml.safe_load(open('$file_path'))" 2>/dev/null; then
                invalid_files+=("$file")
            fi
        fi
    done
    
    # Handle missing files
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        log_warning "Missing configuration files:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        
        if [[ "$GENERATE_DEFAULTS" == "true" ]]; then
            generate_default_configs "${missing_files[@]}"
        fi
    fi
    
    # Handle invalid files
    if [[ ${#invalid_files[@]} -gt 0 ]]; then
        log_error "Invalid configuration files:"
        for file in "${invalid_files[@]}"; do
            echo "  - $file"
        done
        return 1
    fi
    
    log_success "Configuration files validation passed"
}

# Generate default configuration files
generate_default_configs() {
    local files=("$@")
    
    log_info "Generating default configuration files..."
    
    for file in "${files[@]}"; do
        local file_path="$CONFIG_DIR/$file"
        
        case "$file" in
            "database.yaml")
                cat > "$file_path" << 'EOF'
# Database Configuration
database:
  url: ${DATABASE_URL}
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 3600
  echo: false
  
  # Connection retry settings
  retry:
    max_attempts: 3
    backoff_factor: 2
    max_delay: 60

  # Health check settings
  health_check:
    enabled: true
    interval: 30
    timeout: 5
EOF
                ;;
            "redis.yaml")
                cat > "$file_path" << 'EOF'
# Redis Configuration
redis:
  url: ${REDIS_URL}
  max_connections: 10
  retry_on_timeout: true
  socket_timeout: 5
  socket_connect_timeout: 5
  
  # Connection pool settings
  connection_pool:
    max_connections: 50
    retry_on_timeout: true
    
  # Cluster settings (if using Redis Cluster)
  cluster:
    enabled: false
    startup_nodes: []
EOF
                ;;
            "security.yaml")
                cat > "$file_path" << 'EOF'
# Security Configuration
security:
  encryption:
    enabled: true
    algorithm: "AES-256-GCM"
    key_rotation_days: 90
    
  authentication:
    jwt_expiry: 3600
    refresh_token_expiry: 604800
    require_2fa: false
    password_policy:
      min_length: 8
      require_uppercase: true
      require_lowercase: true
      require_numbers: true
      require_symbols: false
      
  rate_limiting:
    enabled: true
    requests_per_minute: 60
    burst_size: 10
    
  cors:
    enabled: true
    allowed_origins: ["http://localhost:3000"]
    allowed_methods: ["GET", "POST", "PUT", "DELETE"]
    allowed_headers: ["Content-Type", "Authorization"]
EOF
                ;;
            "monitoring.yaml")
                cat > "$file_path" << 'EOF'
# Monitoring Configuration
monitoring:
  metrics:
    enabled: true
    port: 9090
    path: "/metrics"
    
  logging:
    level: INFO
    format: json
    file: "/app/logs/application.log"
    max_size: "100MB"
    backup_count: 5
    
  health_checks:
    enabled: true
    interval: 30
    timeout: 10
    endpoints:
      - path: "/health"
        method: "GET"
      - path: "/ready"
        method: "GET"
        
  tracing:
    enabled: false
    jaeger_endpoint: "http://localhost:14268/api/traces"
    
  alerts:
    enabled: true
    webhook_url: ""
    channels: ["email", "slack"]
EOF
                ;;
            "backup.yaml")
                cat > "$file_path" << 'EOF'
# Backup Configuration
backup:
  enabled: true
  schedule: "0 2 * * *"  # Daily at 2 AM
  retention: "30d"
  compression: true
  encryption: true
  
  storage:
    type: "local"  # Options: local, s3, gcs, azure
    path: "/app/backups"
    
    # S3 configuration (if type is s3)
    s3:
      bucket: ""
      region: "us-west-2"
      access_key_id: ""
      secret_access_key: ""
      
  # What to backup
  targets:
    - type: "database"
      connection: "${DATABASE_URL}"
    - type: "files"
      paths: ["/app/data", "/app/logs"]
      
  # Notification settings
  notifications:
    enabled: true
    on_success: false
    on_failure: true
    email: ""
EOF
                ;;
        esac
        
        log_info "Generated default configuration: $file"
    done
}

# Run Python validation
run_python_validation() {
    log_info "Running comprehensive Python validation..."
    
    local python_validator="$SCRIPT_DIR/config_validator.py"
    
    if [[ ! -f "$python_validator" ]]; then
        log_error "Python validator not found: $python_validator"
        return 1
    fi
    
    local validation_args=(
        "--config-dir" "$CONFIG_DIR"
    )
    
    if [[ "$FAIL_ON_WARNINGS" == "true" ]]; then
        validation_args+=("--fail-on-warnings")
    fi
    
    # Run validation and capture exit code
    if python3 "$python_validator" "${validation_args[@]}"; then
        log_success "Python validation passed"
        return 0
    else
        local exit_code=$?
        if [[ $exit_code -eq 1 ]]; then
            log_error "Python validation failed with errors"
            return 1
        elif [[ $exit_code -eq 2 ]]; then
            log_warning "Python validation passed with warnings"
            if [[ "$FAIL_ON_WARNINGS" == "true" ]]; then
                return 1
            fi
            return 0
        fi
    fi
}

# Validate network connectivity
validate_network_connectivity() {
    log_info "Validating network connectivity..."
    
    # Check database connectivity
    if [[ -n "$DATABASE_URL" ]]; then
        log_info "Testing database connectivity..."
        if python3 -c "
import os
from urllib.parse import urlparse
import socket

url = os.getenv('DATABASE_URL')
if url:
    parsed = urlparse(url)
    if parsed.hostname and parsed.port:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((parsed.hostname, parsed.port))
            sock.close()
            if result == 0:
                print('Database connection test: SUCCESS')
            else:
                print('Database connection test: FAILED')
                exit(1)
        except Exception as e:
            print(f'Database connection test: ERROR - {e}')
            exit(1)
" 2>/dev/null; then
            log_success "Database connectivity test passed"
        else
            log_warning "Database connectivity test failed - service may not be running"
        fi
    fi
    
    # Check Redis connectivity
    if [[ -n "$REDIS_URL" ]]; then
        log_info "Testing Redis connectivity..."
        if python3 -c "
import os
from urllib.parse import urlparse
import socket

url = os.getenv('REDIS_URL')
if url:
    parsed = urlparse(url)
    if parsed.hostname and parsed.port:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((parsed.hostname, parsed.port))
            sock.close()
            if result == 0:
                print('Redis connection test: SUCCESS')
            else:
                print('Redis connection test: FAILED')
                exit(1)
        except Exception as e:
            print(f'Redis connection test: ERROR - {e}')
            exit(1)
" 2>/dev/null; then
            log_success "Redis connectivity test passed"
        else
            log_warning "Redis connectivity test failed - service may not be running"
        fi
    fi
}

# Validate file permissions
validate_file_permissions() {
    log_info "Validating file permissions..."
    
    local sensitive_files=(
        ".env"
        ".env.generated"
        "secrets.yaml"
        "private.key"
    )
    
    local permission_issues=()
    
    for file in "${sensitive_files[@]}"; do
        local file_path="$PROJECT_ROOT/$file"
        
        if [[ -f "$file_path" ]]; then
            local perms=$(stat -c "%a" "$file_path" 2>/dev/null || stat -f "%A" "$file_path" 2>/dev/null)
            
            # Check if file is readable by others (last digit > 0)
            if [[ "${perms: -1}" -gt 0 ]]; then
                permission_issues+=("$file (permissions: $perms)")
            fi
        fi
    done
    
    if [[ ${#permission_issues[@]} -gt 0 ]]; then
        log_warning "Files with potentially insecure permissions:"
        for issue in "${permission_issues[@]}"; do
            echo "  - $issue"
        done
        log_info "Consider setting permissions to 600 for sensitive files"
    else
        log_success "File permissions validation passed"
    fi
}

# Generate validation report
generate_validation_report() {
    log_info "Generating validation report..."
    
    local report_file="$PROJECT_ROOT/config_validation_report.json"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    cat > "$report_file" << EOF
{
  "validation_report": {
    "timestamp": "$timestamp",
    "environment": "$ENVIRONMENT",
    "config_directory": "$CONFIG_DIR",
    "validation_status": "completed",
    "checks_performed": [
      "environment_variables",
      "configuration_files",
      "python_validation",
      "network_connectivity",
      "file_permissions"
    ],
    "generated_files": []
  }
}
EOF
    
    # Add generated files if any
    if [[ -f "$PROJECT_ROOT/.env.generated" ]]; then
        python3 -c "
import json
with open('$report_file', 'r') as f:
    data = json.load(f)
data['validation_report']['generated_files'].append('.env.generated')
with open('$report_file', 'w') as f:
    json.dump(data, f, indent=2)
"
    fi
    
    log_success "Validation report generated: $report_file"
}

# Main execution
main() {
    echo "========================================"
    echo "Pixelated Empathy AI Configuration Validation"
    echo "========================================"
    echo "Environment: $ENVIRONMENT"
    echo "Config Directory: $CONFIG_DIR"
    echo "========================================"
    
    local validation_steps=(
        "check_prerequisites"
        "validate_environment_variables"
        "validate_config_files"
        "run_python_validation"
        "validate_network_connectivity"
        "validate_file_permissions"
        "generate_validation_report"
    )
    
    local failed_steps=()
    
    for step in "${validation_steps[@]}"; do
        echo ""
        if ! $step; then
            failed_steps+=("$step")
            log_error "Validation step failed: $step"
        fi
    done
    
    echo ""
    echo "========================================"
    
    if [[ ${#failed_steps[@]} -eq 0 ]]; then
        log_success "All validation checks passed!"
        echo "✅ Configuration is ready for use"
        exit 0
    else
        log_error "Validation failed for ${#failed_steps[@]} step(s):"
        for step in "${failed_steps[@]}"; do
            echo "  - $step"
        done
        echo "❌ Please fix the issues and run validation again"
        exit 1
    fi
}

# Show usage if help requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h              Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  ENVIRONMENT             Target environment (default: development)"
    echo "  CONFIG_DIR              Configuration directory (default: script directory)"
    echo "  FAIL_ON_WARNINGS        Fail validation on warnings (default: false)"
    echo "  GENERATE_DEFAULTS       Generate default configs if missing (default: true)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Validate with defaults"
    echo "  ENVIRONMENT=production $0             # Validate for production"
    echo "  FAIL_ON_WARNINGS=true $0             # Fail on warnings"
    echo "  GENERATE_DEFAULTS=false $0            # Don't generate defaults"
    exit 0
fi

# Run main function
main "$@"
