#!/bin/bash

# Worker Management Script for Pixelated Empathy AI Distributed Processing
# Manages Celery workers for different task types

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
CELERY_APP="distributed_processing.celery_config:celery_app"
WORKER_LOG_DIR="${WORKER_LOG_DIR:-$PROJECT_ROOT/logs/workers}"
WORKER_PID_DIR="${WORKER_PID_DIR:-$PROJECT_ROOT/tmp/workers}"
REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"

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

# Create necessary directories
create_directories() {
    mkdir -p "$WORKER_LOG_DIR"
    mkdir -p "$WORKER_PID_DIR"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check if Celery is installed
    if ! python3 -c "import celery" 2>/dev/null; then
        log_error "Celery is not installed. Install with: pip install celery[redis]"
        exit 1
    fi
    
    # Check if Redis is accessible
    if ! python3 -c "import redis; r=redis.from_url('$REDIS_URL'); r.ping()" 2>/dev/null; then
        log_warning "Redis is not accessible at $REDIS_URL"
        log_info "Make sure Redis is running or update REDIS_URL environment variable"
    fi
    
    log_success "Prerequisites check passed"
}

# Start a worker
start_worker() {
    local worker_name="$1"
    local queue="$2"
    local concurrency="${3:-4}"
    local max_tasks="${4:-1000}"
    
    if [[ -z "$worker_name" || -z "$queue" ]]; then
        log_error "Usage: start_worker <worker_name> <queue> [concurrency] [max_tasks]"
        return 1
    fi
    
    local pid_file="$WORKER_PID_DIR/${worker_name}.pid"
    local log_file="$WORKER_LOG_DIR/${worker_name}.log"
    
    # Check if worker is already running
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        log_warning "Worker $worker_name is already running (PID: $(cat "$pid_file"))"
        return 0
    fi
    
    log_info "Starting worker: $worker_name (queue: $queue, concurrency: $concurrency)"
    
    # Start worker in background
    nohup celery -A "$CELERY_APP" worker \
        --hostname="$worker_name@%h" \
        --queues="$queue" \
        --concurrency="$concurrency" \
        --max-tasks-per-child="$max_tasks" \
        --loglevel=info \
        --logfile="$log_file" \
        --pidfile="$pid_file" \
        --detach \
        --time-limit=600 \
        --soft-time-limit=300 \
        > /dev/null 2>&1
    
    # Wait a moment and check if worker started
    sleep 2
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        log_success "Worker $worker_name started successfully (PID: $(cat "$pid_file"))"
    else
        log_error "Failed to start worker $worker_name"
        return 1
    fi
}

# Stop a worker
stop_worker() {
    local worker_name="$1"
    
    if [[ -z "$worker_name" ]]; then
        log_error "Usage: stop_worker <worker_name>"
        return 1
    fi
    
    local pid_file="$WORKER_PID_DIR/${worker_name}.pid"
    
    if [[ ! -f "$pid_file" ]]; then
        log_warning "Worker $worker_name is not running (no PID file)"
        return 0
    fi
    
    local pid=$(cat "$pid_file")
    
    if ! kill -0 "$pid" 2>/dev/null; then
        log_warning "Worker $worker_name is not running (stale PID file)"
        rm -f "$pid_file"
        return 0
    fi
    
    log_info "Stopping worker: $worker_name (PID: $pid)"
    
    # Send TERM signal
    kill -TERM "$pid"
    
    # Wait for graceful shutdown
    local timeout=30
    while [[ $timeout -gt 0 ]] && kill -0 "$pid" 2>/dev/null; do
        sleep 1
        ((timeout--))
    done
    
    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        log_warning "Worker $worker_name did not stop gracefully, force killing"
        kill -KILL "$pid"
        sleep 1
    fi
    
    # Clean up PID file
    rm -f "$pid_file"
    
    log_success "Worker $worker_name stopped"
}

# Restart a worker
restart_worker() {
    local worker_name="$1"
    local queue="$2"
    local concurrency="${3:-4}"
    local max_tasks="${4:-1000}"
    
    log_info "Restarting worker: $worker_name"
    stop_worker "$worker_name"
    sleep 2
    start_worker "$worker_name" "$queue" "$concurrency" "$max_tasks"
}

# Show worker status
show_worker_status() {
    local worker_name="$1"
    
    if [[ -n "$worker_name" ]]; then
        # Show specific worker status
        local pid_file="$WORKER_PID_DIR/${worker_name}.pid"
        
        if [[ -f "$pid_file" ]]; then
            local pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                log_success "Worker $worker_name is running (PID: $pid)"
                
                # Show process info
                ps -p "$pid" -o pid,ppid,pcpu,pmem,etime,cmd --no-headers
            else
                log_error "Worker $worker_name is not running (stale PID file)"
            fi
        else
            log_error "Worker $worker_name is not running (no PID file)"
        fi
    else
        # Show all workers status
        log_info "Worker Status:"
        echo "=============="
        
        local any_running=false
        
        for pid_file in "$WORKER_PID_DIR"/*.pid; do
            if [[ -f "$pid_file" ]]; then
                local worker_name=$(basename "$pid_file" .pid)
                local pid=$(cat "$pid_file")
                
                if kill -0 "$pid" 2>/dev/null; then
                    echo -e "${GREEN}✓${NC} $worker_name (PID: $pid)"
                    any_running=true
                else
                    echo -e "${RED}✗${NC} $worker_name (stale PID file)"
                fi
            fi
        done
        
        if [[ "$any_running" == "false" ]]; then
            echo "No workers are currently running"
        fi
    fi
}

# Start all default workers
start_all_workers() {
    log_info "Starting all default workers..."
    
    # Quality validation workers
    start_worker "quality_validator_1" "quality_validation" 4 1000
    start_worker "quality_validator_2" "quality_validation" 4 1000
    
    # Data processing workers
    start_worker "data_processor_1" "data_processing" 2 500
    start_worker "data_processor_2" "data_processing" 2 500
    
    # Model training worker (single instance, high resource)
    start_worker "model_trainer" "model_training" 1 100
    
    # Backup worker
    start_worker "backup_worker" "backup" 1 50
    
    # High priority worker
    start_worker "high_priority" "high_priority" 2 200
    
    # Default queue worker
    start_worker "default_worker" "default" 2 500
    
    log_success "All workers started"
}

# Stop all workers
stop_all_workers() {
    log_info "Stopping all workers..."
    
    for pid_file in "$WORKER_PID_DIR"/*.pid; do
        if [[ -f "$pid_file" ]]; then
            local worker_name=$(basename "$pid_file" .pid)
            stop_worker "$worker_name"
        fi
    done
    
    log_success "All workers stopped"
}

# Show worker logs
show_logs() {
    local worker_name="$1"
    local lines="${2:-50}"
    
    if [[ -z "$worker_name" ]]; then
        log_error "Usage: show_logs <worker_name> [lines]"
        return 1
    fi
    
    local log_file="$WORKER_LOG_DIR/${worker_name}.log"
    
    if [[ -f "$log_file" ]]; then
        tail -n "$lines" "$log_file"
    else
        log_error "Log file not found: $log_file"
    fi
}

# Monitor workers
monitor_workers() {
    log_info "Monitoring workers (press Ctrl+C to stop)..."
    
    while true; do
        clear
        echo "Pixelated Empathy AI - Worker Monitor"
        echo "====================================="
        echo "Time: $(date)"
        echo ""
        
        show_worker_status
        
        echo ""
        echo "Queue Status:"
        echo "============="
        
        # Show queue lengths using Celery inspect
        celery -A "$CELERY_APP" inspect active_queues 2>/dev/null || echo "Could not retrieve queue information"
        
        sleep 5
    done
}

# Scale workers
scale_workers() {
    local queue="$1"
    local count="$2"
    
    if [[ -z "$queue" || -z "$count" ]]; then
        log_error "Usage: scale_workers <queue> <count>"
        return 1
    fi
    
    log_info "Scaling $queue workers to $count instances..."
    
    # Stop existing workers for this queue
    for pid_file in "$WORKER_PID_DIR"/*.pid; do
        if [[ -f "$pid_file" ]]; then
            local worker_name=$(basename "$pid_file" .pid)
            if [[ "$worker_name" == *"$queue"* ]]; then
                stop_worker "$worker_name"
            fi
        fi
    done
    
    # Start new workers
    for ((i=1; i<=count; i++)); do
        start_worker "${queue}_${i}" "$queue" 4 1000
    done
    
    log_success "Scaled $queue workers to $count instances"
}

# Show usage
show_usage() {
    cat << EOF
Worker Management for Pixelated Empathy AI

Usage: $0 <command> [options]

Commands:
  start <name> <queue> [concurrency] [max_tasks]  Start a worker
  stop <name>                                     Stop a worker
  restart <name> <queue> [concurrency] [max_tasks] Restart a worker
  status [name]                                   Show worker status
  start-all                                       Start all default workers
  stop-all                                        Stop all workers
  logs <name> [lines]                            Show worker logs
  monitor                                         Monitor workers in real-time
  scale <queue> <count>                          Scale workers for a queue
  help                                           Show this help message

Examples:
  $0 start quality_worker quality_validation 4 1000
  $0 stop quality_worker
  $0 restart quality_worker quality_validation 4 1000
  $0 status quality_worker
  $0 start-all
  $0 logs quality_worker 100
  $0 scale quality_validation 3

Environment Variables:
  REDIS_URL          Redis connection URL (default: redis://localhost:6379/0)
  WORKER_LOG_DIR     Directory for worker logs (default: logs/workers)
  WORKER_PID_DIR     Directory for worker PID files (default: tmp/workers)
EOF
}

# Main execution
main() {
    create_directories
    
    case "${1:-help}" in
        start)
            check_prerequisites
            start_worker "$2" "$3" "$4" "$5"
            ;;
        stop)
            stop_worker "$2"
            ;;
        restart)
            check_prerequisites
            restart_worker "$2" "$3" "$4" "$5"
            ;;
        status)
            show_worker_status "$2"
            ;;
        start-all)
            check_prerequisites
            start_all_workers
            ;;
        stop-all)
            stop_all_workers
            ;;
        logs)
            show_logs "$2" "$3"
            ;;
        monitor)
            monitor_workers
            ;;
        scale)
            check_prerequisites
            scale_workers "$2" "$3"
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
