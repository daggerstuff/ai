#!/bin/bash

# Pixelated Empathy AI Helm Deployment Script
# Usage: ./deploy.sh [environment] [action]
# Environments: dev, staging, prod
# Actions: install, upgrade, uninstall, status

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHART_DIR="$SCRIPT_DIR/pixelated-empathy"

# Default values
ENVIRONMENT="${1:-dev}"
ACTION="${2:-upgrade}"
NAMESPACE="pixelated-empathy-${ENVIRONMENT}"
RELEASE_NAME="pixelated-empathy-${ENVIRONMENT}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Validate environment
validate_environment() {
    case $ENVIRONMENT in
        dev|staging|prod)
            log_info "Deploying to $ENVIRONMENT environment"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_info "Valid environments: dev, staging, prod"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_error "Helm is not installed. Please install Helm first."
        exit 1
    fi
    
    # Check if kubectl is installed and configured
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if we can connect to the cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace $NAMESPACE if it doesn't exist..."
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    log_success "Namespace $NAMESPACE is ready"
}

# Add required Helm repositories
add_helm_repos() {
    log_info "Adding required Helm repositories..."
    
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    
    helm repo update
    log_success "Helm repositories updated"
}

# Install or upgrade the release
install_or_upgrade() {
    local values_file="$CHART_DIR/values-${ENVIRONMENT}.yaml"
    
    if [[ ! -f "$values_file" ]]; then
        log_warning "Environment-specific values file not found: $values_file"
        log_info "Using default values.yaml"
        values_file="$CHART_DIR/values.yaml"
    fi
    
    log_info "Deploying Pixelated Empathy AI to $ENVIRONMENT..."
    
    # Check if release exists
    if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        log_info "Release exists, upgrading..."
        helm upgrade "$RELEASE_NAME" "$CHART_DIR" \
            --namespace "$NAMESPACE" \
            --values "$values_file" \
            --wait \
            --timeout 10m \
            --atomic
    else
        log_info "Installing new release..."
        helm install "$RELEASE_NAME" "$CHART_DIR" \
            --namespace "$NAMESPACE" \
            --values "$values_file" \
            --wait \
            --timeout 10m \
            --atomic \
            --create-namespace
    fi
    
    log_success "Deployment completed successfully"
}

# Uninstall the release
uninstall() {
    log_info "Uninstalling $RELEASE_NAME from $NAMESPACE..."
    
    if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        helm uninstall "$RELEASE_NAME" --namespace "$NAMESPACE"
        log_success "Release $RELEASE_NAME uninstalled"
    else
        log_warning "Release $RELEASE_NAME not found in namespace $NAMESPACE"
    fi
}

# Show status
show_status() {
    log_info "Showing status for $RELEASE_NAME in $NAMESPACE..."
    
    if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        helm status "$RELEASE_NAME" --namespace "$NAMESPACE"
        echo ""
        log_info "Pods status:"
        kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/instance=$RELEASE_NAME"
        echo ""
        log_info "Services:"
        kubectl get services -n "$NAMESPACE" -l "app.kubernetes.io/instance=$RELEASE_NAME"
    else
        log_warning "Release $RELEASE_NAME not found in namespace $NAMESPACE"
    fi
}

# Validate Helm chart
validate_chart() {
    log_info "Validating Helm chart..."
    
    helm lint "$CHART_DIR"
    
    local values_file="$CHART_DIR/values-${ENVIRONMENT}.yaml"
    if [[ -f "$values_file" ]]; then
        helm template "$RELEASE_NAME" "$CHART_DIR" \
            --values "$values_file" \
            --namespace "$NAMESPACE" > /dev/null
    else
        helm template "$RELEASE_NAME" "$CHART_DIR" \
            --namespace "$NAMESPACE" > /dev/null
    fi
    
    log_success "Chart validation passed"
}

# Main execution
main() {
    log_info "Starting Pixelated Empathy AI deployment process..."
    
    validate_environment
    check_prerequisites
    validate_chart
    
    case $ACTION in
        install|upgrade)
            add_helm_repos
            create_namespace
            install_or_upgrade
            show_status
            ;;
        uninstall)
            uninstall
            ;;
        status)
            show_status
            ;;
        *)
            log_error "Invalid action: $ACTION"
            log_info "Valid actions: install, upgrade, uninstall, status"
            exit 1
            ;;
    esac
    
    log_success "Deployment process completed!"
}

# Show usage if no arguments provided
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [environment] [action]"
    echo ""
    echo "Environments:"
    echo "  dev      - Development environment"
    echo "  staging  - Staging environment"
    echo "  prod     - Production environment"
    echo ""
    echo "Actions:"
    echo "  install  - Install new release"
    echo "  upgrade  - Upgrade existing release (default)"
    echo "  uninstall- Uninstall release"
    echo "  status   - Show release status"
    echo ""
    echo "Examples:"
    echo "  $0 dev install"
    echo "  $0 staging upgrade"
    echo "  $0 prod status"
    exit 0
fi

# Run main function
main
