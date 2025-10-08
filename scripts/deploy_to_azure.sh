#!/bin/bash

# Azure Kubernetes Deployment Script
# Deploys database, monitoring, and data migration to Azure AKS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RESOURCE_GROUP=${AZURE_RESOURCE_GROUP:-"pixelated-empathy-rg"}
AKS_CLUSTER=${AKS_CLUSTER_NAME:-"pixelated-empathy-aks"}
LOCATION=${AZURE_LOCATION:-"eastus"}

echo -e "${BLUE}🚀 STARTING AZURE KUBERNETES DEPLOYMENT${NC}"
echo -e "${BLUE}Resource Group: ${RESOURCE_GROUP}${NC}"
echo -e "${BLUE}AKS Cluster: ${AKS_CLUSTER}${NC}"
echo -e "${BLUE}Location: ${LOCATION}${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
echo -e "\n${BLUE}🔍 CHECKING PREREQUISITES${NC}"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    print_error "Azure CLI is not installed. Please install it first."
    exit 1
fi
print_status "Azure CLI is installed"

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed. Please install it first."
    exit 1
fi
print_status "kubectl is installed"

# Check if logged into Azure
if ! az account show &> /dev/null; then
    print_error "Not logged into Azure. Please run 'az login' first."
    exit 1
fi
print_status "Logged into Azure"

# Get AKS credentials
echo -e "\n${BLUE}🔑 GETTING AKS CREDENTIALS${NC}"
az aks get-credentials --resource-group $RESOURCE_GROUP --name $AKS_CLUSTER --overwrite-existing
print_status "AKS credentials configured"

# Verify cluster connection
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster"
    exit 1
fi
print_status "Connected to Kubernetes cluster"

# Deploy database infrastructure
echo -e "\n${BLUE}🗄️ DEPLOYING DATABASE INFRASTRUCTURE${NC}"

# Apply database manifests
kubectl apply -f k8s/database-production.yaml
print_status "Database manifests applied"

# Wait for PostgreSQL to be ready
echo -e "${YELLOW}⏳ Waiting for PostgreSQL to be ready...${NC}"
kubectl wait --for=condition=ready pod -l app=postgres -n pixelated-empathy-data --timeout=300s
print_status "PostgreSQL is ready"

# Check if database initialization job completed
echo -e "${YELLOW}⏳ Waiting for database initialization...${NC}"
kubectl wait --for=condition=complete job/postgres-init -n pixelated-empathy-data --timeout=300s
print_status "Database initialization completed"

# Deploy monitoring infrastructure
echo -e "\n${BLUE}📊 DEPLOYING MONITORING INFRASTRUCTURE${NC}"

# Apply monitoring manifests
kubectl apply -f k8s/monitoring-production.yaml
print_status "Monitoring manifests applied"

# Wait for Prometheus to be ready
echo -e "${YELLOW}⏳ Waiting for Prometheus to be ready...${NC}"
kubectl wait --for=condition=ready pod -l app=prometheus -n pixelated-empathy-monitoring --timeout=300s
print_status "Prometheus is ready"

# Wait for Grafana to be ready
echo -e "${YELLOW}⏳ Waiting for Grafana to be ready...${NC}"
kubectl wait --for=condition=ready pod -l app=grafana -n pixelated-empathy-monitoring --timeout=300s
print_status "Grafana is ready"

# Upload local backup data to Azure (if exists)
echo -e "\n${BLUE}📦 CHECKING FOR LOCAL BACKUP DATA${NC}"

if [ -f "backups/emergency_backup_$(date +%Y%m%d)*.tar.gz" ]; then
    print_warning "Local backup found. You may want to upload it to Azure Storage for migration."
    echo -e "${YELLOW}Consider running:${NC}"
    echo -e "${YELLOW}az storage blob upload --account-name <storage-account> --container-name backups --file backups/*.tar.gz --name backup.tar.gz${NC}"
else
    print_warning "No local backup found. Migration will look for mounted data volumes."
fi

# Deploy data migration job
echo -e "\n${BLUE}🔄 DEPLOYING DATA MIGRATION JOB${NC}"

# Apply migration job
kubectl apply -f k8s/data-migration-job.yaml
print_status "Data migration job created"

# Monitor migration job
echo -e "${YELLOW}⏳ Monitoring data migration progress...${NC}"
kubectl logs -f job/data-migration -n pixelated-empathy-data &
LOGS_PID=$!

# Wait for migration job to complete (with timeout)
if kubectl wait --for=condition=complete job/data-migration -n pixelated-empathy-data --timeout=1800s; then
    print_status "Data migration completed successfully"
else
    print_warning "Data migration timed out or failed. Check logs for details."
fi

# Kill the logs process
kill $LOGS_PID 2>/dev/null || true

# Get service endpoints
echo -e "\n${BLUE}🌐 GETTING SERVICE ENDPOINTS${NC}"

# Get PostgreSQL service endpoint
POSTGRES_IP=$(kubectl get service postgres-service -n pixelated-empathy-data -o jsonpath='{.spec.clusterIP}')
echo -e "${GREEN}PostgreSQL: ${POSTGRES_IP}:5432${NC}"

# Get Prometheus service endpoint
PROMETHEUS_IP=$(kubectl get service prometheus-service -n pixelated-empathy-monitoring -o jsonpath='{.spec.clusterIP}')
echo -e "${GREEN}Prometheus: ${PROMETHEUS_IP}:9090${NC}"

# Get Grafana service endpoint
GRAFANA_IP=$(kubectl get service grafana-service -n pixelated-empathy-monitoring -o jsonpath='{.spec.clusterIP}')
echo -e "${GREEN}Grafana: ${GRAFANA_IP}:3000${NC}"

# Port forwarding instructions
echo -e "\n${BLUE}🔗 PORT FORWARDING INSTRUCTIONS${NC}"
echo -e "${YELLOW}To access services locally, run these commands:${NC}"
echo -e "${GREEN}# PostgreSQL:${NC}"
echo -e "kubectl port-forward service/postgres-service 5432:5432 -n pixelated-empathy-data"
echo -e "${GREEN}# Prometheus:${NC}"
echo -e "kubectl port-forward service/prometheus-service 9090:9090 -n pixelated-empathy-monitoring"
echo -e "${GREEN}# Grafana:${NC}"
echo -e "kubectl port-forward service/grafana-service 3000:3000 -n pixelated-empathy-monitoring"

# Verify deployment
echo -e "\n${BLUE}✅ VERIFYING DEPLOYMENT${NC}"

# Check database
echo -e "${YELLOW}Checking database...${NC}"
if kubectl exec -n pixelated-empathy-data deployment/postgres -- pg_isready -U postgres; then
    print_status "Database is healthy"
else
    print_error "Database health check failed"
fi

# Check monitoring
echo -e "${YELLOW}Checking monitoring...${NC}"
if kubectl get pods -n pixelated-empathy-monitoring | grep -q "Running"; then
    print_status "Monitoring stack is running"
else
    print_error "Monitoring stack health check failed"
fi

# Get final statistics
echo -e "\n${BLUE}📊 DEPLOYMENT STATISTICS${NC}"

# Database statistics
echo -e "${YELLOW}Getting database statistics...${NC}"
kubectl exec -n pixelated-empathy-data deployment/postgres -- psql -U postgres -d pixelated_empathy -c "
SELECT 
    'Conversations' as table_name, COUNT(*) as count 
FROM conversations
UNION ALL
SELECT 
    'Messages' as table_name, COUNT(*) as count 
FROM messages;
" 2>/dev/null || print_warning "Could not retrieve database statistics"

# Resource usage
echo -e "${YELLOW}Resource usage:${NC}"
kubectl top nodes 2>/dev/null || print_warning "Could not retrieve node metrics"

echo -e "\n${GREEN}🎉 AZURE DEPLOYMENT COMPLETED SUCCESSFULLY!${NC}"
echo -e "${GREEN}Your Pixelated Empathy infrastructure is now running on Azure Kubernetes Service.${NC}"

echo -e "\n${BLUE}📋 NEXT STEPS:${NC}"
echo -e "${YELLOW}1. Set up ingress controllers for external access${NC}"
echo -e "${YELLOW}2. Configure SSL certificates${NC}"
echo -e "${YELLOW}3. Set up Azure Monitor integration${NC}"
echo -e "${YELLOW}4. Configure backup schedules${NC}"
echo -e "${YELLOW}5. Deploy your application containers${NC}"

echo -e "\n${BLUE}🔧 USEFUL COMMANDS:${NC}"
echo -e "${GREEN}# Check all pods:${NC}"
echo -e "kubectl get pods --all-namespaces"
echo -e "${GREEN}# Check services:${NC}"
echo -e "kubectl get services --all-namespaces"
echo -e "${GREEN}# Check logs:${NC}"
echo -e "kubectl logs -f deployment/postgres -n pixelated-empathy-data"
echo -e "kubectl logs -f deployment/prometheus -n pixelated-empathy-monitoring"
