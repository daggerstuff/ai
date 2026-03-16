#!/bin/bash

# Deploy to Existing Azure Infrastructure
# Uses your existing Container Apps environment and resources

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Your existing Azure configuration
RESOURCE_GROUP="pixelated-rg"
CONTAINER_ENV="pixel-env-production"
CONTAINER_REGISTRY="pixelatedcr"
LOG_WORKSPACE="pixel-log-production"
LOCATION="eastus"

echo -e "${BLUE}🚀 DEPLOYING TO EXISTING AZURE INFRASTRUCTURE${NC}"
echo -e "${BLUE}Resource Group: ${RESOURCE_GROUP}${NC}"
echo -e "${BLUE}Container Environment: ${CONTAINER_ENV}${NC}"
echo -e "${BLUE}Container Registry: ${CONTAINER_REGISTRY}${NC}"

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

# Step 1: Create PostgreSQL Container App
echo -e "\n${BLUE}🗄️ DEPLOYING POSTGRESQL DATABASE${NC}"

az containerapp create \
  --name pixelated-postgres \
  --resource-group $RESOURCE_GROUP \
  --environment $CONTAINER_ENV \
  --image postgres:15 \
  --target-port 5432 \
  --ingress internal \
  --min-replicas 1 \
  --max-replicas 1 \
  --cpu 1.0 \
  --memory 2.0Gi \
  --env-vars \
    POSTGRES_DB=pixelated_empathy \
    POSTGRES_USER=postgres \
    POSTGRES_PASSWORD=pixelated_empathy_prod_2025 \
    PGDATA=/var/lib/postgresql/data/pgdata

print_status "PostgreSQL Container App created"

# Step 2: Create Redis Container App (for caching/queues)
echo -e "\n${BLUE}🔄 DEPLOYING REDIS CACHE${NC}"

az containerapp create \
  --name pixelated-redis \
  --resource-group $RESOURCE_GROUP \
  --environment $CONTAINER_ENV \
  --image redis:7-alpine \
  --target-port 6379 \
  --ingress internal \
  --min-replicas 1 \
  --max-replicas 1 \
  --cpu 0.5 \
  --memory 1.0Gi

print_status "Redis Container App created"

# Step 3: Create Prometheus Container App
echo -e "\n${BLUE}📊 DEPLOYING PROMETHEUS MONITORING${NC}"

# Create Prometheus config as a secret
PROMETHEUS_CONFIG=$(cat << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'pixelated-web'
    static_configs:
      - targets: ['pixelated-web:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
EOF
)

# Create Prometheus container app
az containerapp create \
  --name pixelated-prometheus \
  --resource-group $RESOURCE_GROUP \
  --environment $CONTAINER_ENV \
  --image prom/prometheus:latest \
  --target-port 9090 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 1 \
  --cpu 0.5 \
  --memory 1.0Gi

print_status "Prometheus Container App created"

# Step 4: Create Grafana Container App
echo -e "\n${BLUE}📈 DEPLOYING GRAFANA DASHBOARDS${NC}"

az containerapp create \
  --name pixelated-grafana \
  --resource-group $RESOURCE_GROUP \
  --environment $CONTAINER_ENV \
  --image grafana/grafana:latest \
  --target-port 3000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 1 \
  --cpu 0.5 \
  --memory 1.0Gi \
  --env-vars \
    GF_SECURITY_ADMIN_PASSWORD=pixelated_admin_2025 \
    GF_USERS_ALLOW_SIGN_UP=false

print_status "Grafana Container App created"

# Step 5: Create Data Migration Job
echo -e "\n${BLUE}🔄 CREATING DATA MIGRATION JOB${NC}"

az containerapp job create \
  --name pixelated-data-migration \
  --resource-group $RESOURCE_GROUP \
  --environment $CONTAINER_ENV \
  --image python:3.11-slim \
  --cpu 1.0 \
  --memory 2.0Gi \
  --replica-timeout 3600 \
  --replica-retry-limit 1 \
  --trigger-type Manual \
  --parallelism 1 \
  --completion-count 1 \
  --env-vars \
    POSTGRES_HOST=pixelated-postgres \
    POSTGRES_PORT=5432 \
    POSTGRES_USER=postgres \
    POSTGRES_PASSWORD=pixelated_empathy_prod_2025 \
    POSTGRES_DB=pixelated_empathy

print_status "Data migration job created"

# Step 6: Get service endpoints
echo -e "\n${BLUE}🌐 GETTING SERVICE ENDPOINTS${NC}"

# Get Container App URLs
POSTGRES_FQDN=$(az containerapp show --name pixelated-postgres --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)
PROMETHEUS_FQDN=$(az containerapp show --name pixelated-prometheus --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)
GRAFANA_FQDN=$(az containerapp show --name pixelated-grafana --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)
REDIS_FQDN=$(az containerapp show --name pixelated-redis --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)

echo -e "${GREEN}📊 SERVICE ENDPOINTS:${NC}"
echo -e "${GREEN}PostgreSQL (internal): ${POSTGRES_FQDN}:5432${NC}"
echo -e "${GREEN}Redis (internal): ${REDIS_FQDN}:6379${NC}"
echo -e "${GREEN}Prometheus: https://${PROMETHEUS_FQDN}${NC}"
echo -e "${GREEN}Grafana: https://${GRAFANA_FQDN}${NC}"
echo -e "${GREEN}  - Username: admin${NC}"
echo -e "${GREEN}  - Password: pixelated_admin_2025${NC}"

# Step 7: Update existing pixelated-web app with database connection
echo -e "\n${BLUE}🔗 UPDATING EXISTING WEB APP WITH DATABASE CONNECTION${NC}"

az containerapp update \
  --name pixelated-web \
  --resource-group $RESOURCE_GROUP \
  --set-env-vars \
    DATABASE_URL="postgresql://postgres:pixelated_empathy_prod_2025@${POSTGRES_FQDN}:5432/pixelated_empathy" \
    REDIS_URL="redis://${REDIS_FQDN}:6379/0"

print_status "Web app updated with database connections"

echo -e "\n${GREEN}🎉 DEPLOYMENT TO EXISTING AZURE INFRASTRUCTURE COMPLETED!${NC}"
echo -e "${GREEN}Your Pixelated Empathy infrastructure is now running on Azure Container Apps.${NC}"

echo -e "\n${BLUE}📋 NEXT STEPS:${NC}"
echo -e "${YELLOW}1. Run data migration job to import your conversations${NC}"
echo -e "${YELLOW}2. Access Grafana to set up monitoring dashboards${NC}"
echo -e "${YELLOW}3. Configure custom domains for external services${NC}"
echo -e "${YELLOW}4. Set up automated backups${NC}"

echo -e "\n${BLUE}🔧 USEFUL COMMANDS:${NC}"
echo -e "${GREEN}# Check container app status:${NC}"
echo -e "az containerapp list --resource-group $RESOURCE_GROUP --output table"
echo -e "${GREEN}# View logs:${NC}"
echo -e "az containerapp logs show --name pixelated-postgres --resource-group $RESOURCE_GROUP"
echo -e "${GREEN}# Run data migration:${NC}"
echo -e "az containerapp job start --name pixelated-data-migration --resource-group $RESOURCE_GROUP"
