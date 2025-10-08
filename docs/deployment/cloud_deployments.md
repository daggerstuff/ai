# Cloud Deployment Guides: Pixelated Empathy AI

**Cloud-specific deployment instructions for AWS, GCP, and Azure.**

## Table of Contents

1. [AWS Deployment](#aws-deployment)
2. [Google Cloud Platform](#google-cloud-platform)
3. [Microsoft Azure](#microsoft-azure)
4. [Multi-Cloud Strategy](#multi-cloud-strategy)
5. [Cost Optimization](#cost-optimization)

---

## AWS Deployment

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CloudFront    │    │   Application   │    │   RDS PostgreSQL│
│   (CDN/WAF)     │───▶│   Load Balancer │───▶│   (Multi-AZ)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                    ┌─────────────────┐
                    │   EKS Cluster   │
                    │   (Auto Scaling)│
                    └─────────────────┘
                                │
                    ┌─────────────────┐
                    │   ElastiCache   │
                    │   (Redis)       │
                    └─────────────────┘
```

### Infrastructure as Code (Terraform)

```hcl
# aws/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "pixelated-empathy-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  
  tags = {
    Environment = var.environment
    Project     = "pixelated-empathy"
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "pixelated-empathy"
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # EKS Managed Node Groups
  eks_managed_node_groups = {
    main = {
      min_size     = 1
      max_size     = 10
      desired_size = 3
      
      instance_types = ["m5.large"]
      capacity_type  = "ON_DEMAND"
      
      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "main"
      }
    }
  }
  
  tags = {
    Environment = var.environment
    Project     = "pixelated-empathy"
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "main" {
  identifier = "pixelated-empathy-db"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.r6g.large"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type         = "gp3"
  storage_encrypted    = true
  
  db_name  = "pixelated_empathy"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.database.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "pixelated-empathy-final-snapshot"
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  
  tags = {
    Name        = "Pixelated Empathy Database"
    Environment = var.environment
  }
}

# ElastiCache Redis
resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "pixelated-empathy-redis"
  description                = "Redis cluster for Pixelated Empathy AI"
  
  node_type                  = "cache.r6g.large"
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = 2
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Name        = "Pixelated Empathy Redis"
    Environment = var.environment
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "pixelated-empathy-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = module.vpc.public_subnets
  
  enable_deletion_protection = true
  
  tags = {
    Environment = var.environment
  }
}

# S3 Bucket for Storage
resource "aws_s3_bucket" "storage" {
  bucket = "pixelated-empathy-storage-${var.environment}"
  
  tags = {
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "storage" {
  bucket = aws_s3_bucket.storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "storage" {
  bucket = aws_s3_bucket.storage.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "main" {
  origin {
    domain_name = aws_lb.main.dns_name
    origin_id   = "ALB-${aws_lb.main.name}"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  enabled = true
  
  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "ALB-${aws_lb.main.name}"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"
    
    forwarded_values {
      query_string = true
      headers      = ["Authorization", "Content-Type"]
      
      cookies {
        forward = "none"
      }
    }
  }
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    acm_certificate_arn = aws_acm_certificate.main.arn
    ssl_support_method  = "sni-only"
  }
  
  tags = {
    Environment = var.environment
  }
}
```

### Deployment Script

```bash
#!/bin/bash
# aws/deploy.sh

set -e

echo "🚀 Deploying Pixelated Empathy AI to AWS..."

# Check prerequisites
command -v terraform >/dev/null 2>&1 || { echo "Terraform is required but not installed. Aborting." >&2; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed. Aborting." >&2; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "AWS CLI is required but not installed. Aborting." >&2; exit 1; }

# Set variables
export AWS_REGION=${AWS_REGION:-us-west-2}
export ENVIRONMENT=${ENVIRONMENT:-production}

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var="environment=$ENVIRONMENT" -var="aws_region=$AWS_REGION"

# Apply infrastructure
echo "Applying Terraform configuration..."
terraform apply -var="environment=$ENVIRONMENT" -var="aws_region=$AWS_REGION" -auto-approve

# Get EKS cluster credentials
aws eks update-kubeconfig --region $AWS_REGION --name pixelated-empathy

# Install AWS Load Balancer Controller
kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"

helm repo add eks https://aws.github.io/eks-charts
helm repo update

helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=pixelated-empathy \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller

# Deploy application
kubectl apply -f ../k8s/

# Wait for deployment
kubectl wait --for=condition=available --timeout=300s deployment/pixelated-empathy-api -n pixelated-empathy

# Get load balancer URL
ALB_URL=$(kubectl get ingress pixelated-empathy-ingress -n pixelated-empathy -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

echo "✅ AWS deployment completed!"
echo "Application URL: https://$ALB_URL"
echo "Health check: curl https://$ALB_URL/health"
```

---

## Google Cloud Platform

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Cloud CDN     │    │   Load Balancer │    │   Cloud SQL     │
│   (Global)      │───▶│   (Regional)    │───▶│   (PostgreSQL)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                    ┌─────────────────┐
                    │   GKE Cluster   │
                    │   (Autopilot)   │
                    └─────────────────┘
                                │
                    ┌─────────────────┐
                    │   Memorystore   │
                    │   (Redis)       │
                    └─────────────────┘
```

### Terraform Configuration

```hcl
# gcp/main.tf
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "container.googleapis.com",
    "sqladmin.googleapis.com",
    "redis.googleapis.com",
    "compute.googleapis.com"
  ])
  
  service = each.value
}

# GKE Autopilot Cluster
resource "google_container_cluster" "primary" {
  name     = "pixelated-empathy"
  location = var.region
  
  enable_autopilot = true
  
  ip_allocation_policy {
    cluster_ipv4_cidr_block  = "/17"
    services_ipv4_cidr_block = "/22"
  }
  
  depends_on = [google_project_service.apis]
}

# Cloud SQL PostgreSQL
resource "google_sql_database_instance" "main" {
  name             = "pixelated-empathy-db"
  database_version = "POSTGRES_15"
  region          = var.region
  
  settings {
    tier = "db-custom-2-8192"
    
    backup_configuration {
      enabled                        = true
      start_time                    = "03:00"
      point_in_time_recovery_enabled = true
      backup_retention_settings {
        retained_backups = 7
      }
    }
    
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.main.id
    }
    
    database_flags {
      name  = "max_connections"
      value = "200"
    }
  }
  
  depends_on = [google_project_service.apis]
}

# Memorystore Redis
resource "google_redis_instance" "cache" {
  name           = "pixelated-empathy-cache"
  tier           = "STANDARD_HA"
  memory_size_gb = 4
  region         = var.region
  
  authorized_network = google_compute_network.main.id
  
  redis_version = "REDIS_7_0"
  
  depends_on = [google_project_service.apis]
}

# VPC Network
resource "google_compute_network" "main" {
  name                    = "pixelated-empathy-network"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "main" {
  name          = "pixelated-empathy-subnet"
  ip_cidr_range = "10.0.0.0/16"
  region        = var.region
  network       = google_compute_network.main.id
  
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }
  
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/16"
  }
}

# Cloud Storage Bucket
resource "google_storage_bucket" "storage" {
  name     = "pixelated-empathy-storage-${var.project_id}"
  location = var.region
  
  versioning {
    enabled = true
  }
  
  encryption {
    default_kms_key_name = google_kms_crypto_key.storage.id
  }
}

# KMS Key for encryption
resource "google_kms_key_ring" "main" {
  name     = "pixelated-empathy-keyring"
  location = var.region
}

resource "google_kms_crypto_key" "storage" {
  name     = "storage-key"
  key_ring = google_kms_key_ring.main.id
}
```

### Deployment Script

```bash
#!/bin/bash
# gcp/deploy.sh

set -e

echo "🚀 Deploying Pixelated Empathy AI to Google Cloud..."

# Set variables
export PROJECT_ID=${PROJECT_ID:-pixelated-empathy}
export REGION=${REGION:-us-west1}

# Authenticate with Google Cloud
gcloud auth login
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable container.googleapis.com
gcloud services enable sqladmin.googleapis.com
gcloud services enable redis.googleapis.com

# Deploy infrastructure with Terraform
terraform init
terraform plan -var="project_id=$PROJECT_ID" -var="region=$REGION"
terraform apply -var="project_id=$PROJECT_ID" -var="region=$REGION" -auto-approve

# Get GKE credentials
gcloud container clusters get-credentials pixelated-empathy --region $REGION

# Deploy application
kubectl apply -f ../k8s/

# Wait for deployment
kubectl wait --for=condition=available --timeout=300s deployment/pixelated-empathy-api -n pixelated-empathy

echo "✅ GCP deployment completed!"
```

---

## Microsoft Azure

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Front Door    │    │   App Gateway   │    │   PostgreSQL    │
│   (Global CDN)  │───▶│   (Regional LB) │───▶│   (Flexible)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                    ┌─────────────────┐
                    │   AKS Cluster   │
                    │   (Auto Scale)  │
                    └─────────────────┘
                                │
                    ┌─────────────────┐
                    │   Redis Cache   │
                    │   (Premium)     │
                    └─────────────────┘
```

### Terraform Configuration

```hcl
# azure/main.tf
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "pixelated-empathy-rg"
  location = var.location
  
  tags = {
    Environment = var.environment
    Project     = "pixelated-empathy"
  }
}

# Virtual Network
resource "azurerm_virtual_network" "main" {
  name                = "pixelated-empathy-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
}

resource "azurerm_subnet" "aks" {
  name                 = "aks-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.1.0/24"]
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "main" {
  name                = "pixelated-empathy-aks"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "pixelatedempathy"
  
  default_node_pool {
    name       = "default"
    node_count = 3
    vm_size    = "Standard_D2_v2"
    
    enable_auto_scaling = true
    min_count          = 1
    max_count          = 10
    
    vnet_subnet_id = azurerm_subnet.aks.id
  }
  
  identity {
    type = "SystemAssigned"
  }
  
  network_profile {
    network_plugin = "azure"
  }
}

# PostgreSQL Flexible Server
resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "pixelated-empathy-db"
  resource_group_name    = azurerm_resource_group.main.name
  location              = azurerm_resource_group.main.location
  version               = "15"
  administrator_login    = var.db_username
  administrator_password = var.db_password
  
  storage_mb = 32768
  sku_name   = "GP_Standard_D2s_v3"
  
  backup_retention_days = 7
  
  high_availability {
    mode = "ZoneRedundant"
  }
}

# Redis Cache
resource "azurerm_redis_cache" "main" {
  name                = "pixelated-empathy-redis"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  capacity            = 2
  family              = "P"
  sku_name            = "Premium"
  
  redis_configuration {
    maxmemory_reserved = 200
    maxmemory_delta    = 200
    maxmemory_policy   = "allkeys-lru"
  }
}

# Storage Account
resource "azurerm_storage_account" "main" {
  name                     = "pixelatedempathystorage"
  resource_group_name      = azurerm_resource_group.main.name
  location                = azurerm_resource_group.main.location
  account_tier            = "Standard"
  account_replication_type = "GRS"
  
  blob_properties {
    versioning_enabled = true
  }
}

# Application Gateway
resource "azurerm_application_gateway" "main" {
  name                = "pixelated-empathy-appgw"
  resource_group_name = azurerm_resource_group.main.name
  location           = azurerm_resource_group.main.location
  
  sku {
    name     = "WAF_v2"
    tier     = "WAF_v2"
    capacity = 2
  }
  
  gateway_ip_configuration {
    name      = "gateway-ip-config"
    subnet_id = azurerm_subnet.gateway.id
  }
  
  frontend_port {
    name = "frontend-port"
    port = 80
  }
  
  frontend_ip_configuration {
    name                 = "frontend-ip-config"
    public_ip_address_id = azurerm_public_ip.gateway.id
  }
  
  backend_address_pool {
    name = "backend-pool"
  }
  
  backend_http_settings {
    name                  = "backend-http-settings"
    cookie_based_affinity = "Disabled"
    port                  = 80
    protocol              = "Http"
    request_timeout       = 60
  }
  
  http_listener {
    name                           = "http-listener"
    frontend_ip_configuration_name = "frontend-ip-config"
    frontend_port_name             = "frontend-port"
    protocol                       = "Http"
  }
  
  request_routing_rule {
    name                       = "routing-rule"
    rule_type                  = "Basic"
    http_listener_name         = "http-listener"
    backend_address_pool_name  = "backend-pool"
    backend_http_settings_name = "backend-http-settings"
  }
}
```

---

## Multi-Cloud Strategy

### Disaster Recovery Setup

```yaml
# multi-cloud/disaster-recovery.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: multi-cloud-config
data:
  primary_region: "us-west-2"
  backup_region: "us-east-1"
  failover_threshold: "5m"
  
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cross-region-backup
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: pixelatedempathy/backup-tool:latest
            env:
            - name: SOURCE_REGION
              value: "us-west-2"
            - name: TARGET_REGION
              value: "us-east-1"
            command:
            - /bin/sh
            - -c
            - |
              # Backup database
              pg_dump $DATABASE_URL | aws s3 cp - s3://backup-bucket/db-$(date +%Y%m%d).sql
              
              # Sync Redis data
              redis-cli --rdb /tmp/dump.rdb
              aws s3 cp /tmp/dump.rdb s3://backup-bucket/redis-$(date +%Y%m%d).rdb
              
              # Replicate to backup region
              aws s3 sync s3://primary-bucket s3://backup-bucket --source-region us-west-2 --region us-east-1
          restartPolicy: OnFailure
```

### Cost Optimization Strategies

```bash
#!/bin/bash
# scripts/cost-optimization.sh

echo "💰 Analyzing cloud costs and optimization opportunities..."

# AWS Cost Analysis
aws ce get-cost-and-usage \
  --time-period Start=2025-01-01,End=2025-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE

# Identify unused resources
aws ec2 describe-instances --query 'Reservations[*].Instances[?State.Name==`stopped`]'
aws rds describe-db-instances --query 'DBInstances[?DBInstanceStatus==`stopped`]'

# Right-sizing recommendations
aws compute-optimizer get-ec2-instance-recommendations
aws compute-optimizer get-rds-database-recommendations

# Reserved Instance opportunities
aws ce get-reservation-purchase-recommendation \
  --service EC2-Instance \
  --account-scope PAYER

echo "✅ Cost analysis completed. Review recommendations above."
```

---

**For detailed infrastructure configurations and advanced deployment scenarios, see our [Infrastructure as Code](infrastructure_as_code.md) and [Monitoring Setup](monitoring_setup.md) guides.**

*Cloud deployment guides are updated with each major release. Last updated: 2025-08-17*
