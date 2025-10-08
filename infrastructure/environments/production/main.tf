# Pixelated Empathy AI - Production Infrastructure
# Phase 4.1: Enterprise Deployment Procedures & Infrastructure as Code

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket         = "pixelated-empathy-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment   = "production"
      Project       = "pixelated-empathy"
      ManagedBy     = "terraform"
      Owner         = "devops-team"
      CostCenter    = "engineering"
      Compliance    = "hipaa-sox2-gdpr"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values
locals {
  name_prefix = "${var.project_name}-${var.environment}"
  
  common_tags = {
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "terraform"
  }
  
  azs = slice(data.aws_availability_zones.available.names, 0, 3)
}

# Networking Module
module "networking" {
  source = "../../modules/networking"
  
  name_prefix = local.name_prefix
  vpc_cidr    = var.vpc_cidr
  azs         = local.azs
  
  public_subnet_cidrs  = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  
  tags = local.common_tags
}

# Security Module
module "security" {
  source = "../../modules/security"
  
  name_prefix = local.name_prefix
  vpc_id      = module.networking.vpc_id
  
  # KMS keys for encryption
  create_kms_keys = true
  
  # WAF configuration
  enable_waf = true
  waf_rules = [
    "AWSManagedRulesCommonRuleSet",
    "AWSManagedRulesOWASPTop10RuleSet",
    "AWSManagedRulesKnownBadInputsRuleSet"
  ]
  
  tags = local.common_tags
}

# Database Module
module "database" {
  source = "../../modules/database"
  
  name_prefix = local.name_prefix
  vpc_id      = module.networking.vpc_id
  
  # Database subnet group
  db_subnet_group_subnet_ids = module.networking.private_subnet_ids
  
  # RDS PostgreSQL configuration
  db_instance_class    = var.db_instance_class
  db_allocated_storage = var.db_allocated_storage
  db_engine_version    = var.db_engine_version
  db_name              = var.db_name
  db_username          = var.db_username
  
  # Multi-AZ deployment for high availability
  multi_az = true
  
  # Backup configuration
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  # Read replica configuration
  create_read_replica = true
  read_replica_count  = 2
  
  # ElastiCache Redis configuration
  redis_node_type          = var.redis_node_type
  redis_num_cache_nodes    = var.redis_num_cache_nodes
  redis_parameter_group    = "default.redis7"
  redis_engine_version     = "7.0"
  
  # DynamoDB configuration
  create_dynamodb_tables = true
  
  # Encryption
  kms_key_id = module.security.kms_key_id
  
  tags = local.common_tags
}

# Compute Module
module "compute" {
  source = "../../modules/compute"
  
  name_prefix = local.name_prefix
  vpc_id      = module.networking.vpc_id
  
  # ECS Cluster configuration
  ecs_cluster_name = "${local.name_prefix}-cluster"
  
  # Subnets for ECS services
  private_subnet_ids = module.networking.private_subnet_ids
  public_subnet_ids  = module.networking.public_subnet_ids
  
  # Application Load Balancer
  alb_security_group_ids = [module.security.alb_security_group_id]
  
  # ECS Service configuration
  ecs_services = {
    pixelated-empathy-api = {
      task_definition_family = "pixelated-empathy-api"
      desired_count         = var.api_desired_count
      min_capacity          = var.api_min_capacity
      max_capacity          = var.api_max_capacity
      
      container_definitions = [
        {
          name  = "api"
          image = "${var.ecr_repository_url}:latest"
          
          portMappings = [
            {
              containerPort = 8000
              protocol      = "tcp"
            }
          ]
          
          environment = [
            {
              name  = "ENVIRONMENT"
              value = "production"
            },
            {
              name  = "DATABASE_URL"
              value = module.database.rds_endpoint
            },
            {
              name  = "REDIS_URL"
              value = module.database.redis_endpoint
            }
          ]
          
          secrets = [
            {
              name      = "DATABASE_PASSWORD"
              valueFrom = module.database.db_password_secret_arn
            },
            {
              name      = "JWT_SECRET"
              valueFrom = module.security.jwt_secret_arn
            }
          ]
          
          logConfiguration = {
            logDriver = "awslogs"
            options = {
              "awslogs-group"         = "/ecs/${local.name_prefix}-api"
              "awslogs-region"        = var.aws_region
              "awslogs-stream-prefix" = "ecs"
            }
          }
          
          healthCheck = {
            command = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
            interval = 30
            timeout = 5
            retries = 3
          }
        }
      ]
      
      # Auto-scaling configuration
      auto_scaling = {
        target_cpu_utilization    = 70
        target_memory_utilization = 80
        scale_up_cooldown        = 300
        scale_down_cooldown      = 300
      }
      
      # Load balancer target group
      target_group = {
        port                = 8000
        protocol            = "HTTP"
        health_check_path   = "/health"
        health_check_matcher = "200"
      }
    }
    
    pixelated-empathy-worker = {
      task_definition_family = "pixelated-empathy-worker"
      desired_count         = var.worker_desired_count
      min_capacity          = var.worker_min_capacity
      max_capacity          = var.worker_max_capacity
      
      container_definitions = [
        {
          name  = "worker"
          image = "${var.ecr_repository_url}:latest"
          
          command = ["python", "manage.py", "runworker"]
          
          environment = [
            {
              name  = "ENVIRONMENT"
              value = "production"
            },
            {
              name  = "DATABASE_URL"
              value = module.database.rds_endpoint
            },
            {
              name  = "REDIS_URL"
              value = module.database.redis_endpoint
            }
          ]
          
          secrets = [
            {
              name      = "DATABASE_PASSWORD"
              valueFrom = module.database.db_password_secret_arn
            }
          ]
          
          logConfiguration = {
            logDriver = "awslogs"
            options = {
              "awslogs-group"         = "/ecs/${local.name_prefix}-worker"
              "awslogs-region"        = var.aws_region
              "awslogs-stream-prefix" = "ecs"
            }
          }
        }
      ]
      
      # Auto-scaling configuration
      auto_scaling = {
        target_cpu_utilization    = 70
        target_memory_utilization = 80
        scale_up_cooldown        = 300
        scale_down_cooldown      = 300
      }
    }
  }
  
  # SSL certificate
  ssl_certificate_arn = var.ssl_certificate_arn
  
  # Domain configuration
  domain_name = var.domain_name
  
  tags = local.common_tags
}

# Monitoring Module
module "monitoring" {
  source = "../../modules/monitoring"
  
  name_prefix = local.name_prefix
  
  # CloudWatch configuration
  log_retention_days = 30
  
  # ECS cluster for monitoring
  ecs_cluster_name = module.compute.ecs_cluster_name
  
  # Database endpoints for monitoring
  rds_instance_id = module.database.rds_instance_id
  redis_cluster_id = module.database.redis_cluster_id
  
  # Load balancer for monitoring
  alb_arn = module.compute.alb_arn
  
  # SNS topics for alerting
  create_sns_topics = true
  alert_email = var.alert_email
  
  # PagerDuty integration
  pagerduty_integration_key = var.pagerduty_integration_key
  
  # Datadog configuration
  datadog_api_key = var.datadog_api_key
  
  tags = local.common_tags
}

# S3 buckets for application data
resource "aws_s3_bucket" "app_data" {
  bucket = "${local.name_prefix}-app-data"
}

resource "aws_s3_bucket_versioning" "app_data" {
  bucket = aws_s3_bucket.app_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = module.security.kms_key_id
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Add lifecycle configuration
resource "aws_s3_bucket_lifecycle_configuration" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  rule {
    id     = "cleanup-old-versions"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 90
    }

    expiration {
      days = 365
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# Add access logging
resource "aws_s3_bucket_logging" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  target_bucket = aws_s3_bucket.app_data.id
  target_prefix = "access-logs/"
}

# Add event notifications
resource "aws_s3_bucket_notification" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  eventbridge {
    events = ["s3:ObjectCreated:*", "s3:ObjectRemoved:*"]
  }
}

# CloudFront distribution for static assets
resource "aws_cloudfront_distribution" "static_assets" {
  origin {
    domain_name = aws_s3_bucket.app_data.bucket_regional_domain_name
    origin_id   = "S3-${aws_s3_bucket.app_data.id}"
    
    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.static_assets.cloudfront_access_identity_path
    }
  }
  
  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"
  
  aliases = ["static.${var.domain_name}"]
  
  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "S3-${aws_s3_bucket.app_data.id}"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"
    
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
    
    min_ttl     = 0
    default_ttl = 3600
    max_ttl     = 86400
  }
  
  price_class = "PriceClass_100"
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    acm_certificate_arn      = var.ssl_certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }
  
  tags = local.common_tags
}

resource "aws_cloudfront_origin_access_identity" "static_assets" {
  comment = "OAI for ${local.name_prefix} static assets"
}

# Route 53 DNS records
data "aws_route53_zone" "main" {
  name         = var.domain_name
  private_zone = false
}

resource "aws_route53_record" "api" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "api.${var.domain_name}"
  type    = "A"
  
  alias {
    name                   = module.compute.alb_dns_name
    zone_id                = module.compute.alb_zone_id
    evaluate_target_health = true
  }
}

resource "aws_route53_record" "static" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "static.${var.domain_name}"
  type    = "A"
  
  alias {
    name                   = aws_cloudfront_distribution.static_assets.domain_name
    zone_id                = aws_cloudfront_distribution.static_assets.hosted_zone_id
    evaluate_target_health = false
  }
}
