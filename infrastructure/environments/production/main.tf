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
      Environment = "production"
      Project     = "pixelated-empathy"
      ManagedBy   = "terraform"
      Owner       = "devops-team"
      CostCenter  = "engineering"
      Compliance  = "hipaa-sox2-gdpr"
    }
  }
}

provider "aws" {
  alias  = "backup"
  region = var.backup_region
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
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"

  # Read replica configuration
  create_read_replica = true
  read_replica_count  = 2

  # ElastiCache Redis configuration
  redis_node_type       = var.redis_node_type
  redis_num_cache_nodes = var.redis_num_cache_nodes
  redis_parameter_group = "default.redis7"
  redis_engine_version  = "7.0"

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
      desired_count          = var.api_desired_count
      min_capacity           = var.api_min_capacity
      max_capacity           = var.api_max_capacity

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
            command  = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
            interval = 30
            timeout  = 5
            retries  = 3
          }
        }
      ]

      # Auto-scaling configuration
      auto_scaling = {
        target_cpu_utilization    = 70
        target_memory_utilization = 80
        scale_up_cooldown         = 300
        scale_down_cooldown       = 300
      }

      # Load balancer target group
      target_group = {
        port                 = 8000
        protocol             = "HTTP"
        health_check_path    = "/health"
        health_check_matcher = "200"
      }
    }

    pixelated-empathy-worker = {
      task_definition_family = "pixelated-empathy-worker"
      desired_count          = var.worker_desired_count
      min_capacity           = var.worker_min_capacity
      max_capacity           = var.worker_max_capacity

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
        scale_up_cooldown         = 300
        scale_down_cooldown       = 300
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
  rds_instance_id  = module.database.rds_instance_id
  redis_cluster_id = module.database.redis_cluster_id

  # Load balancer for monitoring
  alb_arn = module.compute.alb_arn

  # SNS topics for alerting
  create_sns_topics = true
  alert_email       = var.alert_email

  # PagerDuty integration
  pagerduty_integration_key = var.pagerduty_integration_key

  # Datadog configuration
  datadog_api_key = var.datadog_api_key

  tags = local.common_tags
}

# CloudFront logging bucket for access logs
resource "aws_s3_bucket" "cloudfront_access_logs" {
  bucket = "${local.name_prefix}-cloudfront-access-logs"
}

resource "aws_s3_bucket_versioning" "cloudfront_access_logs" {
  bucket = aws_s3_bucket.cloudfront_access_logs.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "cloudfront_access_logs" {
  bucket = aws_s3_bucket.cloudfront_access_logs.id

  rule {
    id     = "cleanup-old-objects"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    expiration {
      days = 90
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket_notification" "cloudfront_access_logs" {
  bucket      = aws_s3_bucket.cloudfront_access_logs.id
  eventbridge = true
}

resource "aws_s3_bucket_public_access_block" "cloudfront_access_logs" {
  bucket = aws_s3_bucket.cloudfront_access_logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "cloudfront_access_logs" {
  bucket = aws_s3_bucket.cloudfront_access_logs.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = module.security.kms_key_id
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_logging" "cloudfront_access_logs" {
  bucket = aws_s3_bucket.cloudfront_access_logs.id

  target_bucket = aws_s3_bucket.cloudfront_access_logs.id
  target_prefix = "access-logs/"
}

resource "aws_s3_bucket" "cloudfront_access_logs_replica" {
  provider = aws.backup
  bucket   = "${local.name_prefix}-cloudfront-access-logs-replica"
}

resource "aws_s3_bucket_versioning" "cloudfront_access_logs_replica" {
  provider = aws.backup
  bucket   = aws_s3_bucket.cloudfront_access_logs_replica.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "cloudfront_access_logs_replica" {
  provider = aws.backup
  bucket   = aws_s3_bucket.cloudfront_access_logs_replica.id

  rule {
    id     = "cleanup-old-objects"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    expiration {
      days = 90
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket_notification" "cloudfront_access_logs_replica" {
  provider    = aws.backup
  bucket      = aws_s3_bucket.cloudfront_access_logs_replica.id
  eventbridge = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "cloudfront_access_logs_replica" {
  provider = aws.backup
  bucket   = aws_s3_bucket.cloudfront_access_logs_replica.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = "alias/aws/s3"
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_logging" "cloudfront_access_logs_replica" {
  provider = aws.backup
  bucket   = aws_s3_bucket.cloudfront_access_logs_replica.id

  target_bucket = aws_s3_bucket.cloudfront_access_logs_replica.id
  target_prefix = "access-logs/"
}

resource "aws_s3_bucket_public_access_block" "cloudfront_access_logs_replica" {
  provider                = aws.backup
  bucket                  = aws_s3_bucket.cloudfront_access_logs_replica.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_replication_configuration" "cloudfront_access_logs" {
  depends_on = [
    aws_iam_role_policy_attachment.s3_replication,
    aws_s3_bucket_versioning.cloudfront_access_logs,
    aws_s3_bucket_versioning.cloudfront_access_logs_replica
  ]

  bucket = aws_s3_bucket.cloudfront_access_logs.id
  role   = aws_iam_role.s3_replication.arn

  rule {
    id     = "replicate-to-backup-region"
    status = "Enabled"

    destination {
      bucket       = aws_s3_bucket.cloudfront_access_logs_replica.arn
      storage_class = "STANDARD_IA"
    }
  }
}

data "aws_iam_policy_document" "s3_replication_assume_role" {
  statement {
    sid     = "S3ReplicationAssumeRole"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["s3.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "s3_replication_access" {
  statement {
    sid = "S3ReplicationSourceAccess"
    actions = [
      "s3:GetReplicationConfiguration",
      "s3:ListBucket",
      "s3:GetBucketVersioning",
      "s3:PutReplicationConfiguration",
      "s3:GetObjectVersionForReplication",
      "s3:GetObjectVersionAcl",
      "s3:GetObjectVersionTagging",
      "s3:ListMultipartUploadParts",
      "s3:GetObjectVersion"
    ]

    resources = [
      aws_s3_bucket.cloudfront_access_logs.arn,
      "${aws_s3_bucket.cloudfront_access_logs.arn}/*",
      aws_s3_bucket.app_data.arn,
      "${aws_s3_bucket.app_data.arn}/*"
    ]
  }

  statement {
    sid = "S3ReplicationDestinationAccess"
    actions = [
      "s3:ReplicateObject",
      "s3:ReplicateDelete",
      "s3:ReplicateTags",
      "s3:ObjectOwnerOverrideToBucketOwner"
    ]

    resources = [
      "${aws_s3_bucket.cloudfront_access_logs_replica.arn}/*",
      "${aws_s3_bucket.app_data_replica.arn}/*"
    ]
  }
}

resource "aws_iam_role" "s3_replication" {
  name               = "${local.name_prefix}-s3-replication-role"
  assume_role_policy = data.aws_iam_policy_document.s3_replication_assume_role.json
}

resource "aws_iam_policy" "s3_replication" {
  name        = "${local.name_prefix}-s3-replication-policy"
  description = "Allow S3 cross-region replication for compliance logs and app data"
  policy      = data.aws_iam_policy_document.s3_replication_access.json
}

resource "aws_iam_role_policy_attachment" "s3_replication" {
  role       = aws_iam_role.s3_replication.name
  policy_arn = aws_iam_policy.s3_replication.arn
}

resource "aws_wafv2_web_acl" "static_assets" {
  name        = "${local.name_prefix}-cloudfront-waf"
  scope       = "CLOUDFRONT"
  description = "CloudFront WAF for static assets"

  default_action {
    allow {}
  }

  visibility_config {
    sampled_requests_enabled   = true
    cloudwatch_metrics_enabled = true
    metric_name                = "${local.name_prefix}-static-assets"
  }

  rule {
    name     = "AWSManagedKnownBadInputs"
    priority = 1

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      sampled_requests_enabled   = true
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.name_prefix}-waf-known-bad-inputs"
    }
  }

  rule {
    name     = "AWSManagedCommonRules"
    priority = 2

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      sampled_requests_enabled   = true
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.name_prefix}-waf-common"
    }
  }

  rule {
    name     = "AWSManagedAnonymousIpList"
    priority = 3

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesAnonymousIpList"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      sampled_requests_enabled   = true
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.name_prefix}-waf-anon-ip-list"
    }
  }
}

resource "aws_cloudwatch_log_group" "waf_static_assets" {
  name              = "/aws/wafv2/${local.name_prefix}/cloudfront-static-assets"
  retention_in_days = 365
  kms_key_id        = "alias/aws/logs"
}

resource "aws_wafv2_web_acl_logging_configuration" "static_assets" {
  log_destination_configs = [aws_cloudwatch_log_group.waf_static_assets.arn]
  resource_arn           = aws_wafv2_web_acl.static_assets.arn
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

resource "aws_s3_bucket" "app_data_replica" {
  #checkov:skip=CKV_AWS_144: "Replication target bucket for cross-region backup destination."
  provider = aws.backup
  bucket   = "${local.name_prefix}-app-data-replica"
}

resource "aws_s3_bucket_versioning" "app_data_replica" {
  provider = aws.backup
  bucket   = aws_s3_bucket.app_data_replica.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "app_data_replica" {
  provider = aws.backup
  bucket   = aws_s3_bucket.app_data_replica.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = "alias/aws/s3"
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_logging" "app_data_replica" {
  provider = aws.backup
  bucket   = aws_s3_bucket.app_data_replica.id

  target_bucket = aws_s3_bucket.app_data_replica.id
  target_prefix = "access-logs/"
}

resource "aws_s3_bucket_public_access_block" "app_data_replica" {
  provider                = aws.backup
  bucket                  = aws_s3_bucket.app_data_replica.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "app_data_replica" {
  provider = aws.backup
  bucket   = aws_s3_bucket.app_data_replica.id

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

resource "aws_s3_bucket_notification" "app_data_replica" {
  provider    = aws.backup
  bucket      = aws_s3_bucket.app_data_replica.id
  eventbridge = true
}

resource "aws_s3_bucket_replication_configuration" "app_data" {
  depends_on = [
    aws_iam_role_policy_attachment.s3_replication,
    aws_s3_bucket_versioning.app_data,
    aws_s3_bucket_versioning.app_data_replica
  ]

  bucket = aws_s3_bucket.app_data.id
  role   = aws_iam_role.s3_replication.arn

  rule {
    id     = "replicate-to-backup-region"
    status = "Enabled"

    destination {
      bucket       = aws_s3_bucket.app_data_replica.arn
      storage_class = "STANDARD_IA"
    }
  }
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
  bucket      = aws_s3_bucket.app_data.id
  eventbridge = true
}

resource "aws_cloudfront_response_headers_policy" "security_headers" {
  name = "${local.name_prefix}-security-headers-policy"

  security_headers_config {
    content_type_options {
      override = true
    }

    frame_options {
      frame_option = "DENY"
      override     = true
    }

    referrer_policy {
      referrer_policy = "strict-origin-when-cross-origin"
      override        = true
    }

    strict_transport_security {
      access_control_max_age_sec = 31536000
      include_subdomains        = true
      preload                   = true
      override                  = true
    }

    xss_protection {
      mode_block = true
      override   = true
      protection = true
    }

    content_security_policy {
      content_security_policy = "default-src 'self'; object-src 'none'; frame-ancestors 'none'; upgrade-insecure-requests; block-all-mixed-content;"
      override               = true
    }
  }
}

# CloudFront distribution for static assets
resource "aws_cloudfront_distribution" "static_assets" {
  origin {
    domain_name = aws_s3_bucket.app_data.bucket_regional_domain_name
    origin_id   = "S3-${aws_s3_bucket.app_data.id}-primary"

    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.static_assets.cloudfront_access_identity_path
    }
  }

  origin {
    domain_name = aws_s3_bucket.app_data.bucket_domain_name
    origin_id   = "S3-${aws_s3_bucket.app_data.id}-secondary"

    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.static_assets.cloudfront_access_identity_path
    }
  }

  origin_group {
    origin_id = "origin-group-${aws_s3_bucket.app_data.id}"

    failover_criteria {
      status_codes = ["500", "502", "503", "504"]
    }

    member {
      origin_id = "S3-${aws_s3_bucket.app_data.id}-primary"
    }

    member {
      origin_id = "S3-${aws_s3_bucket.app_data.id}-secondary"
    }
  }

  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"
  web_acl_id          = aws_wafv2_web_acl.static_assets.arn

  logging_config {
    include_cookies = false
    bucket          = aws_s3_bucket.cloudfront_access_logs.bucket_domain_name
    prefix          = "access-logs/"
  }

  aliases = ["static.${var.domain_name}"]

  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "origin-group-${aws_s3_bucket.app_data.id}"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"
  response_headers_policy_id = aws_cloudfront_response_headers_policy.security_headers.id

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
      restriction_type = "whitelist"
      locations        = ["US", "CA", "GB"]
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
