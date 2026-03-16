#!/bin/bash
# Pixelated Empathy AI - Production Deployment Script
# Phase 4.1: Enterprise Deployment Procedures & Infrastructure as Code

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENVIRONMENT="production"
AWS_REGION="${AWS_REGION:-us-east-1}"
CLUSTER_NAME="pixelated-empathy-production-cluster"

# Colors for output
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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required tools
    local required_tools=("aws" "docker" "jq" "curl")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    # Check required environment variables
    local required_vars=("ECR_REPOSITORY_URL" "BUILD_NUMBER")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Environment variable $var is required"
            exit 1
        fi
    done
    
    log_success "Prerequisites check passed"
}

# Get current deployment state
get_deployment_state() {
    log_info "Getting current deployment state..."
    
    # Get current active environment (blue or green)
    local listener_arn=$(aws elbv2 describe-listeners \
        --load-balancer-arn "$ALB_ARN" \
        --query 'Listeners[0].ListenerArn' \
        --output text)
    
    local current_target_group=$(aws elbv2 describe-listeners \
        --listener-arns "$listener_arn" \
        --query 'Listeners[0].DefaultActions[0].TargetGroupArn' \
        --output text)
    
    if [[ "$current_target_group" == *"blue"* ]]; then
        CURRENT_ENV="blue"
        NEW_ENV="green"
    else
        CURRENT_ENV="green"
        NEW_ENV="blue"
    fi
    
    log_info "Current environment: $CURRENT_ENV"
    log_info "Deploying to: $NEW_ENV"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    local image_tag="${ECR_REPOSITORY_URL}:${BUILD_NUMBER}"
    local latest_tag="${ECR_REPOSITORY_URL}:latest"
    
    # Build Docker image
    log_info "Building Docker image: $image_tag"
    docker build -t "$image_tag" -t "$latest_tag" "$PROJECT_ROOT"
    
    # Login to ECR
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "$ECR_REPOSITORY_URL"
    
    # Push images
    log_info "Pushing Docker image to ECR..."
    docker push "$image_tag"
    docker push "$latest_tag"
    
    log_success "Docker image built and pushed successfully"
}

# Update ECS task definition
update_task_definition() {
    log_info "Updating ECS task definition for $NEW_ENV environment..."
    
    local service_name="pixelated-empathy-api-${NEW_ENV}"
    local task_family="pixelated-empathy-api-${NEW_ENV}"
    
    # Get current task definition
    local current_task_def=$(aws ecs describe-task-definition \
        --task-definition "$task_family" \
        --query 'taskDefinition')
    
    # Update image in task definition
    local new_task_def=$(echo "$current_task_def" | jq --arg image "${ECR_REPOSITORY_URL}:${BUILD_NUMBER}" '
        .containerDefinitions[0].image = $image |
        del(.taskDefinitionArn, .revision, .status, .requiresAttributes, .placementConstraints, .compatibilities, .registeredAt, .registeredBy)
    ')
    
    # Register new task definition
    local new_task_arn=$(echo "$new_task_def" | aws ecs register-task-definition \
        --cli-input-json file:///dev/stdin \
        --query 'taskDefinition.taskDefinitionArn' \
        --output text)
    
    log_success "New task definition registered: $new_task_arn"
    echo "$new_task_arn"
}

# Deploy to ECS service
deploy_to_ecs() {
    local task_definition_arn="$1"
    local service_name="pixelated-empathy-api-${NEW_ENV}"
    
    log_info "Deploying to ECS service: $service_name"
    
    # Update ECS service
    aws ecs update-service \
        --cluster "$CLUSTER_NAME" \
        --service "$service_name" \
        --task-definition "$task_definition_arn" \
        --desired-count 3 > /dev/null
    
    # Wait for deployment to complete
    log_info "Waiting for deployment to complete..."
    aws ecs wait services-stable \
        --cluster "$CLUSTER_NAME" \
        --services "$service_name"
    
    log_success "ECS service deployment completed"
}

# Run database migrations
run_database_migrations() {
    log_info "Running database migrations..."
    
    local task_definition="pixelated-empathy-migration"
    local subnet_ids=$(aws ec2 describe-subnets \
        --filters "Name=tag:Name,Values=*private*" \
        --query 'Subnets[].SubnetId' \
        --output text | tr '\t' ',')
    
    # Run migration task
    local task_arn=$(aws ecs run-task \
        --cluster "$CLUSTER_NAME" \
        --task-definition "$task_definition" \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[$subnet_ids],assignPublicIp=DISABLED}" \
        --query 'tasks[0].taskArn' \
        --output text)
    
    # Wait for migration to complete
    log_info "Waiting for database migration to complete..."
    aws ecs wait tasks-stopped \
        --cluster "$CLUSTER_NAME" \
        --tasks "$task_arn"
    
    # Check migration result
    local exit_code=$(aws ecs describe-tasks \
        --cluster "$CLUSTER_NAME" \
        --tasks "$task_arn" \
        --query 'tasks[0].containers[0].exitCode' \
        --output text)
    
    if [[ "$exit_code" != "0" ]]; then
        log_error "Database migration failed with exit code: $exit_code"
        exit 1
    fi
    
    log_success "Database migrations completed successfully"
}

# Health check function
health_check() {
    local endpoint="$1"
    local max_attempts=30
    local attempt=1
    
    log_info "Performing health check on: $endpoint"
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$endpoint/health" > /dev/null; then
            log_success "Health check passed"
            return 0
        fi
        
        log_info "Health check attempt $attempt/$max_attempts failed, retrying in 10 seconds..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Switch traffic with canary deployment
switch_traffic() {
    log_info "Starting canary deployment..."
    
    local listener_arn=$(aws elbv2 describe-listeners \
        --load-balancer-arn "$ALB_ARN" \
        --query 'Listeners[0].ListenerArn' \
        --output text)
    
    local current_target_group_arn=$(aws elbv2 describe-target-groups \
        --names "pixelated-empathy-${CURRENT_ENV}" \
        --query 'TargetGroups[0].TargetGroupArn' \
        --output text)
    
    local new_target_group_arn=$(aws elbv2 describe-target-groups \
        --names "pixelated-empathy-${NEW_ENV}" \
        --query 'TargetGroups[0].TargetGroupArn' \
        --output text)
    
    # Canary deployment stages
    local stages=(5 25 50 100)
    
    for stage in "${stages[@]}"; do
        local current_weight=$((100 - stage))
        local new_weight=$stage
        
        log_info "Switching ${stage}% traffic to $NEW_ENV environment..."
        
        if [[ $stage -eq 100 ]]; then
            # Final switch - 100% to new environment
            aws elbv2 modify-listener \
                --listener-arn "$listener_arn" \
                --default-actions Type=forward,TargetGroupArn="$new_target_group_arn" > /dev/null
        else
            # Weighted routing
            aws elbv2 modify-listener \
                --listener-arn "$listener_arn" \
                --default-actions Type=forward,ForwardConfig="{
                    \"TargetGroups\":[
                        {\"TargetGroupArn\":\"$current_target_group_arn\",\"Weight\":$current_weight},
                        {\"TargetGroupArn\":\"$new_target_group_arn\",\"Weight\":$new_weight}
                    ]
                }" > /dev/null
        fi
        
        # Monitor metrics for 5 minutes
        log_info "Monitoring metrics for 5 minutes..."
        if ! monitor_metrics 300; then
            log_error "Metrics check failed, initiating rollback..."
            rollback_deployment
            exit 1
        fi
        
        log_success "${stage}% traffic switch completed successfully"
    done
    
    log_success "Traffic switch completed - 100% traffic on $NEW_ENV"
}

# Monitor key metrics
monitor_metrics() {
    local duration="$1"
    local end_time=$(($(date +%s) + duration))
    
    while [[ $(date +%s) -lt $end_time ]]; do
        # Check error rate
        local error_rate=$(aws cloudwatch get-metric-statistics \
            --namespace AWS/ApplicationELB \
            --metric-name HTTPCode_Target_5XX_Count \
            --dimensions Name=LoadBalancer,Value="$ALB_NAME" \
            --start-time "$(date -u -d '5 minutes ago' +%Y-%m-%dT%H:%M:%S)" \
            --end-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
            --period 300 \
            --statistics Sum \
            --query 'Datapoints[0].Sum' \
            --output text 2>/dev/null || echo "0")
        
        # Check response time
        local response_time=$(aws cloudwatch get-metric-statistics \
            --namespace AWS/ApplicationELB \
            --metric-name TargetResponseTime \
            --dimensions Name=LoadBalancer,Value="$ALB_NAME" \
            --start-time "$(date -u -d '5 minutes ago' +%Y-%m-%dT%H:%M:%S)" \
            --end-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
            --period 300 \
            --statistics Average \
            --query 'Datapoints[0].Average' \
            --output text 2>/dev/null || echo "0")
        
        # Check thresholds
        if (( $(echo "$error_rate > 10" | bc -l) )); then
            log_error "Error rate too high: $error_rate"
            return 1
        fi
        
        if (( $(echo "$response_time > 0.5" | bc -l) )); then
            log_error "Response time too high: ${response_time}s"
            return 1
        fi
        
        log_info "Metrics OK - Error rate: $error_rate, Response time: ${response_time}s"
        sleep 30
    done
    
    return 0
}

# Rollback deployment
rollback_deployment() {
    log_warning "Initiating rollback to $CURRENT_ENV environment..."
    
    local listener_arn=$(aws elbv2 describe-listeners \
        --load-balancer-arn "$ALB_ARN" \
        --query 'Listeners[0].ListenerArn' \
        --output text)
    
    local current_target_group_arn=$(aws elbv2 describe-target-groups \
        --names "pixelated-empathy-${CURRENT_ENV}" \
        --query 'TargetGroups[0].TargetGroupArn' \
        --output text)
    
    # Switch 100% traffic back to current environment
    aws elbv2 modify-listener \
        --listener-arn "$listener_arn" \
        --default-actions Type=forward,TargetGroupArn="$current_target_group_arn" > /dev/null
    
    # Scale down new environment
    aws ecs update-service \
        --cluster "$CLUSTER_NAME" \
        --service "pixelated-empathy-api-${NEW_ENV}" \
        --desired-count 0 > /dev/null
    
    # Send alert
    aws sns publish \
        --topic-arn "$ALERT_TOPIC_ARN" \
        --message "ROLLBACK COMPLETED: Production deployment rolled back to $CURRENT_ENV environment" \
        --subject "Production Deployment Rollback" > /dev/null
    
    log_success "Rollback completed successfully"
}

# Cleanup old environment
cleanup_old_environment() {
    log_info "Scaling down old environment: $CURRENT_ENV"
    
    # Scale down old environment
    aws ecs update-service \
        --cluster "$CLUSTER_NAME" \
        --service "pixelated-empathy-api-${CURRENT_ENV}" \
        --desired-count 0 > /dev/null
    
    # Wait for scale down
    aws ecs wait services-stable \
        --cluster "$CLUSTER_NAME" \
        --services "pixelated-empathy-api-${CURRENT_ENV}"
    
    log_success "Old environment scaled down successfully"
}

# Send deployment notification
send_notification() {
    local status="$1"
    local message="Production deployment $status - Build: $BUILD_NUMBER, Environment: $NEW_ENV"
    
    # Send SNS notification
    aws sns publish \
        --topic-arn "$ALERT_TOPIC_ARN" \
        --message "$message" \
        --subject "Production Deployment $status" > /dev/null
    
    # Send Slack notification (if webhook configured)
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$SLACK_WEBHOOK_URL" > /dev/null
    fi
    
    log_info "Deployment notification sent: $status"
}

# Main deployment function
main() {
    log_info "Starting production deployment..."
    log_info "Build Number: $BUILD_NUMBER"
    log_info "Environment: $ENVIRONMENT"
    log_info "AWS Region: $AWS_REGION"
    
    # Load configuration
    source "$SCRIPT_DIR/config/production.env"
    
    # Run deployment steps
    check_prerequisites
    get_deployment_state
    build_and_push_image
    
    local task_definition_arn
    task_definition_arn=$(update_task_definition)
    
    run_database_migrations
    deploy_to_ecs "$task_definition_arn"
    
    # Health check new environment
    local new_endpoint="https://${NEW_ENV}.pixelatedempathy.com"
    if ! health_check "$new_endpoint"; then
        log_error "Health check failed, aborting deployment"
        send_notification "FAILED"
        exit 1
    fi
    
    # Switch traffic with canary deployment
    switch_traffic
    
    # Cleanup old environment
    cleanup_old_environment
    
    # Send success notification
    send_notification "SUCCESS"
    
    log_success "Production deployment completed successfully!"
    log_info "New environment: $NEW_ENV"
    log_info "Build number: $BUILD_NUMBER"
    log_info "Deployment time: $(date)"
}

# Error handling
trap 'log_error "Deployment failed at line $LINENO"' ERR

# Run main function
main "$@"
