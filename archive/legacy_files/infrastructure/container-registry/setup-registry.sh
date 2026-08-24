#!/bin/bash

# Container Registry Setup Script for Pixelated Empathy AI
# Supports Docker Hub, AWS ECR, Google GCR, and Azure ACR

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGISTRY_TYPE="${REGISTRY_TYPE:-docker-hub}"
PROJECT_NAME="pixelated-empathy"
IMAGE_NAME="pixelated/empathy-ai"

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
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Setup Docker Hub registry
setup_docker_hub() {
    log_info "Setting up Docker Hub registry..."
    
    if [[ -z "$DOCKER_USERNAME" || -z "$DOCKER_PASSWORD" ]]; then
        log_error "DOCKER_USERNAME and DOCKER_PASSWORD environment variables are required for Docker Hub"
        exit 1
    fi
    
    echo "$DOCKER_PASSWORD" | docker login --username "$DOCKER_USERNAME" --password-stdin
    
    log_success "Docker Hub registry configured"
    echo "Registry URL: docker.io/$IMAGE_NAME"
}

# Setup AWS ECR
setup_aws_ecr() {
    log_info "Setting up AWS ECR..."
    
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install AWS CLI first."
        exit 1
    fi
    
    # Get AWS account ID and region
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    AWS_REGION="${AWS_REGION:-us-west-2}"
    
    # Create ECR repository if it doesn't exist
    REPO_NAME="$PROJECT_NAME"
    
    if ! aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$AWS_REGION" &> /dev/null; then
        log_info "Creating ECR repository: $REPO_NAME"
        aws ecr create-repository \
            --repository-name "$REPO_NAME" \
            --region "$AWS_REGION" \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=AES256
    fi
    
    # Login to ECR
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
    
    log_success "AWS ECR configured"
    echo "Registry URL: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME"
}

# Setup Google Container Registry
setup_google_gcr() {
    log_info "Setting up Google Container Registry..."
    
    if ! command -v gcloud &> /dev/null; then
        log_error "Google Cloud SDK is not installed. Please install gcloud first."
        exit 1
    fi
    
    # Get project ID
    PROJECT_ID="${GOOGLE_PROJECT_ID:-$(gcloud config get-value project)}"
    
    if [[ -z "$PROJECT_ID" ]]; then
        log_error "Google Cloud project ID not found. Set GOOGLE_PROJECT_ID or configure gcloud."
        exit 1
    fi
    
    # Configure Docker to use gcloud as credential helper
    gcloud auth configure-docker --quiet
    
    log_success "Google Container Registry configured"
    echo "Registry URL: gcr.io/$PROJECT_ID/$PROJECT_NAME"
}

# Setup Azure Container Registry
setup_azure_acr() {
    log_info "Setting up Azure Container Registry..."
    
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI is not installed. Please install Azure CLI first."
        exit 1
    fi
    
    ACR_NAME="${AZURE_ACR_NAME:-pixelatedempathy}"
    RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-pixelated-empathy-rg}"
    
    # Create resource group if it doesn't exist
    if ! az group show --name "$RESOURCE_GROUP" &> /dev/null; then
        log_info "Creating resource group: $RESOURCE_GROUP"
        az group create --name "$RESOURCE_GROUP" --location "West US 2"
    fi
    
    # Create ACR if it doesn't exist
    if ! az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
        log_info "Creating Azure Container Registry: $ACR_NAME"
        az acr create \
            --resource-group "$RESOURCE_GROUP" \
            --name "$ACR_NAME" \
            --sku Standard \
            --admin-enabled true
    fi
    
    # Login to ACR
    az acr login --name "$ACR_NAME"
    
    log_success "Azure Container Registry configured"
    echo "Registry URL: $ACR_NAME.azurecr.io/$PROJECT_NAME"
}

# Generate image versioning configuration
generate_versioning_config() {
    log_info "Generating image versioning configuration..."
    
    cat > "$(dirname "$0")/versioning.env" << EOF
# Image Versioning Configuration
PROJECT_NAME=$PROJECT_NAME
REGISTRY_TYPE=$REGISTRY_TYPE

# Versioning strategy
VERSION_STRATEGY=semantic  # Options: semantic, timestamp, git-sha, build-number

# Tag patterns
LATEST_TAG=latest
DEV_TAG_PREFIX=dev
STAGING_TAG_PREFIX=staging
PROD_TAG_PREFIX=v

# Build metadata
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

# Registry URLs based on type
case \$REGISTRY_TYPE in
    docker-hub)
        REGISTRY_URL="docker.io"
        IMAGE_REPO="$IMAGE_NAME"
        ;;
    aws-ecr)
        AWS_ACCOUNT_ID=\$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "unknown")
        AWS_REGION=\${AWS_REGION:-us-west-2}
        REGISTRY_URL="\$AWS_ACCOUNT_ID.dkr.ecr.\$AWS_REGION.amazonaws.com"
        IMAGE_REPO="$PROJECT_NAME"
        ;;
    google-gcr)
        PROJECT_ID=\${GOOGLE_PROJECT_ID:-\$(gcloud config get-value project 2>/dev/null || echo "unknown")}
        REGISTRY_URL="gcr.io"
        IMAGE_REPO="\$PROJECT_ID/$PROJECT_NAME"
        ;;
    azure-acr)
        ACR_NAME=\${AZURE_ACR_NAME:-pixelatedempathy}
        REGISTRY_URL="\$ACR_NAME.azurecr.io"
        IMAGE_REPO="$PROJECT_NAME"
        ;;
esac

# Full image name
FULL_IMAGE_NAME="\$REGISTRY_URL/\$IMAGE_REPO"
EOF
    
    log_success "Versioning configuration generated: $(dirname "$0")/versioning.env"
}

# Generate build script
generate_build_script() {
    log_info "Generating build and push script..."
    
    cat > "$(dirname "$0")/build-and-push.sh" << 'EOF'
#!/bin/bash

# Build and Push Script for Pixelated Empathy AI
# Usage: ./build-and-push.sh [version] [environment]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load versioning configuration
if [[ -f "$SCRIPT_DIR/versioning.env" ]]; then
    source "$SCRIPT_DIR/versioning.env"
else
    echo "Error: versioning.env not found. Run setup-registry.sh first."
    exit 1
fi

# Parameters
VERSION="${1:-dev-$(date +%Y%m%d-%H%M%S)}"
ENVIRONMENT="${2:-dev}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Generate version tag based on strategy
generate_version_tag() {
    case $VERSION_STRATEGY in
        semantic)
            echo "$VERSION"
            ;;
        timestamp)
            echo "$(date +%Y%m%d-%H%M%S)"
            ;;
        git-sha)
            echo "$(git rev-parse --short HEAD)"
            ;;
        build-number)
            echo "build-${BUILD_NUMBER:-$(date +%s)}"
            ;;
        *)
            echo "$VERSION"
            ;;
    esac
}

# Build Docker image
build_image() {
    local version_tag=$(generate_version_tag)
    local environment_tag="${ENVIRONMENT}-${version_tag}"
    
    log_info "Building Docker image..."
    log_info "Version: $version_tag"
    log_info "Environment: $ENVIRONMENT"
    
    # Build with multiple tags
    docker build \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg GIT_COMMIT="$GIT_COMMIT" \
        --build-arg GIT_BRANCH="$GIT_BRANCH" \
        --build-arg VERSION="$version_tag" \
        --tag "$FULL_IMAGE_NAME:$version_tag" \
        --tag "$FULL_IMAGE_NAME:$environment_tag" \
        --tag "$FULL_IMAGE_NAME:${ENVIRONMENT}-latest" \
        "$PROJECT_ROOT"
    
    log_success "Image built successfully"
    
    # Export tags for pushing
    export IMAGE_TAGS="$version_tag $environment_tag ${ENVIRONMENT}-latest"
    
    # Add latest tag for production
    if [[ "$ENVIRONMENT" == "prod" ]]; then
        docker tag "$FULL_IMAGE_NAME:$version_tag" "$FULL_IMAGE_NAME:latest"
        export IMAGE_TAGS="$IMAGE_TAGS latest"
    fi
}

# Push Docker image
push_image() {
    log_info "Pushing Docker image to registry..."
    
    for tag in $IMAGE_TAGS; do
        log_info "Pushing tag: $tag"
        docker push "$FULL_IMAGE_NAME:$tag"
    done
    
    log_success "All images pushed successfully"
}

# Show image information
show_image_info() {
    log_info "Image Information:"
    echo "Registry: $REGISTRY_URL"
    echo "Repository: $IMAGE_REPO"
    echo "Full Image Name: $FULL_IMAGE_NAME"
    echo "Tags: $IMAGE_TAGS"
    echo "Build Date: $BUILD_DATE"
    echo "Git Commit: $GIT_COMMIT"
    echo "Git Branch: $GIT_BRANCH"
}

# Main execution
main() {
    log_info "Starting build and push process..."
    
    build_image
    push_image
    show_image_info
    
    log_success "Build and push completed successfully!"
}

# Show usage if help requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 [version] [environment]"
    echo ""
    echo "Parameters:"
    echo "  version     - Image version (default: dev-YYYYMMDD-HHMMSS)"
    echo "  environment - Target environment (default: dev)"
    echo ""
    echo "Examples:"
    echo "  $0 1.0.0 prod"
    echo "  $0 dev-feature-auth dev"
    echo "  $0 v1.2.3-rc1 staging"
    exit 0
fi

# Run main function
main
EOF
    
    chmod +x "$(dirname "$0")/build-and-push.sh"
    log_success "Build script generated: $(dirname "$0")/build-and-push.sh"
}

# Main execution
main() {
    log_info "Setting up container registry for Pixelated Empathy AI..."
    
    check_prerequisites
    
    case $REGISTRY_TYPE in
        docker-hub)
            setup_docker_hub
            ;;
        aws-ecr)
            setup_aws_ecr
            ;;
        google-gcr)
            setup_google_gcr
            ;;
        azure-acr)
            setup_azure_acr
            ;;
        *)
            log_error "Unsupported registry type: $REGISTRY_TYPE"
            log_info "Supported types: docker-hub, aws-ecr, google-gcr, azure-acr"
            exit 1
            ;;
    esac
    
    generate_versioning_config
    generate_build_script
    
    log_success "Container registry setup completed!"
    log_info "Next steps:"
    echo "1. Review and customize versioning.env"
    echo "2. Use build-and-push.sh to build and push images"
    echo "3. Update Helm charts with the correct image repository"
}

# Show usage if no registry type specified
if [[ -z "$REGISTRY_TYPE" ]]; then
    echo "Usage: REGISTRY_TYPE=<type> $0"
    echo ""
    echo "Registry Types:"
    echo "  docker-hub  - Docker Hub (requires DOCKER_USERNAME, DOCKER_PASSWORD)"
    echo "  aws-ecr     - Amazon ECR (requires AWS CLI configured)"
    echo "  google-gcr  - Google Container Registry (requires gcloud configured)"
    echo "  azure-acr   - Azure Container Registry (requires Azure CLI configured)"
    echo ""
    echo "Examples:"
    echo "  REGISTRY_TYPE=docker-hub $0"
    echo "  REGISTRY_TYPE=aws-ecr AWS_REGION=us-west-2 $0"
    echo "  REGISTRY_TYPE=google-gcr GOOGLE_PROJECT_ID=my-project $0"
    echo "  REGISTRY_TYPE=azure-acr AZURE_ACR_NAME=myregistry $0"
    exit 0
fi

# Run main function
main
