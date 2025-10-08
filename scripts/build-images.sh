#!/bin/bash

# build-images.sh
# Script to build and tag container images for Pixelated Empathy AI

set -e

# Default values
REGISTRY=${CONTAINER_REGISTRY:-"local"}
VERSION=${IMAGE_VERSION:-"latest"}
PUSH=${PUSH_IMAGES:-"false"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Pixelated Empathy AI Container Images${NC}"
echo "Registry: $REGISTRY"
echo "Version: $VERSION"
echo "Push images: $PUSH"
echo

# Function to build an image
build_image() {
    local component=$1
    local dockerfile=${2:-"Dockerfile"}
    local context=${3:-"."}
    
    echo -e "${YELLOW}Building $component image...${NC}"
    
    # Build the image
    docker build -f "$dockerfile" -t "${REGISTRY}/pixelated-empathy-${component}:${VERSION}" "$context"
    
    if [ "$PUSH" = "true" ]; then
        echo -e "${YELLOW}Pushing $component image...${NC}"
        docker push "${REGISTRY}/pixelated-empathy-${component}:${VERSION}"
    fi
    
    echo -e "${GREEN}Successfully built ${REGISTRY}/pixelated-empathy-${component}:${VERSION}${NC}"
    echo
}

# Build main application image
build_image "api" "Dockerfile" "."

# Build any additional component images if needed
# build_image "component" "path/to/Dockerfile" "build/context"

echo -e "${GREEN}All images built successfully!${NC}"

# Tagging for different environments
echo -e "${YELLOW}Creating environment-specific tags...${NC}"

# Development tag
docker tag "${REGISTRY}/pixelated-empathy-api:${VERSION}" "${REGISTRY}/pixelated-empathy-api:dev"
echo "Tagged as dev"

# Staging tag
docker tag "${REGISTRY}/pixelated-empathy-api:${VERSION}" "${REGISTRY}/pixelated-empathy-api:staging"
echo "Tagged as staging"

echo -e "${GREEN}Environment tags created successfully!${NC}"