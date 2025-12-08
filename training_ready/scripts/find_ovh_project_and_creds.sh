#!/usr/bin/env zsh
# Find OVH project ID and retrieve S3 credentials

set -e

BUCKET_NAME="pixel-data"

echo "🔍 Finding OVH Cloud Project for bucket: $BUCKET_NAME"
echo ""

# Method 1: Try to get bucket info (might reveal project)
echo "Method 1: Getting bucket information..."
BUCKET_INFO=$(ovhcloud cloud storage-s3 get --name "$BUCKET_NAME" --format json 2>&1 || echo "")

if [[ -n "$BUCKET_INFO" && "$BUCKET_INFO" != "null" && "$BUCKET_INFO" != *"error"* ]]; then
    PROJECT_ID=$(echo "$BUCKET_INFO" | jq -r '.projectId // .project_id // empty' 2>/dev/null || echo "")
    if [[ -n "$PROJECT_ID" ]]; then
        echo "✅ Found project ID from bucket: $PROJECT_ID"
        export OVH_CLOUD_PROJECT="$PROJECT_ID"
    fi
fi

# Method 2: Try listing with different project IDs (if we have any hints)
if [[ -z "$PROJECT_ID" ]]; then
    echo ""
    echo "Method 2: Need to specify project ID manually"
    echo ""
    echo "You can find your project ID by:"
    echo "  1. Logging into OVH Manager: https://www.ovh.com/manager/public-cloud/"
    echo "  2. Going to Public Cloud → Project Settings"
    echo "  3. The Project ID is shown there"
    echo ""
    echo "Or enter it now if you know it:"
    read -r PROJECT_ID
    
    if [[ -n "$PROJECT_ID" ]]; then
        export OVH_CLOUD_PROJECT="$PROJECT_ID"
    fi
fi

if [[ -z "$PROJECT_ID" ]]; then
    echo "❌ Project ID required to continue"
    exit 1
fi

echo ""
echo "📋 Listing users for project: $PROJECT_ID"
USERS=$(ovhcloud cloud user list --cloud-project "$PROJECT_ID" --format json 2>/dev/null || echo "[]")

if [[ "$USERS" == "null" || "$USERS" == "[]" ]]; then
    echo "⚠️  No users found. Creating a new S3 user..."
    USER_INFO=$(ovhcloud cloud user create --cloud-project "$PROJECT_ID" --description "S3 Access User" --format json 2>/dev/null || echo "")
    
    if [[ -z "$USER_INFO" || "$USER_INFO" == "null" ]]; then
        echo "❌ Failed to create user"
        exit 1
    fi
    
    USER_ID=$(echo "$USER_INFO" | jq -r '.id' 2>/dev/null || echo "")
    echo "✅ Created user with ID: $USER_ID"
else
    echo "Found users:"
    echo "$USERS" | jq -r '.[] | "  - \(.username // .description) (ID: \(.id))"' 2>/dev/null || echo "$USERS"
    
    USER_ID=$(echo "$USERS" | jq -r '.[0].id' 2>/dev/null || echo "")
    
    if [[ -z "$USER_ID" ]]; then
        echo "❌ Could not determine user ID"
        exit 1
    fi
    
    echo "Using user ID: $USER_ID"
fi

echo ""
echo "🔑 Getting or creating S3 credentials..."

# Check existing credentials
CREDS=$(ovhcloud cloud storage-s3 credentials list --cloud-project "$PROJECT_ID" --user-id "$USER_ID" --format json 2>/dev/null || echo "[]")

if [[ "$CREDS" == "null" || "$CREDS" == "[]" ]]; then
    echo "Creating new credentials..."
    NEW_CREDS=$(ovhcloud cloud storage-s3 credentials create --cloud-project "$PROJECT_ID" --user-id "$USER_ID" --format json 2>/dev/null || echo "")
    
    if [[ -z "$NEW_CREDS" || "$NEW_CREDS" == "null" ]]; then
        echo "❌ Failed to create credentials"
        exit 1
    fi
    
    ACCESS_KEY=$(echo "$NEW_CREDS" | jq -r '.access' 2>/dev/null || echo "")
    SECRET_KEY=$(echo "$NEW_CREDS" | jq -r '.secret' 2>/dev/null || echo "")
else
    echo "Found existing credentials:"
    echo "$CREDS" | jq -r '.[] | "  - Access ID: \(.access)"' 2>/dev/null || echo "$CREDS"
    
    ACCESS_ID=$(echo "$CREDS" | jq -r '.[0].access' 2>/dev/null || echo "")
    
    if [[ -z "$ACCESS_ID" ]]; then
        echo "Creating new credentials..."
        NEW_CREDS=$(ovhcloud cloud storage-s3 credentials create --cloud-project "$PROJECT_ID" --user-id "$USER_ID" --format json 2>/dev/null || echo "")
        ACCESS_KEY=$(echo "$NEW_CREDS" | jq -r '.access' 2>/dev/null || echo "")
        SECRET_KEY=$(echo "$NEW_CREDS" | jq -r '.secret' 2>/dev/null || echo "")
    else
        echo "Retrieving credentials for access ID: $ACCESS_ID"
        CRED_INFO=$(ovhcloud cloud storage-s3 credentials get --cloud-project "$PROJECT_ID" --user-id "$USER_ID" --access-id "$ACCESS_ID" --format json 2>/dev/null || echo "")
        
        ACCESS_KEY=$(echo "$CRED_INFO" | jq -r '.access' 2>/dev/null || echo "")
        SECRET_KEY=$(echo "$CRED_INFO" | jq -r '.secret' 2>/dev/null || echo "")
    fi
fi

if [[ -z "$ACCESS_KEY" || -z "$SECRET_KEY" ]]; then
    echo "❌ Failed to retrieve credentials"
    exit 1
fi

echo ""
echo "✅ Credentials retrieved successfully!"
echo ""
echo "📝 Add these to your ai/.env file:"
echo ""
echo "OVH_S3_ACCESS_KEY=$ACCESS_KEY"
echo "OVH_S3_SECRET_KEY=$SECRET_KEY"
echo ""
echo "Or run these commands to export them:"
echo "export OVH_S3_ACCESS_KEY='$ACCESS_KEY'"
echo "export OVH_S3_SECRET_KEY='$SECRET_KEY'"

