#!/usr/bin/env zsh
# Script to retrieve OVH S3 credentials using the OVH CLI

set -e

echo "🔍 Retrieving OVH S3 credentials..."
echo ""

# First, let's check if we need to specify a cloud project
# Try to list S3 containers to see if we need project ID
echo "📦 Step 1: Finding S3 containers..."
CONTAINERS=$(ovhcloud cloud storage-s3 list --format json 2>/dev/null || echo "[]")

if [[ "$CONTAINERS" == "null" || "$CONTAINERS" == "[]" ]]; then
    echo "⚠️  No containers found or need to specify cloud project"
    echo ""
    echo "Please provide your Cloud Project ID:"
    read -r PROJECT_ID
    
    if [[ -z "$PROJECT_ID" ]]; then
        echo "❌ Project ID required"
        exit 1
    fi
    
    export OVH_CLOUD_PROJECT="$PROJECT_ID"
    CONTAINERS=$(ovhcloud cloud storage-s3 list --cloud-project "$PROJECT_ID" --format json 2>/dev/null || echo "[]")
fi

echo "Found containers:"
echo "$CONTAINERS" | jq -r '.[] | "  - \(.name) (ID: \(.id))"' 2>/dev/null || echo "$CONTAINERS"

echo ""
echo "📋 Step 2: Finding users..."
USERS=$(ovhcloud cloud user list --cloud-project "${OVH_CLOUD_PROJECT:-}" --format json 2>/dev/null || echo "[]")

if [[ "$USERS" == "null" || "$USERS" == "[]" ]]; then
    echo "⚠️  No users found. You may need to create one first."
    echo ""
    echo "Would you like to create a new user? (y/n)"
    read -r CREATE_USER
    
    if [[ "$CREATE_USER" == "y" ]]; then
        echo "Enter username for new user:"
        read -r USERNAME
        ovhcloud cloud user create --cloud-project "${OVH_CLOUD_PROJECT:-}" --description "$USERNAME" --format json
        echo "✅ User created. Please run this script again to get credentials."
        exit 0
    else
        echo "Please provide an existing user ID:"
        read -r USER_ID
    fi
else
    echo "Found users:"
    echo "$USERS" | jq -r '.[] | "  - \(.username) (ID: \(.id))"' 2>/dev/null || echo "$USERS"
    echo ""
    echo "Enter user ID to get credentials for:"
    read -r USER_ID
fi

if [[ -z "$USER_ID" ]]; then
    echo "❌ User ID required"
    exit 1
fi

echo ""
echo "🔑 Step 3: Listing existing credentials for user $USER_ID..."
CREDS=$(ovhcloud cloud storage-s3 credentials list --cloud-project "${OVH_CLOUD_PROJECT:-}" --user-id "$USER_ID" --format json 2>/dev/null || echo "[]")

if [[ "$CREDS" == "null" || "$CREDS" == "[]" ]]; then
    echo "⚠️  No credentials found. Creating new credentials..."
    echo ""
    NEW_CREDS=$(ovhcloud cloud storage-s3 credentials create --cloud-project "${OVH_CLOUD_PROJECT:-}" --user-id "$USER_ID" --format json)
    
    if [[ -z "$NEW_CREDS" || "$NEW_CREDS" == "null" ]]; then
        echo "❌ Failed to create credentials"
        exit 1
    fi
    
    ACCESS_KEY=$(echo "$NEW_CREDS" | jq -r '.access' 2>/dev/null || echo "")
    SECRET_KEY=$(echo "$NEW_CREDS" | jq -r '.secret' 2>/dev/null || echo "")
else
    echo "Found existing credentials:"
    echo "$CREDS" | jq -r '.[] | "  - Access ID: \(.access)"' 2>/dev/null || echo "$CREDS"
    echo ""
    echo "Enter access ID to retrieve:"
    read -r ACCESS_ID
    
    CRED_INFO=$(ovhcloud cloud storage-s3 credentials get --cloud-project "${OVH_CLOUD_PROJECT:-}" --user-id "$USER_ID" --access-id "$ACCESS_ID" --format json)
    
    ACCESS_KEY=$(echo "$CRED_INFO" | jq -r '.access' 2>/dev/null || echo "")
    SECRET_KEY=$(echo "$CRED_INFO" | jq -r '.secret' 2>/dev/null || echo "")
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
echo "Or export them now:"
echo "export OVH_S3_ACCESS_KEY='$ACCESS_KEY'"
echo "export OVH_S3_SECRET_KEY='$SECRET_KEY'"

