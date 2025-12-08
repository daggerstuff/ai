#!/usr/bin/env zsh
# Quick script to get OVH S3 credentials - tries common approaches

set -e

echo "🔍 Quick OVH S3 Credential Retrieval"
echo ""

# Method 1: Try to list all credentials across all users
echo "Method 1: Checking if we can list credentials directly..."
echo ""

# First, get cloud project (might be in config or need to specify)
PROJECT_ID="${OVH_CLOUD_PROJECT:-}"

if [[ -z "$PROJECT_ID" ]]; then
    echo "ℹ️  No OVH_CLOUD_PROJECT set. Trying to detect..."
    # Check if there's a default project
    PROJECTS=$(ovhcloud cloud project list --format json 2>/dev/null || echo "[]")
    
    if [[ "$PROJECTS" != "null" && "$PROJECTS" != "[]" ]]; then
        PROJECT_ID=$(echo "$PROJECTS" | jq -r '.[0].project_id' 2>/dev/null || echo "")
        if [[ -n "$PROJECT_ID" ]]; then
            echo "✅ Found project: $PROJECT_ID"
            export OVH_CLOUD_PROJECT="$PROJECT_ID"
        fi
    fi
fi

# List users
echo "📋 Listing users..."
if [[ -n "$PROJECT_ID" ]]; then
    USERS=$(ovhcloud cloud user list --cloud-project "$PROJECT_ID" --format json 2>/dev/null || echo "[]")
else
    USERS=$(ovhcloud cloud user list --format json 2>/dev/null || echo "[]")
fi

if [[ "$USERS" == "null" || "$USERS" == "[]" ]]; then
    echo "❌ No users found. You may need to:"
    echo "   1. Create a user: ovhcloud cloud user create --cloud-project <PROJECT_ID> --description 'S3 User'"
    echo "   2. Or specify your project: export OVH_CLOUD_PROJECT=<your-project-id>"
    exit 1
fi

echo "Found users:"
echo "$USERS" | jq -r '.[] | "  - \(.username // .description) (ID: \(.id))"' 2>/dev/null || echo "$USERS"

# Get first user ID
USER_ID=$(echo "$USERS" | jq -r '.[0].id' 2>/dev/null || echo "")

if [[ -z "$USER_ID" ]]; then
    echo "❌ Could not determine user ID"
    exit 1
fi

echo ""
echo "🔑 Retrieving credentials for user: $USER_ID"

# List credentials for this user
if [[ -n "$PROJECT_ID" ]]; then
    CREDS=$(ovhcloud cloud storage-s3 credentials list --cloud-project "$PROJECT_ID" --user-id "$USER_ID" --format json 2>/dev/null || echo "[]")
else
    CREDS=$(ovhcloud cloud storage-s3 credentials list --user-id "$USER_ID" --format json 2>/dev/null || echo "[]")
fi

if [[ "$CREDS" == "null" || "$CREDS" == "[]" ]]; then
    echo "⚠️  No existing credentials found. Creating new ones..."
    
    if [[ -n "$PROJECT_ID" ]]; then
        NEW_CREDS=$(ovhcloud cloud storage-s3 credentials create --cloud-project "$PROJECT_ID" --user-id "$USER_ID" --format json 2>/dev/null || echo "")
    else
        NEW_CREDS=$(ovhcloud cloud storage-s3 credentials create --user-id "$USER_ID" --format json 2>/dev/null || echo "")
    fi
    
    if [[ -z "$NEW_CREDS" || "$NEW_CREDS" == "null" ]]; then
        echo "❌ Failed to create credentials. You may need to specify --cloud-project"
        exit 1
    fi
    
    ACCESS_KEY=$(echo "$NEW_CREDS" | jq -r '.access' 2>/dev/null || echo "")
    SECRET_KEY=$(echo "$NEW_CREDS" | jq -r '.secret' 2>/dev/null || echo "")
else
    echo "Found existing credentials:"
    echo "$CREDS" | jq -r '.[] | "  - Access ID: \(.access)"' 2>/dev/null || echo "$CREDS"
    
    # Get first access ID
    ACCESS_ID=$(echo "$CREDS" | jq -r '.[0].access' 2>/dev/null || echo "")
    
    if [[ -z "$ACCESS_ID" ]]; then
        echo "❌ Could not determine access ID"
        exit 1
    fi
    
    echo ""
    echo "Retrieving full credentials for access ID: $ACCESS_ID"
    
    if [[ -n "$PROJECT_ID" ]]; then
        CRED_INFO=$(ovhcloud cloud storage-s3 credentials get --cloud-project "$PROJECT_ID" --user-id "$USER_ID" --access-id "$ACCESS_ID" --format json 2>/dev/null || echo "")
    else
        CRED_INFO=$(ovhcloud cloud storage-s3 credentials get --user-id "$USER_ID" --access-id "$ACCESS_ID" --format json 2>/dev/null || echo "")
    fi
    
    ACCESS_KEY=$(echo "$CRED_INFO" | jq -r '.access' 2>/dev/null || echo "")
    SECRET_KEY=$(echo "$CRED_INFO" | jq -r '.secret' 2>/dev/null || echo "")
fi

if [[ -z "$ACCESS_KEY" || -z "$SECRET_KEY" ]]; then
    echo "❌ Failed to retrieve credentials"
    echo ""
    echo "Try running the interactive script:"
    echo "  ./ai/training_ready/scripts/get_ovh_s3_credentials.sh"
    exit 1
fi

echo ""
echo "✅ Credentials retrieved!"
echo ""
echo "📝 Add to ai/.env:"
echo ""
echo "OVH_S3_ACCESS_KEY=$ACCESS_KEY"
echo "OVH_S3_SECRET_KEY=$SECRET_KEY"
echo ""
echo "Or run these commands:"
echo "export OVH_S3_ACCESS_KEY='$ACCESS_KEY'"
echo "export OVH_S3_SECRET_KEY='$SECRET_KEY'"

