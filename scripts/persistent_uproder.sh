#!/bin/bash

# Configuration
SCRIPT_PATH="ai/scripts/full_ai_sweep_s3.py"
MAX_RESTARTS=50
RESTART_DELAY=30

echo "🛡️ Starting PERSISTENT S3 migration wrapper..."
echo "This script will restart the uploader if it crashes due to network loss."

count=0
while [ $count -lt $MAX_RESTARTS ]; do
    echo "🚀 Run #$((count+1)) started at $(date)"
    
    # Run the uploader
    # We use 'uv run' but pass the env vars in.
    OVH_S3_ACCESS_KEY=$OVH_S3_ACCESS_KEY OVH_S3_SECRET_KEY=$OVH_S3_SECRET_KEY PYTHONUNBUFFERED=1 uv run --with boto3 $SCRIPT_PATH
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "🎉 Migration script finished successfully with exit code 0."
        exit 0
    else
        echo "❌ Migration script crashed with exit code $EXIT_CODE."
        echo "🕒 Restarting in $RESTART_DELAY seconds..."
        sleep $RESTART_DELAY
        count=$((count+1))
    fi
done

echo "🛑 Reached maximum restart limit ($MAX_RESTARTS). Please check the connection manually."
exit 1
