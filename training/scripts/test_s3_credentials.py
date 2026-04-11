#!/usr/bin/env python3
"""
Test S3 credentials for HETZNER S3
"""

import os

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv("ai/.env")  # Try loading explicitly


def test_credentials():
    """Test different credential formats for HETZNER S3"""

    # Check if credentials are set
    access_key = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get(
        "HETZNER_S3_ACCESS_KEY"
    )
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get(
        "HETZNER_S3_SECRET_KEY"
    )

    print("DEBUG: Checking Env Keys:")
    for k, v in os.environ.items():
        if any(x in k for x in ["AWS", "HETZNER", "KEY", "SECRET"]):
            print(f"  {k}: {v[:4]}...")

    if not access_key or not secret_key:
        print("❌ AWS credentials not found in environment")
        print("Set these environment variables:")
        print("  AWS_ACCESS_KEY_ID=your-access-key")
        print("  AWS_SECRET_ACCESS_KEY=your-secret-key")
        return

    print("🔑 Using credentials:")
    print(f"   Access Key: {access_key[:8]}...")
    print(f"   Secret: {'*' * min(len(secret_key), 8)}...")

    # Test with HETZNER S3
    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url="https://hel1.your-objectstorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='hel1',
        )

        # Try to list buckets
        response = s3_client.list_buckets()
        print("✅ Successfully connected to HETZNER S3")
        print("📦 Available buckets:")
        for bucket in response.get("Buckets", []):
            print(f"   - {bucket['Name']}")

    except ClientError as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔧 HETZNER S3 uses these formats:")
        print("   Access Key: <application_key>")
        print("   Secret Key: <application_secret>")
        print("   Get from: HETZNER Control Panel > Public Cloud > Object Storage > Users")


if __name__ == "__main__":
    test_credentials()
