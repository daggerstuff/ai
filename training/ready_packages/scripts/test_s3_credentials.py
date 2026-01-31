#!/usr/bin/env python3
"""
Test S3 credentials for OVH S3
"""

import os

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv("ai/.env")  # Try loading explicitly


def test_credentials():
    """Test different credential formats for OVH S3"""

    # Check if credentials are set
    access_key = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get(
        "OVH_S3_ACCESS_KEY"
    )
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get(
        "OVH_S3_SECRET_KEY"
    )

    print("DEBUG: Checking Env Keys:")
    for k, v in os.environ.items():
        if any(x in k for x in ["AWS", "OVH", "KEY", "SECRET"]):
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

    # Test with OVH S3
    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url="https://s3.us-east-va.io.cloud.ovh.us",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-va",
        )

        # Try to list buckets
        response = s3_client.list_buckets()
        print("✅ Successfully connected to OVH S3")
        print("📦 Available buckets:")
        for bucket in response.get("Buckets", []):
            print(f"   - {bucket['Name']}")

    except ClientError as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔧 OVH S3 uses these formats:")
        print("   Access Key: <application_key>")
        print("   Secret Key: <application_secret>")
        print("   Get from: OVH Control Panel > Public Cloud > Object Storage > Users")


if __name__ == "__main__":
    test_credentials()
