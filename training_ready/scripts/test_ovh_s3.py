#!/usr/bin/env python3
"""Quick test script for OVH S3 connection"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.dataset_pipeline.storage_config import StorageConfig
import boto3

def main():
    print("🔍 Loading StorageConfig from environment...")
    c = StorageConfig.from_env()

    print(f"\n📦 Configuration:")
    print(f"   Bucket: {c.s3_bucket}")
    print(f"   Endpoint: {c.s3_endpoint_url}")
    print(f"   Region: {c.s3_region}")
    print(f"   Access Key ID: {'***' + c.s3_access_key_id[-4:] if c.s3_access_key_id else 'NOT SET'}")
    print(f"   Secret Key: {'SET' if c.s3_secret_access_key else 'NOT SET'}")

    if not c.s3_bucket:
        print("\n❌ S3 bucket not configured!")
        print("   Set OVH_S3_BUCKET environment variable")
        return 1

    if not c.s3_access_key_id or not c.s3_secret_access_key:
        print("\n❌ S3 credentials not configured!")
        print("   Set OVH_S3_ACCESS_KEY and OVH_S3_SECRET_KEY environment variables")
        return 1

    if not c.s3_endpoint_url:
        print("\n❌ S3 endpoint not configured!")
        print("   Set OVH_S3_ENDPOINT environment variable")
        return 1

    print("\n🔌 Creating S3 client...")
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=c.s3_endpoint_url,
            aws_access_key_id=c.s3_access_key_id,
            aws_secret_access_key=c.s3_secret_access_key,
            region_name=c.s3_region
        )

        print("✅ S3 client created successfully")

        print("\n📋 Testing connection (list_buckets)...")
        response = s3.list_buckets()
        print('✅ S3 connection successful!')
        buckets = [b['Name'] for b in response.get('Buckets', [])]
        print(f'   Available buckets: {buckets}')

        if c.s3_bucket in buckets:
            print(f'\n✅ Target bucket "{c.s3_bucket}" found!')

            # Try to list objects in the bucket
            print(f'\n📂 Testing bucket access (listing objects in {c.s3_bucket})...')
            try:
                objects = s3.list_objects_v2(Bucket=c.s3_bucket, MaxKeys=5)
                count = objects.get('KeyCount', 0)
                print(f'   ✅ Bucket accessible! Found {count} objects (showing first 5)')
                if 'Contents' in objects:
                    for obj in objects['Contents'][:5]:
                        print(f'      - {obj["Key"]} ({obj["Size"]} bytes)')
            except Exception as e:
                print(f'   ⚠️  Could not list objects: {e}')
        else:
            print(f'\n⚠️  Target bucket "{c.s3_bucket}" not found in list')
            print(f'   Available buckets: {buckets}')

        return 0

    except Exception as e:
        print(f'\n❌ S3 connection failed: {e}')
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

