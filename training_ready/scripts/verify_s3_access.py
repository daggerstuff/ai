#!/usr/bin/env python3
"""
Verify S3 Access - Quick test for OVH S3 connectivity
Tests that credentials are loaded and S3 access works
"""

import sys
from pathlib import Path

# Add project root to path
# Script is at: ai/training_ready/scripts/verify_s3_access.py
# Project root is: /home/vivi/pixelated/
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]  # Go up 3 levels: scripts -> training_ready -> ai -> project_root
sys.path.insert(0, str(project_root))

def main():
    """Verify S3 access and list available datasets"""
    print("🔍 Verifying S3 Access (OVH)")
    print("=" * 60)

    try:
        from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

        # Initialize loader (will load from .env automatically)
        print("\n1. Initializing S3DatasetLoader...")
        loader = S3DatasetLoader()
        print("   ✅ Loader initialized")
        print(f"   📦 Bucket: {loader.bucket}")
        print(f"   🌐 Endpoint: {loader.endpoint_url}")

        # Test connection by listing datasets
        print("\n2. Testing S3 connection...")
        try:
            datasets = loader.list_datasets(prefix="gdrive/processed/")
            print("   ✅ Connection successful!")
            print(f"   📊 Found {len(datasets)} datasets in gdrive/processed/")

            if datasets:
                print("\n   Sample datasets:")
                for dataset in datasets[:5]:  # Show first 5
                    print(f"      - {dataset}")
                if len(datasets) > 5:
                    print(f"      ... and {len(datasets) - 5} more")
            else:
                print("   ⚠️  No datasets found in gdrive/processed/")
                print("   💡 This is normal if raw sync is still in progress")
                print("   💡 Check gdrive/raw/ for datasets being synced")

                # Try raw structure
                raw_datasets = loader.list_datasets(prefix="gdrive/raw/")
                if raw_datasets:
                    print(f"\n   📊 Found {len(raw_datasets)} datasets in gdrive/raw/")
                    print("   💡 These will be organized into processed/ structure")

        except Exception as e:
            print(f"   ⚠️  Connection test failed: {e}")
            print("   💡 This might be normal if:")
            print("      - Bucket doesn't exist yet")
            print("      - No datasets uploaded yet")
            print("      - Network/credential issue")
            return 1

        # Test raw structure
        print("\n3. Checking raw structure...")
        try:
            raw_datasets = loader.list_datasets(prefix="gdrive/raw/")
            print(f"   📊 Found {len(raw_datasets)} datasets in gdrive/raw/")
        except Exception as e:
            print(f"   ⚠️  Could not list raw datasets: {e}")

        print("\n" + "=" * 60)
        print("✅ S3 Access Verification Complete")
        print("\n💡 Next steps:")
        print("   1. Ensure datasets are synced to S3")
        print("   2. Run: python scripts/update_manifest_s3_paths.py")
        print("   3. Test training: python scripts/train_optimized.py --help")

        return 0

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n💡 Make sure your .env file has:")
        print("   OVH_S3_ACCESS_KEY=...")
        print("   OVH_S3_SECRET_KEY=...")
        print("\n   Or set environment variables:")
        print("   export OVH_S3_ACCESS_KEY=...")
        print("   export OVH_S3_SECRET_KEY=...")
        return 1

    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n💡 Install dependencies:")
        print("   uv pip install boto3 python-dotenv")
        return 1

    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
