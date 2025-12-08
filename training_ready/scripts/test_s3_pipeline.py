#!/usr/bin/env python3
"""
Test S3 Pipeline Integration

Quick test script to verify S3 integration works end-to-end.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def test_s3_loader():
    """Test S3 dataset loader"""
    print("🧪 Test 1: S3 Dataset Loader")
    print("-" * 60)

    try:
        from ai.training_ready.tools.data_preparation.s3_dataset_loader import S3DatasetLoader

        loader = S3DatasetLoader()
        test_path = "s3://pixel-data/datasets/huggingface/huggingface/ShreyaR_DepressionDetection.jsonl"

        if loader.dataset_exists(test_path):
            print(f"✅ Dataset exists: {test_path}")

            count = 0
            for record in loader.load_jsonl(test_path, max_records=3):
                count += 1
                if count == 1:
                    print(f"   Sample record keys: {list(record.keys())[:3]}")

            print(f"✅ Loaded {count} records successfully")
            return True
        else:
            print(f"❌ Dataset not found: {test_path}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_path_resolver():
    """Test path resolver"""
    print("\n🧪 Test 2: Path Resolver")
    print("-" * 60)

    try:
        from ai.training_ready.tools.data_preparation.path_resolver import get_resolver

        resolver = get_resolver()
        test_s3_path = "s3://pixel-data/datasets/huggingface/huggingface/ShreyaR_DepressionDetection.jsonl"

        manifest_entry = {
            "name": "test_dataset",
            "path": "/local/path.jsonl",
            "s3_path": test_s3_path
        }

        resolved, source_type = resolver.resolve_path("/local/path.jsonl", manifest_entry)
        print(f"   Original: /local/path.jsonl")
        print(f"   Resolved: {resolved[:70]}...")
        print(f"   Source: {source_type}")

        if source_type == "s3":
            print("✅ Path resolution to S3 works")

            # Test loading
            count = 0
            for record in resolver.load_dataset(resolved, source_type, max_records=2):
                count += 1
            print(f"✅ Loaded {count} records via resolver")
            return True
        else:
            print("⚠️  Resolved to local (S3 may not be available)")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_s3_sourcing():
    """Test S3 dataset sourcing"""
    print("\n🧪 Test 3: S3 Dataset Sourcing")
    print("-" * 60)

    try:
        from ai.training_ready.tools.data_preparation.source_datasets import DatasetSourcer

        manifest_path = project_root / "ai" / "training_ready" / "TRAINING_MANIFEST.json"
        sourcer = DatasetSourcer(manifest_path)

        test_s3_path = "s3://pixel-data/datasets/huggingface/huggingface/ShreyaR_DepressionDetection.jsonl"
        result = sourcer.source_s3_dataset(test_s3_path, "test_dataset")

        print(f"   Success: {result.success}")
        print(f"   Path: {result.path[:70]}...")
        print(f"   Source type: {result.source_type}")

        if result.success:
            print(f"✅ S3 sourcing works")
            if result.size_bytes:
                print(f"   Size: {result.size_bytes / 1024 / 1024:.2f} MB")
            return True
        else:
            print(f"❌ S3 sourcing failed: {result.error}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("S3 Pipeline Integration Test Suite")
    print("=" * 60)

    results = []
    results.append(("S3 Loader", test_s3_loader()))
    results.append(("Path Resolver", test_path_resolver()))
    results.append(("S3 Sourcing", test_s3_sourcing()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed! S3 integration is working.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

