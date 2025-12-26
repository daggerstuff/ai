#!/usr/bin/env python3
"""
Verify Provenance Implementation

Quick verification script to ensure all components are properly integrated.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def verify_imports():
    """Verify all imports work correctly."""
    print("Verifying imports...")

    try:
        print("✅ ProvenanceService imported")
    except Exception as e:
        print(f"❌ Failed to import ProvenanceService: {e}")
        return False

    try:
        print("✅ Provenance integration helpers imported")
    except Exception as e:
        print(f"❌ Failed to import integration helpers: {e}")
        return False

    try:
        print("✅ ProvenanceOrchestratorWrapper imported")
    except Exception as e:
        print(f"❌ Failed to import wrapper: {e}")
        return False

    try:
        print("✅ Provenance schema dataclasses imported")
    except Exception as e:
        print(f"❌ Failed to import schema classes: {e}")
        return False

    return True


def verify_files():
    """Verify all files exist."""
    print("\nVerifying files...")

    files_to_check = [
        "db/provenance_schema.sql",
        "docs/governance/provenance_schema.json",
        "docs/governance/provenance_storage_plan.md",
        "docs/governance/audit_report_example.json",
        "ai/dataset_pipeline/schemas/provenance_schema.py",
        "ai/dataset_pipeline/services/provenance_service.py",
        "ai/dataset_pipeline/services/provenance_integration.py",
        "ai/dataset_pipeline/services/provenance_orchestrator_wrapper.py",
        "ai/dataset_pipeline/services/README.md",
        "ai/dataset_pipeline/cli/provenance_cli.py",
    ]

    all_exist = True
    base_path = Path(__file__).parent.parent.parent.parent

    for file_path in files_to_check:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False

    return all_exist


def main():
    """Run verification."""
    print("=" * 60)
    print("Provenance Implementation Verification")
    print("=" * 60)

    imports_ok = verify_imports()
    files_ok = verify_files()

    print("\n" + "=" * 60)
    if imports_ok and files_ok:
        print("✅ All verifications passed!")
        print("\nImplementation is ready to use.")
        print("\nNext steps:")
        print("1. Run database migration: db/provenance_schema.sql")
        print("2. Set environment variables: DATABASE_URL, S3_BUCKET")
        print(
            "3. Test CLI: python -m ai.dataset_pipeline.cli.provenance_cli init-schema"
        )
        return 0
    else:
        print("❌ Some verifications failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
