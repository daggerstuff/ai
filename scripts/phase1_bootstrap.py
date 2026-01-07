#!/usr/bin/env python
"""
Phase 1 Bootstrap Script — Initialize NGC Therapeutic Enhancement.

Runs:
1. Development environment validation
2. NeMo microservices status check
3. Therapeutic data pipeline initialization
4. Docker resource validation

Usage:
  uv run python ai/scripts/phase1_bootstrap.py
"""

import sys
from pathlib import Path

from ai.foundation.dev_environment import initialize_dev_environment
from ai.foundation.nemo_orchestration import NeMoMicroservicesManager
from ai.foundation.therapeutic_data_pipeline import TherapeuticDataPipeline

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def _print_step_header(step_num: int, title: str) -> None:
    """Print a formatted step header."""
    print(f"STEP {step_num}: {title}")
    print("-" * 60)


def main():
    """Execute Phase 1 bootstrap."""
    print("\n" + "=" * 60)
    print("PHASE 1: NGC THERAPEUTIC ENHANCEMENT BOOTSTRAP")
    print("=" * 60 + "\n")

    # Step 1: Development environment
    _print_step_header(1, "Development Environment Setup")
    initialize_dev_environment()
    print()

    # Step 2: NeMo Microservices
    _print_step_header(2, "NeMo Microservices Validation")
    nemo_manager = NeMoMicroservicesManager()
    print(nemo_manager.status())
    print()

    # Step 3: Therapeutic Data Pipeline
    _print_step_header(3, "Therapeutic Data Pipeline Initialization")
    pipeline = TherapeuticDataPipeline()
    pipeline.initialize()
    print(pipeline.status())
    print()

    # Summary
    print("=" * 60)
    print("PHASE 1 BOOTSTRAP COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Configure NeMo microservices in docker-compose.yaml")
    print("  2. Download NVIDIA containers (PyTorch, TensorFlow, Triton)")
    print("  3. Begin Phase 2: Model Development")
    print()


if __name__ == "__main__":
    main()
