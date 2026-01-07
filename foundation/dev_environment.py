"""
Development environment configuration and validation for therapeutic AI.

Provides:
- PyTorch/TensorFlow setup validation
- GPU capability detection
- Container environment initialization
- NeMo microservices configuration
"""

import os
import sys
from typing import Dict

import torch


class DevEnvironmentConfig:
    """Configuration for development environment."""

    def __init__(self):
        self.torch_version = torch.__version__
        self.cuda_available = torch.cuda.is_available()
        self.cuda_device_count = torch.cuda.device_count() if self.cuda_available else 0
        self.gpu_device_name = (
            torch.cuda.get_device_name(0) if self.cuda_available else None
        )
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    def validate(self) -> Dict[str, bool]:
        """Validate environment meets Phase 1 requirements."""
        return {
            "python_3_11_plus": self.python_version >= "3.11",
            "torch_installed": True,
            "cuda_available": self.cuda_available,
            "nemo_downloaded": os.path.exists(
                "/home/vivi/pixelated/ngc_public_therapeutic_resources/microservices/nemo-microservices-quickstart_v25.10"
            ),
        }

    def summary(self) -> str:
        """Human-readable environment summary."""
        lines = [
            "=== Development Environment Summary ===",
            f"Python: {self.python_version}",
            f"PyTorch: {self.torch_version}",
            f"CUDA Available: {self.cuda_available}",
        ]
        if self.cuda_available:
            lines.extend(
                [
                    f"CUDA Device Count: {self.cuda_device_count}",
                    f"GPU: {self.gpu_device_name}",
                ]
            )

        nemo_path = (
            "/home/vivi/pixelated/ngc_public_therapeutic_resources/microservices/"
            "nemo-microservices-quickstart_v25.10"
        )
        lines.append(f"NeMo Quickstart: {'✓' if os.path.exists(nemo_path) else '✗'}")

        checks = self.validate()
        lines.append("\nValidation Results:")
        for check_name, result in checks.items():
            status = "✓" if result else "✗"
            lines.append(f"  {status} {check_name}")

        return "\n".join(lines)


def initialize_dev_environment() -> DevEnvironmentConfig:
    """Initialize and validate development environment."""
    config = DevEnvironmentConfig()
    print(config.summary())
    return config


if __name__ == "__main__":
    initialize_dev_environment()
