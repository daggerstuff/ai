"""Configuration for NVIDIA NeMo Data Designer service."""

import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


@dataclass
class DataDesignerConfig:
    """Configuration for NeMo Data Designer client."""

    base_url: str = "http://localhost:8000"  # Default for local Docker Compose
    api_key: Optional[str] = None
    timeout: int = 300  # 5 minutes default timeout
    max_retries: int = 3
    batch_size: int = 1000

    @classmethod
    def from_env(cls) -> "DataDesignerConfig":
        """Create configuration from environment variables.

        Environment Variables:
        - NEMO_DATA_DESIGNER_BASE_URL: Base URL for the NeMo Data Designer service
          - Local Docker Compose: http://localhost:8000
          - Kubernetes deployment: https://nemo-data-designer.your-cluster-domain.com
          - Remote server: http://your-server-ip:8080
        - NVIDIA_API_KEY: Your NVIDIA API key for accessing NeMo services
        - NEMO_DATA_DESIGNER_TIMEOUT: Request timeout in seconds (default: 300)
        - NEMO_DATA_DESIGNER_MAX_RETRIES: Maximum retry attempts (default: 3)
        - NEMO_DATA_DESIGNER_BATCH_SIZE: Batch size for data generation (default: 1000)
        """
        return cls(
            base_url=os.getenv(
                "NEMO_DATA_DESIGNER_BASE_URL",
                "http://localhost:8000",
            ),
            api_key=os.getenv("NVIDIA_API_KEY"),
            timeout=int(os.getenv("NEMO_DATA_DESIGNER_TIMEOUT", "300")),
            max_retries=int(os.getenv("NEMO_DATA_DESIGNER_MAX_RETRIES", "3")),
            batch_size=int(os.getenv("NEMO_DATA_DESIGNER_BATCH_SIZE", "1000")),
        )

    def validate(self) -> None:
        """Validate configuration."""
        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY environment variable is required. "
                "Get your API key from https://build.nvidia.com/nemo/data-designer"
            )
        if not self.base_url:
            raise ValueError("base_url cannot be empty")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

