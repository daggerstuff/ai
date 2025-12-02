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

    base_url: str = "http://localhost:8000"  # For local Docker Compose, use http://localhost:8000
    api_key: Optional[str] = None
    timeout: int = 300  # 5 minutes default timeout
    max_retries: int = 3
    batch_size: int = 1000

    @classmethod
    def from_env(cls) -> "DataDesignerConfig":
        """Create configuration from environment variables."""
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

