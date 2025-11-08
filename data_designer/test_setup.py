"""Quick test to verify NeMo Data Designer setup."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_config():
    """Test configuration loading."""
    print("Testing NeMo Data Designer Configuration...")
    print("=" * 60)

    # Check API key
    api_key = os.getenv("NVIDIA_API_KEY")
    if api_key:
        # Mask the key for display
        masked_key = api_key[:10] + "..." + api_key[-10:] if len(api_key) > 20 else "***"
        print(f"✅ NVIDIA_API_KEY is set: {masked_key}")
    else:
        print("❌ NVIDIA_API_KEY is not set")
        return False

    # Test imports
    try:
        from ai.data_designer.config import DataDesignerConfig
        from ai.data_designer.service import NeMoDataDesignerService
        print("✅ Imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test configuration
    try:
        config = DataDesignerConfig.from_env()
        config.validate()
        print("✅ Configuration is valid")
        print(f"   Base URL: {config.base_url}")
        print(f"   Timeout: {config.timeout}s")
        print(f"   Max Retries: {config.max_retries}")
        print(f"   Batch Size: {config.batch_size}")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

    # Test service initialization
    try:
        service = NeMoDataDesignerService()
        print("✅ Service initialized successfully")
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False

    print("=" * 60)
    print("✅ All tests passed! NeMo Data Designer is ready to use.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1)

