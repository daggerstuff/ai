#!/usr/bin/env python3
"""
Test script for NGC CLI integration.

Run this to verify NGC CLI is properly installed and configured.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.utils.ngc_cli import NGCCLIAuthError, get_ngc_cli

# Configure logging for test output
logging.basicConfig(
    level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def test_ngc_cli() -> bool:
    """Test NGC CLI availability and configuration"""
    logger.info("Testing NGC CLI integration...")
    logger.info("=" * 60)

    # Test 1: Check if NGC CLI is available
    logger.info("\n1. Checking NGC CLI availability...")
    try:
        cli = get_ngc_cli()
        if cli.is_available():
            logger.info("✅ NGC CLI found!")
            logger.info(f"   Command: {cli.ngc_cmd}")
        else:
            logger.error("❌ NGC CLI not found")
            _extracted_from_test_ngc_cli_15(
                "\n   To install:",
                "   1. Visit https://catalog.ngc.nvidia.com",
                "   2. Download NGC CLI",
                "   3. Extract to ~/ngc-cli/ or add to PATH",
            )
            return False
    except Exception as e:
        logger.error(f"❌ Error checking NGC CLI: {e}")
        return False

    # Test 2: Check configuration
    logger.info("\n2. Checking NGC CLI configuration...")
    try:
        config = cli.check_config()
        logger.info("✅ NGC CLI is configured!")
        if config.get("API key"):
            logger.info("   API key: Configured")
        if config.get("Organization"):
            logger.info(f"   Organization: {config.get('Organization')}")
        if config.get("Team"):
            logger.info(f"   Team: {config.get('Team')}")
    except NGCCLIAuthError as e:
        logger.warning("⚠️  NGC CLI not configured")
        logger.warning(f"   {e}")
        _extracted_from_test_ngc_cli_15(
            "\n   To configure:",
            "   1. Get API key from https://catalog.ngc.nvidia.com",
            "   2. Run: ngc config set",
            "   3. Or set: export NGC_API_KEY='your-key'",
        )
        return False
    except Exception as e:
        logger.error(f"❌ Error checking config: {e}")
        return False

    # Test 3: Try listing resources (optional, may require network)
    logger.info("\n3. Testing resource access...")
    try:
        resources = cli.list_resources()
        logger.info(f"✅ Can access NGC catalog (found {len(resources)} resources)")
    except Exception as e:
        logger.warning(f"⚠️  Could not list resources: {e}")
        logger.info("   (This is okay - may require network or specific permissions)")

    logger.info("\n" + "=" * 60)
    _extracted_from_test_ngc_cli_15(
        "✅ NGC CLI integration test complete!",
        "\nNext steps:",
        "  - Use NGCResourceDownloader in training_ready",
        "  - Use NGCIngestor in dataset_pipeline",
    )
    logger.info("  - See docs/NGC_CLI_INTEGRATION.md for usage examples")

    return True


# TODO Rename this here and in `test_ngc_cli`
def _extracted_from_test_ngc_cli_15(arg0, arg1, arg2, arg3):
    logger.info(arg0)
    logger.info(arg1)
    logger.info(arg2)
    logger.info(arg3)


if __name__ == "__main__":
    success = test_ngc_cli()
    sys.exit(0 if success else 1)
