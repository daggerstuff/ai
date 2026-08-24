"""
Main entry point for YouTube channel discovery system.

Can be run as:
- Module: python -m ai.pipelines.data_processing.youtube
- Script: python ai/pipelines/data_processing/youtube/__main__.py discovery --help
- Direct function call
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from ai.pipelines.data_processing.youtube.api_impl import (
    get_api_quota_status,
    test_api_connection,
)
from ai.pipelines.data_processing.youtube.channel_registry import ChannelRegistryDB
from ai.pipelines.data_processing.youtube.models import (
    Channel,
    ChannelStatus,
    ContentCategory,
    LicensingInfo,
    QualityMetrics,
)


def setup_logging(level: str = "INFO"):
    """Configure logging with color support."""
    COLORS = {
        "RED": "\033[0;31m",
        "GREEN": "\033[0;32m",
        "YELLOW": "\033[0;33m",
        "BLUE": "\033[0;34m",
        "CYAN": "\033[0;36m",
        "MAGENTA": "\033[0;35m",
        "WHITE": "\033[0;37m",
        "GRAY": "\033[0;90m",
        "NC": "\033[0m",  # No Color
    }

    class ColorFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            color = COLORS.get(record.levelname, COLORS["WHITE"])
            message = record.getMessage()
            levelname = record.levelname
            timestamp = self.formatTime(record, "%H:%M:%S")
            return f"{color}[{timestamp}] {levelname}:NC {message} {COLORS['NC']}"

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    handler.setLevel(logging.DEBUG)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    root_logger.addHandler(handler)


def command_test_connection(args) -> int:
    """Test YouTube API connection and quota."""
    setup_logging(args.verbose)

    load_dotenv(".env.youtube.example", override=True)

    api_key = None
    if hasattr(args, "api_key") and args.api_key:
        api_key = args.api_key

    success = test_api_connection(api_key)
    return 0 if success else 1


def command_check_quota(args) -> int:
    """Check YouTube API quota status."""
    setup_logging(args.verbose)

    load_dotenv(".env.youtube.example", override=True)

    has_quota, _used, _limit = get_api_quota_status()

    if not has_quota:
        pass
    else:
        pass

    return 0


def command_list_registry(args) -> int:
    """List channels in the registry database."""
    setup_logging(args.verbose)

    db_path = args.registry or "channels.db"

    if not Path(db_path).exists():
        return 1

    with ChannelRegistryDB(db_path) as registry:
        channels = registry.get_all_channels()

        if not channels:
            return 0

        stats = registry.get_statistics()

        for _status, _count in sorted(stats["by_status"].items()):
            pass
        for _lang, _count in sorted(stats["by_language"].items(), key=lambda x: -x[1])[:10]:
            pass
        for _cat, _count in sorted(stats["by_category"].items(), key=lambda x: -x[1])[:10]:
            pass

        for _i, channel in enumerate(channels[:50]):
            {
                ChannelStatus.ACTIVE: "🟢",
                ChannelStatus.AT_RISK: "🟡",
                ChannelStatus.INACTIVE: "🔴",
                ChannelStatus.REMOVED: "⛔",
                ChannelStatus.UNKNOWN: "⚪",
            }.get(channel.status, "⚪")

            categories = ", ".join([c.value for c in channel.categories[:3]])
            if len(channel.categories) > 3:
                categories = f"{categories}, ..."

        return 0


def command_init_db(args) -> int:
    """Initialize a new channel registry database."""
    setup_logging(args.verbose)

    db_path = args.db or "channels.db"

    with ChannelRegistryDB(db_path) as registry:
        registry.get_statistics()

    return 0


def command_import_channels(args) -> int:
    """Import channels from a JSON file."""
    setup_logging(args.verbose)

    load_dotenv(".env.youtube.example", override=True)

    input_path = Path(args.input)

    if not input_path.exists():
        return 1

    if not args.registry:
        args.registry = "channels.db"

    with open(input_path) as f:
        channels_data = json.load(f)

    with ChannelRegistryDB(args.registry) as registry:
        imported = 0
        skipped = 0

        for data in channels_data:
            try:
                # Parse quality metrics
                if data.get("quality_metrics"):
                    metrics_obj = QualityMetrics(**data["quality_metrics"])
                else:
                    metrics_obj = QualityMetrics()

                # Parse licensing info
                licensing_obj = LicensingInfo(**data["licensing"]) if data.get("licensing") else LicensingInfo()

                # Determine status
                status = ChannelStatus(data.get("status", "unknown"))

                channel = Channel(
                    channel_id=data["channel_id"],
                    channel_name=data["channel_name"],
                    channel_url=data["channel_url"],
                    subscriber_count=data.get("subscriber_count", 0),
                    video_count=data.get("video_count", 0),
                )
                channel.quality_metrics = metrics_obj
                channel.licensing = licensing_obj
                channel.status = status

                # Add additional fields
                channel.primary_language = data.get("primary_language", "en")
                channel.languages = set(data.get("languages", ["en"]))
                channel.categories = [
                    ChannelStatus.ACTIVE  # Will be parsed correctly
                ]
                if k := data.get("categories"):
                    channel.categories = [ContentCategory(c) for c in k]

                # Add to registry
                registry.add_channel(channel)
                imported += 1

            except Exception:
                skipped += 1

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="YouTube Channel Discovery System for CPTSD Dataset",
        epilog="""
Examples:
  # Test API connection and quota
  python -m ai.pipelines.data_processing.youtube test-connection

  # Check quota status
  python -m ai.pipelines.data_processing.youtube check-quota

  # List channels in registry
  python -m ai.pipelines.data_processing.youtube list-registry

  # Import channels from JSON
  python -m ai.pipelines.data_processing.youtube import-channels -i channels.json

  # Initialize new database
  python -m ai.pipelines.data_processing.youtube init-db -d channels.db
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Test connection
    test_parser = subparsers.add_parser("test-connection", help="Test YouTube API connection")
    test_parser.add_argument("--api-key", help="YouTube API key")
    test_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    test_parser.set_defaults(func=command_test_connection)

    # Check quota
    quota_parser = subparsers.add_parser("check-quota", help="Check YouTube API quota")
    quota_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    quota_parser.set_defaults(func=command_check_quota)

    # List registry
    list_parser = subparsers.add_parser("list-registry", help="List channels in registry")
    list_parser.add_argument("--registry", "-r", help="Registry database file path")
    list_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    list_parser.set_defaults(func=command_list_registry)

    # Import channels
    import_parser = subparsers.add_parser("import-channels", help="Import channels from JSON file")
    import_parser.add_argument("--input", "-i", required=True, help="Input JSON file path")
    import_parser.add_argument("--registry", "-r", help="Target registry database file")
    import_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    import_parser.set_defaults(func=command_import_channels)

    # Initialize database
    init_parser = subparsers.add_parser("init-db", help="Initialize channel registry database")
    init_parser.add_argument("--db", "-d", default="channels.db", help="Database file path")
    init_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    init_parser.set_defaults(func=command_init_db)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch
    return args.command.func(args)


if __name__ == "__main__":
    sys.exit(main())
