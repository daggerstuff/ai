"""
Main entry point for YouTube channel discovery system.

Can be run as:
- Module: python -m ai.sourcing.youtube
- Script: python ai/sourcing/youtube/__main__.py discovery --help
- Direct function call
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from ai.sourcing.youtube.api_impl import (
    get_api_quota_status,
    test_api_connection,
)
from ai.sourcing.youtube.channel_registry import ChannelRegistryDB
from ai.sourcing.youtube.models import (
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

    print("Testing YouTube API connection...\n")

    success = test_api_connection(api_key)
    return 0 if success else 1


def command_check_quota(args) -> int:
    """Check YouTube API quota status."""
    setup_logging(args.verbose)

    load_dotenv(".env.youtube.example", override=True)

    has_quota, used, limit = get_api_quota_status()

    print("Quota Status Check:")
    print(f"  Available: {'Yes' if has_quota else 'No'}")
    print(f"  Used Today: {used}/{limit} units")
    print(f"  Remaining: {limit - used} units")

    if not has_quota:
        print("\n⚠️  Recommendations:")
        print("  - Wait 24h for quota reset")
        print("  - Increase project quota in Google Cloud Console")
    else:
        print(f"\n  Estimated capacity: ~{(limit - used) // 100} searches")
        print(f"  Estimated capacity: ~{(limit - used) // 10} channel lookups")

    return 0


def command_list_registry(args) -> int:
    """List channels in the registry database."""
    setup_logging(args.verbose)

    db_path = args.registry or "channels.db"

    if not Path(db_path).exists():
        print(f"Error: Registry database not found: {db_path}")
        return 1

    print(f"Loading registry from: {db_path}\n")

    with ChannelRegistryDB(db_path) as registry:
        channels = registry.get_all_channels()

        if not channels:
            print("No channels found in registry.")
            return 0

        stats = registry.get_statistics()

        print("=" * 80)
        print(f"Channel Registry: {stats['total']:.0f} channels")
        print("=" * 80)
        print()
        print(f"By Status: {stats['total']:.0f} total")
        for status, count in sorted(stats["by_status"].items()):
            print(f"  {status}: {count}")
        print()
        print("Top Languages:")
        for lang, count in sorted(stats["by_language"].items(), key=lambda x: -x[1])[
            :10
        ]:
            print(f"  {lang}: {count}")
        print()
        print("Top Categories:")
        for cat, count in sorted(stats["by_category"].items(), key=lambda x: -x[1])[
            :10
        ]:
            print(f"  {cat}: {count}")
        print()
        print("=" * 80)
        print(f"{' #':<4} {'Channel':<40} {'Quality':<8} {'Videos':<8}")
        print("-" * 80)

        for i, channel in enumerate(channels[:50]):
            status_emoji = {
                ChannelStatus.ACTIVE: "🟢",
                ChannelStatus.AT_RISK: "🟡",
                ChannelStatus.INACTIVE: "🔴",
                ChannelStatus.REMOVED: "⛔",
                ChannelStatus.UNKNOWN: "⚪",
            }.get(channel.status, "⚪")

            categories = ", ".join([c.value for c in channel.categories[:3]])
            if len(channel.categories) > 3:
                categories = f"{categories}, ..."

            print(
                f"{status_emoji} {i + 1:<3} "
                f"{channel.channel_name:<40} "
                f"{channel.quality_score:<8.2f} "
                f"{channel.video_count:<8,}"
            )

        return 0


def command_init_db(args) -> int:
    """Initialize a new channel registry database."""
    setup_logging(args.verbose)

    db_path = args.db or "channels.db"

    print(f"Creating channel registry at: {db_path}")

    with ChannelRegistryDB(db_path) as registry:
        stats = registry.get_statistics()

    print("Registry initialized successfully.")
    print(f"  Total: {stats['total']} channels")
    print("  Ready for channel imports.")

    return 0


def command_import_channels(args) -> int:
    """Import channels from a JSON file."""
    setup_logging(args.verbose)

    load_dotenv(".env.youtube.example", override=True)

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    if not args.registry:
        args.registry = "channels.db"

    print(f"Importing channels from: {input_path}")
    print(f"Target registry: {args.registry}\n")

    import json

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
                if data.get("licensing"):
                    licensing_obj = LicensingInfo(**data["licensing"])
                else:
                    licensing_obj = LicensingInfo()

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
                row_id = registry.add_channel(channel)
                imported += 1

                print(f"  ✓ {channel.channel_name}: {row_id}")

            except Exception as e:
                print(f"  ✗ Skipping {data.get('channel_name', 'unknown')}: {e}")
                skipped += 1

        print(f"\nImport complete: {imported} channels, {skipped} skipped")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="YouTube Channel Discovery System for CPTSD Dataset",
        epilog="""
Examples:
  # Test API connection and quota
  python -m ai.sourcing.youtube test-connection

  # Check quota status
  python -m ai.sourcing.youtube check-quota

  # List channels in registry
  python -m ai.sourcing.youtube list-registry

  # Import channels from JSON
  python -m ai.sourcing.youtube import-channels -i channels.json

  # Initialize new database
  python -m ai.sourcing.youtube init-db -d channels.db
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Test connection
    test_parser = subparsers.add_parser(
        "test-connection", help="Test YouTube API connection"
    )
    test_parser.add_argument("--api-key", help="YouTube API key")
    test_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    test_parser.set_defaults(func=command_test_connection)

    # Check quota
    quota_parser = subparsers.add_parser("check-quota", help="Check YouTube API quota")
    quota_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    quota_parser.set_defaults(func=command_check_quota)

    # List registry
    list_parser = subparsers.add_parser(
        "list-registry", help="List channels in registry"
    )
    list_parser.add_argument("--registry", "-r", help="Registry database file path")
    list_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    list_parser.set_defaults(func=command_list_registry)

    # Import channels
    import_parser = subparsers.add_parser(
        "import-channels", help="Import channels from JSON file"
    )
    import_parser.add_argument(
        "--input", "-i", required=True, help="Input JSON file path"
    )
    import_parser.add_argument("--registry", "-r", help="Target registry database file")
    import_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    import_parser.set_defaults(func=command_import_channels)

    # Initialize database
    init_parser = subparsers.add_parser(
        "init-db", help="Initialize channel registry database"
    )
    init_parser.add_argument(
        "--db", "-d", default="channels.db", help="Database file path"
    )
    init_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    init_parser.set_defaults(func=command_init_db)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch
    return args.command.func(args)


if __name__ == "__main__":
    sys.exit(main())
