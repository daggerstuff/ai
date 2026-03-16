"""
Command-line interface for YouTube channel discovery.

Provides CLI commands for:
- Running discovery pipeline
- Managing channel registry
- Health monitoring
- Report generation
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from ai.sourcing.youtube.api import ChannelHunterConfig, ChannelQualityThresholds
from ai.sourcing.youtube.models import Channel, ChannelRegistry, ChannelStatus
from ai.sourcing.youtube.monitoring import ChannelMonitor, health_check_channel
from ai.sourcing.youtube.processor import run_pipeline

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )


def cmd_discover(args):
    """Run channel discovery pipeline."""
    setup_logging(args.verbose)

    if not args.api_key:
        print("Error: YouTube API key required. Use --api-key or set YOUTUBE_API_KEY environment variable.")
        return 1

    print(f"Starting channel discovery...")
    print(f"  Target channels: {args.channels}")
    print(f"  Min subscribers: {args.min_subscribers:,}")
    print(f"  Min videos: {args.min_videos}")
    print(f"  Output: {args.output}\n")

    progress = None
    if args.verbose:
        def progress_callback(percent, step):
            print(f"  [{percent*100:5.1f}%] {step}")
    else:
        def progress_callback(percent, step):
            # No output during progress to keep it clean
            pass

    try:
        results, report = run_pipeline(
            api_key=args.api_key,
            target_channels=args.channels,
            output_path=args.output,
            progress_callback=progress_callback,
        )

        print(f"\n{'-'*60}")
        print(f"Discovery Complete!")
        print(f"{'-'*60}")
        print(f"\nResults:")
        print(f"  Qualified: {len(results.qualified_channels)} / {args.channels} target")
        print(f"  Rejected: {len(results.rejected_channels)}")
        print(f"  Total evaluated: {len(results.found_channels)}")
        print(f"  Languages: {len(results.languages)}")
        print(f"  Categories: {len(results.categories)}")
        print(f"\n  Statistics:")
        print(f"    Total subscribers: {results.total_subscribers:,}")
        print(f"    Total videos: {results.total_videos:,}")
        print(f"    Professional sources: {results.professional_count}")

        if args.report:
            report_path = Path(args.report)
            report_path.write_text(report)
            print(f"\nReport saved to: {report_path}")

        # Show top 5 channels
        if results.qualified_channels:
            print(f"\n{'-'*60}")
            print("Top 5 Qualified Channels:")
            print(f"{'-'*60}")
            for i, channel in enumerate(results.qualified_channels[:5]):
                cats = ", ".join([c.value for c in channel.categories])
                print(f"{i+1}. {channel.channel_name}")
                print(f"   URL: {channel.channel_url}")
                print(f"   Quality: {channel.quality_score:.2f} | "
                      f"Subs: {channel.subscriber_count:,} | "
                      f"Videos: {channel.video_count:,}")
                print(f"   Categories: {cats}")
                print(f"   Professional: {'Yes' if channel.is_professional else 'No'}")
                print()

        return 0

    except Exception as e:
        logger.error(f"Discovery failed: {e}", exc_info=args.verbose)
        print(f"Error: {e}")
        return 1


def cmd_check(args):
    """Health check a specific channel."""
    setup_logging(args.verbose)

    if not args.channel_id:
        print("Error: --channel-id required")
        return 1

    print(f"Checking channel health: {args.channel_id}\n")

    # Create a minimal channel object for checking
    # In practice, you'd fetch actual channel data
    channel = Channel(
        channel_id=args.channel_id,
        channel_name=args.channel_id,
        channel_url=f"https://www.youtube.com/channel/{args.channel_id}",
    )

    result = health_check_channel(channel)

    print(f"Status: {result['status']}")
    print(f"Activity Status: {result['activity_status']}")
    print(f"Health Score: {result['health_score']:.2f}")
    print(f"Quality Score: {result['quality_score']:.2f}")
    print(f"\nNotes:")
    for note in result['notes']:
        print(f"  • {note}")

    if result['alerts']:
        print(f"\nAlerts:")
        for alert in result['alerts']:
            print(f"  [{alert['severity'].upper()}] {alert['name']}: {alert['description']}")

    return 0


def cmd_import(args):
    """Import channels from JSON file."""
    setup_logging(args.verbose)

    if not args.input:
        print("Error: --input required")
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {args.input}")
        return 1

    print(f"Importing channels from {input_path}\n")

    with open(input_path) as f:
        channels_data = json.load(f)

    registry = ChannelRegistry()

    for data in channels_data:
        channel = Channel(
            channel_id=data['channel_id'],
            channel_name=data['channel_name'],
            channel_url=data['channel_url'],
            subscriber_count=data.get('subscriber_count', 0),
            video_count=data.get('video_count', 0),
            total_views=data.get('total_views', 0),
        )
        channel.categories = [
            ChannelStatus.ACTIVE  # We'll just mark as active
        ]
        channel.status = ChannelStatus[data.get('status', 'UNKNOWN')]
        channel.quality_score = data.get('quality_score', 0.0)

        registry.add_channel(channel)

    print(f"{'-'*60}")
    print(f"Import Complete!")
    print(f"{'-'*60}")
    print(registry.summary())

    return 0


def cmd_list(args):
    """List channels from an existing registry file."""
    setup_logging(args.verbose)

    if not args.registry:
        # Try default location
        registry_path = Path("qualified_channels.json")
        if not registry_path.exists():
            print("Error: Registry file not found. Use --registry or ensure qualified_channels.json exists.")
            return 1
    else:
        registry_path = Path(args.registry)

    print(f"Loading registry from: {registry_path}\n")

    with open(registry_path) as f:
        channels_data = json.load(f)

    print(f"{" #":<4} {"Channel":<40} {"Quality":<8} {"Subs":<12} {"Videos":<8} {"License"}")
    print("-" * 100)

    for i, data in enumerate(channels_data):
        # Get licensing info
        lic = data.get('licensing', {})
        lic_status = "CC" if lic and lic.get('cc_license') else "Unknown"

        print(f"{i+1:<4} {data['channel_name']:<40} {data['quality_score']:<8.2f} "
              f"{data['subscriber_count']:<12,} {data['video_count']:<8,} {lic_status}")

    print(f"\nTotal: {len(channels_data)} channels")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="YouTube Channel Discovery for CPTSD Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run discovery pipeline
  python -m ai.sourcing.youtube.cli discover --api-key YOUR_KEY

  # Check a specific channel
  python -m ai.sourcing.youtube.cli check --channel-id UCxxxxxxxxxxxxxx

  # Import existing channels
  python -m ai.sourcing.youtube.cli import --input channels.json

  # List channels in registry
  python -m ai.sourcing.youtube.cli list
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Discover command
    discover_parser = subparsers.add_parser('discover', help='Run channel discovery')
    discover_parser.add_argument(
        '--api-key',
        help='YouTube Data API key (or set YOUTUBE_API_KEY env var)',
    )
    discover_parser.add_argument(
        '--channels',
        type=int,
        default=50,
        help='Target number of channels to discover (default: 50)',
    )
    discover_parser.add_argument(
        '--min-subs', '--min-subscribers',
        type=int,
        default=1000,
        dest='min_subscribers',
        help='Minimum subscriber count (default: 1000)',
    )
    discover_parser.add_argument(
        '--min-videos',
        type=int,
        default=20,
        help='Minimum video count (default: 20)',
    )
    discover_parser.add_argument(
        '--output',
        '-o',
        default='qualified_channels.json',
        help='Output JSON file path (default: qualified_channels.json)',
    )
    discover_parser.add_argument(
        '--report',
        '-r',
        help='Save markdown report to file',
    )
    discover_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output',
    )

    # Check command
    check_parser = subparsers.add_parser('check', help='Channel health check')
    check_parser.add_argument(
        '--channel-id',
        help='YouTube channel ID',
    )
    check_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output',
    )

    # Import command
    import_parser = subparsers.add_parser('import', help='Import channels from JSON')
    import_parser.add_argument(
        '--input',
        '-i',
        help='Input JSON file path',
    )
    import_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output',
    )

    # List command
    list_parser = subparsers.add_parser('list', help='List channels in registry')
    list_parser.add_argument(
        '--registry',
        help='Registry JSON file path (default: qualified_channels.json)',
    )
    list_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output',
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handler
    if args.command == 'discover':
        return cmd_discover(args)
    elif args.command == 'check':
        return cmd_check(args)
    elif args.command == 'import':
        return cmd_import(args)
    elif args.command == 'list':
        return cmd_list(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
