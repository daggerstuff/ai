#!/usr/bin/env python3
"""
YouTube Rate Limiting and Proxy Support Example.

This example demonstrates how to use the enhanced YouTube processor
with rate limiting, proxy rotation, and anti-detection measures.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


from ai.tools.utilities.core.pipelines.youtube_processor import (
    AntiDetectionConfig,
    ProxyConfig,
    RateLimitConfig,
    YouTubePlaylistProcessor,
)


async def example_conservative_rate_limiting():
    """Example with conservative rate limiting for production use."""

    # Conservative rate limiting configuration
    rate_config = RateLimitConfig(
        enabled=True,
        requests_per_minute=10,  # Very conservative
        requests_per_hour=300,  # 5 requests per minute average
        burst_limit=3,  # Max 3 concurrent requests
        backoff_factor=2.0,  # Double backoff on errors
        max_backoff=600,  # Max 10 minutes backoff
        jitter=True,  # Add randomness
        respect_429=True,  # Always respect rate limit responses
    )

    # Anti-detection configuration
    anti_detection = AntiDetectionConfig(
        enabled=True,
        randomize_user_agents=True,
        randomize_delays=True,
        min_delay=2.0,  # Minimum 2 seconds between requests
        max_delay=8.0,  # Maximum 8 seconds between requests
        use_cookies=True,
        simulate_browser=True,
        geo_bypass=True,
    )

    # Initialize processor
    processor = YouTubePlaylistProcessor(
        output_dir="conservative_output",
        max_concurrent=2,  # Reduced concurrency
        rate_limit_config=rate_config,
        anti_detection_config=anti_detection,
    )

    # Example URLs (replace with actual URLs)
    urls = [
        "https://www.youtube.com/playlist?list=PLexample1",
        "https://www.youtube.com/playlist?list=PLexample2",
    ]

    try:
        result = await processor.process_playlists_batch(urls)

        if result.success:
            pass
        else:
            for _error in result.errors[:3]:
                pass

    except Exception:
        pass


async def example_proxy_rotation():
    """Example with proxy rotation for high-volume processing."""

    # Proxy configuration with rotation
    proxy_config = ProxyConfig(
        enabled=True,
        proxy_list=[
            "http://proxy1.example.com:8080",
            "http://proxy2.example.com:8080",
            "http://proxy3.example.com:8080",
            "socks5://proxy4.example.com:1080",
        ],
        rotation_strategy="random",  # Random proxy selection
        proxy_timeout=30,
        max_retries_per_proxy=2,
        test_url="https://httpbin.org/ip",
    )

    # Moderate rate limiting with proxies
    rate_config = RateLimitConfig(
        enabled=True,
        requests_per_minute=20,  # Higher rate with proxies
        requests_per_hour=800,
        burst_limit=5,
        backoff_factor=1.5,
        max_backoff=300,
        jitter=True,
        respect_429=True,
    )

    # Initialize processor with proxy support
    processor = YouTubePlaylistProcessor(
        output_dir="proxy_output",
        max_concurrent=4,  # Higher concurrency with proxies
        rate_limit_config=rate_config,
        proxy_config=proxy_config,
    )

    # Example URLs
    urls = [
        "https://www.youtube.com/playlist?list=PLlarge_playlist1",
        "https://www.youtube.com/playlist?list=PLlarge_playlist2",
        "https://www.youtube.com/playlist?list=PLlarge_playlist3",
    ]

    try:
        result = await processor.process_playlists_batch(urls)

        # Generate detailed report
        processor.generate_processing_report(result)

    except Exception:
        pass


async def example_high_volume_processing():
    """Example for high-volume processing with all optimizations."""

    # Aggressive rate limiting for high volume
    rate_config = RateLimitConfig(
        enabled=True,
        requests_per_minute=30,  # Push the limits
        requests_per_hour=1200,  # 20 per minute average
        burst_limit=8,  # Higher burst
        backoff_factor=1.2,  # Gentle backoff
        max_backoff=120,  # Max 2 minutes backoff
        jitter=True,
        respect_429=True,
    )

    # Multiple proxy pools
    proxy_config = ProxyConfig(
        enabled=True,
        proxy_list=[
            # Residential proxies
            "http://residential1.proxy.com:8080",
            "http://residential2.proxy.com:8080",
            "http://residential3.proxy.com:8080",
            # Datacenter proxies
            "http://datacenter1.proxy.com:8080",
            "http://datacenter2.proxy.com:8080",
            # SOCKS5 proxies
            "socks5://socks1.proxy.com:1080",
            "socks5://socks2.proxy.com:1080",
        ],
        rotation_strategy="round_robin",  # Systematic rotation
        proxy_timeout=45,
        max_retries_per_proxy=3,
        test_url="https://httpbin.org/ip",
    )

    # Full anti-detection suite
    anti_detection = AntiDetectionConfig(
        enabled=True,
        randomize_user_agents=True,
        randomize_delays=True,
        min_delay=0.5,  # Faster for high volume
        max_delay=3.0,
        use_cookies=True,
        simulate_browser=True,
        geo_bypass=True,
    )

    # High-performance processor
    processor = YouTubePlaylistProcessor(
        output_dir="high_volume_output",
        max_concurrent=6,  # Maximum concurrency
        retry_attempts=5,  # More retries
        rate_limit_config=rate_config,
        proxy_config=proxy_config,
        anti_detection_config=anti_detection,
    )

    # Large batch of URLs
    urls = [
        f"https://www.youtube.com/playlist?list=PLbatch_{i}"
        for i in range(1, 11)  # 10 playlists
    ]

    try:
        result = await processor.process_playlists_batch(urls)

        # Comprehensive reporting
        report = processor.generate_processing_report(result)

        # Save report to file
        report_path = Path("high_volume_output") / "processing_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

    except Exception:
        pass


def example_configuration_templates():
    """Show configuration templates for different use cases."""

    templates = {
        "Development/Testing": {
            "rate_config": RateLimitConfig(
                enabled=True,
                requests_per_minute=5,
                requests_per_hour=100,
                burst_limit=2,
            ),
            "description": "Safe settings for development and testing",
        },
        "Production Conservative": {
            "rate_config": RateLimitConfig(
                enabled=True,
                requests_per_minute=15,
                requests_per_hour=500,
                burst_limit=3,
                backoff_factor=2.0,
                max_backoff=600,
            ),
            "description": "Conservative settings for production use",
        },
        "High-Volume with Proxies": {
            "rate_config": RateLimitConfig(
                enabled=True,
                requests_per_minute=25,
                requests_per_hour=1000,
                burst_limit=6,
                backoff_factor=1.3,
                max_backoff=300,
            ),
            "proxy_config": ProxyConfig(enabled=True, rotation_strategy="random", max_retries_per_proxy=2),
            "description": "Optimized for high-volume processing with proxy rotation",
        },
        "Stealth Mode": {
            "rate_config": RateLimitConfig(
                enabled=True,
                requests_per_minute=8,
                requests_per_hour=200,
                burst_limit=2,
                jitter=True,
            ),
            "anti_detection": AntiDetectionConfig(
                enabled=True,
                randomize_user_agents=True,
                randomize_delays=True,
                min_delay=3.0,
                max_delay=10.0,
                use_cookies=True,
                simulate_browser=True,
            ),
            "description": "Maximum stealth with human-like behavior patterns",
        },
    }

    for _name, config in templates.items():
        if "rate_config" in config:
            config["rate_config"]


async def main():
    """Run all examples."""

    # Note: These examples use placeholder URLs and proxies

    # Show configuration templates
    example_configuration_templates()

    # Run examples (commented out to avoid actual processing)
    # await example_conservative_rate_limiting()
    # await example_proxy_rotation()
    # await example_high_volume_processing()


if __name__ == "__main__":
    asyncio.run(main())
