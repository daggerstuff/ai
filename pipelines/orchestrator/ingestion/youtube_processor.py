"""
YouTube Playlist Processing Infrastructure for Voice Training Data.

This module provides comprehensive YouTube playlist processing capabilities
for the voice training pipeline, including batch processing, error handling,
progress tracking, and integration with the dataset pipeline.
"""

import asyncio
import json
import random
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from logger import setup_logger


@dataclass
class ProxyConfig:
    """Configuration for proxy usage."""

    enabled: bool = False
    proxy_list: list[str] = field(default_factory=list)
    rotation_strategy: str = "random"  # "random", "round_robin", "sticky"
    proxy_timeout: int = 30
    max_retries_per_proxy: int = 2
    test_url: str = "https://httpbin.org/ip"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    enabled: bool = True
    requests_per_minute: int = 30  # Conservative default
    requests_per_hour: int = 1000
    burst_limit: int = 5  # Max concurrent requests
    backoff_factor: float = 1.5  # Exponential backoff multiplier
    max_backoff: int = 300  # Max backoff time in seconds
    jitter: bool = True  # Add random jitter to requests
    respect_429: bool = True  # Respect HTTP 429 responses


@dataclass
class AntiDetectionConfig:
    """Configuration for anti-detection measures."""

    enabled: bool = True
    randomize_user_agents: bool = True
    randomize_delays: bool = True
    min_delay: float = 1.0  # Minimum delay between requests
    max_delay: float = 5.0  # Maximum delay between requests
    use_cookies: bool = True
    simulate_browser: bool = True
    geo_bypass: bool = True


@dataclass
class PlaylistInfo:
    """Information about a YouTube playlist."""

    id: str
    url: str
    title: str | None = None
    video_count: int | None = None
    duration: str | None = None
    uploader: str | None = None


@dataclass
class ProcessingResult:
    """Result of processing a single playlist."""

    playlist_id: str
    success: bool
    audio_files: list[Path] = field(default_factory=list)
    failed_videos: list[str] = field(default_factory=list)
    processing_time: float = 0.0
    error_message: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class BatchProcessingResult:
    """Result of batch processing multiple playlists."""

    total_playlists: int
    successful_playlists: int
    failed_playlists: int
    total_audio_files: int
    total_processing_time: float
    results: list[ProcessingResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class RateLimiter:
    """Rate limiting implementation for YouTube requests."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_times = []
        self.last_request_time = 0
        self.current_backoff = 1

    async def acquire(self):
        """Acquire permission to make a request."""
        if not self.config.enabled:
            return None

        now = time.time()

        # Clean old request times (older than 1 hour)
        cutoff_time = now - 3600
        self.request_times = [t for t in self.request_times if t > cutoff_time]

        # Check hourly limit
        if len(self.request_times) >= self.config.requests_per_hour:
            wait_time = 3600 - (now - self.request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                return await self.acquire()

        # Check per-minute limit
        minute_cutoff = now - 60
        recent_requests = [t for t in self.request_times if t > minute_cutoff]

        if len(recent_requests) >= self.config.requests_per_minute:
            wait_time = 60 - (now - recent_requests[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                return await self.acquire()

        # Add jitter and minimum delay
        if self.config.jitter and self.last_request_time > 0:
            min_interval = 60.0 / self.config.requests_per_minute
            elapsed = now - self.last_request_time
            if elapsed < min_interval:
                jitter = random.uniform(0, min_interval * 0.1)
                await asyncio.sleep(min_interval - elapsed + jitter)

        self.request_times.append(time.time())
        self.last_request_time = time.time()
        return None

    async def handle_rate_limit_error(self):
        """Handle rate limit errors with exponential backoff."""
        if self.config.respect_429:
            backoff_time = min(
                self.current_backoff * self.config.backoff_factor,
                self.config.max_backoff,
            )

            if self.config.jitter:
                backoff_time *= random.uniform(0.8, 1.2)

            await asyncio.sleep(backoff_time)
            self.current_backoff = min(
                self.current_backoff * self.config.backoff_factor,
                self.config.max_backoff,
            )

    def reset_backoff(self):
        """Reset backoff after successful request."""
        self.current_backoff = 1


class ProxyManager:
    """Proxy rotation and management for YouTube requests."""

    def __init__(self, config: ProxyConfig):
        self.config = config
        self.proxies = config.proxy_list.copy()
        self.current_proxy_index = 0
        self.proxy_failures = {}
        self.logger = setup_logger("proxy_manager")

    def get_current_proxy(self) -> str | None:
        """Get the current proxy to use."""
        if not self.config.enabled or not self.proxies:
            return None

        if self.config.rotation_strategy == "round_robin":
            proxy = self.proxies[self.current_proxy_index]
            self.current_proxy_index = (self.current_proxy_index + 1) % len(
                self.proxies
            )
            return proxy
        if self.config.rotation_strategy == "random":
            return random.choice(self.proxies)
        if self.config.rotation_strategy == "sticky":
            return self.proxies[0] if self.proxies else None

        return None

    def mark_proxy_failed(self, proxy: str):
        """Mark a proxy as failed."""
        if proxy:
            self.proxy_failures[proxy] = self.proxy_failures.get(proxy, 0) + 1
            if (
                self.proxy_failures[proxy] >= self.config.max_retries_per_proxy
                and proxy in self.proxies
            ):
                self.proxies.remove(proxy)
                self.logger.warning(f"Removed failed proxy: {proxy}")

    def mark_proxy_success(self, proxy: str):
        """Mark a proxy as successful."""
        if proxy and proxy in self.proxy_failures:
            self.proxy_failures[proxy] = max(0, self.proxy_failures[proxy] - 1)


class YouTubePlaylistProcessor:
    """
    Comprehensive YouTube playlist processor for voice training data.

    Features:
    - Batch processing of multiple playlists
    - Robust error handling and retry logic
    - Progress tracking and detailed logging
    - Audio quality control and preprocessing
    - Integration with dataset pipeline
    """

    def __init__(
        self,
        output_dir: str = "voice_data",
        audio_format: str = "wav",
        audio_quality: str = "0",  # Best quality
        max_concurrent: int = 3,
        retry_attempts: int = 3,
        retry_delay: float = 5.0,
        rate_limit_config: RateLimitConfig | None = None,
        proxy_config: ProxyConfig | None = None,
        anti_detection_config: AntiDetectionConfig | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.audio_format = audio_format
        self.audio_quality = audio_quality
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Initialize rate limiting and proxy management
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.proxy_config = proxy_config or ProxyConfig()
        self.anti_detection_config = anti_detection_config or AntiDetectionConfig()

        self.rate_limiter = RateLimiter(self.rate_limit_config)
        self.proxy_manager = ProxyManager(self.proxy_config)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_file = (
            self.output_dir
            / f"youtube_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.logger = setup_logger("youtube_processor", str(log_file))

        # User agents for anti-detection
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
        ]

        # Track processing state
        self.processed_playlists: dict[str, ProcessingResult] = {}
        self.failed_playlists: list[str] = []

    def validate_youtube_url(self, url: str) -> bool:
        """Validate YouTube URL format and safety."""
        try:
            parsed = urlparse(url)
            if parsed.netloc not in ["www.youtube.com", "youtube.com", "youtu.be"]:
                return False
            # Check for playlist or video URL
            return (
                "playlist" in parsed.query
                or "watch" in parsed.path
                or "youtu.be" in parsed.netloc
            )
        except Exception:
            return False

    def extract_playlist_id(self, url: str) -> str | None:
        """Extract playlist ID from YouTube URL."""
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)

            return query_params["list"][0] if "list" in query_params else None
        except Exception:
            return None

    def get_playlist_info(self, url: str) -> PlaylistInfo | None:
        """Get metadata information about a playlist."""
        try:
            cmd = [
                "yt-dlp",
                "--dump-json",
                "--flat-playlist",
                "--playlist-end",
                "1",  # Just get playlist info, not all videos
                url,
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=False
            )

            if result.returncode == 0 and result.stdout:
                # Parse first line of JSON output
                first_line = result.stdout.strip().split("\n")[0]
                data = json.loads(first_line)

                playlist_id = self.extract_playlist_id(url) or data.get(
                    "playlist_id", "unknown"
                )

                return PlaylistInfo(
                    id=playlist_id,
                    url=url,
                    title=data.get("playlist_title") or data.get("title"),
                    video_count=data.get("playlist_count"),
                    uploader=data.get("uploader") or data.get("channel"),
                )

            return None

        except Exception as e:
            self.logger.warning(f"Failed to get playlist info for {url}: {e}")
            return None

    def _build_yt_dlp_command(
        self, url: str, output_path: Path, proxy: str | None = None
    ) -> list[str]:
        """Build yt-dlp command with rate limiting and anti-detection options."""
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format",
            self.audio_format,
            "--audio-quality",
            self.audio_quality,
            "--output",
            str(output_path / "%(title)s.%(ext)s"),
            "--write-info-json",
            "--write-description",
            "--write-thumbnail",
        ]

        # Anti-detection measures
        if self.anti_detection_config.enabled:
            if self.anti_detection_config.randomize_user_agents:
                user_agent = random.choice(self.user_agents)
                cmd.extend(["--user-agent", user_agent])

            if self.anti_detection_config.simulate_browser:
                cmd.extend(
                    [
                        "--add-header",
                        "Accept:text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "--add-header",
                        "Accept-Language:en-US,en;q=0.5",
                        "--add-header",
                        "Accept-Encoding:gzip, deflate",
                        "--add-header",
                        "DNT:1",
                        "--add-header",
                        "Connection:keep-alive",
                        "--add-header",
                        "Upgrade-Insecure-Requests:1",
                    ]
                )

            if self.anti_detection_config.use_cookies:
                cmd.extend(["--cookies-from-browser", "chrome"])

            if self.anti_detection_config.geo_bypass:
                cmd.append("--geo-bypass")

        # Proxy configuration
        if proxy:
            cmd.extend(["--proxy", proxy])

        # Rate limiting options
        if self.rate_limit_config.enabled:
            # Add sleep interval between downloads
            sleep_interval = 60.0 / self.rate_limit_config.requests_per_minute
            if self.anti_detection_config.randomize_delays:
                min_delay = max(
                    sleep_interval * 0.5, self.anti_detection_config.min_delay
                )
                max_delay = min(
                    sleep_interval * 1.5, self.anti_detection_config.max_delay
                )
                sleep_interval = random.uniform(min_delay, max_delay)

            cmd.extend(["--sleep-interval", str(sleep_interval)])

        cmd.append(url)
        return cmd

    async def download_playlist_audio(
        self, url: str, output_path: Path
    ) -> ProcessingResult:
        """Download audio from a single playlist with error handling."""
        start_time = time.time()
        playlist_id = self.extract_playlist_id(url) or f"unknown_{int(time.time())}"

        self.logger.info(f"Starting download for playlist: {playlist_id}")

        # Create playlist-specific directory
        playlist_dir = output_path / f"playlist_{playlist_id}"
        playlist_dir.mkdir(exist_ok=True)

        # Apply rate limiting
        await self.rate_limiter.acquire()

        # Get proxy if configured
        proxy = self.proxy_manager.get_current_proxy()
        if proxy:
            self.logger.info(f"Using proxy: {proxy}")

        # Build yt-dlp command with enhanced options
        cmd = self._build_yt_dlp_command(url, playlist_dir, proxy)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                check=False,
            )

            processing_time = time.time() - start_time

            # Check for audio files
            audio_files = list(playlist_dir.glob(f"*.{self.audio_format}"))

            if result.returncode == 0 and audio_files:
                self.logger.info(
                    f"Successfully downloaded {len(audio_files)} audio files for playlist {playlist_id}"
                )

                # Mark proxy as successful if used
                if proxy:
                    self.proxy_manager.mark_proxy_success(proxy)

                # Reset rate limiter backoff on success
                self.rate_limiter.reset_backoff()

                return ProcessingResult(
                    playlist_id=playlist_id,
                    success=True,
                    audio_files=audio_files,
                    processing_time=processing_time,
                    metadata={
                        "output_dir": str(playlist_dir),
                        "audio_format": self.audio_format,
                        "command": " ".join(cmd),
                        "proxy_used": proxy,
                    },
                )
            error_msg = (
                f"Download failed for playlist {playlist_id}: {result.stderr[:500]}"
            )
            self.logger.error(error_msg)

            # Handle rate limiting and proxy errors
            if "429" in result.stderr or "rate limit" in result.stderr.lower():
                await self.rate_limiter.handle_rate_limit_error()
                self.logger.warning("Rate limit detected, applied backoff")

            if proxy and (
                "proxy" in result.stderr.lower() or result.returncode in [1, 2]
            ):
                self.proxy_manager.mark_proxy_failed(proxy)
                self.logger.warning(f"Marked proxy as failed: {proxy}")

            return ProcessingResult(
                playlist_id=playlist_id,
                success=False,
                processing_time=processing_time,
                error_message=error_msg,
            )

        except subprocess.TimeoutExpired:
            error_msg = f"Download timeout for playlist {playlist_id}"
            self.logger.error(error_msg)
            return ProcessingResult(
                playlist_id=playlist_id,
                success=False,
                processing_time=time.time() - start_time,
                error_message=error_msg,
            )
        except Exception as e:
            error_msg = f"Unexpected error downloading playlist {playlist_id}: {e}"
            self.logger.error(error_msg)
            return ProcessingResult(
                playlist_id=playlist_id,
                success=False,
                processing_time=time.time() - start_time,
                error_message=error_msg,
            )

    async def process_single_playlist(self, url: str) -> ProcessingResult:
        """Process a single playlist with retry logic."""
        if not self.validate_youtube_url(url):
            return ProcessingResult(
                playlist_id="invalid",
                success=False,
                error_message=f"Invalid YouTube URL: {url}",
            )

        if playlist_info := self.get_playlist_info(url):
            self.logger.info(
                f"Processing playlist: {playlist_info.title} ({playlist_info.video_count} videos)"
            )

        # Attempt download with retries
        last_result = None
        for attempt in range(self.retry_attempts):
            if attempt > 0:
                self.logger.info(f"Retry attempt {attempt + 1} for {url}")
                await asyncio.sleep(self.retry_delay)

            result = await self.download_playlist_audio(url, self.output_dir)

            if result.success:
                self.processed_playlists[result.playlist_id] = result
                return result

            last_result = result

        # All attempts failed
        if last_result:
            self.failed_playlists.append(last_result.playlist_id)
            return last_result

        return ProcessingResult(
            playlist_id="unknown",
            success=False,
            error_message="All retry attempts failed",
        )

    async def process_playlists_batch(self, urls: list[str]) -> BatchProcessingResult:
        """Process multiple playlists concurrently with progress tracking."""
        start_time = time.time()
        self.logger.info(f"Starting batch processing of {len(urls)} playlists")

        # Validate all URLs first
        valid_urls = []
        for url in urls:
            if self.validate_youtube_url(url):
                valid_urls.append(url)
            else:
                self.logger.warning(f"Skipping invalid URL: {url}")

        if not valid_urls:
            return BatchProcessingResult(
                total_playlists=len(urls),
                successful_playlists=0,
                failed_playlists=len(urls),
                total_audio_files=0,
                total_processing_time=0.0,
                errors=["No valid URLs provided"],
            )

        # Process playlists with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_with_semaphore(url: str) -> ProcessingResult:
            async with semaphore:
                return await self.process_single_playlist(url)

        # Execute all tasks
        tasks = [process_with_semaphore(url) for url in valid_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_results = []
        failed_results = []
        total_audio_files = 0
        errors = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Exception processing {valid_urls[i]}: {result}"
                errors.append(error_msg)
                self.logger.error(error_msg)
                failed_results.append(
                    ProcessingResult(
                        playlist_id=f"exception_{i}",
                        success=False,
                        error_message=str(result),
                    )
                )
            elif isinstance(result, ProcessingResult):
                if result.success:
                    successful_results.append(result)
                    total_audio_files += len(result.audio_files)
                else:
                    failed_results.append(result)
                    errors.append(result.error_message or "Unknown error")
            else:
                # Unexpected type, treat as failed
                error_msg = f"Unknown result type for {valid_urls[i]}: {type(result)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
                failed_results.append(
                    ProcessingResult(
                        playlist_id=f"unknown_type_{i}",
                        success=False,
                        error_message=error_msg,
                    )
                )

        total_processing_time = time.time() - start_time

        batch_result = BatchProcessingResult(
            total_playlists=len(urls),
            successful_playlists=len(successful_results),
            failed_playlists=len(failed_results),
            total_audio_files=total_audio_files,
            total_processing_time=total_processing_time,
            results=successful_results + failed_results,
            errors=errors,
        )

        self.logger.info(
            f"Batch processing complete: {batch_result.successful_playlists}/{batch_result.total_playlists} successful"
        )
        return batch_result

    def process_playlists_from_file(self, file_path: str) -> BatchProcessingResult:
        """Process playlists from a file containing URLs."""
        try:
            with open(file_path, encoding="utf-8") as f:
                urls = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]

            self.logger.info(f"Loaded {len(urls)} URLs from {file_path}")
            return asyncio.run(self.process_playlists_batch(urls))

        except Exception as e:
            error_msg = f"Failed to read playlist file {file_path}: {e}"
            self.logger.error(error_msg)
            return BatchProcessingResult(
                total_playlists=0,
                successful_playlists=0,
                failed_playlists=0,
                total_audio_files=0,
                total_processing_time=0.0,
                errors=[error_msg],
            )

    def generate_processing_report(self, result: BatchProcessingResult) -> str:
        """Generate a detailed processing report."""
        report = [
            "=" * 60,
            "YOUTUBE PLAYLIST PROCESSING REPORT",
            "=" * 60,
            f"Total Playlists: {result.total_playlists}",
            f"Successful: {result.successful_playlists}",
            f"Failed: {result.failed_playlists}",
            f"Total Audio Files: {result.total_audio_files}",
            f"Processing Time: {result.total_processing_time:.2f} seconds",
            f"Success Rate: {result.successful_playlists / result.total_playlists * 100:.1f}%",
            "",
        ]
        if result.results:
            report.extend(("DETAILED RESULTS:", "-" * 40))
            for res in result.results:
                status = "✅ SUCCESS" if res.success else "❌ FAILED"
                report.append(
                    f"{status} | {res.playlist_id} | {len(res.audio_files)} files | {res.processing_time:.1f}s"
                )
                if not res.success and res.error_message:
                    report.append(f"    Error: {res.error_message[:100]}...")

        if result.errors:
            report.extend(("", "ERRORS:", "-" * 40))
            report.extend(f"• {error[:150]}..." for error in result.errors[:10])
        report.append("=" * 60)
        return "\n".join(report)

    def save_processing_metadata(self, result: BatchProcessingResult) -> Path:
        """Save processing metadata to JSON file."""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_playlists": result.total_playlists,
                "successful_playlists": result.successful_playlists,
                "failed_playlists": result.failed_playlists,
                "total_audio_files": result.total_audio_files,
                "total_processing_time": result.total_processing_time,
            },
            "results": [
                {
                    "playlist_id": res.playlist_id,
                    "success": res.success,
                    "audio_files_count": len(res.audio_files),
                    "audio_files": [str(f) for f in res.audio_files],
                    "processing_time": res.processing_time,
                    "error_message": res.error_message,
                    "metadata": res.metadata,
                }
                for res in result.results
            ],
            "errors": result.errors,
        }

        metadata_file = (
            self.output_dir
            / f"processing_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Processing metadata saved to {metadata_file}")
        return metadata_file


# Backward compatibility function for existing scripts
def process_youtube_playlists(
    urls: list[str],
    output_dir: str = "voice_data",
    audio_format: str = "wav",
    max_concurrent: int = 3,
) -> BatchProcessingResult:
    """
    Process YouTube playlists with enhanced capabilities.

    This function provides backward compatibility while offering
    the enhanced features of the new YouTubePlaylistProcessor.
    """
    processor = YouTubePlaylistProcessor(
        output_dir=output_dir, audio_format=audio_format, max_concurrent=max_concurrent
    )

    return asyncio.run(processor.process_playlists_batch(urls))


# Alias for backward compatibility
YouTubeProcessor = YouTubePlaylistProcessor
