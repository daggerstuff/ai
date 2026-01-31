"""YouTube connector for dataset pipeline ingestion.

Implements the IngestionConnector interface for YouTube content access,
with enhanced retry, rate limiting, and security features.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, List
import urllib.parse
import requests

from .ingestion_interface import IngestionConnector, IngestRecord, IngestionError
from .ingest_utils import RateLimiter
from .validation import validate_record
from .quarantine import get_quarantine_store
from .youtube_processor import YouTubePlaylistProcessor, RateLimitConfig, ProxyConfig, AntiDetectionConfig


@dataclass
class YouTubeConfig:
    """Configuration for YouTube connector."""
    playlist_urls: List[str]
    max_concurrent: int = 3
    retry_attempts: int = 3
    retry_delay: float = 5.0
    rate_limit_config: Optional[RateLimitConfig] = None
    proxy_config: Optional[ProxyConfig] = None
    anti_detection_config: Optional[AntiDetectionConfig] = None
    audio_format: str = "mp3"  # Default audio format
    audio_quality: str = "0"   # Best quality
    output_dir: str = "temp_youtube_data"


class YouTubeConnector(IngestionConnector):
    """YouTube connector implementing the IngestionConnector interface."""

    def __init__(
        self,
        config: YouTubeConfig,
        name: Optional[str] = None,
        allowed_domains: Optional[List[str]] = None
    ):
        super().__init__(name=name or "youtube")
        self.config = config
        self.allowed_domains = allowed_domains or ['youtube.com', 'www.youtube.com', 'youtu.be']
        
        # Initialize YouTube processor
        self.processor = YouTubePlaylistProcessor(
            output_dir=config.output_dir,
            audio_format=config.audio_format,
            audio_quality=config.audio_quality,
            max_concurrent=config.max_concurrent,
            retry_attempts=config.retry_attempts,
            retry_delay=config.retry_delay,
            rate_limit_config=config.rate_limit_config,
            proxy_config=config.proxy_config,
            anti_detection_config=config.anti_detection_config
        )
        
        # Set up rate limiting
        if config.rate_limit_config:
            self.rate_limiter = RateLimiter(
                capacity=config.rate_limit_config.requests_per_minute,
                refill_rate=config.rate_limit_config.requests_per_minute / 60.0
            )
        else:
            self.rate_limiter = None

    def _validate_youtube_url(self, url: str) -> bool:
        """Validate YouTube URL format and check against allow-list."""
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.netloc not in self.allowed_domains:
                return False
            
            # Additional SSRF protection: ensure it's a valid YouTube URL structure
            if not parsed.scheme in ['http', 'https']:
                return False
                
            # Check for playlist or video URL
            is_valid = (
                "playlist" in parsed.query
                or "watch" in parsed.path
                or "youtu.be" in parsed.netloc
            )
            
            if is_valid:
                # Additional validation to prevent SSRF
                # Check for suspicious patterns
                suspicious_patterns = ['127.0.0.1', 'localhost', 'internal', 'metadata.google.internal']
                url_lower = url.lower()
                for pattern in suspicious_patterns:
                    if pattern in url_lower:
                        return False
            
            return is_valid
        except Exception:
            return False

    def connect(self) -> None:
        """Validate YouTube connectivity and check if required tools are available."""
        try:
            # Check if yt-dlp is available
            result = subprocess.run(["yt-dlp", "--version"], 
                                    capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise IngestionError("yt-dlp is not installed or not accessible")
        except FileNotFoundError:
            raise IngestionError("yt-dlp is not installed")
        except Exception as e:
            raise IngestionError(f"YouTube connectivity check failed: {e}")

    def fetch(self) -> Iterable[IngestRecord]:
        """Fetch content from YouTube playlists/videos."""
        try:
            for url in self.config.playlist_urls:
                # Validate URL against allow-list
                if not self._validate_youtube_url(url):
                    continue
                
                try:
                    # Honor rate limiting if configured
                    if self.rate_limiter:
                        acquired = self.rate_limiter.acquire(blocking=True, timeout=10)
                        if not acquired:
                            raise IngestionError(f"Rate limiter timeout for URL: {url}")
                    
                    # Process the URL through the existing YouTube processor
                    result = asyncio.run(self.processor.process_single_playlist(url))
                    
                    if not result.success:
                        # Quarantine failed processing attempts
                        error_record = IngestRecord(
                            id=f"youtube_error_{int(time.time())}_{url}",
                            payload={"url": url, "error": result.error_message},
                            metadata={
                                'source_type': 'youtube_error',
                                'processing_url': url,
                                'error_message': result.error_message
                            }
                        )
                        store = get_quarantine_store()
                        errors = [f"YouTube processing failed: {result.error_message}"]
                        store.quarantine_record(error_record, errors)
                        continue
                    
                    # Process the results and create IngestRecords
                    for audio_file in result.audio_files:
                        try:
                            # Read the audio file content
                            content = audio_file.read_bytes()
                            
                            # Deduplication check at ingestion stage
                            from .ingestion_deduplication import add_content_check_duplicate
                            if not add_content_check_duplicate(content):
                                # Content is a duplicate, skip
                                continue
                            
                            # Create IngestRecord for each audio file
                            rec = IngestRecord(
                                id=f"youtube://{result.playlist_id}/{audio_file.name}",
                                payload=content,
                                metadata={
                                    'source_type': 'youtube_audio',
                                    'playlist_id': result.playlist_id,
                                    'file_path': str(audio_file),
                                    'size': len(content),
                                    'audio_format': self.config.audio_format,
                                    'processing_metadata': result.metadata
                                }
                            )
                            
                            # Source-level validation
                            if not self.validate(rec):
                                store = get_quarantine_store()
                                errors = [f"Source validation failed for {self.name} connector: {audio_file.name}"]
                                store.quarantine_record(rec, errors)
                                continue
                            
                            # Schema validation
                            try:
                                validated = validate_record(rec)
                                yield validated
                            except Exception as ve:
                                # Quarantine validation failures
                                store = get_quarantine_store()
                                errors = [str(ve)]
                                store.quarantine_record(rec, errors)
                                continue
                                
                        except Exception as e:
                            # Log individual file processing error and continue
                            error_rec = IngestRecord(
                                id=f"youtube_file_error_{int(time.time())}_{audio_file.name}",
                                payload={"file_path": str(audio_file), "error": str(e)},
                                metadata={
                                    'source_type': 'youtube_file_error',
                                    'file_path': str(audio_file),
                                    'error_message': str(e)
                                }
                            )
                            store = get_quarantine_store()
                            errors = [f"YouTube file processing error: {e}"]
                            store.quarantine_record(error_rec, errors)
                            continue

                except Exception as e:
                    # Log and continue processing other URLs
                    error_rec = IngestRecord(
                        id=f"youtube_error_{int(time.time())}_{url}",
                        payload={"url": url, "error": str(e)},
                        metadata={
                            'source_type': 'youtube_error',
                            'processing_url': url,
                            'error_message': str(e)
                        }
                    )
                    store = get_quarantine_store()
                    errors = [f"YouTube fetch error: {e}"]
                    store.quarantine_record(error_rec, errors)
                    continue

        except Exception as e:
            raise IngestionError(f"Error processing YouTube content: {e}")

    def close(self) -> None:
        """Close any open resources."""
        # Nothing specific to close for the YouTube connector
        pass


# Register the YouTube connector
from .ingestion_interface import register_connector

register_connector('youtube', YouTubeConnector)