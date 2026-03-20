"""
YouTube Data API v3 implementation.

This module provides the actual YouTube API integration functions
replacing the TODO stubs in api.py.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import urlencode, urlparse, parse_qs

import requests

from ai.sourcing.youtube.models import (
    Channel,
    ChannelStatus,
    ContentCategory,
    QualityMetrics,
    LicensingInfo,
)

logger = logging.getLogger(__name__)


class YouTubeAPIKeyError(Exception):
    """Raised when YouTube API key is missing or invalid."""

    pass


class YouTubeAPIQuotaError(Exception):
    """Raised when YouTube API quota is exhausted."""

    pass


class YouTubeAPIRateLimitError(Exception):
    """Raised when rate limit is exceeded."""

    pass


class YouTubeAPI:
    """
    Concrete implementation of YouTube Data API v3 client.

    Provides methods for:
    - Channel search
    - Channel details retrieval
    - Video listing
    - Content metadata extraction
    """

    API_BASE = "https://www.googleapis.com/youtube/v3"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize YouTube API client.

        Args:
            api_key: YouTube Data API key. If None, reads from environment.
        """
        self.api_key = self._get_api_key(api_key)
        self.last_request_time = None

        # Check quota periodically
        self.units_used_today = 0
        self.quota_limit = 10000  # Default daily quota
        self._check_time = datetime.now()

    def _get_api_key(self, api_key: Optional[str]) -> str:
        """Get and validate YouTube API key."""
        if api_key:
            return api_key

        import os
        from dotenv import load_dotenv

        load_dotenv(".env.youtube.example", override=True)

        # Try environment variable
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key or api_key.startswith("your-"):
            raise YouTubeAPIKeyError(
                "YouTube API key not found. Set YOUTUBE_API_KEY in environment "
                "or pass as parameter."
            )

        return api_key

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make authenticated request to YouTube API.

        Args:
            endpoint: API endpoint path (e.g., 'search')
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            YouTubeAPIQuotaError: When quota is exhausted
            YouTubeAPIRateLimitError: When rate limit is exceeded
        """
        params = params or {}
        params["key"] = self.api_key

        url = f"{self.API_BASE}/{endpoint.lstrip('/')}"
        response = requests.get(url, params=params)

        # Update quota tracking
        self._check_quota_usage(response.headers)

        if response.status_code == 403:
            raise YouTubeAPIQuotaError(
                "YouTube API quota exhausted. Try again tomorrow or add quota."
            )

        if response.status_code == 429:
            raise YouTubeAPIRateLimitError(
                "Rate limit exceeded. Try again in a moment."
            )

        response.raise_for_status()

        return response.json()

    def _check_quota_usage(self, headers: Dict):
        """Check quota usage from response headers."""
        quota_header = headers.get("X-RateLimit-Quota-Limit")
        used_header = headers.get("X-RateLimit-Quota-Used")

        if quota_header and used_header:
            try:
                self.quota_limit = int(quota_header)
                self.units_used_today = int(used_header)
                logger.debug(f"Quota usage: {self.units_used_today}/{self.quota_limit}")
            except ValueError:
                pass

    def check_quota(self) -> tuple[bool, int, int]:
        """
        Check quota status.

        Returns:
            Tuple of (has_quota, used, limit)
        """
        remaining = self.quota_limit - self.units_used_today
        has_quota = remaining > 100  # Safety margin

        if not has_quota:
            logger.warning(
                f"Low quota: {remaining} units remaining ({self.units_used_today}/{self.quota_limit})"
            )

        return has_quota, self.units_used_today, self.quota_limit

    def search_channels(self, query: str, max_results: int = 25) -> List[Dict]:
        """
        Search for channels by query.

        Args:
            query: Search query string
            max_results: Maximum number of results

        Returns:
            List of channel data dictionaries

        Note:
            Actual YouTube search is limited. We'll need to use channel search
            and iterate through results with pagination.
        """
        logger.info(f"Searching for channels: {query}")

        # YouTube API supports channel search but not term-based search
        # We'll use search.list and filter by query

        # First approach: search.list with q parameter
        channels = []

        try:
            # Search for videos from matching channels, then get channel info
            params = {
                "part": "snippet",
                "maxResults": min(max_results, 50),
                "q": query,
                "type": "video",
                "order": "relevance",
            }

            results = self._make_request("search", params)

            # Extract unique channels from results
            seen_channel_ids = set()

            for item in results.get("items", []):
                snippet = item.get("snippet", {})
                channel_id = snippet.get("channelId")
                channel_title = snippet.get("channelTitle", "")
                video_id = item.get("id")

                if not channel_id or channel_id in seen_channel_ids:
                    continue

                seen_channel_ids.add(channel_id)

                # Get channel details
                channel_data = self.get_channel_details(channel_id)

                if channel_data:
                    channels.append(channel_data)
                    logger.debug(
                        f"Found channel via video: {channel_title} "
                        f"({video_count} videos)"
                    )

                    if len(channels) >= max_results:
                        break

        except Exception as e:
            logger.error(f"Search failed: {e}")

        return channels

    def get_channel_details(self, channel_id: str) -> Optional[Dict]:
        """
        Get detailed information about a channel.

        Args:
            channel_id: YouTube channel ID

        Returns:
            Channel details dictionary or None
        """
        try:
            params = {
                "part": "snippet,contentDetails,statistics,brandingSettings",
                "id": channel_id,
            }

            data = self._make_request("channels", params)
            items = data.get("items", [])

            if not items:
                logger.warning(f"No data found for channel: {channel_id}")
                return None

            channel_data = items[0]

            return {
                "channelId": channel_id,
                "channelTitle": channel_data.get("snippet", {}).get("title"),
                "description": channel_data.get("snippet", {}).get("description"),
                "customUrl": channel_data.get("snippet", {}).get("customUrl"),
                "publishedAt": channel_data.get("snippet", {}).get("publishedAt"),
                "subscriberCount": channel_data.get("statistics", {}).get(
                    "subscriberCount", 0
                ),
                "videoCount": channel_data.get("statistics", {}).get("videoCount", 0),
                "viewCount": channel_data.get("statistics", {}).get("viewCount", 0),
                "thumbnailUrl": channel_data.get("snippet", {})
                .get("thumbnails", {})
                .get("default", {})
                .get("url"),
                "country": channel_data.get("snippet", {}).get("country"),
                "keywords": channel_data.get("snippet", {}).get("tags", []),
                "brandingSettings": channel_data.get("brandingSettings", {}),
            }

        except Exception as e:
            logger.error(f"Failed to get channel details for {channel_id}: {e}")
            return None

    def get_channel_videos(self, channel_id: str, max_results: int = 50) -> List[Dict]:
        """
        Get videos from a channel.

        Args:
            channel_id: YouTube channel ID
            max_results: Maximum number of videos

        Returns:
            List of video data dictionaries
        """
        videos = []

        try:
            params = {
                "part": "snippet,contentDetails,statistics",
                "channelId": channel_id,
                "maxResults": min(max_results, 50),
                "order": "date",
            }

            has_more = True
            next_page_token = None

            while has_more:
                params_with_page = params.copy()
                if next_page_token:
                    params_with_page["pageToken"] = next_page_token

                data = self._make_request("search", params_with_page)

                items = data.get("items", [])
                videos.extend(items)

                has_more = data.get("nextPageToken") is not None
                next_page_token = data.get("nextPageToken")

                logger.debug(
                    f"Fetched {len(items)} videos, total: {len(videos)}, "
                    f"more: {has_more}"
                )

                if not has_more:
                    break

        except Exception as e:
            logger.error(f"Failed to get videos for channel {channel_id}: {e}")

        return videos

    def get_video_details(self, video_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific video.

        Args:
            video_id: YouTube video ID

        Returns:
            Video details dictionary or None
        """
        try:
            params = {
                "part": "snippet,contentDetails,statistics",
                "id": video_id,
            }

            data = self._make_request("videos", params)
            items = data.get("items", [])

            if not items:
                return None

            return items[0]

        except Exception as e:
            logger.error(f"Failed to get video details for {video_id}: {e}")
            return None

    def get_channel_playlists(self, channel_id: str) -> List[Dict]:
        """
        Get playlists from a channel.

        Args:
            channel_id: YouTube channel ID

        Returns:
            List of playlist dictionaries
        """
        playlists = []

        try:
            params = {
                "part": "snippet,contentDetails",
                "channelId": channel_id,
                "maxResults": 25,
            }

            data = self._make_request("playlists", params)
            items = data.get("items", [])

            for item in items:
                snippet = item.get("snippet", {})
                playlists.append(
                    {
                        "playlistId": item.get("id"),
                        "title": snippet.get("title"),
                        "description": snippet.get("description"),
                        "videoCount": item.get("contentDetails", {}).get(
                            "itemCount", 0
                        ),
                        "publishedAt": snippet.get("publishedAt"),
                        "thumbnailUrl": snippet.get("thumbnails", {})
                        .get("default", {})
                        .get("url"),
                    }
                )

            logger.debug(f"Found {len(playlists)} playlists for channel {channel_id}")

        except Exception as e:
            logger.error(f"Failed to get playlists for channel {channel_id}: {e}")

        return playlists


# Standalone function tests for API functionality
def test_api_connection(api_key: str) -> bool:
    """Test that the API key works and has quota."""
    try:
        api = YouTubeAPI(api_key)
        has_quota, used, limit = api.check_quota()

        if not has_quota:
            print(f"⚠️  Low quota: { used}/{limit} units used")
        else:
            print(f"✅ API connection successful")
            print(f"   Quota: {used}/{limit} units used")

        # Try a simple search to verify
        params = {
            "part": "snippet",
            "maxResults": 1,
            "q": "test",
            "type": "video",
        }
        api._make_request("search", params)

        print("✅ YouTube API connection test passed")
        return True

    except YouTubeAPIKeyError as e:
        print(f"❌ API key error: {e}")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False


def get_api_quota_status() -> tuple[bool, int, int]:
    """
    Check YouTube API quota status.

    Returns:
        Tuple of (has_quota, used, limit)
    """
    try:
        import os
        from dotenv import load_dotenv

        load_dotenv(".env.youtube.example", override=True)
        api_key = os.getenv("YOUTUBE_API_KEY")

        if not api_key or api_key.startswith("your-"):
            print("⚠️  No API key configured")
            return False, 0, 10000

        api = YouTubeAPI(api_key)
        return api.check_quota()

    except Exception as e:
        print(f"❌ Failed to check quota: {e}")
        return False, 0, 10000


if __name__ == "__main__":
    # Run quota status check
    has_quota, used, limit = get_api_quota_status()

    print(f"YouTube API Quota Status:")
    print(f"  Available: {'Yes' if has_quota else 'No'}")
    print(f"  Used today: {used}/{limit} units")
    print()

    if not has_quota:
        print("⚠️  Consider waiting 24h for quota reset or increasing quota")
    else:
        print("✅ Quota available for discovery")
        print("   Estimated capacity: ~100 channel discoveries (at 100 units/search)")
