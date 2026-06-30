import logging
import re

logger = logging.getLogger(__name__)


class YouTubeAPIKeyError(Exception):
    pass


class YouTubeAPIQuotaError(Exception):
    pass


class YouTubeAPIRateLimitError(Exception):
    pass


class YouTubeAPI:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.units_used_today = 0
        self.quota_limit = float("inf")

    def _parse_int(self, value: str | int | None, default: int = 0) -> int:
        if value is None:
            return default
        if isinstance(value, int):
            return value
        digits = re.sub(r"[^\d]", "", str(value))
        return int(digits) if digits else default

    def _extract_channel_id(self, url: str) -> str:
        m = re.search(r"UC[A-Za-z0-9_-]{22}", url or "")
        return m.group(0) if m else ""

    def search_channels(self, query: str, max_results: int = 25) -> list[dict]:
        logger.info(f"Searching for channels: {query}")
        channels = []
        try:
            from youtubesearchpython import ChannelsSearch

            search = ChannelsSearch(query, limit=max_results)
            results = search.result()

            for item in results.get("result", []):
                if item.get("type") != "channel":
                    continue
                channel_id = item.get("id", "")
                if not channel_id:
                    channel_id = self._extract_channel_id(item.get("link", ""))
                if not channel_id:
                    continue

                channels.append(
                    {
                        "channelId": channel_id,
                        "channelTitle": item.get("title") or "Unknown",
                        "description": self._join_snippet(item.get("descriptionSnippet")) or "",
                        "customUrl": item.get("subscribers") or "",
                        "publishedAt": None,
                        "subscriberCount": 0,
                        "videoCount": self._parse_int(item.get("videoCount")),
                        "viewCount": 0,
                        "thumbnailUrl": self._first_thumbnail(item.get("thumbnails")),
                        "country": None,
                        "keywords": [],
                        "brandingSettings": {},
                    }
                )

                if len(channels) >= max_results:
                    break

        except Exception as e:
            logger.error(f"Search failed: {e}")

        return channels

    def get_channel_details(self, channel_id: str) -> dict | None:
        try:
            from youtubesearchpython import Channel

            data = Channel.get(channel_id)
            if not data:
                logger.warning(f"No data found for channel: {channel_id}")
                return None

            subs = data.get("subscribers", {}) or {}
            sub_label = subs.get("label") or subs.get("simpleText") or ""
            sub_count = self._parse_int(sub_label)

            return {
                "channelId": channel_id,
                "channelTitle": data.get("title") or "Unknown",
                "description": data.get("description") or "",
                "customUrl": None,
                "publishedAt": None,
                "subscriberCount": sub_count,
                "videoCount": 0,
                "viewCount": self._parse_int(data.get("views")),
                "thumbnailUrl": self._first_thumbnail(data.get("thumbnails")),
                "country": data.get("country"),
                "keywords": (data.get("keywords") or "").split(),
                "brandingSettings": {},
            }

        except Exception as e:
            logger.error(f"Failed to get channel details for {channel_id}: {e}")
            return None

    def get_channel_videos(self, channel_id: str, max_results: int = 50) -> list[dict]:
        videos = []
        try:
            from youtubesearchpython import Playlist

            uploads_id = "UU" + channel_id[2:]
            pl = Playlist.get(f"https://www.youtube.com/playlist?list={uploads_id}")
            for entry in (pl.get("videos") or [])[:max_results]:
                videos.append(
                    {
                        "id": entry.get("id"),
                        "title": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "thumbnails": entry.get("thumbnails", []),
                        "viewCount": self._parse_int(entry.get("views", {}).get("text")),
                        "publishedAt": None,
                        "duration": entry.get("duration"),
                        "snippet": {"title": entry.get("title", ""), "description": "", "tags": []},
                        "description": "",
                        "tags": [],
                    }
                )
        except Exception as e:
            logger.error(f"Failed to get videos for channel {channel_id}: {e}")

        return videos

    def get_video_details(self, video_id: str) -> dict | None:
        try:
            from youtubesearchpython import Video

            return Video.get(f"https://www.youtube.com/watch?v={video_id}")

        except Exception as e:
            logger.error(f"Failed to get video details for {video_id}: {e}")
            return None

    def get_channel_playlists(self, channel_id: str) -> list[dict]:
        playlists = []
        try:
            from youtubesearchpython import Channel

            data = Channel.get(channel_id)
            for pl in data.get("playlists", []):
                playlists.append(
                    {
                        "playlistId": pl.get("id"),
                        "title": pl.get("title"),
                        "description": "",
                        "videoCount": self._parse_int(pl.get("videoCount")),
                        "publishedAt": None,
                        "thumbnailUrl": self._first_thumbnail(pl.get("thumbnails")),
                    }
                )

        except Exception as e:
            logger.error(f"Failed to get playlists for channel {channel_id}: {e}")

        return playlists

    def check_quota(self) -> tuple[bool, int, int]:
        return True, 0, 0

    @staticmethod
    def _join_snippet(snippet: list | None) -> str:
        if not snippet:
            return ""
        return "".join(part.get("text", "") for part in snippet if isinstance(part, dict))

    @staticmethod
    def _first_thumbnail(thumbnails: list | None) -> str | None:
        if not thumbnails:
            return None
        t = thumbnails[0]
        url = t.get("url", "") if isinstance(t, dict) else ""
        return f"https:{url}" if url and url.startswith("//") else url or None


def test_api_connection(api_key: str | None = None) -> bool:
    try:
        api = YouTubeAPI(api_key)
        results = api.search_channels("test", max_results=1)
        return len(results) > 0
    except Exception:
        return False


def get_api_quota_status() -> tuple[bool, int, int]:
    return True, 0, 0


if __name__ == "__main__":
    ok = test_api_connection()
