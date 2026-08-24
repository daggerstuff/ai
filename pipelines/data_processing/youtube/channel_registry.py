"""
Channel registry and database management.

Provides persistent storage for discovered channels with CRUD operations.

Uses SQLite for lightweight, embedded database storage.
"""

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from ai.sourcing.youtube.models import (
    Channel,
    ChannelStatus,
    ContentCategory,
)

logger = logging.getLogger(__name__)


class ChannelRegistryDB:
    """
    Persistent database for channel registry.

    Uses SQLite for lightweight, embedded database storage.
    """

    def __init__(self, db_path: str = "channels.db"):
        """
        Initialize channel registry database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = None
        self._initialize_db()

    def _initialize_db(self):
        """Create database schema."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        # Create channels table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT UNIQUE NOT NULL,
                channel_name TEXT NOT NULL,
                channel_url TEXT NOT NULL,
                subscriber_count INTEGER DEFAULT 0,
                video_count INTEGER DEFAULT 0,
                total_views INTEGER DEFAULT 0,
                created_date TEXT,
                primary_language TEXT DEFAULT 'en',
                languages TEXT DEFAULT '[]',
                categories TEXT DEFAULT '[]',
                description TEXT,
                quality_score REAL DEFAULT 0.0,
                quality_metrics TEXT,
                is_professional INTEGER DEFAULT 0,
                credentials TEXT DEFAULT '[]',
                organization TEXT,
                notes TEXT,
                status TEXT DEFAULT 'unknown',
                health_score REAL DEFAULT 0.0,
                last_monitored TEXT,
                tags TEXT DEFAULT '[]',
                source TEXT DEFAULT 'api',
                first_seen TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)

        # Create indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_quality ON channels (quality_score)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON channels (status)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_language ON channels (primary_language)")

        self.conn.commit()

    def add_channel(self, channel: Channel) -> int:
        """
        Add or update a channel in the registry.

        Args:
            channel: Channel to add/update

        Returns:
            Database row ID (integer)
        """
        now = datetime.now(UTC).isoformat()

        data = {
            "channel_id": channel.channel_id,
            "channel_name": channel.channel_name,
            "channel_url": channel.channel_url,
            "subscriber_count": channel.subscriber_count,
            "video_count": channel.video_count,
            "total_views": channel.total_views,
            "created_date": channel.created_date.isoformat() if channel.created_date else None,
            "last_updated": channel.last_updated.isoformat() if channel.last_updated else None,
            "primary_language": channel.primary_language,
            "languages": json.dumps(list(channel.languages)),
            "categories": json.dumps([c.value for c in channel.categories]),
            "description": channel.description,
            "quality_score": channel.quality_score,
            "quality_metrics": json.dumps(channel.quality_metrics.__dict__()) if channel.quality_metrics else None,
            "is_professional": 1 if channel.is_professional else 0,
            "credentials": json.dumps(channel.credentials),
            "organization": channel.organization,
            "licensing_info": json.dumps(channel.licensing.__dict__()) if channel.licensing else None,
            "status": channel.status.value,
            "health_score": channel.health_score,
            "last_monitored": channel.last_monitored.isoformat() if channel.last_monitored else None,
            "tags": json.dumps(channel.tags),
            "notes": channel.notes,
            "source": channel.source,
            "first_seen": now,
            "last_updated": now,
        }

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO channels (
                    channel_id, channel_name, channel_url, subscriber_count,
                    video_count, total_views, created_date, last_updated,
                    primary_language, languages, categories, description,
                    quality_score, quality_metrics, is_professional, credentials,
                    organization, licensing_info, status, health_score,
                    last_monitored, tags, notes, source,
                    first_seen, last_updated
                ) VALUES (
                    :channel_id, :channel_name, :channel_url, :subscriber_count,
                    :video_count, :total_views, :created_date, :last_updated,
                    :primary_language, :languages, :categories, :description,
                    :quality_score, :quality_metrics, :is_professional, :credentials,
                    :organization, :licensing_info, :status, :health_score,
                    :last_monitored, :tags, :notes, :source,
                    :first_seen, :last_updated
                )
                ON CONFLICT(channel_id) DO UPDATE SET
                    channel_name = EXCLUDED.channel_name,
                    channel_url = EXCLUDED.channel_url,
                    subscriber_count = EXCLUDED.subscriber_count,
                    video_count = EXCLUDED.video_count,
                    total_views = EXCLUDED.total_views,
                    created_date = COALESCE(EXCLUDED.created_date, :created_date),
                    last_updated = COALESCE(EXCLUDED.last_updated, :last_updated),
                    primary_language = COALESCE(EXCLUDED.primary_language, :primary_language),
                    languages = COALESCE(EXCLUDED.languages, :languages),
                    categories = COALESCE(EXCLUDED.categories, :categories),
                    description = COALESCE(EXCLUDED.description, :description),
                    quality_score = COALESCE(EXCLUDED.quality_score, :quality_score),
                    quality_metrics = COALESCE(EXCLUDED.quality_metrics, :quality_metrics),
                    is_professional = COALESCE(EXCLUDED.is_professional, :is_professional),
                    credentials = COALESCE(EXCLUDED.credentials, :credentials),
                    organization = COALESCE(EXCLUDED.organization, :organization),
                    licensing_info = COALESCE(EXCLUDED.licensing_info, :licensing_info),
                    status = COALESCE(EXCLUDED.status, :status),
                    health_score = COALESCE(EXCLUDED.health_score, :health_score),
                    last_monitored = COALESCE(EXCLUDED.last_monitored, :last_monitored),
                    tags = COALESCE(EXCLUDED.tags, :tags),
                    notes = COALESCE(EXCLUDED.notes, :notes),
                    source = COALESCE(EXCLUDED.source, :source),
                    last_updated = :last_updated
            """,
                data,
            )

            self.conn.commit()
            return cursor.lastrowid

        except sqlite3.Error as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to add channel: {e}") from e

    def get_channel(self, channel_id: str) -> Channel | None:
        """
        Retrieve a channel from registry by channel_id.

        Args:
            channel_id: YouTube channel ID

        Returns:
            Channel object or None if not found
        """
        self.conn.row_factory = sqlite3.Row

        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM channels WHERE channel_id = ?
            """,
            (channel_id,),
        )

        row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_channel(row)

    def get_all_channels(self, status_filter: ChannelStatus | None = None) -> list[Channel]:
        """
        Get all channels, optionally filtered by status.

        Args:
            status_filter: Optional status filter

        Returns:
            List of Channel objects
        """
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()

        if status_filter:
            cursor.execute(
                "SELECT * FROM channels WHERE status = ? ORDER BY quality_score DESC",
                (status_filter.value,),
            )
        else:
            cursor.execute("SELECT * FROM channels ORDER BY quality_score DESC")

        rows = cursor.fetchall()
        return [self._row_to_channel(row) for row in rows]

    def get_channels_by_language(self, language: str) -> list[Channel]:
        """
        Get channels that support a specific language.

        Args:
            language: Language code (e.g., 'en', 'es', 'fr')

        Returns:
            List of Channel objects
        """
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM channels
            WHERE languages LIKE ?
            ORDER BY quality_score DESC
            """,
            (f'%"{language}"%',),
        )

        rows = cursor.fetchall()
        return [self._row_to_channel(row) for row in rows]

    def get_channels_by_category(self, category: ContentCategory) -> list[Channel]:
        """
        Get channels that belong to a specific category.

        Args:
            category: ContentCategory enum

        Returns:
            List of Channel objects
        """
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM channels
            WHERE categories LIKE ?
            ORDER BY quality_score DESC
            """,
            (f'%"{category.value}"%',),
        )

        rows = cursor.fetchall()
        return [self._row_to_channel(row) for row in rows]

    def update_channel_status(self, channel_id: str, status: ChannelStatus) -> bool:
        """
        Update status of a channel.

        Args:
            channel_id: YouTube channel ID
            status: New status

        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                UPDATE channels
                SET status = ?, last_monitored = ?
                WHERE channel_id = ?
                """,
                (status.value, datetime.now(UTC).isoformat(), channel_id),
            )

            self.conn.commit()
            return cursor.rowcount > 0

        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to update status: {e}")
            return False

    def update_channel_health(self, channel_id: str, health_score: float) -> bool:
        """
        Update health score of a channel.

        Args:
            channel_id: YouTube channel ID
            health_score: New health score (0.0-1.0)

        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                UPDATE channels
                SET health_score = ?, last_monitored = ?
                WHERE channel_id = ?
                """,
                (health_score, datetime.now(UTC).isoformat(), channel_id),
            )

            self.conn.commit()
            return cursor.rowcount > 0

        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to update health score: {e}")
            return False

    def get_statistics(self) -> dict:
        """Get overall registry statistics."""
        cursor = self.conn.cursor()

        # Total channels
        cursor.execute("SELECT COUNT(*) FROM channels")
        total = cursor.fetchone()[0]

        # By status
        cursor.execute("SELECT status, COUNT(*) FROM channels GROUP BY status")
        by_status = {row[0]: row[1] for row in cursor.fetchall()}

        # Quality distribution
        cursor.execute(
            "SELECT "
            "    quality_score * 10 AS quality_bucket, "
            "    COUNT(*) as count "
            "FROM channels "
            "GROUP BY quality_score * 10 "
            "ORDER BY quality_bucket"
        )
        quality_dist_raw = cursor.fetchall()
        quality_dist = {f"{row[0] / 10:.1f}-{(row[0] + 1) / 10:.1f}": row[1] for row in quality_dist_raw}

        # By language
        cursor.execute("""
            SELECT json_each(json_extract(languages, '$'))
            FROM channels
            GROUP BY 1
            ORDER BY 2 DESC
            LIMIT 10
        """)
        by_language = {row[0]: row[1] for row in cursor.fetchall()}

        # By category
        cursor.execute("""
            SELECT json_each(json_extract(categories, '$'))
            FROM channels
            GROUP BY 1
            ORDER BY 2 DESC
            LIMIT 10
        """)
        by_category = {row[0]: row[1] for row in cursor.fetchall()}

        return {
            "total": total,
            "by_status": by_status,
            "quality_distribution": quality_dist,
            "by_language": by_language,
            "by_category": by_category,
        }

    def _row_to_channel(self, row) -> Channel:
        """Convert database row to Channel object."""
        return Channel(
            channel_id=row["channel_id"],
            channel_name=row["channel_name"],
            channel_url=row["channel_url"],
            subscriber_count=row["subscriber_count"],
            video_count=row["video_count"],
            total_views=row["total_views"],
            created_date=datetime.fromisoformat(row["created_date"]) if row["created_date"] else None,
            last_updated=datetime.fromisoformat(row["last_updated"]) if row["last_updated"] else None,
            primary_language=row["primary_language"],
            languages=set(json.loads(row["languages"])) if row["languages"] else set(),
            categories=[ContentCategory(c) for c in json.loads(row["categories"])] if row["categories"] else [],
            description=row["description"],
            quality_score=row["quality_score"],
            is_professional=bool(row["is_professional"]),
            credentials=json.loads(row["credentials"]) if row["credentials"] else [],
            organization=row["organization"],
            status=ChannelStatus(row["status"]),
            health_score=row["health_score"],
            last_monitored=datetime.fromisoformat(row["last_monitored"]) if row["last_monitored"] else None,
            tags=json.loads(row["tags"]) if row["tags"] else [],
            notes=row["notes"],
            source=row["source"],
        )

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_registry(db_path: str = "channels.db") -> ChannelRegistryDB:
    """Convenience function to create a channel registry."""
    return ChannelRegistryDB(db_path)
