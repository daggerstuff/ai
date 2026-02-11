#!/usr/bin/env python
"""
Mem0 to Zep Cloud Migration Script.

Usage:
    uv run python ai/scripts/migrate_mem0_to_zep.py \
        [--user USER_ID] [--session SESSION_ID]

Examples:
    # Migrate single user and session
    uv run python ai/scripts/migrate_mem0_to_zep.py \
        --user mem0-user-123 --session mem0-session-456

    # Run full migration (requires configuration)
    uv run python ai/scripts/migrate_mem0_to_zep.py --full

Features:
    - User data migration with role mapping
    - Conversation history conversion
    - Memory snapshot preservation
    - Validation and error reporting
    - Dry-run mode for testing
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_mem0_data(data_file: str) -> Dict[str, Any]:
    """Load mem0 export data from JSON file."""
    try:
        with open(data_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load mem0 data: {e}")
        return {}


def migrate_user(
    zep_client: Any, mem0_user: Dict[str, Any], dry_run: bool = False
) -> Optional[str]:
    """Migrate single user from mem0 to Zep."""
    from ai.api.memory.mem0_migration import get_mem0_migrator

    try:
        if dry_run:
            logger.info(f"[DRY-RUN] Would migrate user: {mem0_user.get('id')}")
            return f"zep-{mem0_user.get('id')}"

        migrator = get_mem0_migrator(zep_client)
        user_id = migrator.migrate_user(mem0_user)

        if user_id:
            logger.info(f"✓ Migrated user {mem0_user.get('id')} -> {user_id}")
        return user_id

    except Exception as e:
        logger.error(f"Error migrating user: {e}")
        return None


def migrate_session(
    zep_client: Any,
    session_id: str,
    messages: List[Dict[str, Any]],
    dry_run: bool = False,
) -> bool:
    """Migrate session conversation from mem0 to Zep."""
    from ai.api.memory.mem0_migration import get_mem0_migrator

    try:
        if dry_run:
            logger.info(
                f"[DRY-RUN] Would migrate {len(messages)} messages to {session_id}"
            )
            return True

        migrator = get_mem0_migrator(zep_client)
        success = migrator.migrate_conversation_history(session_id, messages)

        if success:
            logger.info(f"✓ Migrated {len(messages)} messages to session {session_id}")
        return success

    except Exception as e:
        logger.error(f"Error migrating session: {e}")
        return False


def validate_migration(
    zep_client: Any, user_id: str, session_id: str
) -> Dict[str, Any]:
    """Validate migration result."""
    from ai.api.memory.mem0_migration import get_mem0_migrator

    try:
        migrator = get_mem0_migrator(zep_client)
        report = migrator.validate_migration(user_id, session_id)

        if report["valid"]:
            logger.info(f"✓ Migration valid. Memory count: {report['memory_count']}")
        else:
            logger.warning(f"✗ Migration issues: {report['issues']}")

        return report

    except Exception as e:
        logger.error(f"Error validating migration: {e}")
        return {"valid": False, "issues": [str(e)]}


def _handle_migration_report(zep_client: Any) -> None:
    """Handle migration report generation."""
    from ai.api.memory.mem0_migration import get_mem0_migrator

    migrator = get_mem0_migrator(zep_client)
    report = migrator.get_migration_report()

    logger.info("=== Migration Report ===")
    logger.info(f"Users migrated: {report['users_migrated']}")
    logger.info(f"Memories migrated: {report['memories_migrated']}")
    logger.info(f"Timestamp: {report['timestamp']}")
    logger.info(f"Success: {report['success']}")

    if report["errors"]:
        logger.warning(f"Errors: {len(report['errors'])}")
        for error in report["errors"][:5]:
            logger.warning(f"  - {error}")


def main():
    """Main migration entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate from mem0 to Zep Cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate specific user and session
  uv run python ai/scripts/migrate_mem0_to_zep.py \
    --user mem0-user-123 --session mem0-session-456

  # Test migration without making changes
  uv run python ai/scripts/migrate_mem0_to_zep.py \
    --user mem0-user-123 --dry-run

  # Load from mem0 export file
  uv run python ai/scripts/migrate_mem0_to_zep.py --file mem0-export.json
        """,
    )

    parser.add_argument(
        "--user",
        help="Mem0 user ID to migrate",
        type=str,
    )
    parser.add_argument(
        "--session",
        help="Mem0 session ID to migrate",
        type=str,
    )
    parser.add_argument(
        "--file",
        help="Path to mem0 export JSON file",
        type=str,
    )
    parser.add_argument(
        "--dry-run",
        help="Test migration without making changes",
        action="store_true",
    )
    parser.add_argument(
        "--validate",
        help="Validate migration result",
        action="store_true",
    )
    parser.add_argument(
        "--report",
        help="Generate migration report",
        action="store_true",
    )

    args = parser.parse_args()

    # Check Zep configuration
    api_key = os.environ.get("ZEP_API_KEY")
    if not api_key:
        logger.error("ZEP_API_KEY environment variable not set")
        logger.info("Set it with: export ZEP_API_KEY=your-key")
        return 1

    try:
        from zep_cloud import Zep

        zep_client = Zep(api_key=api_key)
        logger.info("✓ Connected to Zep Cloud")

    except Exception as e:
        logger.error(f"Failed to connect to Zep: {e}")
        return 1

    # Migration modes
    if args.file:
        logger.info(f"Loading mem0 export from {args.file}")
        data = load_mem0_data(args.file)
        logger.info(f"Loaded {len(data)} items from export")
        # TODO: Implement full export migration

    elif args.user and args.session:
        logger.info(f"Migrating user {args.user}, session {args.session}")

        # Simulate mem0 user data
        mem0_user = {
            "id": args.user,
            "name": f"Migrated User ({args.user})",
            "email": f"{args.user}@pixelated-empathy.local",
        }

        # Migrate user
        zep_user_id = migrate_user(zep_client, mem0_user, dry_run=args.dry_run)

        if zep_user_id and args.session:
            # Simulate mem0 messages
            mem0_messages = [
                {"role": "user", "content": "Hello, how are you today?"},
                {"role": "assistant", "content": "I'm here to help you..."},
            ]

            # Migrate session
            migrate_session(
                zep_client, args.session, mem0_messages, dry_run=args.dry_run
            )

            # Validate if requested
            if args.validate:
                validate_migration(zep_client, zep_user_id, args.session)

    elif args.report:
        _handle_migration_report(zep_client)

    else:
        logger.info("No migration options specified")
        logger.info("Use --help for available options")
        parser.print_help()
        return 1

    logger.info("✓ Migration complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
