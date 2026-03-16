#!/usr/bin/env python3
"""
Configuration Change Tracking and Rollback System for Pixelated Empathy AI
Tracks configuration changes and provides rollback capabilities
"""

import os
import sys
import json
import yaml
import hashlib
import shutil
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import subprocess
import tempfile
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ConfigChange:
    """Represents a configuration change"""
    timestamp: str
    change_id: str
    file_path: str
    change_type: str  # 'create', 'update', 'delete'
    old_hash: Optional[str]
    new_hash: Optional[str]
    old_content: Optional[str]
    new_content: Optional[str]
    user: str
    description: str
    environment: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigChange':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ConfigSnapshot:
    """Represents a configuration snapshot"""
    snapshot_id: str
    timestamp: str
    description: str
    environment: str
    files: Dict[str, str]  # file_path -> content_hash
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigSnapshot':
        """Create from dictionary"""
        return cls(**data)


class ConfigTracker:
    """Main configuration tracking system"""
    
    def __init__(self, config_dir: str = None, tracking_dir: str = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self.tracking_dir = Path(tracking_dir) if tracking_dir else self.config_dir / '.config_tracking'
        
        # Create tracking directory structure
        self.tracking_dir.mkdir(exist_ok=True)
        (self.tracking_dir / 'changes').mkdir(exist_ok=True)
        (self.tracking_dir / 'snapshots').mkdir(exist_ok=True)
        (self.tracking_dir / 'backups').mkdir(exist_ok=True)
        
        self.changes_file = self.tracking_dir / 'changes.json'
        self.snapshots_file = self.tracking_dir / 'snapshots.json'
        
        # Initialize tracking files if they don't exist
        if not self.changes_file.exists():
            self._save_changes([])
        if not self.snapshots_file.exists():
            self._save_snapshots([])
    
    def track_change(self, file_path: str, change_type: str, description: str = "",
                    user: str = None, environment: str = None) -> str:
        """Track a configuration change"""
        file_path = str(Path(file_path).resolve())
        
        # Generate change ID
        change_id = self._generate_change_id()
        
        # Get current user and environment
        if user is None:
            user = os.getenv('USER', 'unknown')
        if environment is None:
            environment = os.getenv('ENVIRONMENT', 'unknown')
        
        # Get file content and hash
        old_content = None
        old_hash = None
        new_content = None
        new_hash = None
        
        if change_type in ['update', 'delete']:
            # Get old content from backup or current file
            old_content, old_hash = self._get_file_content_and_hash(file_path)
        
        if change_type in ['create', 'update']:
            # Get new content
            if Path(file_path).exists():
                new_content, new_hash = self._get_file_content_and_hash(file_path)
        
        # Create change record
        change = ConfigChange(
            timestamp=datetime.now(timezone.utc).isoformat(),
            change_id=change_id,
            file_path=file_path,
            change_type=change_type,
            old_hash=old_hash,
            new_hash=new_hash,
            old_content=old_content,
            new_content=new_content,
            user=user,
            description=description,
            environment=environment
        )
        
        # Save change
        self._add_change(change)
        
        # Create backup of the file
        if change_type in ['update', 'delete'] and old_content:
            self._create_backup(file_path, change_id, old_content)
        
        logger.info(f"Tracked configuration change: {change_id} - {description}")
        return change_id
    
    def create_snapshot(self, description: str = "", environment: str = None) -> str:
        """Create a configuration snapshot"""
        if environment is None:
            environment = os.getenv('ENVIRONMENT', 'unknown')
        
        snapshot_id = self._generate_snapshot_id()
        
        # Get all configuration files
        config_files = self._get_all_config_files()
        files_dict = {}
        
        for file_path in config_files:
            try:
                _, file_hash = self._get_file_content_and_hash(file_path)
                files_dict[str(file_path)] = file_hash
            except Exception as e:
                logger.warning(f"Could not include file in snapshot: {file_path} - {e}")
        
        # Create snapshot
        snapshot = ConfigSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            description=description,
            environment=environment,
            files=files_dict,
            metadata={
                'total_files': len(files_dict),
                'config_dir': str(self.config_dir)
            }
        )
        
        # Save snapshot
        self._add_snapshot(snapshot)
        
        # Create snapshot backup
        self._create_snapshot_backup(snapshot_id, config_files)
        
        logger.info(f"Created configuration snapshot: {snapshot_id} - {description}")
        return snapshot_id
    
    def rollback_to_change(self, change_id: str) -> bool:
        """Rollback to a specific change"""
        changes = self._load_changes()
        
        # Find the change
        target_change = None
        for change in changes:
            if change['change_id'] == change_id:
                target_change = ConfigChange.from_dict(change)
                break
        
        if not target_change:
            logger.error(f"Change not found: {change_id}")
            return False
        
        try:
            # Create backup of current state
            current_backup_id = self.create_snapshot(f"Pre-rollback backup for {change_id}")
            
            # Restore the file
            if target_change.change_type == 'delete':
                # Restore deleted file
                if target_change.old_content:
                    with open(target_change.file_path, 'w') as f:
                        f.write(target_change.old_content)
                    logger.info(f"Restored deleted file: {target_change.file_path}")
                else:
                    logger.error(f"Cannot restore deleted file - no backup content")
                    return False
            
            elif target_change.change_type in ['create', 'update']:
                # Rollback to previous version
                if target_change.old_content:
                    with open(target_change.file_path, 'w') as f:
                        f.write(target_change.old_content)
                    logger.info(f"Rolled back file: {target_change.file_path}")
                else:
                    # This was a create operation, delete the file
                    if Path(target_change.file_path).exists():
                        os.remove(target_change.file_path)
                        logger.info(f"Removed created file: {target_change.file_path}")
            
            # Track the rollback as a new change
            self.track_change(
                target_change.file_path,
                'rollback',
                f"Rollback to change {change_id}",
                environment=target_change.environment
            )
            
            logger.info(f"Successfully rolled back to change: {change_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def rollback_to_snapshot(self, snapshot_id: str) -> bool:
        """Rollback to a specific snapshot"""
        snapshots = self._load_snapshots()
        
        # Find the snapshot
        target_snapshot = None
        for snapshot in snapshots:
            if snapshot['snapshot_id'] == snapshot_id:
                target_snapshot = ConfigSnapshot.from_dict(snapshot)
                break
        
        if not target_snapshot:
            logger.error(f"Snapshot not found: {snapshot_id}")
            return False
        
        try:
            # Create backup of current state
            current_backup_id = self.create_snapshot(f"Pre-rollback backup for snapshot {snapshot_id}")
            
            # Restore files from snapshot backup
            snapshot_backup_dir = self.tracking_dir / 'snapshots' / snapshot_id
            
            if not snapshot_backup_dir.exists():
                logger.error(f"Snapshot backup directory not found: {snapshot_backup_dir}")
                return False
            
            # Restore each file
            restored_files = []
            for file_path in target_snapshot.files.keys():
                backup_file = snapshot_backup_dir / Path(file_path).name
                
                if backup_file.exists():
                    # Restore the file
                    shutil.copy2(backup_file, file_path)
                    restored_files.append(file_path)
                    logger.info(f"Restored file: {file_path}")
                else:
                    logger.warning(f"Backup file not found: {backup_file}")
            
            # Track the rollback
            for file_path in restored_files:
                self.track_change(
                    file_path,
                    'rollback',
                    f"Rollback to snapshot {snapshot_id}",
                    environment=target_snapshot.environment
                )
            
            logger.info(f"Successfully rolled back to snapshot: {snapshot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Snapshot rollback failed: {e}")
            return False
    
    def get_change_history(self, file_path: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Get change history"""
        changes = self._load_changes()
        
        # Filter by file path if specified
        if file_path:
            file_path = str(Path(file_path).resolve())
            changes = [c for c in changes if c['file_path'] == file_path]
        
        # Sort by timestamp (newest first)
        changes.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Apply limit if specified
        if limit:
            changes = changes[:limit]
        
        return changes
    
    def get_snapshots(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get snapshot history"""
        snapshots = self._load_snapshots()
        
        # Sort by timestamp (newest first)
        snapshots.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Apply limit if specified
        if limit:
            snapshots = snapshots[:limit]
        
        return snapshots
    
    def compare_configurations(self, snapshot_id1: str, snapshot_id2: str) -> Dict[str, Any]:
        """Compare two configuration snapshots"""
        snapshots = self._load_snapshots()
        
        snapshot1 = None
        snapshot2 = None
        
        for snapshot in snapshots:
            if snapshot['snapshot_id'] == snapshot_id1:
                snapshot1 = ConfigSnapshot.from_dict(snapshot)
            elif snapshot['snapshot_id'] == snapshot_id2:
                snapshot2 = ConfigSnapshot.from_dict(snapshot)
        
        if not snapshot1 or not snapshot2:
            raise ValueError("One or both snapshots not found")
        
        # Compare files
        all_files = set(snapshot1.files.keys()) | set(snapshot2.files.keys())
        
        differences = {
            'added': [],
            'removed': [],
            'modified': [],
            'unchanged': []
        }
        
        for file_path in all_files:
            hash1 = snapshot1.files.get(file_path)
            hash2 = snapshot2.files.get(file_path)
            
            if hash1 and not hash2:
                differences['removed'].append(file_path)
            elif not hash1 and hash2:
                differences['added'].append(file_path)
            elif hash1 != hash2:
                differences['modified'].append(file_path)
            else:
                differences['unchanged'].append(file_path)
        
        return {
            'snapshot1': snapshot1.to_dict(),
            'snapshot2': snapshot2.to_dict(),
            'differences': differences,
            'summary': {
                'total_files': len(all_files),
                'added': len(differences['added']),
                'removed': len(differences['removed']),
                'modified': len(differences['modified']),
                'unchanged': len(differences['unchanged'])
            }
        }
    
    def cleanup_old_backups(self, days: int = 30) -> int:
        """Clean up old backups and snapshots"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (days * 24 * 60 * 60)
        cleaned_count = 0
        
        # Clean up old change backups
        backup_dir = self.tracking_dir / 'backups'
        if backup_dir.exists():
            for backup_file in backup_dir.iterdir():
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    cleaned_count += 1
        
        # Clean up old snapshot backups
        snapshot_dir = self.tracking_dir / 'snapshots'
        if snapshot_dir.exists():
            for snapshot_backup in snapshot_dir.iterdir():
                if snapshot_backup.is_dir() and snapshot_backup.stat().st_mtime < cutoff_time:
                    shutil.rmtree(snapshot_backup)
                    cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} old backup files")
        return cleaned_count
    
    def export_tracking_data(self, output_file: str) -> bool:
        """Export all tracking data to a file"""
        try:
            export_data = {
                'export_timestamp': datetime.now(timezone.utc).isoformat(),
                'config_dir': str(self.config_dir),
                'changes': self._load_changes(),
                'snapshots': self._load_snapshots()
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported tracking data to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def import_tracking_data(self, input_file: str) -> bool:
        """Import tracking data from a file"""
        try:
            with open(input_file, 'r') as f:
                import_data = json.load(f)
            
            # Validate import data
            if 'changes' not in import_data or 'snapshots' not in import_data:
                raise ValueError("Invalid import data format")
            
            # Backup current tracking data
            backup_file = self.tracking_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.export_tracking_data(str(backup_file))
            
            # Import changes and snapshots
            self._save_changes(import_data['changes'])
            self._save_snapshots(import_data['snapshots'])
            
            logger.info(f"Imported tracking data from: {input_file}")
            return True
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False
    
    @contextmanager
    def track_changes(self, description: str = "Batch configuration changes"):
        """Context manager for tracking multiple changes"""
        initial_snapshot = self.create_snapshot(f"Pre-change snapshot: {description}")
        
        try:
            yield
            
            # Create post-change snapshot
            final_snapshot = self.create_snapshot(f"Post-change snapshot: {description}")
            
            logger.info(f"Tracked batch changes: {description}")
            logger.info(f"Initial snapshot: {initial_snapshot}")
            logger.info(f"Final snapshot: {final_snapshot}")
            
        except Exception as e:
            logger.error(f"Error during tracked changes: {e}")
            
            # Rollback to initial snapshot
            logger.info(f"Rolling back to initial snapshot: {initial_snapshot}")
            self.rollback_to_snapshot(initial_snapshot)
            raise
    
    def _generate_change_id(self) -> str:
        """Generate unique change ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"change_{timestamp}_{random_suffix}"
    
    def _generate_snapshot_id(self) -> str:
        """Generate unique snapshot ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"snapshot_{timestamp}_{random_suffix}"
    
    def _get_file_content_and_hash(self, file_path: str) -> Tuple[str, str]:
        """Get file content and its hash"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        file_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return content, file_hash
    
    def _get_all_config_files(self) -> List[Path]:
        """Get all configuration files"""
        config_files = []
        
        # Common configuration file patterns
        patterns = [
            '*.yaml', '*.yml', '*.json', '*.toml', '*.ini', '*.conf',
            '.env*', '*.config'
        ]
        
        for pattern in patterns:
            config_files.extend(self.config_dir.glob(pattern))
        
        # Also check subdirectories
        for subdir in self.config_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                for pattern in patterns:
                    config_files.extend(subdir.glob(pattern))
        
        return config_files
    
    def _create_backup(self, file_path: str, change_id: str, content: str):
        """Create backup of file content"""
        backup_file = self.tracking_dir / 'backups' / f"{change_id}_{Path(file_path).name}"
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _create_snapshot_backup(self, snapshot_id: str, config_files: List[Path]):
        """Create backup of all files in snapshot"""
        snapshot_backup_dir = self.tracking_dir / 'snapshots' / snapshot_id
        snapshot_backup_dir.mkdir(exist_ok=True)
        
        for file_path in config_files:
            if file_path.exists():
                backup_file = snapshot_backup_dir / file_path.name
                shutil.copy2(file_path, backup_file)
    
    def _load_changes(self) -> List[Dict[str, Any]]:
        """Load changes from file"""
        try:
            with open(self.changes_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_changes(self, changes: List[Dict[str, Any]]):
        """Save changes to file"""
        with open(self.changes_file, 'w') as f:
            json.dump(changes, f, indent=2)
    
    def _add_change(self, change: ConfigChange):
        """Add a change to the tracking file"""
        changes = self._load_changes()
        changes.append(change.to_dict())
        self._save_changes(changes)
    
    def _load_snapshots(self) -> List[Dict[str, Any]]:
        """Load snapshots from file"""
        try:
            with open(self.snapshots_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_snapshots(self, snapshots: List[Dict[str, Any]]):
        """Save snapshots to file"""
        with open(self.snapshots_file, 'w') as f:
            json.dump(snapshots, f, indent=2)
    
    def _add_snapshot(self, snapshot: ConfigSnapshot):
        """Add a snapshot to the tracking file"""
        snapshots = self._load_snapshots()
        snapshots.append(snapshot.to_dict())
        self._save_snapshots(snapshots)


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Change Tracking System")
    parser.add_argument('--config-dir', help="Configuration directory")
    parser.add_argument('--tracking-dir', help="Tracking data directory")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Track command
    track_parser = subparsers.add_parser('track', help='Track a configuration change')
    track_parser.add_argument('file_path', help='Path to configuration file')
    track_parser.add_argument('change_type', choices=['create', 'update', 'delete'])
    track_parser.add_argument('--description', default='', help='Change description')
    track_parser.add_argument('--user', help='User making the change')
    track_parser.add_argument('--environment', help='Environment')
    
    # Snapshot command
    snapshot_parser = subparsers.add_parser('snapshot', help='Create a configuration snapshot')
    snapshot_parser.add_argument('--description', default='', help='Snapshot description')
    snapshot_parser.add_argument('--environment', help='Environment')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback configuration')
    rollback_group = rollback_parser.add_mutually_exclusive_group(required=True)
    rollback_group.add_argument('--change-id', help='Change ID to rollback to')
    rollback_group.add_argument('--snapshot-id', help='Snapshot ID to rollback to')
    
    # History command
    history_parser = subparsers.add_parser('history', help='Show change history')
    history_parser.add_argument('--file-path', help='Filter by file path')
    history_parser.add_argument('--limit', type=int, help='Limit number of results')
    
    # Snapshots command
    snapshots_parser = subparsers.add_parser('snapshots', help='List snapshots')
    snapshots_parser.add_argument('--limit', type=int, help='Limit number of results')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare snapshots')
    compare_parser.add_argument('snapshot1', help='First snapshot ID')
    compare_parser.add_argument('snapshot2', help='Second snapshot ID')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old backups')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Days to keep')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export tracking data')
    export_parser.add_argument('output_file', help='Output file path')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import tracking data')
    import_parser.add_argument('input_file', help='Input file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create tracker
    tracker = ConfigTracker(args.config_dir, args.tracking_dir)
    
    # Execute command
    if args.command == 'track':
        change_id = tracker.track_change(
            args.file_path,
            args.change_type,
            args.description,
            args.user,
            args.environment
        )
        print(f"Change tracked: {change_id}")
    
    elif args.command == 'snapshot':
        snapshot_id = tracker.create_snapshot(args.description, args.environment)
        print(f"Snapshot created: {snapshot_id}")
    
    elif args.command == 'rollback':
        if args.change_id:
            success = tracker.rollback_to_change(args.change_id)
        else:
            success = tracker.rollback_to_snapshot(args.snapshot_id)
        
        if success:
            print("Rollback completed successfully")
        else:
            print("Rollback failed")
            sys.exit(1)
    
    elif args.command == 'history':
        changes = tracker.get_change_history(args.file_path, args.limit)
        print(json.dumps(changes, indent=2))
    
    elif args.command == 'snapshots':
        snapshots = tracker.get_snapshots(args.limit)
        print(json.dumps(snapshots, indent=2))
    
    elif args.command == 'compare':
        comparison = tracker.compare_configurations(args.snapshot1, args.snapshot2)
        print(json.dumps(comparison, indent=2))
    
    elif args.command == 'cleanup':
        count = tracker.cleanup_old_backups(args.days)
        print(f"Cleaned up {count} old backup files")
    
    elif args.command == 'export':
        success = tracker.export_tracking_data(args.output_file)
        if success:
            print(f"Tracking data exported to: {args.output_file}")
        else:
            print("Export failed")
            sys.exit(1)
    
    elif args.command == 'import':
        success = tracker.import_tracking_data(args.input_file)
        if success:
            print(f"Tracking data imported from: {args.input_file}")
        else:
            print("Import failed")
            sys.exit(1)


if __name__ == '__main__':
    main()
