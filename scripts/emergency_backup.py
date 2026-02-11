#!/usr/bin/env python3
"""
Emergency Backup System for Processed Conversations
Creates immediate backup of all processed conversation data to prevent data loss.
"""

import os
import json
import shutil
import tarfile
import gzip
import hashlib
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emergency_backup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmergencyBackupSystem:
    """Emergency backup system for conversation data."""
    
    def __init__(self, source_dir: str = "data", backup_dir: str = "backups"):
        self.source_dir = Path(source_dir)
        self.backup_dir = Path(backup_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_name = f"emergency_backup_{self.timestamp}"
        
        # Create backup directory
        self.backup_dir.mkdir(exist_ok=True)
        
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum for file integrity verification."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def create_manifest(self, backed_up_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create backup manifest with file inventory and checksums."""
        manifest = {
            "backup_name": self.backup_name,
            "timestamp": self.timestamp,
            "source_directory": str(self.source_dir),
            "total_files": len(backed_up_files),
            "files": backed_up_files,
            "backup_size_bytes": sum(f["size_bytes"] for f in backed_up_files),
            "backup_completed": datetime.now().isoformat()
        }
        return manifest
    
    def backup_processed_conversations(self) -> bool:
        """Backup all processed conversation data."""
        logger.info(f"Starting emergency backup: {self.backup_name}")
        
        try:
            # Create backup subdirectory
            backup_path = self.backup_dir / self.backup_name
            backup_path.mkdir(exist_ok=True)
            
            backed_up_files = []
            
            # Backup processed data
            processed_dir = self.source_dir / "processed"
            if processed_dir.exists():
                logger.info("Backing up processed conversation data...")
                processed_backup = backup_path / "processed"
                shutil.copytree(processed_dir, processed_backup)
                
                # Calculate checksums for processed files
                for file_path in processed_backup.rglob("*"):
                    if file_path.is_file():
                        checksum = self.calculate_checksum(file_path)
                        relative_path = file_path.relative_to(backup_path)
                        backed_up_files.append({
                            "path": str(relative_path),
                            "size_bytes": file_path.stat().st_size,
                            "checksum": checksum,
                            "backup_time": datetime.now().isoformat()
                        })
            
            # Backup batch processing data
            batch_dir = self.source_dir / "batch_processing"
            if batch_dir.exists():
                logger.info("Backing up batch processing data...")
                batch_backup = backup_path / "batch_processing"
                shutil.copytree(batch_dir, batch_backup)
                
                # Calculate checksums for batch files
                for file_path in batch_backup.rglob("*"):
                    if file_path.is_file():
                        checksum = self.calculate_checksum(file_path)
                        relative_path = file_path.relative_to(backup_path)
                        backed_up_files.append({
                            "path": str(relative_path),
                            "size_bytes": file_path.stat().st_size,
                            "checksum": checksum,
                            "backup_time": datetime.now().isoformat()
                        })
            
            # Backup psychology knowledge base
            psychology_dir = self.source_dir / "psychology"
            if psychology_dir.exists():
                logger.info("Backing up psychology knowledge base...")
                psychology_backup = backup_path / "psychology"
                shutil.copytree(psychology_dir, psychology_backup)
                
                # Calculate checksums for psychology files
                for file_path in psychology_backup.rglob("*"):
                    if file_path.is_file():
                        checksum = self.calculate_checksum(file_path)
                        relative_path = file_path.relative_to(backup_path)
                        backed_up_files.append({
                            "path": str(relative_path),
                            "size_bytes": file_path.stat().st_size,
                            "checksum": checksum,
                            "backup_time": datetime.now().isoformat()
                        })
            
            # Create manifest
            manifest = self.create_manifest(backed_up_files)
            manifest_path = backup_path / "backup_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Create compressed archive
            logger.info("Creating compressed backup archive...")
            archive_path = self.backup_dir / f"{self.backup_name}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(backup_path, arcname=self.backup_name)
            
            # Calculate archive checksum
            archive_checksum = self.calculate_checksum(archive_path)
            
            # Create archive manifest
            archive_manifest = {
                "archive_name": f"{self.backup_name}.tar.gz",
                "archive_path": str(archive_path),
                "archive_size_bytes": archive_path.stat().st_size,
                "archive_checksum": archive_checksum,
                "created": datetime.now().isoformat(),
                "contains": manifest
            }
            
            archive_manifest_path = self.backup_dir / f"{self.backup_name}_archive_manifest.json"
            with open(archive_manifest_path, 'w') as f:
                json.dump(archive_manifest, f, indent=2)
            
            # Clean up uncompressed backup
            shutil.rmtree(backup_path)
            
            logger.info(f"Emergency backup completed successfully!")
            logger.info(f"Archive: {archive_path}")
            logger.info(f"Size: {archive_path.stat().st_size / (1024*1024*1024):.2f} GB")
            logger.info(f"Files backed up: {len(backed_up_files)}")
            logger.info(f"Checksum: {archive_checksum}")
            
            return True
            
        except Exception as e:
            logger.error(f"Emergency backup failed: {e}")
            return False
    
    def verify_backup(self, archive_path: str) -> bool:
        """Verify backup integrity."""
        logger.info(f"Verifying backup: {archive_path}")
        
        try:
            # Load archive manifest
            manifest_path = archive_path.replace('.tar.gz', '_archive_manifest.json')
            with open(manifest_path, 'r') as f:
                archive_manifest = json.load(f)
            
            # Verify archive checksum
            current_checksum = self.calculate_checksum(Path(archive_path))
            expected_checksum = archive_manifest["archive_checksum"]
            
            if current_checksum != expected_checksum:
                logger.error(f"Archive checksum mismatch! Expected: {expected_checksum}, Got: {current_checksum}")
                return False
            
            logger.info("Backup verification successful!")
            return True
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False

def main():
    """Run emergency backup."""
    backup_system = EmergencyBackupSystem()
    
    logger.info("🚨 STARTING EMERGENCY BACKUP OF CONVERSATION DATA 🚨")
    logger.info("This backup protects against data loss during infrastructure migration")
    
    success = backup_system.backup_processed_conversations()
    
    if success:
        logger.info("✅ Emergency backup completed successfully!")
        logger.info("Data is now protected against system failures during migration")
    else:
        logger.error("❌ Emergency backup failed!")
        logger.error("CRITICAL: Data remains at risk!")
    
    return success

if __name__ == "__main__":
    main()
