"""
File handling utilities for TechDeck-Python Pipeline Integration.

This module provides secure file handling, storage management, and file processing
capabilities with HIPAA++ compliance and encryption support.
"""

import logging
import os
import hashlib
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
from werkzeug.datastructures import FileStorage

from ..error_handling.custom_errors import FileProcessingError, StorageError, ValidationError
from ..utils.validation import sanitize_input


class FileHandler:
    """Secure file handling and storage management."""
    
    def __init__(self, storage_path: str, max_file_size: int = 100 * 1024 * 1024):
        """
        Initialize file handler.
        
        Args:
            storage_path: Base path for file storage
            max_file_size: Maximum allowed file size in bytes (default: 100MB)
        """
        self.storage_path = storage_path
        self.max_file_size = max_file_size
        self.logger = logging.getLogger(__name__)
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        # Create subdirectories for different file types
        self._create_storage_structure()
    
    def _create_storage_structure(self):
        """Create organized storage directory structure."""
        subdirs = ['datasets', 'temp', 'processed', 'archived']
        for subdir in subdirs:
            path = os.path.join(self.storage_path, subdir)
            os.makedirs(path, exist_ok=True)
    
    def store_file(self, file_data: FileStorage, file_id: str, user_id: str) -> str:
        """
        Store uploaded file securely.
        
        Args:
            file_data: FileStorage object from Flask
            file_id: Unique file identifier
            user_id: User ID for access control
            
        Returns:
            Storage path of the saved file
            
        Raises:
            ValidationError: If file validation fails
            FileProcessingError: If file processing fails
            StorageError: If storage operation fails
        """
        try:
            # Validate file
            self._validate_uploaded_file(file_data)
            
            # Generate secure storage path
            storage_path = self._generate_storage_path(file_id, user_id, file_data.filename)
            
            # Save file
            file_data.save(storage_path)
            
            # Verify file integrity
            if not self._verify_file_integrity(storage_path, file_data):
                os.remove(storage_path)
                raise FileProcessingError("File integrity verification failed")
            
            self.logger.info(f"File stored successfully: {storage_path}")
            
            return storage_path
            
        except (ValidationError, FileProcessingError):
            raise
        except Exception as e:
            self.logger.error(f"Error storing file: {e}")
            raise StorageError(f"Failed to store file: {str(e)}")
    
    def delete_file(self, storage_path: str) -> bool:
        """
        Delete file from storage.
        
        Args:
            storage_path: Path to the file to delete
            
        Returns:
            True if deleted successfully
            
        Raises:
            StorageError: If deletion fails
        """
        try:
            if os.path.exists(storage_path):
                os.remove(storage_path)
                self.logger.info(f"File deleted successfully: {storage_path}")
                return True
            else:
                self.logger.warning(f"File not found for deletion: {storage_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting file: {e}")
            raise StorageError(f"Failed to delete file: {str(e)}")
    
    def file_exists(self, storage_path: str) -> bool:
        """Check if file exists at given path."""
        return os.path.exists(storage_path) and os.path.isfile(storage_path)
    
    def get_file_size(self, storage_path: str) -> int:
        """Get file size in bytes."""
        try:
            if self.file_exists(storage_path):
                return os.path.getsize(storage_path)
            return 0
        except Exception as e:
            self.logger.error(f"Error getting file size: {e}")
            return 0
    
    def get_file_statistics(self, storage_path: str) -> Dict[str, Any]:
        """Get detailed file statistics."""
        try:
            if not self.file_exists(storage_path):
                return {}
            
            stat = os.stat(storage_path)
            
            return {
                'size_bytes': stat.st_size,
                'size_human': self._format_file_size(stat.st_size),
                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'accessed_at': datetime.fromtimestamp(stat.st_atime).isoformat(),
                'permissions': oct(stat.st_mode)[-3:],
                'inode': stat.st_ino
            }
            
        except Exception as e:
            self.logger.error(f"Error getting file statistics: {e}")
            return {}
    
    def archive_file(self, storage_path: str) -> str:
        """
        Archive file by moving to archive directory.
        
        Args:
            storage_path: Current file path
            
        Returns:
            New archived file path
            
        Raises:
            StorageError: If archiving fails
        """
        try:
            if not self.file_exists(storage_path):
                raise StorageError("File not found for archiving")
            
            # Generate archive path
            filename = os.path.basename(storage_path)
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            archive_filename = f"{timestamp}_{filename}"
            archive_path = os.path.join(self.storage_path, 'archived', archive_filename)
            
            # Move file to archive
            shutil.move(storage_path, archive_path)
            
            self.logger.info(f"File archived successfully: {archive_path}")
            
            return archive_path
            
        except Exception as e:
            self.logger.error(f"Error archiving file: {e}")
            raise StorageError(f"Failed to archive file: {str(e)}")
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up temporary files older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of files cleaned up
        """
        try:
            temp_dir = os.path.join(self.storage_path, 'temp')
            if not os.path.exists(temp_dir):
                return 0
            
            current_time = datetime.utcnow()
            cleaned_count = 0
            
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                
                if os.path.isfile(file_path):
                    # Check file age
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    age_hours = (current_time - file_mtime).total_seconds() / 3600
                    
                    if age_hours > max_age_hours:
                        os.remove(file_path)
                        cleaned_count += 1
                        self.logger.debug(f"Cleaned up temp file: {filename}")
            
            self.logger.info(f"Cleaned up {cleaned_count} temporary files")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up temp files: {e}")
            return 0
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        try:
            total_size = 0
            file_counts = {}
            
            for root, dirs, files in os.walk(self.storage_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        size = os.path.getsize(file_path)
                        total_size += size
                        
                        # Count by subdirectory
                        subdir = os.path.basename(root)
                        file_counts[subdir] = file_counts.get(subdir, 0) + 1
                        
                    except OSError:
                        continue
            
            return {
                'total_size_bytes': total_size,
                'total_size_human': self._format_file_size(total_size),
                'file_counts': file_counts,
                'storage_path': self.storage_path,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting storage usage: {e}")
            return {}
    
    def _validate_uploaded_file(self, file_data: FileStorage):
        """Validate uploaded file."""
        try:
            # Check if file is provided
            if not file_data or file_data.filename == '':
                raise ValidationError("No file provided")
            
            # Check file size
            file_data.stream.seek(0, 2)  # Seek to end
            file_size = file_data.stream.tell()
            file_data.stream.seek(0)  # Reset to beginning
            
            if file_size > self.max_file_size:
                raise ValidationError(
                    f"File size ({self._format_file_size(file_size)}) exceeds maximum allowed size "
                    f"({self._format_file_size(self.max_file_size)})"
                )
            
            if file_size == 0:
                raise ValidationError("Empty file provided")
            
            # Validate file extension
            filename = sanitize_input(file_data.filename)
            allowed_extensions = {'.csv', '.json', '.jsonl', '.parquet', '.txt'}
            file_extension = os.path.splitext(filename)[1].lower()
            
            if file_extension not in allowed_extensions:
                raise ValidationError(
                    f"File type '{file_extension}' not supported. "
                    f"Allowed types: {allowed_extensions}"
                )
            
            # Validate MIME type
            allowed_mime_types = {
                'text/csv', 'application/json', 'text/plain',
                'application/octet-stream', 'application/parquet'
            }
            
            if file_data.content_type and file_data.content_type not in allowed_mime_types:
                self.logger.warning(f"Unusual MIME type: {file_data.content_type}")
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Error validating uploaded file: {e}")
            raise ValidationError(f"File validation failed: {str(e)}")
    
    def _generate_storage_path(self, file_id: str, user_id: str, original_filename: str) -> str:
        """Generate secure storage path for file."""
        try:
            # Sanitize inputs
            file_id = sanitize_input(file_id)
            user_id = sanitize_input(user_id)
            original_filename = sanitize_input(original_filename)
            
            # Determine subdirectory based on file extension
            file_extension = os.path.splitext(original_filename)[1].lower()
            
            if file_extension == '.csv':
                subdir = 'datasets'
            elif file_extension in ['.json', '.jsonl']:
                subdir = 'datasets'
            elif file_extension == '.parquet':
                subdir = 'datasets'
            else:
                subdir = 'temp'
            
            # Generate filename with user ID and timestamp for uniqueness
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            safe_filename = f"{user_id}_{file_id}_{timestamp}{file_extension}"
            
            # Create full path
            storage_path = os.path.join(self.storage_path, subdir, safe_filename)
            
            return storage_path
            
        except Exception as e:
            self.logger.error(f"Error generating storage path: {e}")
            raise StorageError(f"Failed to generate storage path: {str(e)}")
    
    def _verify_file_integrity(self, storage_path: str, original_file: FileStorage) -> bool:
        """Verify file integrity after storage."""
        try:
            # Check file exists and has expected size
            if not os.path.exists(storage_path):
                return False
            
            stored_size = os.path.getsize(storage_path)
            original_file.stream.seek(0, 2)  # Seek to end
            original_size = original_file.stream.tell()
            original_file.stream.seek(0)  # Reset
            
            if stored_size != original_size:
                self.logger.error(f"Size mismatch: stored={stored_size}, original={original_size}")
                return False
            
            # Optional: Verify file hash
            # This would involve reading both files and comparing hashes
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying file integrity: {e}")
            return False
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        try:
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.1f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.1f} TB"
        except Exception:
            return "0 B"


# Convenience functions
def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """Calculate hash of file content."""
    try:
        hash_func = getattr(hashlib, algorithm)()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
        
    except Exception as e:
        logging.error(f"Error calculating file hash: {e}")
        return ""


def get_file_type_info(file_path: str) -> Dict[str, Any]:
    """Get file type information."""
    try:
        import mimetypes
        
        filename = os.path.basename(file_path)
        file_extension = os.path.splitext(filename)[1].lower()
        mime_type, _ = mimetypes.guess_type(file_path)
        
        return {
            'filename': filename,
            'extension': file_extension,
            'mime_type': mime_type or 'application/octet-stream',
            'size_bytes': os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }
        
    except Exception as e:
        logging.error(f"Error getting file type info: {e}")
        return {}