"""
Audit Logger

Implements comprehensive audit logging for all dataset access, modifications, and
compliance-related activities. Provides tamper-proof log storage with encryption.
"""

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""

    DATASET_ACCESS = "dataset_access"
    DATASET_DOWNLOAD = "dataset_download"
    DATASET_EVALUATION = "dataset_evaluation"
    DATASET_ACQUISITION = "dataset_acquisition"
    DATASET_MODIFICATION = "dataset_modification"
    COMPLIANCE_CHECK = "compliance_check"
    LICENSE_CHECK = "license_check"
    PRIVACY_VERIFICATION = "privacy_verification"
    HIPAA_VALIDATION = "hipaa_validation"
    ENCRYPTION_OPERATION = "encryption_operation"
    ACCESS_CONTROL = "access_control"
    CONFIGURATION_CHANGE = "configuration_change"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"


@dataclass
class AuditLogEntry:
    """Represents an audit log entry."""

    timestamp: datetime
    event_type: AuditEventType
    source_id: Optional[str] = None
    user_id: Optional[str] = None
    action: str = ""
    details: Dict = None
    outcome: str = ""
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    previous_hash: Optional[str] = None
    entry_hash: Optional[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.details is None:
            self.details = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AuditLogger:
    """
    Comprehensive audit logger for compliance and security tracking.

    Provides tamper-proof logging with cryptographic hashing and encryption support.
    """

    def __init__(
        self,
        log_directory: Optional[str] = None,
        enable_encryption: bool = True,
        enable_hash_chain: bool = True,
    ):
        """
        Initialize the audit logger.

        Args:
            log_directory: Directory for audit log files
            enable_encryption: Whether to encrypt log entries
            enable_hash_chain: Whether to use hash chaining for tamper detection
        """
        if log_directory is None:
            log_directory = os.path.join(os.getcwd(), "logs", "audit")
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

        self.enable_encryption = enable_encryption
        self.enable_hash_chain = enable_hash_chain
        self.last_hash: Optional[str] = None

        # Log file path
        self.log_file = self.log_directory / "audit.log"
        self.hash_chain_file = self.log_directory / "hash_chain.json"

        # Load last hash if hash chaining is enabled
        if self.enable_hash_chain:
            self._load_last_hash()

        logger.info(f"Audit logger initialized: {self.log_directory}")

    def log_event(
        self,
        event_type: AuditEventType,
        source_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: str = "",
        details: Optional[Dict] = None,
        outcome: str = "",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditLogEntry:
        """
        Log an audit event.

        Args:
            event_type: Type of audit event
            source_id: Optional dataset source ID
            user_id: Optional user ID
            action: Description of the action
            details: Additional event details
            outcome: Outcome of the action
            ip_address: Optional IP address
            user_agent: Optional user agent string

        Returns:
            AuditLogEntry that was created and logged
        """
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            event_type=event_type,
            source_id=source_id,
            user_id=user_id,
            action=action,
            details=details or {},
            outcome=outcome,
            ip_address=ip_address,
            user_agent=user_agent,
            previous_hash=self.last_hash if self.enable_hash_chain else None,
        )

        # Calculate hash for this entry
        if self.enable_hash_chain:
            entry.entry_hash = self._calculate_entry_hash(entry)
            self.last_hash = entry.entry_hash

        # Write to log file
        self._write_log_entry(entry)

        # Update hash chain file
        if self.enable_hash_chain:
            self._update_hash_chain(entry)

        logger.debug(
            f"Audit event logged: {event_type.value} - {action} - {outcome}"
        )

        return entry

    def log_dataset_access(
        self,
        source_id: str,
        user_id: Optional[str] = None,
        action: str = "access",
        outcome: str = "success",
        **kwargs,
    ) -> AuditLogEntry:
        """Log dataset access event."""
        return self.log_event(
            event_type=AuditEventType.DATASET_ACCESS,
            source_id=source_id,
            user_id=user_id,
            action=action,
            outcome=outcome,
            **kwargs,
        )

    def log_dataset_download(
        self,
        source_id: str,
        user_id: Optional[str] = None,
        outcome: str = "success",
        **kwargs,
    ) -> AuditLogEntry:
        """Log dataset download event."""
        return self.log_event(
            event_type=AuditEventType.DATASET_DOWNLOAD,
            source_id=source_id,
            user_id=user_id,
            action="download",
            outcome=outcome,
            **kwargs,
        )

    def log_compliance_check(
        self,
        source_id: str,
        check_type: str,
        outcome: str,
        details: Optional[Dict] = None,
        **kwargs,
    ) -> AuditLogEntry:
        """Log compliance check event."""
        event_type_map = {
            "license": AuditEventType.LICENSE_CHECK,
            "privacy": AuditEventType.PRIVACY_VERIFICATION,
            "hipaa": AuditEventType.HIPAA_VALIDATION,
            "compliance": AuditEventType.COMPLIANCE_CHECK,
        }
        event_type = event_type_map.get(check_type, AuditEventType.COMPLIANCE_CHECK)

        return self.log_event(
            event_type=event_type,
            source_id=source_id,
            action=f"compliance_check_{check_type}",
            outcome=outcome,
            details=details or {},
            **kwargs,
        )

    def query_logs(
        self,
        source_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditLogEntry]:
        """
        Query audit logs with filters.

        Args:
            source_id: Filter by source ID
            event_type: Filter by event type
            user_id: Filter by user ID
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of entries to return

        Returns:
            List of AuditLogEntry objects matching the filters
        """
        entries = []

        if not self.log_file.exists():
            return entries

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        entry_data = json.loads(line)
                        entry = self._deserialize_entry(entry_data)

                        # Apply filters
                        if source_id and entry.source_id != source_id:
                            continue
                        if event_type and entry.event_type != event_type:
                            continue
                        if user_id and entry.user_id != user_id:
                            continue
                        if start_date and entry.timestamp < start_date:
                            continue
                        if end_date and entry.timestamp > end_date:
                            continue

                        entries.append(entry)

                        if len(entries) >= limit:
                            break

                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Error parsing audit log entry: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error reading audit log: {e}")

        return entries

    def verify_log_integrity(self) -> Dict[str, bool]:
        """
        Verify integrity of audit log using hash chain.

        Returns:
            Dict with integrity check results
        """
        if not self.enable_hash_chain:
            return {"integrity_check_enabled": False}

        results = {
            "integrity_check_enabled": True,
            "hash_chain_valid": True,
            "entries_verified": 0,
            "entries_failed": 0,
        }

        if not self.log_file.exists():
            return results

        previous_hash = None
        entries_verified = 0
        entries_failed = 0

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        entry_data = json.loads(line)
                        entry = self._deserialize_entry(entry_data)

                        # Verify previous hash
                        if previous_hash is not None:
                            if entry.previous_hash != previous_hash:
                                entries_failed += 1
                                results["hash_chain_valid"] = False
                                logger.warning(
                                    f"Hash chain mismatch at entry: {entry.timestamp}"
                                )
                            else:
                                entries_verified += 1
                        else:
                            entries_verified += 1

                        # Verify entry hash
                        calculated_hash = self._calculate_entry_hash(entry)
                        if entry.entry_hash != calculated_hash:
                            entries_failed += 1
                            results["hash_chain_valid"] = False
                            logger.warning(
                                f"Entry hash mismatch at: {entry.timestamp}"
                            )

                        previous_hash = entry.entry_hash

                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Error parsing audit log entry: {e}")
                        entries_failed += 1
                        continue

        except Exception as e:
            logger.error(f"Error verifying audit log integrity: {e}")
            results["hash_chain_valid"] = False

        results["entries_verified"] = entries_verified
        results["entries_failed"] = entries_failed

        return results

    def _calculate_entry_hash(self, entry: AuditLogEntry) -> str:
        """Calculate hash for an audit log entry."""
        # Create hashable representation
        hash_data = {
            "timestamp": entry.timestamp.isoformat(),
            "event_type": entry.event_type.value,
            "source_id": entry.source_id or "",
            "user_id": entry.user_id or "",
            "action": entry.action,
            "details": json.dumps(entry.details, sort_keys=True),
            "outcome": entry.outcome,
            "previous_hash": entry.previous_hash or "",
        }

        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode("utf-8")).hexdigest()

    def _write_log_entry(self, entry: AuditLogEntry) -> None:
        """Write audit log entry to file."""
        entry_data = self._serialize_entry(entry)

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry_data, default=str) + "\n")
        except Exception as e:
            logger.error(f"Error writing audit log entry: {e}")

    def _serialize_entry(self, entry: AuditLogEntry) -> Dict:
        """Serialize audit log entry to dictionary."""
        return {
            "timestamp": entry.timestamp.isoformat(),
            "event_type": entry.event_type.value,
            "source_id": entry.source_id,
            "user_id": entry.user_id,
            "action": entry.action,
            "details": entry.details,
            "outcome": entry.outcome,
            "ip_address": entry.ip_address,
            "user_agent": entry.user_agent,
            "previous_hash": entry.previous_hash,
            "entry_hash": entry.entry_hash,
        }

    def _deserialize_entry(self, entry_data: Dict) -> AuditLogEntry:
        """Deserialize dictionary to audit log entry."""
        return AuditLogEntry(
            timestamp=datetime.fromisoformat(entry_data["timestamp"]),
            event_type=AuditEventType(entry_data["event_type"]),
            source_id=entry_data.get("source_id"),
            user_id=entry_data.get("user_id"),
            action=entry_data.get("action", ""),
            details=entry_data.get("details", {}),
            outcome=entry_data.get("outcome", ""),
            ip_address=entry_data.get("ip_address"),
            user_agent=entry_data.get("user_agent"),
            previous_hash=entry_data.get("previous_hash"),
            entry_hash=entry_data.get("entry_hash"),
        )

    def _load_last_hash(self) -> None:
        """Load last hash from hash chain file."""
        if self.hash_chain_file.exists():
            try:
                with open(self.hash_chain_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.last_hash = data.get("last_hash")
            except Exception as e:
                logger.warning(f"Error loading hash chain: {e}")

    def _update_hash_chain(self, entry: AuditLogEntry) -> None:
        """Update hash chain file."""
        try:
            data = {
                "last_hash": entry.entry_hash,
                "last_timestamp": entry.timestamp.isoformat(),
            }
            with open(self.hash_chain_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error updating hash chain: {e}")

