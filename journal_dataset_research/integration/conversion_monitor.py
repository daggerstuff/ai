"""
Conversion Monitor

Monitor conversion progress for journal research datasets.
Tracks conversion status, progress, and provides status reporting.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConversionMonitor:
    """
    Monitor conversion operations for journal research datasets.
    
    Provides:
    1. Real-time conversion status tracking
    2. Progress reporting
    3. Status persistence
    4. Dashboard data generation
    """

    def __init__(self, status_file: Path = Path("data/journal_research/conversion_status.json")):
        """
        Initialize conversion monitor.
        
        Args:
            status_file: Path to status persistence file
        """
        self.status_file = Path(status_file)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory status tracking
        self.conversion_status: Dict[str, Dict[str, Any]] = {}
        
        # Load existing status if available
        self._load_status()
        
        logger.info(f"Initialized ConversionMonitor: status_file={status_file}")

    def _load_status(self) -> None:
        """Load conversion status from file."""
        if self.status_file.exists():
            try:
                with open(self.status_file, "r", encoding="utf-8") as f:
                    self.conversion_status = json.load(f)
                logger.info(f"Loaded {len(self.conversion_status)} conversion statuses")
            except Exception as e:
                logger.warning(f"Failed to load conversion status: {e}")
                self.conversion_status = {}

    def _save_status(self) -> None:
        """Save conversion status to file."""
        try:
            with open(self.status_file, "w", encoding="utf-8") as f:
                json.dump(self.conversion_status, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save conversion status: {e}")

    def start_conversion(self, source_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark conversion as started.
        
        Args:
            source_id: Source ID of the dataset
            metadata: Optional metadata about the conversion
        """
        self.conversion_status[source_id] = {
            "source_id": source_id,
            "status": "in_progress",
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "progress": 0.0,
        }
        self._save_status()
        logger.debug(f"Started conversion tracking for {source_id}")

    def update_progress(
        self, source_id: str, progress: float, message: Optional[str] = None
    ) -> None:
        """
        Update conversion progress.
        
        Args:
            source_id: Source ID of the dataset
            progress: Progress percentage (0.0-1.0)
            message: Optional progress message
        """
        if source_id not in self.conversion_status:
            self.start_conversion(source_id)
        
        self.conversion_status[source_id].update({
            "progress": progress,
            "updated_at": datetime.now().isoformat(),
        })
        
        if message:
            self.conversion_status[source_id]["message"] = message
        
        self._save_status()
        logger.debug(f"Updated progress for {source_id}: {progress:.1%}")

    def complete_conversion(
        self, source_id: str, result: Dict[str, Any]
    ) -> None:
        """
        Mark conversion as completed.
        
        Args:
            source_id: Source ID of the dataset
            result: Conversion result dictionary
        """
        if source_id not in self.conversion_status:
            self.start_conversion(source_id)
        
        self.conversion_status[source_id].update({
            "status": "completed" if result.get("success") else "failed",
            "progress": 1.0,
            "completed_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "result": result,
        })
        
        if not result.get("success"):
            self.conversion_status[source_id]["error"] = result.get("error")
        
        self._save_status()
        logger.info(
            f"Completed conversion tracking for {source_id}: "
            f"status={self.conversion_status[source_id]['status']}"
        )

    def get_status(self, source_id: str) -> Optional[Dict[str, Any]]:
        """
        Get conversion status for a dataset.
        
        Args:
            source_id: Source ID of the dataset
            
        Returns:
            Status dictionary or None if not found
        """
        return self.conversion_status.get(source_id)

    def list_conversions(
        self,
        status_filter: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all conversions with optional filtering.
        
        Args:
            status_filter: Optional status filter ("in_progress", "completed", "failed")
            limit: Optional limit on number of results
            
        Returns:
            List of conversion status dictionaries
        """
        conversions = list(self.conversion_status.values())
        
        if status_filter:
            conversions = [
                c for c in conversions if c.get("status") == status_filter
            ]
        
        # Sort by updated_at (most recent first)
        conversions.sort(
            key=lambda x: x.get("updated_at", ""), reverse=True
        )
        
        if limit:
            conversions = conversions[:limit]
        
        return conversions

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data for conversion dashboard.
        
        Returns:
            Dictionary with dashboard statistics and data
        """
        total = len(self.conversion_status)
        in_progress = sum(
            1 for s in self.conversion_status.values()
            if s.get("status") == "in_progress"
        )
        completed = sum(
            1 for s in self.conversion_status.values()
            if s.get("status") == "completed"
        )
        failed = sum(
            1 for s in self.conversion_status.values()
            if s.get("status") == "failed"
        )
        
        # Calculate average progress
        total_progress = sum(
            s.get("progress", 0.0) for s in self.conversion_status.values()
        )
        avg_progress = total_progress / total if total > 0 else 0.0
        
        return {
            "total": total,
            "in_progress": in_progress,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total if total > 0 else 0.0,
            "average_progress": avg_progress,
            "recent_conversions": self.list_conversions(limit=10),
        }


