"""
Comprehensive Dataset Inventory System

Advanced inventory management for datasets with metadata tracking,
version control, dependency management, and automated discovery.
"""

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


class DatasetStatus(Enum):
    """Dataset status enumeration."""

    AVAILABLE = "available"
    PROCESSING = "processing"
    CORRUPTED = "corrupted"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class DatasetType(Enum):
    """Dataset type enumeration."""

    MENTAL_HEALTH = "mental_health"
    VOICE_TRAINING = "voice_training"
    PSYCHOLOGY = "psychology"
    REASONING = "reasoning"
    PERSONALITY = "personality"
    QUALITY = "quality"
    MIXED = "mixed"


@dataclass
class DatasetVersion:
    """Dataset version information."""

    version: str
    created_date: datetime
    file_path: str
    file_size: int
    checksum: str
    changes: str
    is_current: bool = False


@dataclass
class DatasetDependency:
    """Dataset dependency information."""

    dataset_id: str
    dependency_type: str  # 'requires', 'enhances', 'conflicts'
    version_constraint: str | None = None
    description: str | None = None


@dataclass
class DatasetMetrics:
    """Dataset quality and usage metrics."""

    quality_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    usage_count: int = 0
    last_accessed: datetime | None = None
    processing_time: float = 0.0
    error_count: int = 0


@dataclass
class DatasetRecord:
    """Complete dataset record."""

    id: str
    name: str
    description: str
    dataset_type: DatasetType
    status: DatasetStatus
    created_date: datetime
    updated_date: datetime
    file_path: str
    file_size: int
    format: str
    source: str
    license: str
    tags: list[str] = field(default_factory=list)
    versions: list[DatasetVersion] = field(default_factory=list)
    dependencies: list[DatasetDependency] = field(default_factory=list)
    metrics: DatasetMetrics = field(default_factory=DatasetMetrics)
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetInventoryDB:
    """SQLite database for dataset inventory."""

    def __init__(self, db_path: str = "dataset_inventory.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.logger = get_logger(__name__)
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                # Main datasets table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datasets (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        dataset_type TEXT,
                        status TEXT,
                        created_date TEXT,
                        updated_date TEXT,
                        file_path TEXT,
                        file_size INTEGER,
                        format TEXT,
                        source TEXT,
                        license TEXT,
                        tags TEXT,
                        metadata TEXT
                    )
                """
                )

                # Versions table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS dataset_versions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dataset_id TEXT,
                        version TEXT,
                        created_date TEXT,
                        file_path TEXT,
                        file_size INTEGER,
                        checksum TEXT,
                        changes TEXT,
                        is_current BOOLEAN,
                        FOREIGN KEY (dataset_id) REFERENCES datasets (id)
                    )
                """
                )

                # Dependencies table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS dataset_dependencies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dataset_id TEXT,
                        dependency_id TEXT,
                        dependency_type TEXT,
                        version_constraint TEXT,
                        description TEXT,
                        FOREIGN KEY (dataset_id) REFERENCES datasets (id)
                    )
                """
                )

                # Metrics table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS dataset_metrics (
                        dataset_id TEXT PRIMARY KEY,
                        quality_score REAL,
                        completeness_score REAL,
                        consistency_score REAL,
                        usage_count INTEGER,
                        last_accessed TEXT,
                        processing_time REAL,
                        error_count INTEGER,
                        FOREIGN KEY (dataset_id) REFERENCES datasets (id)
                    )
                """
                )

                # Create indexes
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_dataset_type ON datasets (dataset_type)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_dataset_status ON datasets (status)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_dataset_tags ON datasets (tags)"
                )

                conn.commit()

            finally:
                conn.close()

        logger.info("Dataset inventory database initialized")

    def insert_dataset(self, record: DatasetRecord) -> bool:
        """Insert new dataset record."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                # Insert main record
                conn.execute(
                    """
                    INSERT OR REPLACE INTO datasets
                    (id, name, description, dataset_type, status, created_date, updated_date,
                     file_path, file_size, format, source, license, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        record.id,
                        record.name,
                        record.description,
                        record.dataset_type.value,
                        record.status.value,
                        record.created_date.isoformat(),
                        record.updated_date.isoformat(),
                        record.file_path,
                        record.file_size,
                        record.format,
                        record.source,
                        record.license,
                        json.dumps(record.tags),
                        json.dumps(record.metadata),
                    ),
                )

                # Insert metrics
                conn.execute(
                    """
                    INSERT OR REPLACE INTO dataset_metrics
                    (dataset_id, quality_score, completeness_score, consistency_score,
                     usage_count, last_accessed, processing_time, error_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        record.id,
                        record.metrics.quality_score,
                        record.metrics.completeness_score,
                        record.metrics.consistency_score,
                        record.metrics.usage_count,
                        (
                            record.metrics.last_accessed.isoformat()
                            if record.metrics.last_accessed
                            else None
                        ),
                        record.metrics.processing_time,
                        record.metrics.error_count,
                    ),
                )

                # Insert versions
                for version in record.versions:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO dataset_versions
                        (dataset_id, version, created_date, file_path, file_size, checksum, changes, is_current)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            record.id,
                            version.version,
                            version.created_date.isoformat(),
                            version.file_path,
                            version.file_size,
                            version.checksum,
                            version.changes,
                            version.is_current,
                        ),
                    )

                # Insert dependencies
                for dep in record.dependencies:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO dataset_dependencies
                        (dataset_id, dependency_id, dependency_type, version_constraint, description)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            record.id,
                            dep.dataset_id,
                            dep.dependency_type,
                            dep.version_constraint,
                            dep.description,
                        ),
                    )

                conn.commit()
                return True

            except Exception as e:
                logger.error(f"Failed to insert dataset record: {e}")
                conn.rollback()
                return False
            finally:
                conn.close()

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        """Get dataset record by ID."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                # Get main record
                cursor = conn.execute(
                    "SELECT * FROM datasets WHERE id = ?", (dataset_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return None

                # Parse main record
                record = DatasetRecord(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    dataset_type=DatasetType(row[3]),
                    status=DatasetStatus(row[4]),
                    created_date=datetime.fromisoformat(row[5]),
                    updated_date=datetime.fromisoformat(row[6]),
                    file_path=row[7],
                    file_size=row[8],
                    format=row[9],
                    source=row[10],
                    license=row[11],
                    tags=json.loads(row[12]) if row[12] else [],
                    metadata=json.loads(row[13]) if row[13] else {},
                )

                # Get metrics
                cursor = conn.execute(
                    "SELECT * FROM dataset_metrics WHERE dataset_id = ?", (dataset_id,)
                )
                metrics_row = cursor.fetchone()
                if metrics_row:
                    record.metrics = DatasetMetrics(
                        quality_score=metrics_row[1],
                        completeness_score=metrics_row[2],
                        consistency_score=metrics_row[3],
                        usage_count=metrics_row[4],
                        last_accessed=(
                            datetime.fromisoformat(metrics_row[5])
                            if metrics_row[5]
                            else None
                        ),
                        processing_time=metrics_row[6],
                        error_count=metrics_row[7],
                    )

                # Get versions
                cursor = conn.execute(
                    "SELECT * FROM dataset_versions WHERE dataset_id = ? ORDER BY created_date DESC",
                    (dataset_id,),
                )
                for version_row in cursor.fetchall():
                    version = DatasetVersion(
                        version=version_row[2],
                        created_date=datetime.fromisoformat(version_row[3]),
                        file_path=version_row[4],
                        file_size=version_row[5],
                        checksum=version_row[6],
                        changes=version_row[7],
                        is_current=bool(version_row[8]),
                    )
                    record.versions.append(version)

                # Get dependencies
                cursor = conn.execute(
                    "SELECT * FROM dataset_dependencies WHERE dataset_id = ?",
                    (dataset_id,),
                )
                for dep_row in cursor.fetchall():
                    dependency = DatasetDependency(
                        dataset_id=dep_row[2],
                        dependency_type=dep_row[3],
                        version_constraint=dep_row[4],
                        description=dep_row[5],
                    )
                    record.dependencies.append(dependency)

                return record

            finally:
                conn.close()

    def search_datasets(self, filters: dict[str, Any]) -> list[DatasetRecord]:
        """Search datasets with filters."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                query = "SELECT id FROM datasets WHERE 1=1"
                params = []

                if "dataset_type" in filters:
                    query += " AND dataset_type = ?"
                    params.append(filters["dataset_type"])

                if "status" in filters:
                    query += " AND status = ?"
                    params.append(filters["status"])

                if "tags" in filters:
                    for tag in filters["tags"]:
                        query += " AND tags LIKE ?"
                        params.append(f'%"{tag}"%')

                if "min_quality" in filters:
                    query += " AND id IN (SELECT dataset_id FROM dataset_metrics WHERE quality_score >= ?)"
                    params.append(filters["min_quality"])

                cursor = conn.execute(query, params)
                dataset_ids = [row[0] for row in cursor.fetchall()]

                return [self.get_dataset(dataset_id) for dataset_id in dataset_ids]

            finally:
                conn.close()


class ComprehensiveDatasetInventory:
    """Main dataset inventory system."""

    def __init__(self, inventory_dir: str = "./dataset_inventory"):
        self.inventory_dir = Path(inventory_dir)
        self.inventory_dir.mkdir(exist_ok=True)

        self.db = DatasetInventoryDB(str(self.inventory_dir / "inventory.db"))
        self.logger = get_logger(__name__)

        # Auto-discovery settings
        self.watch_directories: set[str] = set()
        self.auto_discovery_enabled = True

        logger.info("ComprehensiveDatasetInventory initialized")

    def register_dataset(
        self,
        file_path: str,
        name: str,
        description: str,
        dataset_type: DatasetType,
        source: str = "manual",
        license: str = "unknown",
        tags: list[str] | None = None,
    ) -> str:
        """Register a new dataset."""
        file_path = Path(file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Generate dataset ID
        dataset_id = self._generate_dataset_id(name, file_path)

        # Calculate file metrics
        file_size = file_path.stat().st_size
        checksum = self._calculate_checksum(file_path)

        # Create initial version
        initial_version = DatasetVersion(
            version="1.0.0",
            created_date=datetime.now(),
            file_path=str(file_path),
            file_size=file_size,
            checksum=checksum,
            changes="Initial version",
            is_current=True,
        )

        # Create dataset record
        record = DatasetRecord(
            id=dataset_id,
            name=name,
            description=description,
            dataset_type=dataset_type,
            status=DatasetStatus.AVAILABLE,
            created_date=datetime.now(),
            updated_date=datetime.now(),
            file_path=str(file_path),
            file_size=file_size,
            format=file_path.suffix.lower().lstrip("."),
            source=source,
            license=license,
            tags=tags or [],
            versions=[initial_version],
        )

        # Analyze dataset quality
        self._analyze_dataset_quality(record)

        # Store in database
        if self.db.insert_dataset(record):
            logger.info(f"Registered dataset: {dataset_id}")
            return dataset_id
        raise RuntimeError(f"Failed to register dataset: {dataset_id}")

    def update_dataset_version(
        self, dataset_id: str, file_path: str, version: str, changes: str
    ) -> bool:
        """Update dataset with new version."""
        record = self.db.get_dataset(dataset_id)
        if not record:
            logger.error(f"Dataset not found: {dataset_id}")
            return False

        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"New version file not found: {file_path}")
            return False

        # Mark current version as not current
        for v in record.versions:
            v.is_current = False

        # Create new version
        new_version = DatasetVersion(
            version=version,
            created_date=datetime.now(),
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            checksum=self._calculate_checksum(file_path),
            changes=changes,
            is_current=True,
        )

        record.versions.append(new_version)
        record.updated_date = datetime.now()
        record.file_path = str(file_path)
        record.file_size = new_version.file_size

        # Re-analyze quality
        self._analyze_dataset_quality(record)

        return self.db.insert_dataset(record)

    def add_dependency(
        self,
        dataset_id: str,
        dependency_id: str,
        dependency_type: str,
        version_constraint: str | None = None,
        description: str | None = None,
    ) -> bool:
        """Add dependency between datasets."""
        record = self.db.get_dataset(dataset_id)
        if not record:
            return False

        dependency = DatasetDependency(
            dataset_id=dependency_id,
            dependency_type=dependency_type,
            version_constraint=version_constraint,
            description=description,
        )

        record.dependencies.append(dependency)
        return self.db.insert_dataset(record)

    def get_dataset_info(self, dataset_id: str) -> DatasetRecord | None:
        """Get complete dataset information."""
        return self.db.get_dataset(dataset_id)

    def search_datasets(self, **filters) -> list[DatasetRecord]:
        """Search datasets with various filters."""
        return self.db.search_datasets(filters)

    def get_dataset_dependencies(
        self, dataset_id: str, recursive: bool = False
    ) -> list[str]:
        """Get dataset dependencies."""
        record = self.db.get_dataset(dataset_id)
        if not record:
            return []

        dependencies = [dep.dataset_id for dep in record.dependencies]

        if recursive:
            all_deps = set(dependencies)
            for dep_id in dependencies:
                sub_deps = self.get_dataset_dependencies(dep_id, recursive=True)
                all_deps.update(sub_deps)
            return list(all_deps)

        return dependencies

    def validate_dependencies(self, dataset_id: str) -> dict[str, bool]:
        """Validate all dependencies for a dataset."""
        dependencies = self.get_dataset_dependencies(dataset_id)
        validation_results = {}

        for dep_id in dependencies:
            dep_record = self.db.get_dataset(dep_id)
            validation_results[dep_id] = (
                dep_record is not None and dep_record.status == DatasetStatus.AVAILABLE
            )

        return validation_results

    def discover_datasets(
        self, directory: str, auto_register: bool = False
    ) -> list[dict[str, Any]]:
        """Discover datasets in a directory."""
        directory = Path(directory)
        discovered = []

        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return discovered

        # Common dataset file patterns
        dataset_patterns = ["*.json", "*.jsonl", "*.csv", "*.parquet", "*.pkl"]

        for pattern in dataset_patterns:
            for file_path in directory.rglob(pattern):
                if file_path.is_file() and file_path.stat().st_size > 1024:  # > 1KB
                    dataset_info = {
                        "file_path": str(file_path),
                        "name": file_path.stem,
                        "size": file_path.stat().st_size,
                        "format": file_path.suffix.lower().lstrip("."),
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime),
                    }

                    # Try to infer dataset type
                    dataset_info["inferred_type"] = self._infer_dataset_type(file_path)

                    discovered.append(dataset_info)

                    if auto_register:
                        try:
                            self.register_dataset(
                                str(file_path),
                                file_path.stem,
                                f"Auto-discovered dataset from {directory}",
                                dataset_info["inferred_type"],
                                source="auto_discovery",
                            )
                        except Exception as e:
                            logger.error(f"Failed to auto-register {file_path}: {e}")

        logger.info(f"Discovered {len(discovered)} datasets in {directory}")
        return discovered

    def generate_inventory_report(
        self, output_path: str = "inventory_report.json"
    ) -> str:
        """Generate comprehensive inventory report."""
        all_datasets = self.search_datasets()

        # Calculate statistics
        stats = {
            "total_datasets": len(all_datasets),
            "by_type": {},
            "by_status": {},
            "total_size": 0,
            "average_quality": 0,
            "last_updated": datetime.now().isoformat(),
        }

        for record in all_datasets:
            # Type statistics
            type_name = record.dataset_type.value
            stats["by_type"][type_name] = stats["by_type"].get(type_name, 0) + 1

            # Status statistics
            status_name = record.status.value
            stats["by_status"][status_name] = stats["by_status"].get(status_name, 0) + 1

            # Size and quality
            stats["total_size"] += record.file_size
            stats["average_quality"] += record.metrics.quality_score

        if all_datasets:
            stats["average_quality"] /= len(all_datasets)

        # Create detailed report
        report = {
            "statistics": stats,
            "datasets": [
                {
                    "id": record.id,
                    "name": record.name,
                    "type": record.dataset_type.value,
                    "status": record.status.value,
                    "size": record.file_size,
                    "quality_score": record.metrics.quality_score,
                    "usage_count": record.metrics.usage_count,
                    "last_accessed": (
                        record.metrics.last_accessed.isoformat()
                        if record.metrics.last_accessed
                        else None
                    ),
                    "versions": len(record.versions),
                    "dependencies": len(record.dependencies),
                }
                for record in all_datasets
            ],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Inventory report generated: {output_path}")
        return output_path

    def _generate_dataset_id(self, name: str, file_path: Path) -> str:
        """Generate unique dataset ID."""
        content = f"{name}_{file_path}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _infer_dataset_type(self, file_path: Path) -> DatasetType:
        """Infer dataset type from file name and content."""
        name_lower = file_path.name.lower()

        if any(
            keyword in name_lower
            for keyword in ["mental", "health", "therapy", "counseling"]
        ):
            return DatasetType.MENTAL_HEALTH
        if any(keyword in name_lower for keyword in ["voice", "audio", "speech"]):
            return DatasetType.VOICE_TRAINING
        if any(keyword in name_lower for keyword in ["psychology", "psych"]):
            return DatasetType.PSYCHOLOGY
        if any(keyword in name_lower for keyword in ["reasoning", "logic", "cot"]):
            return DatasetType.REASONING
        if any(keyword in name_lower for keyword in ["personality", "trait"]):
            return DatasetType.PERSONALITY
        if any(keyword in name_lower for keyword in ["quality", "grade"]):
            return DatasetType.QUALITY
        return DatasetType.MIXED

    def _analyze_dataset_quality(self, record: DatasetRecord) -> None:
        """Analyze dataset quality metrics."""
        try:
            Path(record.file_path)

            # Basic quality metrics
            quality_score = 0.5  # Base score
            completeness_score = 0.5
            consistency_score = 0.5

            # File size factor
            if record.file_size > 1024 * 1024:  # > 1MB
                quality_score += 0.2

            # Format factor
            if record.format in ["json", "jsonl", "parquet"]:
                quality_score += 0.2

            # Name quality
            if len(record.name) > 5 and "_" in record.name:
                quality_score += 0.1

            # Update metrics
            record.metrics.quality_score = min(1.0, quality_score)
            record.metrics.completeness_score = completeness_score
            record.metrics.consistency_score = consistency_score

        except Exception as e:
            logger.error(f"Quality analysis failed for {record.id}: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize inventory
    inventory = ComprehensiveDatasetInventory()

    # Create test dataset file
    test_file = Path("test_mental_health_dataset.json")
    test_data = [
        {"id": 1, "content": "Test mental health conversation"},
        {"id": 2, "content": "Another test conversation"},
    ]

    with open(test_file, "w") as f:
        json.dump(test_data, f)

    try:
        # Register dataset
        dataset_id = inventory.register_dataset(
            str(test_file),
            "Test Mental Health Dataset",
            "A test dataset for mental health conversations",
            DatasetType.MENTAL_HEALTH,
            source="test",
            license="MIT",
            tags=["test", "mental_health"],
        )


        # Get dataset info
        info = inventory.get_dataset_info(dataset_id)

        # Search datasets
        results = inventory.search_datasets(dataset_type="mental_health")

        # Generate report
        report_path = inventory.generate_inventory_report()

    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()
        if Path("inventory_report.json").exists():
            Path("inventory_report.json").unlink()
