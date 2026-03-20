"""
Lazy-loading Dataset Registry.

Instead of loading the entire registry JSON into memory, this module provides:
- Index-based access (load only what you need)
- Streaming iteration for bulk operations
- Query by stage, quality profile, or status
- Cache individual entries, not full registry
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class DatasetRef:
    """Reference to a single dataset with metadata."""

    name: str
    path: str
    stage: str
    quality_profile: str
    dataset_type: str = ""
    size_mb: float = 0.0
    status: str = "active"
    fallback_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "DatasetRef":
        """Create DatasetRef from registry dictionary entry."""
        return cls(
            name=name,
            path=data.get("path", ""),
            stage=data.get("stage", ""),
            quality_profile=data.get("quality_profile", ""),
            dataset_type=data.get("type", ""),
            size_mb=data.get("size_mb", 0.0),
            status=data.get("status", "active"),
            fallback_paths=data.get("fallback_paths", {}),
            metadata={k: v for k, v in data.items() if k not in [
                'path', 'stage', 'quality_profile', 'type', 'size_mb',
                'status', 'fallback_paths'
            ]}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": self.path,
            "stage": self.stage,
            "quality_profile": self.quality_profile,
            "type": self.dataset_type,
            "size_mb": self.size_mb,
            "status": self.status,
            "fallback_paths": self.fallback_paths,
            **self.metadata
        }


class DatasetRegistry:
    """
    Lazy-loading dataset registry.

    Usage:
        registry = DatasetRegistry('ai/data/dataset_registry.json')

        # Query by stage (lazy - doesn't load full registry)
        stage3_datasets = registry.by_stage('stage3_edge_stress_test')

        # Query by quality profile
        edge_datasets = registry.by_quality_profile('edge_crisis')

        # Iterate all (streaming)
        for ref in registry.iter_refs():
            process(ref)
    """

    def __init__(self, registry_path: str | Path):
        """
        Initialize registry with lazy loading.

        Args:
            registry_path: Path to dataset_registry.json
        """
        self.registry_path = Path(registry_path)
        self._index: Optional[Dict[str, Dict[str, Any]]] = None
        self._cache: Dict[str, DatasetRef] = {}
        self._index_loaded = False

    @property
    def index(self) -> Dict[str, Dict[str, Any]]:
        """Lazy-load index on first access."""
        if not self._index_loaded:
            self._load_index()
        return self._index

    def _load_index(self):
        """Load only the index (dataset names and categories), not full data."""
        if self._index_loaded:
            return

        with open(self.registry_path, 'r') as f:
            data = json.load(f)

        # Build index: category -> dataset_name -> metadata stub
        self._index = {}
        datasets = data.get('datasets', {})

        for category, entries in datasets.items():
            if isinstance(entries, dict):
                self._index[category] = {}
                for name, metadata in entries.items():
                    # Store minimal metadata for index
                    self._index[category][name] = {
                        'stage': metadata.get('stage', ''),
                        'quality_profile': metadata.get('quality_profile', ''),
                        'status': metadata.get('status', 'active'),
                        'type': metadata.get('type', ''),
                        'size_mb': metadata.get('size_mb', 0),
                    }

        self._index_loaded = True
        logger.debug(f"Loaded registry index: {len(datasets)} categories")

    def _get_category_for_stage(self, stage: str) -> Optional[str]:
        """Map stage to registry category."""
        stage_to_category = {
            'stage1_foundation': 'professional_therapeutic',
            'stage2_therapeutic_expertise': 'cot_reasoning',
            'stage3_edge_stress_test': 'edge_case_sources',
            'stage4_voice_persona': 'voice_persona',
        }
        return stage_to_category.get(stage)

    def by_stage(self, stage: str) -> Iterator[DatasetRef]:
        """
        Get all datasets for a stage (lazy loading).

        Args:
            stage: Stage name (e.g., 'stage3_edge_stress_test')

        Yields:
            DatasetRef objects
        """
        # Load full category data only when needed
        with open(self.registry_path, 'r') as f:
            data = json.load(f)

        datasets = data.get('datasets', {})

        # Search ALL categories for matching stage
        for category, entries in datasets.items():
            if isinstance(entries, dict):
                for name, metadata in entries.items():
                    if metadata.get('stage') == stage:
                        yield DatasetRef.from_dict(name, metadata)

    def by_quality_profile(self, profile: str) -> Iterator[DatasetRef]:
        """
        Get all datasets with given quality profile (lazy loading).

        Args:
            profile: Quality profile name

        Yields:
            DatasetRef objects
        """
        with open(self.registry_path, 'r') as f:
            data = json.load(f)

        datasets = data.get('datasets', {})

        for category, entries in datasets.items():
            if isinstance(entries, dict):
                for name, metadata in entries.items():
                    if metadata.get('quality_profile') == profile:
                        yield DatasetRef.from_dict(name, metadata)

    def by_status(self, status: str) -> Iterator[DatasetRef]:
        """
        Get all datasets with given status (lazy loading).

        Args:
            status: Status (e.g., 'active', 'pending_sync')

        Yields:
            DatasetRef objects
        """
        with open(self.registry_path, 'r') as f:
            data = json.load(f)

        datasets = data.get('datasets', {})

        for category, entries in datasets.items():
            if isinstance(entries, dict):
                for name, metadata in entries.items():
                    if metadata.get('status') == status:
                        yield DatasetRef.from_dict(name, metadata)

    def get(self, name: str, category: str) -> Optional[DatasetRef]:
        """
        Get single dataset by name and category.

        Args:
            name: Dataset name
            category: Category name

        Returns:
            DatasetRef or None
        """
        cache_key = f"{category}/{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        with open(self.registry_path, 'r') as f:
            data = json.load(f)

        datasets = data.get('datasets', {})
        if category in datasets and name in datasets[category]:
            ref = DatasetRef.from_dict(name, datasets[category][name])
            self._cache[cache_key] = ref
            return ref

        return None

    def iter_refs(self) -> Iterator[DatasetRef]:
        """
        Iterate all datasets (streaming - doesn't cache).

        Yields:
            DatasetRef objects
        """
        with open(self.registry_path, 'r') as f:
            data = json.load(f)

        datasets = data.get('datasets', {})

        for category, entries in datasets.items():
            if isinstance(entries, dict):
                for name, metadata in entries.items():
                    yield DatasetRef.from_dict(name, metadata)

    def get_counts_by_stage(self) -> Dict[str, Dict[str, Any]]:
        """
        Get sample counts per stage.

        Returns:
            Dict with stage -> {target, available, gap}
        """
        # Load MTGC targets from registry metadata or plan
        targets = {
            'stage1_foundation': 63000,
            'stage2_therapeutic_expertise': 40000,
            'stage3_edge_stress_test': 32000,
            'stage4_voice_persona': 1200,
        }

        counts = {}
        for stage, target in targets.items():
            count = sum(1 for _ in self.by_stage(stage))
            counts[stage] = {
                'target': target,
                'current': count,
                'gap': max(0, target - count),
                'percent': (count / target * 100) if target > 0 else 0
            }

        return counts
