"""
Dataset Source Managers.

Provides abstractions for different dataset sources:
- JournalSource: DOAJ, ClinicalTrials.gov (wraps existing journal_ingestor)
- EdgeCaseSource: Edge case generation pipeline
- VoiceSource: Voice pipeline integration
- HuggingFaceSource: Hugging Face dataset discovery
- LocalSource: Local filesystem scanning
- GDriveSource: Google Drive via rclone
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .lazy_registry import DatasetRef

logger = logging.getLogger(__name__)


class DatasetSource(ABC):
    """Base class for dataset sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Source name."""
        pass

    @abstractmethod
    def discover(self, query: Optional[str] = None) -> Iterator[DatasetRef]:
        """
        Discover datasets from this source.

        Args:
            query: Optional search query

        Yields:
            DatasetRef objects
        """
        pass

    @abstractmethod
    def fill_gap(self, gap: int, **kwargs) -> Iterator[DatasetRef]:
        """
        Generate/discover datasets to fill a gap.

        Args:
            gap: Number of samples needed

        Yields:
            DatasetRef objects
        """
        pass


class DatasetSourceManager:
    """
    Manage multiple dataset sources.

    Usage:
        manager = DatasetSourceManager()
        manager.register_source(JournalSource())
        manager.register_source(EdgeCaseSource())

        for ref in manager.discover(stage='stage2_therapeutic_expertise'):
            print(ref.name)
    """

    def __init__(self):
        self._sources: Dict[str, DatasetSource] = {}

    def register_source(self, source: DatasetSource):
        """Register a dataset source."""
        self._sources[source.name] = source
        logger.info(f"Registered dataset source: {source.name}")

    def get_source(self, name: str) -> Optional[DatasetSource]:
        """Get source by name."""
        return self._sources.get(name)

    def discover(
        self,
        stage: Optional[str] = None,
        quality_profile: Optional[str] = None,
    ) -> Iterator[DatasetRef]:
        """
        Discover datasets across all sources.

        Args:
            stage: Filter by stage
            quality_profile: Filter by quality profile

        Yields:
            DatasetRef objects
        """
        for source in self._sources.values():
            for ref in source.discover():
                if stage and ref.stage != stage:
                    continue
                if quality_profile and ref.quality_profile != quality_profile:
                    yield ref
                elif not stage and not quality_profile:
                    yield ref

    def fill_gap(
        self,
        source_name: str,
        gap: int,
        **kwargs
    ) -> Iterator[DatasetRef]:
        """
        Fill a gap using specified source.

        Args:
            source_name: Name of source to use
            gap: Number of samples needed

        Yields:
            DatasetRef objects
        """
        source = self.get_source(source_name)
        if not source:
            raise ValueError(f"Unknown source: {source_name}")

        for ref in source.fill_gap(gap, **kwargs):
            yield ref


class JournalSource(DatasetSource):
    """
    Journal research source - wraps existing journal_ingestor.

    Discovers research abstracts from:
    - DOAJ (Directory of Open Access Journals)
    - ClinicalTrials.gov
    """

    @property
    def name(self) -> str:
        return "journal_research"

    def discover(self, query: Optional[str] = None) -> Iterator[DatasetRef]:
        """
        Discover journal datasets.

        Args:
            query: Search query (e.g., "CBT psychotherapy")

        Yields:
            DatasetRef objects
        """
        from ai.pipelines.orchestrator.sourcing.journal_ingestor import (
            JournalResearchIngestor,
        )

        ingestor = JournalResearchIngestor()

        # DOAJ queries
        doaj_queries = ["CBT psychotherapy", "DBT mental health", "PTSD therapy"]
        if query:
            doaj_queries.insert(0, query)

        for q in doaj_queries:
            logger.info(f"Discovering journal datasets for: {q}")
            # Note: Would need to extend ingestor to return refs
            # For now, this is a placeholder
            yield DatasetRef(
                name=f"doaj_{q.replace(' ', '_')}",
                path=f"drive:backups/S3-Complete/gdrive/processed/research/doaj_{q.replace(' ', '_')}.jsonl",
                stage="stage2_therapeutic_expertise",
                quality_profile="research",
                dataset_type="journal_abstract",
                status="pending_sync",
            )

        # ClinicalTrials queries
        ct_queries = ["depression", "anxiety", "PTSD"]
        for q in ct_queries:
            yield DatasetRef(
                name=f"clinicaltrials_{q}",
                path=f"drive:backups/S3-Complete/gdrive/processed/research/clinicaltrials_{q}.jsonl",
                stage="stage2_therapeutic_expertise",
                quality_profile="research",
                dataset_type="clinical_trial",
                status="pending_sync",
            )

    def fill_gap(self, gap: int, **kwargs) -> Iterator[DatasetRef]:
        """
        Fill gap by running journal ingestion.

        Args:
            gap: Number of samples needed

        Yields:
            DatasetRef objects
        """
        logger.info(f"JournalSource filling gap of {gap} samples")

        # Run ingestion
        from ai.pipelines.orchestrator.sourcing.journal_ingestor import (
            JournalResearchIngestor,
        )

        ingestor = JournalResearchIngestor()
        count = ingestor.run_all(dry_run=False)

        # Return refs for ingested data
        if count > 0:
            yield DatasetRef(
                name=f"journal_ingest_{count}",
                path="drive:backups/S3-Complete/gdrive/processed/research/journal_abstracts.jsonl",
                stage="stage2_therapeutic_expertise",
                quality_profile="research",
                dataset_type="journal_abstract",
                size_mb=count * 0.001,  # Estimate
                status="active",
            )


class EdgeCaseSource(DatasetSource):
    """
    Edge case generation source.

    Generates edge case samples for Stage 3 stress testing.
    """

    @property
    def name(self) -> str:
        return "edge_case_generator"

    def discover(self, query: Optional[str] = None) -> Iterator[DatasetRef]:
        """Discover existing edge case datasets."""
        # Check for existing edge case files
        edge_dirs = [
            Path("ai/training/ready_packages/datasets/synthetic"),
            Path("ai/pipelines/edge_case_pipeline_standalone/output"),
        ]

        for edge_dir in edge_dirs:
            if edge_dir.exists():
                for f in edge_dir.glob("*.jsonl"):
                    yield DatasetRef(
                        name=f.stem,
                        path=f"file://{f}",
                        stage="stage3_edge_stress_test",
                        quality_profile="edge_crisis",
                        dataset_type="edge_case",
                        status="active",
                    )

    def fill_gap(self, gap: int, **kwargs) -> Iterator[DatasetRef]:
        """
        Generate edge case samples to fill gap.

        Args:
            gap: Number of samples needed

        Yields:
            DatasetRef objects
        """
        logger.info(f"EdgeCaseSource generating {gap} samples")

        # Categories to generate
        categories = kwargs.get('categories', [
            'suicidality',
            'homicidal_ideation',
            'psychotic_episodes',
            'severe_dissociation',
        ])

        # Would call edge case generator here
        # For now, return placeholder
        yield DatasetRef(
            name=f"edge_case_generated_{gap}",
            path="drive:backups/S3-Complete/gdrive/processed/edge_cases/edge_cases_training_format.jsonl",
            stage="stage3_edge_stress_test",
            quality_profile="edge_crisis",
            dataset_type="synthetic_edge",
            size_mb=gap * 0.001,
            status="generated",
            metadata={'categories': categories},
        )


class VoiceSource(DatasetSource):
    """
    Voice pipeline source.

    Processes transcripts and generates voice-signatured samples.
    """

    @property
    def name(self) -> str:
        return "voice_pipeline"

    def discover(self, query: Optional[str] = None) -> Iterator[DatasetRef]:
        """Discover existing voice datasets."""
        voice_dirs = [
            Path("ai/data/tim_fletcher_voice/exports"),
            Path("ai/data/tim_fletcher_transcripts"),
        ]

        for voice_dir in voice_dirs:
            if voice_dir.exists():
                for f in voice_dir.glob("*.jsonl"):
                    yield DatasetRef(
                        name=f.stem,
                        path=f"file://{f}",
                        stage="stage4_voice_persona",
                        quality_profile="voice",
                        dataset_type="voice_transcript",
                        status="active",
                    )

    def fill_gap(self, gap: int, **kwargs) -> Iterator[DatasetRef]:
        """
        Generate voice samples to fill gap.

        Args:
            gap: Number of samples needed

        Yields:
            DatasetRef objects
        """
        logger.info(f"VoiceSource generating {gap} samples")

        # Would run voice pipeline here
        yield DatasetRef(
            name=f"voice_generated_{gap}",
            path="drive:backups/S3-Complete/voice/exports/voice_signatured_conversations.jsonl",
            stage="stage4_voice_persona",
            quality_profile="voice",
            dataset_type="synthetic_voice",
            size_mb=gap * 0.001,
            status="generated",
            metadata={'persona_id': 'dual:therapist_mentor'},
        )
