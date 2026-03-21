"""
Journal Research Sourcing.

Wraps existing journal_ingestor.py to provide active sourcing of
research abstracts from DOAJ and ClinicalTrials.gov.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class JournalSource:
    """
    Source research abstracts from academic databases.

    Uses existing journal_ingestor infrastructure to ingest:
    - DOAJ (Directory of Open Access Journals)
    - ClinicalTrials.gov

    Usage:
        source = JournalSource()
        for ref in source.fill_gap(1000):
            print(f"Generated: {ref}")
    """

    # Research queries organized by therapeutic focus
    THERAPEUTIC_QUERIES = {
        'cbt': ['CBT cognitive behavioral therapy', 'cognitive behavioral therapy randomized trial'],
        'trauma': ['trauma therapy EMDR', 'PTSD treatment outcomes', 'complex trauma CPTSD'],
        'depression': ['depression therapy outcomes', 'behavioral activation depression'],
        'anxiety': ['anxiety disorder therapy', 'GAD treatment efficacy'],
        'dbt': ['DBT dialectical behavior therapy', 'borderline personality DBT'],
        'grief': ['grief counseling therapy', 'complicated grief treatment'],
        'addiction': ['substance abuse therapy', 'addiction treatment outcomes'],
    }

    def __init__(self, output_dir: Optional[str | Path] = None):
        """
        Initialize journal source.

        Args:
            output_dir: Directory for ingested abstracts
        """
        self.output_dir = Path(output_dir) if output_dir else Path("ai/data/acquired_datasets/journal_research")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "journal_abstracts.jsonl"

    def ingest_doaj(
        self,
        query: str = "psychotherapy",
        limit: int = 100
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetch articles from Directory of Open Access Journals.

        Args:
            query: Search query
            limit: Max results

        Yields:
            Abstract records
        """
        # Use existing ingestor if available
        try:
            from ai.pipelines.orchestrator.sourcing.journal_ingestor import JournalResearchIngestor
            ingestor = JournalResearchIngestor(output_dir=self.output_dir)

            # Call DOAJ ingestion
            count = ingestor.ingest_doaj(query=query, limit=limit)
            logger.info(f"Ingested {count} DOAJ articles for '{query}'")

            # Read and yield results
            if self.output_file.exists():
                with open(self.output_file) as f:
                    for line in f:
                        yield json.loads(line)

        except ImportError:
            logger.warning("journal_ingestor not available, using direct API")
            self._ingest_doaj_direct(query, limit)

    def _ingest_doaj_direct(self, query: str, limit: int):
        """Direct DOAJ API ingestion (fallback)."""
        try:
            import requests
            from urllib.parse import quote

            # DOAJ API v2
            url = f"https://doaj.org/api/v3/search/journals/{quote(query)}"
            params = {"pageSize": limit}

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            for item in data.get('results', []):
                bibjson = item.get('bibjson', {})
                abstract = bibjson.get('abstract')
                title = bibjson.get('title')

                if abstract and title:
                    yield {
                        'source': 'DOAJ',
                        'id': item.get('id'),
                        'title': title,
                        'abstract': abstract,
                        'query': query,
                        'tier': 'tier5_research',
                        'stage': 'stage2_therapeutic_expertise',
                        'quality_profile': 'research',
                    }

        except Exception as e:
            logger.error(f"DOAJ ingestion failed: {e}")

    def ingest_clinical_trials(
        self,
        condition: str = "Depression",
        limit: int = 50
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetch studies from ClinicalTrials.gov.

        Args:
            condition: Medical condition to search
            limit: Max results

        Yields:
            Study records
        """
        try:
            from ai.pipelines.orchestrator.sourcing.journal_ingestor import JournalResearchIngestor
            ingestor = JournalResearchIngestor(output_dir=self.output_dir)

            count = ingestor.ingest_clinical_trials(condition=condition, limit=limit)
            logger.info(f"Ingested {count} clinical trials for '{condition}'")

        except ImportError:
            logger.warning("journal_ingestor not available, using direct API")
            self._ingest_clinical_trials_direct(condition, limit)

    def _ingest_clinical_trials_direct(self, condition: str, limit: int):
        """Direct ClinicalTrials.gov API (fallback)."""
        try:
            import requests

            url = "https://clinicaltrials.gov/api/v2/studies"
            params = {
                "query.cond": condition,
                "pageSize": limit,
                "format": "json"
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            for study in data.get('studies', []):
                protocol = study.get('protocolSection', {})
                desc = protocol.get('descriptionModule', {})
                summary = desc.get('briefSummary', '')
                title = protocol.get('identificationModule', {}).get('officialTitle')

                if summary:
                    yield {
                        'source': 'ClinicalTrials.gov',
                        'id': protocol.get('identificationModule', {}).get('nctId'),
                        'title': title or f"{condition} study",
                        'abstract': summary,
                        'query': condition,
                        'tier': 'tier5_research',
                        'stage': 'stage2_therapeutic_expertise',
                        'quality_profile': 'research',
                    }

        except Exception as e:
            logger.error(f"ClinicalTrials ingestion failed: {e}")

    def fill_gap(self, gap: int, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Fill gap by ingesting research abstracts.

        Args:
            gap: Number of samples needed

        Yields:
            Abstract records
        """
        logger.info(f"JournalSource filling gap of {gap} samples")

        # Ingest from multiple queries
        count = 0
        queries = kwargs.get('queries', list(self.THERAPEUTIC_QUERIES.values())[:3])

        for query_list in [self.THERAPEUTIC_QUERIES[k] for k in list(self.THERAPEUTIC_QUERIES.keys())[:3]]:
            if count >= gap:
                break

            for query in query_list:
                if count >= gap:
                    break

                logger.info(f"Ingesting research for: {query}")

                # DOAJ
                for record in self.ingest_doaj(query=query, limit=50):
                    yield record
                    count += 1
                    if count >= gap:
                        break

                # ClinicalTrials
                if count < gap:
                    for record in self.ingest_clinical_trials(condition=query.split()[0], limit=25):
                        yield record
                        count += 1
                        if count >= gap:
                            break

        logger.info(f"Journal sourcing complete: {count} abstracts")

    def discover(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """Discover available research sources."""
        # Return metadata about available queries
        for category, queries in self.THERAPEUTIC_QUERIES.items():
            yield {
                'category': category,
                'queries': queries,
                'sources': ['DOAJ', 'ClinicalTrials.gov'],
                'stage': 'stage2_therapeutic_expertise',
                'quality_profile': 'research',
            }


if __name__ == "__main__":
    # Test ingestion
    source = JournalSource()

    print("Testing journal research sourcing...")
    count = 0
    for record in source.fill_gap(10):
        print(f"  {record['source']}: {record.get('title', 'N/A')[:60]}...")
        count += 1

    print(f"\nIngested {count} research abstracts")
