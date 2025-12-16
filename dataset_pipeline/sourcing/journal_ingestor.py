import json
import logging
import time
from pathlib import Path

import requests

from ai.dataset_pipeline.storage_config import get_dataset_pipeline_output_root

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JournalResearchIngestor:
    def __init__(self, output_dir: str | Path | None = None):
        if output_dir is None:
            output_dir = get_dataset_pipeline_output_root() / "data" / "tier5_research"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "journal_abstracts.jsonl"

        # Base URLs
        self.doaj_url = "https://doaj.org/api/search/articles"
        self.clinical_trials_url = "https://clinicaltrials.gov/api/v2/studies"

    def ingest_doaj(self, query: str = "psychotherapy", limit: int = 20) -> int:
        """
        Fetch articles from Directory of Open Access Journals.
        Ref: https://doaj.org/api/v4/docs
        """
        logger.info(f"Querying DOAJ for '{query}'...")

        # Construct the query URL (simplified for v8 API structure or search route)
        # Note: DOAJ API structure can be complex; using basic search endpoint
        url = f"{self.doaj_url}/{query}?pageSize={limit}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            logger.info(f"Found {len(results)} DOAJ results.")

            processed_count = 0
            with open(self.output_file, "a") as f:
                for item in results:
                    bibjson = item.get("bibjson", {})
                    abstract = bibjson.get("abstract")
                    title = bibjson.get("title")

                    if abstract and title:
                        record = {
                            "source": "DOAJ",
                            "id": item.get("id"),
                            "title": title,
                            "abstract": abstract,
                            "query": query,
                            "tier": "tier5_research",
                        }
                        f.write(json.dumps(record) + "\n")
                        processed_count += 1

            logger.info(f"Ingested {processed_count} DOAJ abstracts.")
            return processed_count

        except Exception as e:
            logger.error(f"DOAJ Ingestion failed: {e}")
            return 0

    def ingest_clinical_trials(self, condition: str = "Depression", limit: int = 10) -> int:
        """
        Fetch studies from ClinicalTrials.gov API v2.
        """
        logger.info(f"Querying ClinicalTrials.gov for '{condition}'...")

        params = {"query.cond": condition, "pageSize": limit, "format": "json"}

        try:
            response = requests.get(self.clinical_trials_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            studies = data.get("studies", [])
            logger.info(f"Found {len(studies)} ClinicalTrials results.")

            processed_count = 0
            with open(self.output_file, "a") as f:
                for study in studies:
                    protocol = study.get("protocolSection", {})
                    desc = protocol.get("descriptionModule", {})
                    summary = desc.get("briefSummary", "")
                    title = protocol.get("identificationModule", {}).get("officialTitle")

                    if summary:
                        record = {
                            "source": "ClinicalTrials.gov",
                            "id": protocol.get("identificationModule", {}).get("nctId"),
                            "title": title,
                            "abstract": summary,
                            "query": condition,
                            "tier": "tier5_research",
                        }
                        f.write(json.dumps(record) + "\n")
                        processed_count += 1

            logger.info(f"Ingested {processed_count} Clinical trials.")
            return processed_count

        except Exception as e:
            logger.error(f"ClinicalTrials Ingestion failed: {e}")
            return 0

    def run_all(self, dry_run: bool = False):
        if dry_run:
            logger.info("[DRY RUN] Would ingest from DOAJ and ClinicalTrials.gov")
            return

        total = 0
        total += self.ingest_doaj(query="CBT psychotherapy", limit=10)
        total += self.ingest_doaj(query="DBT mental health", limit=10)
        time.sleep(1)  # Respect rate limits
        total += self.ingest_clinical_trials(condition="PTSD", limit=10)
        total += self.ingest_clinical_trials(condition="Anxiety", limit=10)

        logger.info(f"Total Research Items Ingested: {total}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ingestor = JournalResearchIngestor()
    ingestor.run_all(dry_run=args.dry_run)
