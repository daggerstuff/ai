#!/usr/bin/env python3
"""
Persistence Layer for AI Pipelines (PIX-4).
Handles metadata storage for datasets, processing runs, and evaluation results.
"""

import hashlib
import json
import logging
import os
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("persistence")


class DatasetPersistence:
    """
    Handles persistence of dataset metadata and pipeline state.
    Supports MongoDB for flexible schema and local fallback if DB is unavailable.
    """

    def __init__(self, db_uri: str | None = None):
        self.db_uri = db_uri or os.environ.get("MONGODB_URI")
        self.client = None
        self.db = None

        if self.db_uri:
            try:
                self.client = MongoClient(self.db_uri, serverSelectionTimeoutMS=5000)
                self.client.admin.command("ping")
                self.db = self.client["pixelated_ai"]
                logger.info("Connected to MongoDB successfully.")
            except (ConnectionFailure, Exception) as e:
                logger.error(f"Failed to connect to MongoDB: {e}. Falling back to local file state.")
                self.client = None
                self.db = None
        else:
            logger.warning("No MONGODB_URI provided. Using local file persistence.")

    def log_dataset_version(self, name: str, version: str, metadata: dict[str, Any]):
        """Logs a new dataset version metadata."""
        record = {
            "name": name,
            "version": version,
            "created_at": datetime.now(UTC).isoformat(),
            **metadata,
        }

        if self.db is not None:
            self.db.datasets.insert_one(record)

        # Always save local shadow copy for zero-trust auditability
        local_path = Path(f"ai/data/versions/{name}_{version}.json")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "w") as f:
            json.dump(record, f, indent=2)

        logger.info(f"Dataset {name} (v{version}) logged successfully.")

    def get_dataset_metadata(self, name: str, version: str) -> dict[str, Any] | None:
        """Retrieves metadata for a specific dataset version."""
        if self.db is not None:
            return self.db.datasets.find_one({"name": name, "version": version})

        # Check local fallback
        local_path = Path(f"ai/data/versions/{name}_{version}.json")
        if local_path.exists():
            with open(local_path) as f:
                return json.load(f)
        return None

    def update_pipeline_state(self, pipeline_id: str, state: str, details: dict[str, Any]):
        """Updates the state of a running pipeline."""
        record = {
            "pipeline_id": pipeline_id,
            "state": state,
            "updated_at": datetime.now(UTC).isoformat(),
            "details": details,
        }

        if self.db is not None:
            self.db.pipeline_states.update_one({"pipeline_id": pipeline_id}, {"$set": record}, upsert=True)
        logger.info(f"Pipeline {pipeline_id} state updated to {state}.")

    def store_training_records(
        self,
        dataset: str,
        records: Iterable[Mapping[str, Any]],
        *,
        version: str | None = None,
        collection: str = "training_records",
        local_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Persist training records with required provenance.

        MongoDB is used when configured; a local JSONL audit copy is always
        written so provenance can be inspected even in local/offline runs.
        """
        persisted_at = datetime.now(UTC).isoformat()
        dataset_version = version or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        local_base = local_dir or Path(__file__).resolve().parents[1] / "data" / "versions"
        local_path = local_base / f"{dataset}_{dataset_version}_training_records.jsonl"
        local_path.parent.mkdir(parents=True, exist_ok=True)

        seen = 0
        stored = 0
        with open(local_path, "w", encoding="utf-8") as f:
            for record in records:
                if not isinstance(record.get("provenance"), dict):
                    raise ValueError("training records must include a provenance object")

                record_hash = self._record_hash(record)
                document = {
                    "dataset": dataset,
                    "version": dataset_version,
                    "record_hash": record_hash,
                    "record": dict(record),
                    "provenance": dict(record["provenance"]),
                    "persisted_at": persisted_at,
                }

                if self.db is not None:
                    self.db[collection].update_one(
                        {"dataset": dataset, "record_hash": record_hash},
                        {"$set": document},
                        upsert=True,
                    )

                f.write(json.dumps(document, sort_keys=True) + "\n")
                seen += 1
                stored += 1

        logger.info("Stored %d training records for dataset %s.", stored, dataset)
        return {
            "dataset": dataset,
            "version": dataset_version,
            "records_seen": seen,
            "records_stored": stored,
            "mongo_enabled": self.db is not None,
            "collection": collection,
            "local_path": str(local_path),
        }

    def query_training_records_by_provenance(
        self,
        *,
        source_type: str | None = None,
        license_id: str | None = None,
        collection: str = "training_records",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query MongoDB training records by provenance fields."""
        if self.db is None:
            return []

        query: dict[str, Any] = {}
        if source_type is not None:
            query["provenance.source_type"] = source_type
        if license_id is not None:
            query["provenance.license"] = license_id

        cursor = self.db[collection].find(query).limit(limit)
        return list(cursor)

    @staticmethod
    def _record_hash(record: Mapping[str, Any]) -> str:
        canonical = json.dumps(record, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    # Self-test
    persist = DatasetPersistence()
    persist.log_dataset_version("test_audit", "1.0.0", {"records": 1000, "source": "audit"})
