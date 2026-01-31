"""
Multi-Source Dataset Ingestor with Direct S3 Streaming.

Supports:
- HuggingFace Datasets
- NGC (NVIDIA GPU Cloud) Catalog
- Zenodo
- GitHub Archives
- Roleplay/Persona Datasets
- Model Selection/Evaluation Datasets
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from dotenv import load_dotenv

if TYPE_CHECKING:
    import boto3
else:
    try:
        import boto3
    except ImportError:
        boto3 = None  # type: ignore[assignment]

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # type: ignore[assignment]

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

try:
    import kaggle
except ImportError:
    kaggle = None  # type: ignore[assignment]

try:
    import datadotworld as dw
except ImportError:
    dw = None  # type: ignore[assignment]

try:
    import openml
except ImportError:
    openml = None  # type: ignore[assignment]

try:
    from ai.pipelines.orchestrator.sourcing.local_consolidated_ingestor import (
        LocalConsolidatedIngestor,
    )
except ImportError:
    LocalConsolidatedIngestor = None  # type: ignore[assignment, misc]

try:
    from ai.pipelines.orchestrator.sourcing.ngc_ingestor import NGCIngestor
except ImportError:
    NGCIngestor = None  # type: ignore[assignment, misc]

load_dotenv("ai/.env")

logger = logging.getLogger(__name__)


def get_s3_client() -> Any | None:
    """Creates an S3 client configured for OVH Object Storage."""
    if boto3 is None:
        logger.error("boto3 not installed. Install with: uv pip install boto3")
        return None

    endpoint = os.environ.get("OVH_S3_ENDPOINT")
    access_key = os.environ.get("OVH_S3_ACCESS_KEY")
    secret_key = os.environ.get("OVH_S3_SECRET_KEY")
    region = os.environ.get("OVH_S3_REGION", "us-east-va")

    if not all([endpoint, access_key, secret_key]):
        logger.error("Missing OVH S3 credentials in environment. Check ai/.env")
        return None

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )


def stream_to_s3(data: Iterator[dict], bucket: str, key: str, _batch_size: int = 100) -> int:
    """Streams data directly to S3 as JSONL without local storage."""
    s3 = get_s3_client()
    if not s3:
        return 0

    buffer = BytesIO()
    count = 0

    for record in data:
        line = json.dumps(record) + "\n"
        buffer.write(line.encode("utf-8"))
        count += 1

    buffer.seek(0)
    s3.upload_fileobj(buffer, bucket, key)
    logger.info(f"Streamed {count} records to s3://{bucket}/{key}")
    return count


class HuggingFaceIngestor:
    """Ingests datasets from HuggingFace, streaming directly to S3."""

    DATASETS: ClassVar[dict] = {
        # Foundation
        "Amod/mental_health_counseling_conversations": {"target": "tier2_professional"},
        "heliosbrahma/mental_health_chatbot_dataset": {"target": "tier1_foundation"},
        # Addiction
        "fadodr/mental_health_therapy": {"target": "tier2_professional"},
        # PTSD/Trauma
        "yenopoya/thousand-voices-trauma": {"target": "tier2_professional"},
        # Personality Disorders
        "Kanakmi/mental-disorders": {"target": "tier2_professional"},
        # Chain-of-Thought Reasoning (NEW - from spec, 30K+ entries)
        "Cartinoe5930/CoT-Clinical-Reasoning": {"target": "tier3_cot_reasoning"},
        "Cartinoe5930/CoT-Emotional-Reasoning": {"target": "tier3_cot_reasoning"},
        "WizardLM/WizardLM-70b-V1.0-evol-cot": {"target": "tier3_cot_reasoning"},
        "jondurbin/bagel-dpo-34b-v0.2": {"target": "tier3_cot_reasoning"},
        # Roleplay & Persona
        "google/Synthetic-Persona-Chat": {"target": "tier4_voice_persona"},
        "nazlicanto/persona-based-chat": {"target": "tier4_voice_persona"},
        "hieunguyenminh/roleplay": {"target": "tier4_voice_persona"},
        "NousResearch/CharacterCodex": {"target": "tier4_voice_persona"},
        # Model Selection / Evaluation
        "HuggingFaceH4/ultrafeedback_binarized": {"target": "tier5_research"},
        "argilla/ultrafeedback-multi-binarized": {"target": "tier5_research"},
    }

    def process_and_stream(self, bucket: str = "pixel-data", prefix: str = "datasets/training_v3/"):
        """Downloads from HF and streams directly to S3."""
        if load_dataset is None:
            logger.warning("datasets library not installed. Skipping HuggingFace ingestion.")
            return {}

        results = {}
        for ds_name, config in self.DATASETS.items():
            target = config.get("target")
            split = config.get("split", "train")

            try:
                logger.info(f"Loading {ds_name}...")
                ds = load_dataset(ds_name, split=split)

                # Stream to S3
                safe_name = ds_name.replace("/", "_")
                s3_key = f"{prefix}{target}/{safe_name}.jsonl"

                # Capture loop variables in closure defaults
                def data_generator(dataset=ds, source_name=ds_name):
                    for i, row in enumerate(dataset):
                        if i >= 500:  # Limit per dataset
                            break
                        yield {"source": source_name, "data": dict(row)}

                count = stream_to_s3(data_generator(), bucket, s3_key)
                results[ds_name] = count

            except Exception as e:
                logger.error(f"Failed to load {ds_name}: {e}")
                results[ds_name] = 0

        return results


class ZenodoIngestor:
    """Ingests datasets from Zenodo, streaming directly to S3."""

    DATASETS: ClassVar[dict] = {
        # Format: "record_id": {"name": "...", "target": "..."}
        "4611251": {
            "name": "reddit_mental_health_2018_2020",
            "target": "stage1_foundation",
        },
        "7612674": {
            "name": "mental_health_lifestyle_sentiment",
            "target": "stage1_foundation",
        },
        "7678225": {
            "name": "multi_class_depression_tweets",
            "target": "stage3_edge_crisis",
        },
    }

    def fetch_and_stream(self, bucket: str = "pixel-data", prefix: str = "datasets/training_v3/"):
        """Fetches from Zenodo API and streams to S3."""
        if requests is None:
            logger.warning("requests library not installed. Skipping Zenodo ingestion.")
            return {}

        results = {}
        for record_id, config in self.DATASETS.items():
            name = config["name"]
            target = config["target"]

            try:
                # Get record metadata
                url = f"https://zenodo.org/api/records/{record_id}"
                resp = requests.get(url)
                resp.raise_for_status()
                record = resp.json()

                # Find downloadable files
                files = record.get("files", [])
                for f in files:
                    if f["key"].endswith((".csv", ".json", ".jsonl")):
                        file_url = f["links"]["self"]

                        # Download and stream
                        logger.info(f"Downloading Zenodo {name}/{f['key']}...")
                        file_resp = requests.get(file_url, stream=True)

                        s3_key = f"{prefix}{target}/zenodo_{name}_{f['key']}"
                        s3 = get_s3_client()
                        if s3:
                            s3.upload_fileobj(file_resp.raw, bucket, s3_key)
                            results[f"{record_id}/{f['key']}"] = s3_key
                            logger.info(f"Streamed to {s3_key}")
                        break  # Only first matching file

            except Exception as e:
                logger.error(f"Failed to fetch Zenodo {record_id}: {e}")

        return results


class GitHubDatasetIngestor:
    """Ingests datasets from GitHub repositories."""

    DATASETS: ClassVar[dict] = {
        # Format: "owner/repo": {"file": "path/to/file.json", "target": "..."}
        "kharrigian/mental-health-datasets": {
            "file": "README.md",  # Just metadata for now
            "target": "stage1_foundation",
        }
    }

    def fetch_and_stream(self, bucket: str = "pixel-data", prefix: str = "datasets/training_v3/"):
        """Fetches from GitHub raw content."""
        if requests is None:
            logger.warning("requests library not installed. Skipping GitHub ingestion.")
            return {}

        results = {}
        for repo, config in self.DATASETS.items():
            file_path = config["file"]
            target = config["target"]

            try:
                url = f"https://raw.githubusercontent.com/{repo}/main/{file_path}"
                resp = requests.get(url)
                resp.raise_for_status()

                s3_key = f"{prefix}{target}/github_{repo.replace('/', '_')}_{file_path.replace('/', '_')}"
                if s3 := get_s3_client():
                    s3.put_object(Bucket=bucket, Key=s3_key, Body=resp.content)
                    results[repo] = s3_key
                    logger.info(f"Uploaded {s3_key}")

            except Exception as e:
                logger.error(f"Failed to fetch GitHub {repo}: {e}")

        return results


class KaggleIngestor:
    """Ingests datasets from Kaggle, streaming directly to S3."""

    DATASETS: ClassVar[dict] = {
        "ruchi798/depression-and-anxiety-on-twitter": "stage1_foundation",
        "osmi/mental-health-in-tech-survey": "stage1_foundation",
        "nikhileswarkomati/suicide-watch": "stage3_edge_crisis",
        "thedevastator/nlp-mental-health-conversations": "stage1_foundation",
        "suchintikasarkar/sentiment-analysis-for-mental-health": "stage1_foundation",
    }

    def fetch_and_stream(self, bucket: str = "pixel-data", prefix: str = "datasets/training_v3/"):
        """Fetches from Kaggle and streams to S3."""
        if kaggle is None:
            logger.warning("Kaggle library not installed. Skipping.")
            return {}

        results = {}
        s3 = get_s3_client()
        if not s3:
            return {}

        for slug, target in self.DATASETS.items():
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    logger.info(f"Downloading Kaggle: {slug}...")
                    kaggle.api.dataset_download_files(slug, path=tmpdir, unzip=True)

                    # Upload each file to S3
                    for file_path in Path(tmpdir).rglob("*"):
                        if file_path.is_file() and file_path.suffix in [
                            ".csv",
                            ".json",
                            ".jsonl",
                        ]:
                            s3_key = (
                                f"{prefix}{target}/kaggle_{slug.replace('/', '_')}_{file_path.name}"
                            )
                            s3.upload_file(str(file_path), bucket, s3_key)
                            results[f"{slug}/{file_path.name}"] = s3_key
                            logger.info(f"Uploaded {s3_key}")
            except Exception as e:
                logger.error(f"Kaggle {slug} failed: {e}")

        return results


class DataWorldIngestor:
    """Ingests datasets from data.world, streaming to S3."""

    DATASETS: ClassVar[dict] = {
        "crowdflower/sentiment-analysis-societal-topics": "stage1_foundation",
        # For scam detection edge cases
        "shivamb/real-or-fake-fake-jobposting-prediction": "stage1_foundation",
    }

    def fetch_and_stream(self, bucket: str = "pixel-data", prefix: str = "datasets/training_v3/"):
        """Fetches from data.world and streams to S3."""
        if dw is None:
            logger.warning("data.world library not installed. Skipping.")
            return {}

        results = {}
        s3 = get_s3_client()
        if not s3:
            return {}

        for ds_id, target in self.DATASETS.items():
            try:
                logger.info(f"Loading data.world: {ds_id}...")
                dataset = dw.load_dataset(ds_id)

                for table_name, df in dataset.dataframes.items():
                    # Convert to JSONL and stream
                    buffer = BytesIO()
                    for _, row in df.iterrows():
                        buffer.write((json.dumps(row.to_dict()) + "\n").encode("utf-8"))
                    buffer.seek(0)

                    s3_key = (
                        f"{prefix}{target}/dataworld_{ds_id.replace('/', '_')}_{table_name}.jsonl"
                    )
                    s3.upload_fileobj(buffer, bucket, s3_key)
                    results[f"{ds_id}/{table_name}"] = s3_key
                    logger.info(f"Streamed {s3_key}")

            except Exception as e:
                logger.error(f"data.world {ds_id} failed: {e}")

        return results


class OpenMLIngestor:
    """Ingests datasets from OpenML."""

    DATASETS: ClassVar[dict] = {
        # OpenML dataset IDs
        "43141": {"name": "mental_health_survey", "target": "stage1_foundation"}
    }

    def fetch_and_stream(self, bucket: str = "pixel-data", prefix: str = "datasets/training_v3/"):
        """Fetches from OpenML and streams to S3."""
        if openml is None:
            logger.warning("OpenML library not installed. Skipping.")
            return {}

        results = {}
        s3 = get_s3_client()
        if not s3:
            return {}

        for ds_id, config in self.DATASETS.items():
            try:
                logger.info(f"Loading OpenML dataset {ds_id}...")
                dataset = openml.datasets.get_dataset(int(ds_id))
                features, target_col, _, _ = dataset.get_data()

                # Convert to JSONL
                buffer = BytesIO()
                for i, row in features.iterrows():
                    record = row.to_dict()
                    if target_col is not None:
                        record["target"] = (
                            target_col.iloc[i] if hasattr(target_col, "iloc") else target_col[i]
                        )
                    buffer.write((json.dumps(record) + "\n").encode("utf-8"))
                buffer.seek(0)

                s3_key = f"{prefix}{config['target']}/openml_{config['name']}.jsonl"
                s3.upload_fileobj(buffer, bucket, s3_key)
                results[ds_id] = s3_key
                logger.info(f"Streamed {s3_key}")

            except Exception as e:
                logger.error(f"OpenML {ds_id} failed: {e}")

        return results


def run_all_ingestors(bucket: str = "pixel-data", prefix: str = "datasets/training_v3/"):
    """Runs all ingestors and streams to S3."""
    results = {}

    # Local Consolidated (NEW - highest priority, 3.2GB of finalized data)
    if LocalConsolidatedIngestor is not None:
        try:
            local = LocalConsolidatedIngestor()
            local_results = local.run_full_upload(bucket, "datasets/consolidated/", skip_large=True)
            results["local_consolidated"] = local_results.get("total_success", 0)
        except Exception as e:
            logger.warning(f"Local consolidated ingestion failed: {e}")
            results["local_consolidated"] = 0
    else:
        logger.warning("LocalConsolidatedIngestor not available. Skipping.")
        results["local_consolidated"] = 0

    # HuggingFace
    hf = HuggingFaceIngestor()
    results["huggingface"] = hf.process_and_stream(bucket, prefix)

    # NGC (NVIDIA GPU Cloud)
    if NGCIngestor is not None:
        try:
            ngc = NGCIngestor()
            if ngc.is_available():
                ngc_results = ngc.process_all_configured()
                results["ngc"] = len(
                    [r for r in ngc_results.values() if r.get("status") == "success"]
                )
            else:
                logger.info("NGC CLI not available, skipping NGC dataset ingestion")
                results["ngc"] = 0
        except Exception as e:
            logger.warning(f"NGC ingestion failed: {e}")
            results["ngc"] = 0
    else:
        logger.warning("NGCIngestor not available. Skipping.")
        results["ngc"] = 0

    # Zenodo
    zenodo = ZenodoIngestor()
    results["zenodo"] = zenodo.fetch_and_stream(bucket, prefix)

    # GitHub
    github = GitHubDatasetIngestor()
    results["github"] = github.fetch_and_stream(bucket, prefix)

    # Kaggle
    kaggle_ing = KaggleIngestor()
    results["kaggle"] = kaggle_ing.fetch_and_stream(bucket, prefix)

    # data.world
    dw_ing = DataWorldIngestor()
    results["dataworld"] = dw_ing.fetch_and_stream(bucket, prefix)

    # OpenML
    openml_ing = OpenMLIngestor()
    results["openml"] = openml_ing.fetch_and_stream(bucket, prefix)

    total = sum(len(v) if isinstance(v, dict) else v for v in results.values())
    return f"Streamed {total} datasets to s3://{bucket}/"
