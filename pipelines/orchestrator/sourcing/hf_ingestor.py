import logging
import json
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class HuggingFaceIngestor:
    """
    Ingests public datasets from HuggingFace using the `datasets` library.
    """

    DATASETS = {
        # Format: "owner/name": {"target": "folder", "config": "config_name", "split": "train"}

        # Foundation - General
        "Amod/mental_health_counseling_conversations": {
            "target": "stage1_foundation", "split": "train"
        },
        "heliosbrahma/mental_health_chatbot_dataset": {
            "target": "stage1_foundation", "split": "train"
        },

        # Specialist - Addiction & Recovery
        "fadodr/mental_health_therapy": {
            "target": "stage2_specialist_addiction", "split": "train"
        },

        # Specialist - Trauma & PTSD
        "yenopoya/thousand-voices-trauma": {
            "target": "stage2_specialist_ptsd", "split": "train"
        },

        # Specialist - Personality Disorders
        "Kanakmi/mental-disorders": {
            "target": "stage2_specialist_personality", "split": "train"
        },

        # Edge Cases - Crisis
        "AIMH/SWMH": {
            "target": "stage3_edge_crisis", "split": "train"
        }
    }

    def __init__(self):
        self.base_output_path = Path("ai/training/ready_packages/datasets")

    def fetch_dataset(self, dataset_name: str, config: Dict) -> List[Dict]:
        """
        Fetches dataset using HuggingFace `datasets` library.
        """
        try:
            from datasets import load_dataset

            target = config.get("target")
            split = config.get("split", "train")
            config_name = config.get("config", None)

            logger.info(f"Loading {dataset_name} (split={split}, config={config_name})...")

            if config_name:
                ds = load_dataset(dataset_name, config_name, split=split, trust_remote_code=True)
            else:
                ds = load_dataset(dataset_name, split=split, trust_remote_code=True)

            # Convert to list of dicts, limit to 500 rows for speed
            rows = [row for row in ds.select(range(min(500, len(ds))))]
            logger.info(f"  -> Loaded {len(rows)} rows.")
            return rows, target

        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            return [], config.get("target")

    def process_and_save(self, limit_per_dataset: int = 500):
        """
        Downloads and saves datasets to their respective folders.
        """
        results = {}

        for ds_name, config in self.DATASETS.items():
            rows, target_folder = self.fetch_dataset(ds_name, config)

            if rows:
                output_dir = self.base_output_path / target_folder
                output_dir.mkdir(parents=True, exist_ok=True)

                safe_name = ds_name.replace("/", "_")
                output_file = output_dir / f"{safe_name}.jsonl"

                with open(output_file, "w") as f:
                    for row in rows[:limit_per_dataset]:
                        # Normalize
                        record = {
                            "source": ds_name,
                            "data": dict(row),
                            "type": "external_foundation"
                        }
                        f.write(json.dumps(record) + "\n")

                results[ds_name] = str(output_file)
                logger.info(f"Saved {len(rows)} items to {output_file}")
            else:
                logger.warning(f"No data found for {ds_name}")

        return results


class KaggleIngestor:
    """
    Ingests datasets from Kaggle.
    Requires KAGGLE_USERNAME and KAGGLE_KEY in environment or ~/.kaggle/kaggle.json.
    """

    DATASETS = {
        # Format: "owner/dataset-slug"
        "ruchi798/depression-and-anxiety-on-twitter": "stage1_foundation",
        "osmi/mental-health-in-tech-survey": "stage1_foundation",
        "nikhileswarkomati/suicide-watch": "stage3_edge_crisis"
    }

    def __init__(self):
        self.base_output_path = Path("ai/training/ready_packages/datasets")

    def fetch_dataset(self, dataset_slug: str, target_folder: str) -> str:
        """
        Downloads a Kaggle dataset.
        """
        try:
            import kaggle

            output_dir = self.base_output_path / target_folder / "kaggle"
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading Kaggle dataset: {dataset_slug}...")
            kaggle.api.dataset_download_files(dataset_slug, path=str(output_dir), unzip=True)

            logger.info(f"  -> Downloaded to {output_dir}")
            return str(output_dir)

        except Exception as e:
            logger.error(f"Failed to download Kaggle dataset {dataset_slug}: {e}")
            return None

    def process_and_save(self):
        """Downloads all configured Kaggle datasets."""
        results = {}
        for slug, target in self.DATASETS.items():
            output = self.fetch_dataset(slug, target)
            if output:
                results[slug] = output
        return results


class DataWorldIngestor:
    """
    Ingests datasets from data.world.
    Requires DW_AUTH_TOKEN in environment.
    """

    DATASETS = {
        # Format: "owner/dataset-id"
        "crowdflower/sentiment-analysis-societal-topics": "stage1_foundation"
    }

    def __init__(self):
        self.base_output_path = Path("ai/training/ready_packages/datasets")

    def fetch_dataset(self, dataset_id: str, target_folder: str) -> str:
        """
        Downloads a data.world dataset.
        """
        try:
            import datadotworld as dw

            output_dir = self.base_output_path / target_folder / "dataworld"
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Loading data.world dataset: {dataset_id}...")
            dataset = dw.load_dataset(dataset_id)

            # Export tables as CSVs
            for table_name, df in dataset.dataframes.items():
                output_file = output_dir / f"{dataset_id.replace('/', '_')}_{table_name}.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"  -> Saved {table_name} to {output_file}")

            return str(output_dir)

        except Exception as e:
            logger.error(f"Failed to download data.world dataset {dataset_id}: {e}")
            return None

    def process_and_save(self):
        """Downloads all configured data.world datasets."""
        results = {}
        for ds_id, target in self.DATASETS.items():
            output = self.fetch_dataset(ds_id, target)
            if output:
                results[ds_id] = output
        return results
