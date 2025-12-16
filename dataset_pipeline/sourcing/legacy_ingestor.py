import csv
import json
import logging
from pathlib import Path

from ai.dataset_pipeline.storage_config import get_dataset_pipeline_output_root

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegacyDatasetIngestor:
    def __init__(self, output_base: str | Path | None = None):
        if output_base is None:
            output_base = get_dataset_pipeline_output_root() / "data"
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)

        self.priority_path = Path("ai/lightning/ghost/datasets")
        self.consolidated_path = Path("ai/training_data_consolidated/final_datasets")

    def convert_csv_to_jsonl(
        self, input_path: Path, output_file: Path, text_col: str, label_col: str = None
    ):
        """Generic CSV to JSONL converter."""
        if not input_path.exists():
            logger.warning(f"File not found: {input_path}")
            return

        count = 0
        with open(output_file, "a") as out_f:
            try:
                with open(input_path, encoding="utf-8", errors="replace") as in_f:
                    reader = csv.DictReader(in_f)
                    for row in reader:
                        text = row.get(text_col)
                        if not text:
                            continue

                        record = {
                            "instruction": "Analyze this text for mental health signals.",
                            "input": text,
                            "output": row.get(label_col, "No label") if label_col else "N/A",
                            "source": input_path.name,
                        }
                        out_f.write(json.dumps(record) + "\n")
                        count += 1
            except Exception as e:
                logger.error(f"Failed to process {input_path}: {e}")

        logger.info(f"Converted {count} rows from {input_path.name}")

    def ingest_priority_files(self):
        """Ingest curated Wendy datasets."""
        tier_dir = self.output_base / "tier1_priority"
        tier_dir.mkdir(exist_ok=True)
        output_file = tier_dir / "legacy_priority_ingested.jsonl"

        total_count = 0

        filename_candidates = {
            1: ["wendy_set_alpha_therapeutic_core.jsonl"],
            2: ["wendy_set_beta_high_quality_core.jsonl"],
            3: ["wendy_set_gamma_specialized_therapy.jsonl"],
        }

        for i in range(1, 4):
            candidate_paths = [
                self.priority_path / f"priority_{i}" / filename
                for filename in filename_candidates.get(i, [])
            ]
            if not candidate_paths:
                continue
            path = next((p for p in candidate_paths if p.exists()), None)
            if path is not None:
                logger.info(f"Ingesting {path}...")
                with open(output_file, "a") as out_f, open(path) as in_f:
                    for line in in_f:
                        try:
                            data = json.loads(line)
                            # Ensure standard format
                            if "instruction" not in data and "text" in data:
                                # Handle raw text or mismatched schema
                                data["instruction"] = "User/Assistant interaction"
                                data["input"] = ""
                                data["output"] = data["text"]

                            data["source"] = f"wendy_set_{i}"
                            out_f.write(json.dumps(data) + "\n")
                            total_count += 1
                        except json.JSONDecodeError:
                            continue

        logger.info(f"Ingested {total_count} records from Priority datasets.")

    def ingest_suicide_watch(self):
        """Ingest Suicide Watch CSV."""
        tier_dir = self.output_base / "tier3_edge_crisis"
        tier_dir.mkdir(exist_ok=True)
        output_file = tier_dir / "suicidewatch_ingested.jsonl"

        csv_path = self.priority_path / "suicide-watch/human_labeled_Suicidewatch.csv"
        # Assuming column 'text' and 'label' based on typical structure
        self.convert_csv_to_jsonl(csv_path, output_file, "text", "label")

    def run_all(self):
        logger.info("Starting legacy ingestion...")
        self.ingest_priority_files()
        self.ingest_suicide_watch()
        logger.info("Legacy ingestion complete.")


if __name__ == "__main__":
    ingestor = LegacyDatasetIngestor()
    ingestor.run_all()
