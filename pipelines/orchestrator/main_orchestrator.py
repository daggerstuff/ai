import logging
import sqlite3
import sys
import threading
import uuid
from pathlib import Path

from pipelines.voice.audio_ingestion import AudioIngestionPipeline
from pipelines.voice.feature_extraction import FeatureExtractionPipeline
from pipelines.voice.pair_generation import TherapeuticPairGenerator
from pipelines.voice.transcription import TranscriptionOrchestrator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MainOrchestrator")

AI_ROOT = Path(__file__).resolve().parents[2]


class PipelineOrchestrator:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else AI_ROOT / "training_corpus/assets/registry.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # timeout avoids "database is locked" under concurrent access; a lock serializes writes.
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self._lock = threading.Lock()
        self._init_db()

        self.ingestion = AudioIngestionPipeline(max_workers=2)
        self.transcription = TranscriptionOrchestrator()
        self.feature_extraction = FeatureExtractionPipeline()
        self.pair_generation = TherapeuticPairGenerator()

    def _init_db(self) -> None:
        with self._lock, self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS training_shards (
                    shard_id TEXT PRIMARY KEY,
                    source_file TEXT,
                    pair_count INTEGER,
                    export_path TEXT
                )
            """)

    def register_shard(self, source_file: str, pair_count: int, export_path: str) -> None:
        shard_id = str(uuid.uuid4())
        with self._lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO training_shards (shard_id, source_file, pair_count, export_path)
                VALUES (?, ?, ?, ?)
            """,
                (shard_id, source_file, pair_count, export_path),
            )
        logger.info(f"Registered shard {shard_id} for {source_file} with {pair_count} pairs.")

    def register_all_shards(self) -> None:
        pairs_dir = AI_ROOT / "data/pairs"
        if not pairs_dir.exists():
            return

        for pair_file in pairs_dir.glob("*.jsonl"):
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT shard_id FROM training_shards WHERE export_path = ?", (str(pair_file),)
            )
            if cursor.fetchone():
                continue

            try:
                # Count pairs
                with open(pair_file, encoding="utf-8") as f:
                    count = sum(1 for line in f if line.strip())
                self.register_shard(pair_file.stem, count, str(pair_file))
            except Exception as e:
                logger.error(f"Failed to register shard {pair_file}: {e}")

    def run_channel(self, channel_url: str) -> None:
        logger.info(f"--- Starting End-to-End Pipeline for {channel_url} ---")

        # 1. Ingestion
        logger.info("Step 1: Audio Ingestion")
        self.ingestion.ingest_channel(channel_url)

        # 2. Transcription & Diarization
        logger.info("Step 2: Transcription & Diarization")
        self.transcription.process_all()

        # 3. Feature Extraction
        logger.info("Step 3: Feature Extraction (Emotion & Clustering)")
        self.feature_extraction.process_all()

        # 4. Pair Generation
        logger.info("Step 4: ML Validation & Therapeutic Pair Generation")
        self.pair_generation.process_all()

        # 5. Registry Update
        logger.info("Step 5: Registering JSONL Shards")
        self.register_all_shards()

        logger.info("--- Pipeline Execution Complete ---")

    def batch_run_all_channels(self, channel_list_path: str) -> None:
        try:
            with open(channel_list_path, encoding="utf-8") as f:
                channels = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Could not read channel list {channel_list_path}: {e}")
            return

        logger.info(f"Loaded {len(channels)} channels for batch processing.")
        for i, channel in enumerate(channels, 1):
            logger.info(f"Processing channel {i}/{len(channels)}")
            # Isolate per-channel failures so one bad channel does not abort the batch.
            try:
                self.run_channel(channel)
            except Exception as e:
                logger.error(f"Channel {channel} failed, skipping: {e}")


if __name__ == "__main__":
    orchestrator = PipelineOrchestrator()
    if len(sys.argv) > 1:
        if sys.argv[1] == "--batch":
            orchestrator.batch_run_all_channels(sys.argv[2])
        else:
            orchestrator.run_channel(sys.argv[1])
    else:
        logger.error(
            "Usage: uv run python -m pipelines.orchestrator.main_orchestrator "
            "<channel_url> OR --batch <channels_list.txt>"
        )
