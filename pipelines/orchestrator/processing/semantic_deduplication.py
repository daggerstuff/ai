import json
import logging
from pathlib import Path
from typing import Any, Dict

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


class SemanticDeduplicator:
    """
    Semantic deduplication for text transcripts.
    Computes BERT embeddings and finds near-duplicates.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.92):
        self.threshold = threshold
        self.model_name = model_name
        self.model = None

        if HAS_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load sentence-transformer: {e}")
                # Note: Not re-assigning HAS_TRANSFORMERS here to avoid
                # local variable scope issues
        else:
            logger.warning(
                "sentence-transformers not installed. Semantic deduplication disabled."
            )

    def run_deduplication(self, transcripts_dir: Path) -> Dict[str, Any]:
        """
        Scans a directory for markdown/text transcripts and finds semantic duplicates.
        """
        if not HAS_TRANSFORMERS or not self.model:
            return {"error": "Transformers not available"}

        transcript_files = list(transcripts_dir.glob("**/*.md")) + list(
            transcripts_dir.glob("**/*.txt")
        )
        if not transcript_files:
            return {"total_files": 0, "duplicate_groups": []}

        logger.info(f"Computing embeddings for {len(transcript_files)} transcripts...")

        # Load contents
        contents = []
        valid_files = []
        for f in transcript_files:
            try:
                text = f.read_text(encoding="utf-8")
                # Strip metadata if it's a markdown with headers
                if "## Transcript" in text:
                    text = text.split("## Transcript")[-1]

                if len(text.strip()) > 100:  # Ignore tiny files
                    contents.append(text)
                    valid_files.append(f)
            except Exception as e:
                logger.warning(f"Error reading {f}: {e}")

        if not contents:
            return {"total_files": 0, "duplicate_groups": []}

        # Compute embeddings in batch
        embeddings = self.model.encode(contents, batch_size=8, show_progress_bar=True)

        # Find duplicates via cosine similarity
        similarities = cosine_similarity(embeddings)

        duplicate_groups = []
        processed = set()

        for i in range(len(valid_files)):
            if i in processed:
                continue

            group = [str(valid_files[i])]
            for j in range(i + 1, len(valid_files)):
                if j in processed:
                    continue

                if similarities[i][j] >= self.threshold:
                    group.append(str(valid_files[j]))
                    processed.add(j)

            if len(group) > 1:
                duplicate_groups.append(group)
                processed.add(i)

        total_removed = sum(len(group) - 1 for group in duplicate_groups)

        return {
            "total_files": len(valid_files),
            "duplicate_groups": duplicate_groups,
            "total_duplicates_found": total_removed,
            "threshold": self.threshold,
            "model": self.model_name,
        }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python semantic_deduplication.py <transcripts_dir>")
        sys.exit(1)

    path = Path(sys.argv[1])
    deduper = SemanticDeduplicator()
    results = deduper.run_deduplication(path)
    print(json.dumps(results, indent=2))
