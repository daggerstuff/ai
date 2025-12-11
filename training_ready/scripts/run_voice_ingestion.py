#!/usr/bin/env python3
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from ai.dataset_pipeline.processing.transcript_ingestor import TranscriptIngestor

logging.basicConfig(level=logging.INFO)

def main():
    ingestor = TranscriptIngestor()
    # Process up to 50 files for now as a test
    output = ingestor.process_batch(batch_size=50)
    print(f"Ingestion Complete. Output: {output}")

if __name__ == "__main__":
    main()
