import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)


class AcousticDeduplicator:
    """
    Acoustic deduplication for audio files.
    Extracts features and compares them to find near-duplicates.
    """

    def __init__(self, threshold: float = 0.95, sample_rate: int = 16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.fingerprints: Dict[str, np.ndarray] = {}

        if not LIBROSA_AVAILABLE:
            logger.warning(
                "librosa not available. Acoustic deduplication will be disabled."
            )

    def compute_fingerprint(self, file_path: Path) -> Optional[np.ndarray]:
        """
        Computes an acoustic fingerprint (MFCC mean/std) for an audio file.
        """
        if not LIBROSA_AVAILABLE:
            return None

        try:
            # Load only a portion if it's too long
            y, sr = librosa.load(str(file_path), sr=self.sample_rate, duration=300)

            # Extract MFCCs
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            # Mean and Std of MFCCs across time as a compact descriptor
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            return np.concatenate([mfcc_mean, mfcc_std])
        except Exception as e:
            logger.error(f"Error computing fingerprint for {file_path}: {e}")
            return None

    def compare_fingerprints(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """
        Computes cosine similarity between two fingerprints.
        """
        norm1 = np.linalg.norm(fp1)
        norm2 = np.linalg.norm(fp2)
        return np.dot(fp1, fp2) / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0.0

    def find_duplicates(self, audio_files: List[Path]) -> List[List[Path]]:
        """
        Finds groups of duplicate audio files.
        """
        if not LIBROSA_AVAILABLE:
            return []

        fps = {}
        for f in audio_files:
            fp = self.compute_fingerprint(f)
            if fp is not None:
                fps[f] = fp

        duplicate_groups = []
        processed = set()

        file_list = list(fps.keys())
        for i, f1 in enumerate(file_list):
            if f1 in processed:
                continue

            group = [f1]
            for f2 in file_list[i + 1 :]:
                if f2 in processed:
                    continue

                similarity = self.compare_fingerprints(fps[f1], fps[f2])
                if similarity >= self.threshold:
                    group.append(f2)
                    processed.add(f2)

            if len(group) > 1:
                duplicate_groups.append(group)
                processed.add(f1)

        return duplicate_groups

    def run_deduplication(self, input_dir: Path) -> Dict[str, Any]:
        """
        Scans a directory for audio files and finds duplicates.
        """
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a"}
        audio_files = [
            f for f in input_dir.glob("**/*") if f.suffix.lower() in audio_extensions
        ]

        logger.info(f"Scanning {len(audio_files)} files in {input_dir}")
        duplicates = self.find_duplicates(audio_files)

        total_removed = sum(len(group) - 1 for group in duplicates)

        return {
            "total_files": len(audio_files),
            "duplicate_groups": [[str(p) for p in g] for g in duplicates],
            "total_duplicates_found": total_removed,
        }


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python acoustic_deduplication.py <directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    deduper = AcousticDeduplicator()
    results = deduper.run_deduplication(input_dir)
    print(json.dumps(results, indent=2))
