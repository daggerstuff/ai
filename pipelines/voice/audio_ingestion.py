import logging
import os
import sqlite3
import threading
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import yt_dlp
from faster_whisper import WhisperModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AudioIngestion")

# Fixed full-scale reference for float audio in [-1.0, 1.0].
FULL_SCALE = 1.0


class PipelineRegistry:
    def __init__(self, db_path="ai/training_corpus/assets/registry.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock, self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_audio (
                    video_id TEXT PRIMARY KEY,
                    status TEXT,
                    qc_passed BOOLEAN,
                    snr REAL,
                    loudness REAL,
                    clipping_ratio REAL,
                    language TEXT,
                    chunks_created INTEGER,
                    error_message TEXT
                )
            """)

    def update_status(
        self,
        video_id,
        status,
        qc_passed=None,
        snr=None,
        loudness=None,
        clipping_ratio=None,
        language=None,
        chunks_created=0,
        error_message=None,
    ):
        with self._lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO processed_audio (video_id, status, qc_passed, snr, loudness, clipping_ratio, language, chunks_created, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    status=excluded.status,
                    qc_passed=excluded.qc_passed,
                    snr=excluded.snr,
                    loudness=excluded.loudness,
                    clipping_ratio=excluded.clipping_ratio,
                    language=excluded.language,
                    chunks_created=excluded.chunks_created,
                    error_message=excluded.error_message
            """,
                (
                    video_id,
                    status,
                    qc_passed,
                    snr,
                    loudness,
                    clipping_ratio,
                    language,
                    chunks_created,
                    error_message,
                ),
            )

    def get_status(self, video_id):
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT status FROM processed_audio WHERE video_id = ?", (video_id,))
            row = cursor.fetchone()
        return row[0] if row else None


class AudioDownloader:
    def __init__(self, output_dir="ai/data/raw_audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_audio(self, video_url: str) -> tuple[str, str]:
        """Download audio from a YouTube URL and return the (path, video_id)."""
        ydl_opts: typing.Any = {
            "format": "bestaudio/best",
            "outtmpl": str(self.output_dir / "%(id)s.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_id = info["id"]
            # after post-processing, yt-dlp saves as .wav
            return str(self.output_dir / f"{video_id}.wav"), video_id

    def get_channel_videos(self, channel_url: str) -> list[str]:
        ydl_opts: typing.Any = {
            "extract_flat": True,
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = typing.cast(dict[str, typing.Any], ydl.extract_info(channel_url, download=False))
            if info and "entries" in info:
                return [
                    str(entry["url"])
                    for entry in info["entries"]
                    if isinstance(entry, dict) and entry.get("url")
                ]
            return []


def _safe_qc(qc_results: dict) -> dict:
    """Return a copy of qc_results safe to log (no raw waveform / sample-rate)."""
    safe = {k: v for k, v in qc_results.items() if k not in ("y", "sr")}
    return safe


class QualityControl:
    def __init__(self, target_sr=16000, min_snr=15.0, min_loudness=-30.0, max_clipping=0.01):
        self.target_sr = target_sr
        self.min_snr = min_snr
        self.min_loudness = min_loudness
        self.max_clipping = max_clipping
        # Load tiny model for fast QC language detection
        self.whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

    def evaluate_audio(self, filepath: str) -> dict:
        y, sr = librosa.load(filepath, sr=self.target_sr)

        # 1. Loudness calculation
        rms = librosa.feature.rms(y=y)[0]
        loudness_db = 20 * np.log10(np.mean(rms) + 1e-9)

        # 2. SNR Estimation (heuristic: compare 95th percentile to 5th percentile energy)
        signal_energy = np.percentile(rms, 95)
        noise_energy = np.percentile(rms, 5)
        snr = 20 * np.log10((signal_energy + 1e-9) / (noise_energy + 1e-9))

        # 3. Clipping detection measured against the fixed full-scale reference.
        #    Comparing against the signal's own peak (0.99 * max) is meaningless
        #    because the peak is always ~the max, so the ratio would be ~1.0.
        clipping_ratio = float(np.mean(np.abs(y) >= 0.99 * FULL_SCALE))

        # 4. Language detection using faster-whisper
        # We only need the info object to get the detected language
        _, info = self.whisper_model.transcribe(filepath, vad_filter=True)
        language = info.language

        passed = (
            snr >= self.min_snr
            and loudness_db >= self.min_loudness
            and clipping_ratio <= self.max_clipping
            and language == "en"
        )

        return {
            "passed": bool(passed),
            "snr": float(snr),
            "loudness": float(loudness_db),
            "clipping_ratio": float(clipping_ratio),
            "language": language,
            "y": y,
            "sr": sr,
        }


class AudioSegmenter:
    def __init__(self, output_dir="ai/data/segmented_audio", chunk_length_s=30.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_length_s = chunk_length_s

    def process_and_segment(self, y: np.ndarray, sr: int, video_id: str) -> int:
        """Removes noise/silence and segments audio into consistent chunks."""
        # 1. Silence removal
        non_silent_intervals = librosa.effects.split(y, top_db=30)
        if non_silent_intervals:
            y_clean = np.concatenate([y[start:end] for start, end in non_silent_intervals])
        else:
            # No non-silent intervals detected; keep the full signal rather
            # than crashing on np.concatenate([]).
            y_clean = y

        # 2. Segmentation (emit a trailing partial chunk so no audio is dropped)
        samples_per_chunk = int(self.chunk_length_s * sr)
        total_chunks = len(y_clean) // samples_per_chunk

        for i in range(total_chunks):
            chunk = y_clean[i * samples_per_chunk : (i + 1) * samples_per_chunk]
            chunk_path = self.output_dir / f"{video_id}_chunk_{i:04d}.wav"
            sf.write(str(chunk_path), chunk, sr)

        remainder = len(y_clean) % samples_per_chunk
        if remainder > 0:
            tail = y_clean[total_chunks * samples_per_chunk :]
            tail_path = self.output_dir / f"{video_id}_chunk_{total_chunks:04d}.wav"
            sf.write(str(tail_path), tail, sr)
            total_chunks += 1

        return total_chunks


class AudioIngestionPipeline:
    def __init__(self, max_workers=4):
        self.registry = PipelineRegistry()
        self.downloader = AudioDownloader()
        self.qc = QualityControl()
        self.segmenter = AudioSegmenter()
        self.max_workers = max_workers

    def process_single_video(self, video_url: str):
        video_id = "unknown"
        try:
            logger.info(f"Downloading {video_url}")
            filepath, video_id = self.downloader.download_audio(video_url)

            if self.registry.get_status(video_id) == "COMPLETED":
                logger.info(f"Skipping {video_id}, already completed.")
                return True

            self.registry.update_status(video_id, "DOWNLOADING")

            logger.info(f"Evaluating QC for {video_id}")
            qc_results = self.qc.evaluate_audio(filepath)

            if not qc_results["passed"]:
                logger.warning(f"QC Failed for {video_id}: {_safe_qc(qc_results)}")
                self.registry.update_status(
                    video_id,
                    "QC_FAILED",
                    qc_passed=False,
                    snr=qc_results["snr"],
                    loudness=qc_results["loudness"],
                    clipping_ratio=qc_results["clipping_ratio"],
                    language=qc_results["language"],
                )
                return False

            logger.info(f"Segmenting {video_id}")
            chunks_created = self.segmenter.process_and_segment(
                qc_results["y"], qc_results["sr"], video_id
            )

            self.registry.update_status(
                video_id,
                "COMPLETED",
                qc_passed=True,
                snr=qc_results["snr"],
                loudness=qc_results["loudness"],
                clipping_ratio=qc_results["clipping_ratio"],
                language=qc_results["language"],
                chunks_created=chunks_created,
            )
            logger.info(f"Successfully processed {video_id}. Created {chunks_created} chunks.")

            # Clean up raw file to save space
            os.remove(filepath)
            return True

        except Exception as e:
            logger.error(f"Error processing {video_url}: {e}")
            if video_id != "unknown":
                self.registry.update_status(video_id, "ERROR", error_message=str(e))
            return False

    def ingest_channel(self, channel_url: str):
        logger.info(f"Fetching videos for channel: {channel_url}")
        video_urls = self.downloader.get_channel_videos(channel_url)
        logger.info(f"Found {len(video_urls)} videos.")

        success_count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self.process_single_video, url): url for url in video_urls
            }
            for future in as_completed(future_to_url):
                if future.result():
                    success_count += 1

        logger.info(
            f"Channel ingestion complete. Successfully processed {success_count}/{len(video_urls)} videos."
        )
        return success_count


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pipeline = AudioIngestionPipeline(max_workers=4)
        pipeline.ingest_channel(sys.argv[1])
