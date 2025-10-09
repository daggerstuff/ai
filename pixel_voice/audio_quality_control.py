import glob
import json
import logging
import os
from pathlib import Path

import librosa
import numpy as np
import webrtcvad
from langdetect import detect
from pydub import AudioSegment  # type: ignore[import]

# Configuration
AUDIO_DIR = "data/voice_raw"
SEGMENT_DIR = "data/voice_segments"
METRICS_DIR = "data/voice_metrics"
LOG_FILE = "logs/audio_quality_control.log"
SAMPLE_RATE = 16000
VAD_MODE = 2  # 0-3, 3 is most aggressive

os.makedirs(SEGMENT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def compute_snr(y: np.ndarray) -> float:
    """
    Compute the signal-to-noise ratio (SNR) of an audio signal.

    Args:
        y (np.ndarray): Audio signal array.

    Returns:
        float: SNR value in decibels (dB).
    """
    signal_power = np.mean(y**2)
    noise_power = np.var(y - np.mean(y))
    return 10 * np.log10(signal_power / (noise_power + 1e-8))


def compute_loudness(audio_path):
    audio = AudioSegment.from_file(audio_path)
    return audio.dBFS


def compute_silence_ratio(y, _sr, threshold_db=-40):
    s = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    return np.mean(s < threshold_db)


CLIPPING_THRESHOLD = 0.99


def compute_quality_metrics(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    snr = compute_snr(y)
    loudness = compute_loudness(audio_path)
    silence_ratio = compute_silence_ratio(y, sr)
    clipping_ratio = np.mean(np.abs(y) > CLIPPING_THRESHOLD)
    language = detect_language(audio_path)
    return {
        "snr": snr,
        "loudness": loudness,
        "silence_ratio": silence_ratio,
        "clipping_ratio": clipping_ratio,
        "language": language,
    }


def detect_language(audio_path):
    # Use Whisper or langdetect on transcript for production; here, placeholder
    try:
        y, sr = librosa.load(audio_path, sr=None)
        text = ""  # Placeholder: run ASR for real language detection
        return detect(text) if text else "unknown"
    except Exception:
        return "unknown"


def segment_audio_vad(audio_path, out_dir, sample_rate=SAMPLE_RATE, vad_mode=VAD_MODE):
    audio = AudioSegment.from_file(audio_path).set_frame_rate(sample_rate).set_channels(1)
    raw_audio = np.array(audio.get_array_of_samples()).tobytes()
    vad = webrtcvad.Vad(vad_mode)
    frame_ms = 30
    frame_bytes = int(sample_rate * frame_ms / 1000) * 2
    segments = []
    start = None
    for i in range(0, len(raw_audio), frame_bytes):
        frame = raw_audio[i : i + frame_bytes]
        if len(frame) < frame_bytes:
            break
        is_speech = vad.is_speech(frame, sample_rate)
        if is_speech and start is None:
            start = i
        elif not is_speech and start is not None:
            end = i
            seg = audio[start // 2 : end // 2]
            seg_path = os.path.join(out_dir, f"{Path(audio_path).stem}_seg_{start}_{end}.wav")
            seg.export(seg_path, format="wav")
            segments.append(seg_path)
            start = None
    return segments


def assess_file(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    snr = compute_snr(y)
    loudness = compute_loudness(audio_path)
    silence_ratio = compute_silence_ratio(y, sr)
    language = detect_language(audio_path)
    segments = segment_audio_vad(audio_path, SEGMENT_DIR)
    return {
        "file": audio_path,
        "snr": snr,
        "loudness_db": loudness,
        "silence_ratio": silence_ratio,
        "language": language,
        "num_segments": len(segments),
        "segments": segments,
    }


def main():
    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    for audio_path in audio_files:
        try:
            metrics = assess_file(audio_path)
            base = Path(audio_path).stem
            metrics_path = os.path.join(METRICS_DIR, f"{base}_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            logging.info(f"Processed: {audio_path}")
        except Exception as e:
            logging.error(f"Error processing {audio_path}: {e}")


if __name__ == "__main__":
    main()
