import glob
import json
import os

import librosa
import numpy as np
from pydub import AudioSegment

# Configuration
AUDIO_DIR = "data/voice_raw"
OUTPUT_DIR = "data/voice_quality"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_snr(y):
    # Signal-to-noise ratio (simple estimate)
    signal_power = np.mean(y**2)
    noise_power = np.var(y - np.mean(y))
    return 10 * np.log10(signal_power / (noise_power + 1e-8))


def compute_loudness(audio_path):
    audio = AudioSegment.from_file(audio_path)
    return audio.dBFS


def compute_silence_ratio(y, threshold_db=-40):
    # Ratio of silent frames
    s = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    return np.mean(threshold_db > s)


def assess_file(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    snr = compute_snr(y)
    loudness = compute_loudness(audio_path)
    silence_ratio = compute_silence_ratio(y, sr)
    return {
        "file": audio_path,
        "snr": snr,
        "loudness_db": loudness,
        "silence_ratio": silence_ratio,
    }


def main():
    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    results = []
    for audio_path in audio_files:
        try:
            metrics = assess_file(audio_path)
            results.append(metrics)
            print(f"Assessed: {audio_path}")
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    # Save report
    report_path = os.path.join(OUTPUT_DIR, "voice_quality_report.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Voice quality report written: {report_path}")


if __name__ == "__main__":
    main()
