import glob
import os
import subprocess

# Configuration
AUDIO_DIR = "data/voice_raw"
OUTPUT_DIR = "data/voice_transcripts"
PIPELINE = "faster-whisper"  # Change to "whisperx" if desired

os.makedirs(OUTPUT_DIR, exist_ok=True)


def transcribe_with_faster_whisper(audio_path, output_path):
    # Example: Use CLI for simplicity; replace with API if needed
    output_dir = os.path.dirname(output_path)
    cmd = [
        "faster-whisper",
        audio_path,
        "--output_dir",
        output_dir,
        "--output_format",
        "json",
        "--task",
        "transcribe",
    ]
    print(f"Transcribing {audio_path} to {output_path} ...")
    subprocess.run(cmd, check=True)


def transcribe_with_whisperx(audio_path, output_path):
    # Example: Use CLI for simplicity; replace with API if needed
    output_dir = os.path.dirname(output_path)
    cmd = [
        "whisperx",
        audio_path,
        "--output_dir",
        output_dir,
        "--output_format",
        "json",
        "--task",
        "transcribe",
    ]
    print(f"Transcribing {audio_path} to {output_path} ...")
    subprocess.run(cmd, check=True)


def main():
    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    if not audio_files:
        print(f"No audio files found in {AUDIO_DIR}")
        return

    for audio_path in audio_files:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base}.json")
        if os.path.exists(output_path):
            print(f"Transcript already exists for {audio_path}, skipping.")
            continue
        try:
            if PIPELINE == "faster-whisper":
                transcribe_with_faster_whisper(audio_path, output_path)
            elif PIPELINE == "whisperx":
                transcribe_with_whisperx(audio_path, output_path)
            else:
                print(f"Unknown pipeline: {PIPELINE}")
        except subprocess.CalledProcessError as e:
            print(f"Error transcribing {audio_path}: {e}")


if __name__ == "__main__":
    main()
