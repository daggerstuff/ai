import os
import json
import glob

# Configuration
TRANSCRIPT_DIR = "data/voice_transcripts_filtered"
MARKERS_DIR = "data/personality_markers"
OUTPUT_DIR = "data/dialogue_format"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def build_dialogue(transcript, markers):
    # Simple heuristic: each segment is a turn; attach marker metadata
    dialogue = []
    for seg, meta in zip(transcript.get("segments", []), markers):
        turn = {
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg.get("text"),
            "personality_metadata": meta,
        }
        dialogue.append(turn)
    return dialogue


def main():
    transcript_files = glob.glob(os.path.join(TRANSCRIPT_DIR, "*.json"))
    for tfile in transcript_files:
        base = os.path.splitext(os.path.basename(tfile))[0]
        mfile = os.path.join(MARKERS_DIR, f"{base}_markers.json")
        if not os.path.exists(mfile):
            print(f"Warning: No marker file for {base}, skipping.")
            continue
        transcript = load_json(tfile)
        markers = load_json(mfile)
        dialogue = build_dialogue(transcript, markers)
        output_path = os.path.join(OUTPUT_DIR, f"{base}.jsonl")
        save_jsonl(dialogue, output_path)
        print(f"Dialogue format written: {output_path}")


if __name__ == "__main__":
    main()
