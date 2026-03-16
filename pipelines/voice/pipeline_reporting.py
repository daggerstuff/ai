import csv
import glob
import json
import logging
import os
from collections import Counter, defaultdict

# Configuration
LOGS_DIR = "logs"
DATA_DIRS = {
    "quality": "data/voice_quality",
    "transcripts": "data/voice_transcripts_filtered",
    "features": "data/voice",
    "clusters": "data/voice_clusters",
    "pairs": "data/dialogue_pairs",
    "therapeutic": "data/therapeutic_pairs",
}
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "pipeline_reporting.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def count_files(directory, pattern="*.json"):
    return len(glob.glob(os.path.join(directory, pattern)))


def aggregate_metrics():
    report = {}
    # Quality control
    if os.path.exists(DATA_DIRS["quality"]):
        report["audio_quality_files"] = count_files(DATA_DIRS["quality"])
    # Transcripts
    if os.path.exists(DATA_DIRS["transcripts"]):
        report["filtered_transcripts"] = count_files(DATA_DIRS["transcripts"])
    # Features
    if os.path.exists(DATA_DIRS["features"]):
        report["feature_files"] = count_files(DATA_DIRS["features"])
    # Clusters
    if os.path.exists(DATA_DIRS["clusters"]):
        cluster_file = os.path.join(
            DATA_DIRS["clusters"], "personality_emotion_clusters.json"
        )
        if os.path.exists(cluster_file):
            with open(cluster_file, "r", encoding="utf-8") as f:
                clusters = json.load(f)
            cluster_counts = Counter([c.get("cluster") for c in clusters])
            report["cluster_counts"] = dict(cluster_counts)
            report["total_clustered_segments"] = len(clusters)
    # Dialogue pairs
    if os.path.exists(DATA_DIRS["pairs"]):
        pair_file = os.path.join(DATA_DIRS["pairs"], "dialogue_pairs.json")
        validated_file = os.path.join(
            DATA_DIRS["pairs"], "dialogue_pairs_validated.json"
        )
        if os.path.exists(pair_file):
            with open(pair_file, "r", encoding="utf-8") as f:
                pairs = json.load(f)
            report["dialogue_pairs"] = len(pairs)
        if os.path.exists(validated_file):
            with open(validated_file, "r", encoding="utf-8") as f:
                validated = json.load(f)
            report["validated_pairs"] = len(validated)
    # Therapeutic pairs
    if os.path.exists(DATA_DIRS["therapeutic"]):
        therapeutic_file = os.path.join(
            DATA_DIRS["therapeutic"], "therapeutic_pairs.json"
        )
        if os.path.exists(therapeutic_file):
            with open(therapeutic_file, "r", encoding="utf-8") as f:
                therapeutic = json.load(f)
            report["therapeutic_pairs"] = len(therapeutic)
    # Error summary from logs
    error_counts = defaultdict(int)
    for log_file in glob.glob(os.path.join(LOGS_DIR, "*.log")):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if "[ERROR]" in line:
                    error_counts[os.path.basename(log_file)] += 1
    report["error_counts"] = dict(error_counts)
    return report


def write_json_report(report, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def write_csv_report(report, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for k, v in report.items():
            writer.writerow([k, v])


def main():
    report = aggregate_metrics()
    json_path = os.path.join(REPORTS_DIR, "pipeline_report.json")
    csv_path = os.path.join(REPORTS_DIR, "pipeline_report.csv")
    write_json_report(report, json_path)
    write_csv_report(report, csv_path)
    logging.info(f"Pipeline report generated: {json_path}, {csv_path}")


if __name__ == "__main__":
    main()
