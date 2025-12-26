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


def _load_json_list(file_path):
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    return []


def _count_errors_in_logs():
    error_counts = defaultdict(int)
    for log_file in glob.glob(os.path.join(LOGS_DIR, "*.log")):
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                if "[ERROR]" in line:
                    error_counts[os.path.basename(log_file)] += 1
    return dict(error_counts)


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
    cluster_file = os.path.join(
        DATA_DIRS["clusters"], "personality_emotion_clusters.json"
    )
    clusters = _load_json_list(cluster_file)
    if clusters:
        cluster_counts = Counter([c.get("cluster") for c in clusters])
        report["cluster_counts"] = dict(cluster_counts)
        report["total_clustered_segments"] = len(clusters)

    # Dialogue pairs
    pair_file = os.path.join(DATA_DIRS["pairs"], "dialogue_pairs.json")
    pairs = _load_json_list(pair_file)
    if pairs:
        report["dialogue_pairs"] = len(pairs)

    validated_file = os.path.join(DATA_DIRS["pairs"], "dialogue_pairs_validated.json")
    validated = _load_json_list(validated_file)
    if validated:
        report["validated_pairs"] = len(validated)

    # Therapeutic pairs
    therapeutic_file = os.path.join(DATA_DIRS["therapeutic"], "therapeutic_pairs.json")
    therapeutic = _load_json_list(therapeutic_file)
    if therapeutic:
        report["therapeutic_pairs"] = len(therapeutic)

    # Error summary from logs
    report["error_counts"] = _count_errors_in_logs()

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
