import argparse
import json

import pandas as pd


def csv_to_jsonl(input_csv, output_jsonl, limit=5000):
    # pandas can handle indexing and basic loading
    df = pd.read_csv(input_csv)

    if limit and limit < len(df):
        df = df.sample(limit, random_state=42)

    with open(output_jsonl, "w") as f:
        for idx, row in df.iterrows():
            # Standardizing format for multi-agent system
            # The system expects 'data' field to work with
            record = {
                "id": f"reddit_suicide_{idx}",
                "text": row["text"],
                "metadata": {
                    "source_class": row["class"],
                    "source_family": "reddit_suicide_detection",
                    "original_id": str(row.get("Unnamed: 0", idx)),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Reddit CSV to JSONL for annotation")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument("--limit", type=int, default=5000, help="Max rows to process")

    args = parser.parse_args()
    csv_to_jsonl(args.input, args.output, args.limit)
