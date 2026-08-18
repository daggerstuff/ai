#!/usr/bin/env python3
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--source-type")
    args = parser.parse_args()

    with open(args.file) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if "provenance" not in record:
                continue
            if args.source_type is not None and record["provenance"].get("source_type") != args.source_type:
                continue


if __name__ == "__main__":
    main()
