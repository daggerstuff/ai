"""Example CLI demonstrating how to use ai.hf_client.HuggingFaceClient.

This is intentionally tiny and dependency-free; it demonstrates best
practices for reading an API token from the environment and calling the
client in a script.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ai.hf_client import HuggingFaceClient


logger = logging.getLogger("ai.examples.download_model")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a file from the Hugging Face Hub")
    parser.add_argument("repo_id", help="Repo id, for example 'owner/model' or 'owner/dataset'")
    parser.add_argument("--filename", help="Filename to download (optional)")
    parser.add_argument("--out", help="Output directory", default=".")
    args = parser.parse_args()

    client = HuggingFaceClient()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.filename:
            path = client.download_file(args.repo_id, filename=args.filename, cache_dir=str(out_dir))
            print(f"Downloaded file to: {path}")
        else:
            path = client.snapshot_download(args.repo_id, cache_dir=str(out_dir))
            print(f"Downloaded repository snapshot to: {path}")
    except Exception as exc:
        logger.exception("Failed to download: %s", exc)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
