#!/usr/bin/env python3
"""Batch extraction script for GitHub repos identified in the hackathon research map.

Reads the github_hackathon_map.json produced by PIX-4238, processes the top-N
repos using the GitHubRepoAdapter, and outputs standardized ChatML JSONL files.

Usage:
    uv run python -m ai.pipelines.data_processing.scripts.extract_github_repos [options]

Options:
    --map PATH        Path to github_hackathon_map.json (default: ai/training/research/github_hackathon_map.json)
    --output-dir DIR  Output directory for converted JSONL (default: ai/data/raw/github_extracted)
    --top N           Process only the top N repos by score (default: 10)
    --repo NAME       Process a specific repo by full_name (e.g., "owner/repo")
    --dry-run         List repos that would be processed without downloading
    --list            List repos in the map and exit
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from ai.pipelines.data_processing.dataset_adapters.github_repo_adapter import GitHubRepoAdapter

_DEFAULT_MAP = "ai/training/research/github_hackathon_map.json"
_DEFAULT_OUTPUT = "ai/data/raw/github_extracted"


def load_repo_map(map_path: Path) -> list[dict]:
    """Load and return repos from the hackathon map, sorted by score."""
    with open(map_path, encoding="utf-8") as f:
        data = json.load(f)

    repos = data.get("repos", [])
    repos.sort(key=lambda r: r.get("score", 0), reverse=True)
    return repos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract data from GitHub repos identified in the hackathon research map",
    )
    parser.add_argument("--map", default=_DEFAULT_MAP, help="Path to github_hackathon_map.json")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--top", type=int, default=10, help="Process only top N repos by score")
    parser.add_argument("--repo", default=None, help="Process a specific repo by full_name")
    parser.add_argument("--dry-run", action="store_true", help="List repos without downloading")
    parser.add_argument("--list", action="store_true", help="List repos and exit")
    args = parser.parse_args()

    map_path = Path(args.map)
    if not map_path.exists():
        print(f"Error: Map file not found: {map_path}", file=sys.stderr)
        sys.exit(1)

    repos = load_repo_map(map_path)
    print(f"Loaded {len(repos)} repos from {map_path}")

    if args.repo:
        repos = [r for r in repos if r["full_name"] == args.repo]
        if not repos:
            print(f"Repo '{args.repo}' not found in map.", file=sys.stderr)
            sys.exit(1)
    else:
        repos = repos[: args.top]

    if args.list:
        print(f"\n{'Score':>6}  {'Repo':<60} {'Categories':<30}")
        print(f"{'─' * 6}  {'─' * 60} {'─' * 30}")
        for r in repos:
            cats = ", ".join(r.get("matched_categories", []))
            print(f"{r.get('score', 0):>6.1f}  {r['full_name']:<60} {cats:<30}")
        return

    if args.dry_run:
        print(f"\nWould process {len(repos)} repos:")
        for r in repos:
            size_kb = r.get("size_kb", 0)
            print(f"  [{r.get('score', 0):.1f}] {r['full_name']} ({size_kb} KB)")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  GitHub Repo Extraction Pipeline — {len(repos)} repo(s)")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 70}\n")

    succeeded: list[tuple[str, int]] = []
    failed: list[tuple[str, str]] = []
    skipped: list[str] = []

    for repo in repos:
        full_name = repo["full_name"]
        score = repo.get("score", 0)
        branch = repo.get("default_branch", "main")
        license_name = repo.get("license", "")
        categories = repo.get("matched_categories", [])
        size_kb = repo.get("size_kb", 0)

        print(f"--- [{score:.1f}] {full_name} ({size_kb} KB) ---")
        print(f"  Categories: {', '.join(categories)}")
        print(f"  License: {license_name or 'none'} | Branch: {branch}")

        # Safe dataset name: owner_repo (sanitized)
        safe_name = full_name.replace("/", "_")
        if not safe_name.isidentifier():
            safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in safe_name)

        try:
            adapter = GitHubRepoAdapter(
                safe_name,
                output_dir,
                repo_full_name=full_name,
                branch=branch,
                license_name=license_name,
                matched_categories=categories,
            )
            result = adapter.run()
            if result and result.exists():
                count = sum(1 for _ in open(result, encoding="utf-8"))
                if count > 0:
                    print(f"  OK: {count} records -> {result}")
                    succeeded.append((full_name, count))
                else:
                    print("  SKIP: 0 records (no recognized data files)")
                    skipped.append(full_name)
            else:
                print("  SKIP: no output")
                skipped.append(full_name)
        except Exception as e:
            print(f"  FAIL: {e}")
            failed.append((full_name, str(e)))
            traceback.print_exc()
        print()

    print(f"{'=' * 70}")
    print(f"  Summary: {len(succeeded)} succeeded, {len(skipped)} skipped, {len(failed)} failed")
    if succeeded:
        total_records = sum(c for _, c in succeeded)
        print(f"  Total records extracted: {total_records}")
        for name, count in succeeded:
            print(f"    {name}: {count} records")
    if skipped:
        print(f"  Skipped (no data found): {', '.join(skipped)}")
    if failed:
        print(f"  Failed: {', '.join(n for n, _ in failed)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
