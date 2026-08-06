#!/usr/bin/env python3
"""Asset consolidation script.

Two modes:

* ``--format legacy`` (default, back-compat): manifest-driven file copier that
  consolidates training configs, datasets, models, pipelines, and
  infrastructure into ``ai/training_ready/`` using symlinks for large files.
  Reads ``ai/training_ready/TRAINING_MANIFEST.json`` produced by
  ``generate_manifest.py``.

* ``--format v7``: V7 MASTER Dataset consolidation (PIX-4241).  Reads JSONL
  training records from one or more input directories, applies SHA-256
  exact-hash deduplication (preserving every record flagged
  ``is_training_edge_case: true``), optional Jaccard near-dedup, ChatML
  boundary verification, and writes a single sharded
  ``ai/training/output/v7/V7_MASTER.jsonl`` plus a ``dedup_report.json``.

The V7 path reuses the proven helpers in ``training.dedup_normalize`` so the
dedup semantics stay identical to the existing normalization pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Make ``training.*`` imports work when this script is run directly.
# Script is at ai/training/scripts/ — go up 2 levels so ``training`` package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training.dedup_normalize import (  # noqa: E402
    ProcessingContext,
    _attempt_reformat,
    _content_hash,
    _extract_text,
    _is_edge_case,
    _jaccard_similarity,
    _token_set,
    _verify_chatml_boundary,
    process_file,
)

logger = logging.getLogger("consolidate_assets")

# Size threshold for using symlinks (100 MB) — legacy mode only.
SYMLINK_THRESHOLD = 100 * 1024 * 1024


# ---------------------------------------------------------------------------
# Legacy manifest-driven consolidation (unchanged behavior).
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load training manifest."""
    with open(manifest_path, "r") as f:
        return json.load(f)


def consolidate_configs(manifest: dict, target_dir: Path, base_path: Path) -> int:
    """Consolidate training configs."""
    configs = manifest.get("training_configurations", [])
    consolidated = 0

    print(f"📋 Consolidating {len(configs)} config files...")

    for config in configs:
        source_path = Path(config.get("path", ""))
        if not source_path.exists():
            continue

        config_type = config.get("type", "other")
        if config_type == "stage":
            target_subdir = target_dir / "stage_configs"
        elif config_type == "model":
            target_subdir = target_dir / "model_configs"
        elif config_type == "hyperparameter":
            target_subdir = target_dir / "hyperparameters"
        elif config_type == "infrastructure":
            target_subdir = target_dir / "infrastructure"
        else:
            target_subdir = target_dir / "stage_configs"

        target_subdir.mkdir(parents=True, exist_ok=True)
        target_file = target_subdir / source_path.name

        if target_file.exists():
            dir_prefix = source_path.parent.name
            target_file = target_subdir / f"{dir_prefix}_{source_path.name}"

        try:
            shutil.copy2(source_path, target_file)
            consolidated += 1
        except Exception as e:
            print(f"  ⚠️  Failed to copy {source_path.name}: {e}")

    print(f"  ✅ Consolidated {consolidated} config files")
    return consolidated


def consolidate_datasets(manifest: dict, target_dir: Path, base_path: Path, use_symlinks: bool = True) -> int:
    """Consolidate datasets with symlink strategy for large files."""
    datasets = manifest.get("datasets", [])
    consolidated = 0
    symlinked = 0

    print(f"📊 Consolidating {len(datasets)} datasets...")

    for dataset in datasets:
        source_path = Path(dataset.get("path", ""))
        if not source_path.exists():
            continue

        stage = dataset.get("stage", "unassigned")
        if stage == "unassigned":
            stage = "stage1_foundation"

        target_subdir = target_dir / stage
        target_subdir.mkdir(parents=True, exist_ok=True)
        target_file = target_subdir / source_path.name

        if target_file.exists():
            dir_prefix = source_path.parent.name
            target_file = target_subdir / f"{dir_prefix}_{source_path.name}"

        file_size = dataset.get("size", 0)
        use_symlink = use_symlinks and file_size > SYMLINK_THRESHOLD

        try:
            if use_symlink:
                if target_file.exists():
                    target_file.unlink()
                target_file.symlink_to(source_path.resolve())
                symlinked += 1
            else:
                shutil.copy2(source_path, target_file)
            consolidated += 1
        except Exception as e:
            print(f"  ⚠️  Failed to {'symlink' if use_symlink else 'copy'} {source_path.name}: {e}")

    print(f"  ✅ Consolidated {consolidated} datasets ({symlinked} symlinked)")
    return consolidated


def consolidate_models(manifest: dict, target_dir: Path, base_path: Path) -> int:
    """Consolidate model architectures."""
    models = manifest.get("model_architectures", [])
    consolidated = 0

    print(f"🤖 Consolidating {len(models)} model architectures...")

    for model in models:
        source_path = Path(model.get("path", ""))
        if not source_path.exists():
            continue

        model_type = model.get("type", "other")
        if model_type == "moe":
            target_subdir = target_dir / "moe"
        elif model_type == "experimental":
            target_subdir = target_dir / "experimental"
        else:
            target_subdir = target_dir / "base"

        target_subdir.mkdir(parents=True, exist_ok=True)
        target_file = target_subdir / source_path.name

        if target_file.exists():
            dir_prefix = source_path.parent.name
            target_file = target_subdir / f"{dir_prefix}_{source_path.name}"

        try:
            shutil.copy2(source_path, target_file)
            consolidated += 1
        except Exception as e:
            print(f"  ⚠️  Failed to copy {source_path.name}: {e}")

    print(f"  ✅ Consolidated {consolidated} model files")
    return consolidated


def consolidate_pipelines(manifest: dict, target_dir: Path, base_path: Path) -> int:
    """Consolidate pipeline components."""
    pipelines = manifest.get("pipelines", [])
    consolidated = 0

    print(f"🔧 Consolidating {len(pipelines)} pipeline components...")

    for pipeline in pipelines:
        source_path = Path(pipeline.get("path", ""))
        if not source_path.exists():
            continue

        pipeline_type = pipeline.get("type", "integrated")
        if pipeline_type == "edge":
            target_subdir = target_dir / "edge"
        elif pipeline_type == "voice":
            target_subdir = target_dir / "voice"
        else:
            target_subdir = target_dir / "integrated"

        target_subdir.mkdir(parents=True, exist_ok=True)
        target_file = target_subdir / source_path.name

        if target_file.exists():
            dir_prefix = source_path.parent.name
            target_file = target_subdir / f"{dir_prefix}_{source_path.name}"

        try:
            shutil.copy2(source_path, target_file)
            consolidated += 1
        except Exception as e:
            print(f"  ⚠️  Failed to copy {source_path.name}: {e}")

    print(f"  ✅ Consolidated {consolidated} pipeline files")
    return consolidated


def consolidate_infrastructure(manifest: dict, target_dir: Path, base_path: Path) -> int:
    """Consolidate infrastructure configs."""
    infrastructure = manifest.get("infrastructure", [])
    consolidated = 0

    print(f"🏗️  Consolidating {len(infrastructure)} infrastructure configs...")

    for infra in infrastructure:
        source_path = Path(infra.get("path", ""))
        if not source_path.exists():
            continue

        infra_type = infra.get("type", "kubernetes")
        if infra_type == "helm":
            target_subdir = target_dir / "helm"
        elif infra_type == "docker":
            target_subdir = target_dir / "docker"
        else:
            target_subdir = target_dir / "kubernetes"

        target_subdir.mkdir(parents=True, exist_ok=True)
        target_file = target_subdir / source_path.name

        if target_file.exists():
            dir_prefix = source_path.parent.name
            target_file = target_subdir / f"{dir_prefix}_{source_path.name}"

        try:
            shutil.copy2(source_path, target_file)
            consolidated += 1
        except Exception as e:
            print(f"  ⚠️  Failed to copy {source_path.name}: {e}")

    print(f"  ✅ Consolidated {consolidated} infrastructure files")
    return consolidated


def run_legacy_consolidation() -> int:
    """Run the original manifest-driven consolidation."""
    base_path = Path.cwd()
    training_ready = base_path / "ai" / "training_ready"
    manifest_path = training_ready / "TRAINING_MANIFEST.json"

    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        print("Please run generate_manifest.py first")
        return 1

    print("🚀 Starting asset consolidation...\n")

    manifest = load_manifest(manifest_path)

    consolidate_configs(manifest, training_ready / "configs", base_path)
    consolidate_datasets(manifest, training_ready / "datasets", base_path, use_symlinks=True)
    consolidate_models(manifest, training_ready / "models", base_path)
    consolidate_pipelines(manifest, training_ready / "pipelines", base_path)
    consolidate_infrastructure(manifest, training_ready / "infrastructure", base_path)

    print("\n✅ Asset consolidation complete!")
    return 0


# ---------------------------------------------------------------------------
# V7 MASTER Dataset consolidation (PIX-4241).
# ---------------------------------------------------------------------------


def _to_chatml(record: dict) -> dict:
    """Normalize a record to ChatML ``{messages, ...}`` shape.

    Preserves ``scenario``, ``provenance``, ``is_training_edge_case``, and
    arbitrary ``metadata`` so downstream tooling keeps full provenance.
    """
    if "messages" in record:
        out: dict[str, Any] = {"messages": record["messages"]}
    else:
        # instruction/output → messages
        instruction = record.get("instruction", "")
        output = record.get("output", "")
        out = {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},
            ]
        }
        if not instruction or not output:
            # text → single user turn (rare; preserves data)
            text = record.get("text", "")
            if text:
                out = {"messages": [{"role": "user", "content": text}]}

    for k in ("scenario", "provenance", "is_training_edge_case", "metadata", "source_channel", "language"):
        if k in record:
            out[k] = record[k]

    return out


def _iter_jsonl(path: Path):
    """Yield parsed JSON records from a JSONL file, skipping blank/bad lines."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_no, json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in %s line %d", path, line_no)


def consolidate_v7(
    input_dirs: list[Path],
    output_dir: Path,
    jaccard_threshold: float = 0.92,
    near_dedup_window: int = 5000,
    shard_size: int = 10000,
) -> dict[str, Any]:
    """V7 MASTER Dataset consolidation.

    Reads every ``*.jsonl`` file under ``input_dirs`` (recursively), applies
    SHA-256 exact-hash dedup, optional Jaccard near-dedup (preserving all
    records flagged ``is_training_edge_case: true``), ChatML boundary
    verification, and writes ``V7_MASTER.jsonl`` (sharded when large) plus
    ``dedup_report.json`` to ``output_dir``.

    Returns the dedup report dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    edge_case_hashes: set[str] = set()
    token_sets: list[tuple[frozenset[str], str]] = []
    rejection_log: list[dict] = []

    kept: list[dict] = []
    total_in = 0
    exact_dupes = 0
    near_dupes = 0
    chatml_failures = 0
    reformatted = 0
    edge_preserved = 0
    files_processed = 0

    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.warning("Input directory not found: %s", input_path)
            continue
        for jsonl_file in sorted(input_path.rglob("*.jsonl")):
            files_processed += 1
            logger.info("V7 consolidate: processing %s", jsonl_file.name)
            ctx = ProcessingContext(
                seen_hashes=seen_hashes,
                edge_case_hashes=edge_case_hashes,
                token_sets=token_sets,
            )
            stats = process_file(
                jsonl_file,
                jaccard_threshold=jaccard_threshold,
                rejection_log=rejection_log,
                ctx=ctx,
                near_dedup_window=near_dedup_window,
            )
            kept.extend(stats.kept)
            total_in += stats.total_read
            exact_dupes += stats.exact_dupes
            near_dupes += stats.near_dupes
            chatml_failures += stats.chatml_failures
            reformatted += stats.reformatted
            edge_preserved += sum(1 for r in stats.kept if _is_edge_case(r))
            logger.info(
                "  %s: %d read, %d kept, %d exact, %d near dup, %d edge",
                jsonl_file.name,
                stats.total_read,
                len(stats.kept),
                stats.exact_dupes,
                stats.near_dupes,
                sum(1 for r in stats.kept if _is_edge_case(r)),
            )

    # Normalize every kept record to ChatML.
    chatml_records = [_to_chatml(r) for r in kept]

    # Write V7_MASTER.jsonl (sharded when the record count exceeds shard_size).
    if shard_size > 0 and len(chatml_records) > shard_size:
        shard_count = (len(chatml_records) + shard_size - 1) // shard_size
        for i in range(shard_count):
            shard = chatml_records[i * shard_size : (i + 1) * shard_size]
            shard_path = output_dir / f"V7_MASTER_{i:04d}.jsonl"
            with open(shard_path, "w", encoding="utf-8") as f:
                for rec in shard:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        master_path = output_dir / "V7_MASTER.jsonl"
        # Also write a single concatenated master for convenience.
        with open(master_path, "w", encoding="utf-8") as f:
            for rec in chatml_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        shard_count = 1
        master_path = output_dir / "V7_MASTER.jsonl"
        with open(master_path, "w", encoding="utf-8") as f:
            for rec in chatml_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if rejection_log:
        rej_path = output_dir / "rejection_log.jsonl"
        with open(rej_path, "w", encoding="utf-8") as f:
            for entry in rejection_log:
                f.write(json.dumps(entry) + "\n")

    report = {
        "schema": "v7_master",
        "generated_at": datetime.now(UTC).isoformat(),
        "input_dirs": [str(p) for p in input_dirs],
        "files_processed": files_processed,
        "total_samples_in": total_in,
        "exact_duplicates": exact_dupes,
        "near_duplicates": near_dupes,
        "chatml_failures": chatml_failures,
        "reformatted": reformatted,
        "edge_cases_preserved": edge_preserved,
        "total_samples_out": len(chatml_records),
        "shard_count": shard_count,
        "jaccard_threshold": jaccard_threshold,
        "near_dedup_window": near_dedup_window,
        "output_master": str(master_path),
    }
    report_path = output_dir / "dedup_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    logger.info(
        "V7 consolidation complete: %d in → %d out (%d exact dup, %d near dup, %d ChatML fail, %d edge preserved)",
        total_in,
        len(chatml_records),
        exact_dupes,
        near_dupes,
        chatml_failures,
        edge_preserved,
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolidate training assets. Use --format v7 for V7 MASTER Dataset consolidation (PIX-4241).",
    )
    parser.add_argument(
        "--format",
        choices=["legacy", "v7"],
        default="legacy",
        help="Consolidation format. 'legacy' = manifest-driven file copier (default). 'v7' = V7 MASTER Dataset JSONL consolidation.",
    )
    # V7-only args
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        default=None,
        help="V7: directories containing JSONL files to consolidate (recursive). Defaults to ai/training/output/.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="V7: output directory for V7_MASTER.jsonl and dedup_report.json. Defaults to ai/training/output/v7/.",
    )
    parser.add_argument(
        "--jaccard_threshold",
        type=float,
        default=0.92,
        help="V7: Jaccard similarity threshold for near-dedup (default 0.92).",
    )
    parser.add_argument(
        "--near_dedup_window",
        type=int,
        default=5000,
        help="V7: max prior token sets to compare for near-dedup (default 5000).",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=10000,
        help="V7: max records per output shard. 0 = no sharding (single V7_MASTER.jsonl). Default 10000.",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    if args.format == "legacy":
        return run_legacy_consolidation()

    # V7 mode
    if args.input_dirs:
        input_dirs = [Path(p) for p in args.input_dirs]
    else:
        input_dirs = [Path.cwd() / "ai" / "training" / "output"]
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path.cwd() / "ai" / "training" / "output" / "v7"

    report = consolidate_v7(
        input_dirs=input_dirs,
        output_dir=output_dir,
        jaccard_threshold=args.jaccard_threshold,
        near_dedup_window=args.near_dedup_window,
        shard_size=args.shard_size,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
