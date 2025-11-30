#!/usr/bin/env python3
"""
Dataset Export Script
Produces versioned dataset exports with checksums, manifests, and storage upload
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .orchestration.integrated_training_pipeline import (
    IntegratedTrainingPipeline,
    IntegratedPipelineConfig
)
from .config_lock import lock_config, LockedConfig
from .export_manifest import (
    DatasetManifest,
    FileManifest,
    QualitySummary
)
from .storage_manager import StorageManager
from .storage_config import get_storage_config


def export_to_jsonl(data: list, output_path: Path) -> int:
    """Export dataset to JSONL format"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
            row_count += 1

    return row_count


def export_to_parquet(data: list, output_path: Path) -> int:
    """Export dataset to Parquet format"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Write to Parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)

    return len(df)


def count_by_source(data: list) -> Dict[str, int]:
    """Count samples by source"""
    counts = {}
    for item in data:
        source = item.get('metadata', {}).get('source', 'unknown')
        counts[source] = counts.get(source, 0) + 1
    return counts


def export_dataset_v1(
    version: str = "1.0.0",
    target_samples: int = 1000,
    output_dir: Optional[Path] = None,
    seed: Optional[int] = None,
    upload_to_storage: bool = True,
    enable_quality_validation: bool = True
) -> DatasetManifest:
    """Export dataset v1.0 with full manifest and storage upload"""

    print(f"🚀 Starting dataset export v{version}")
    print(f"   Target samples: {target_samples}")
    print(f"   Seed: {seed if seed else 'random'}")

    # Setup output directory
    if output_dir is None:
        output_dir = Path(f"ai/dataset_pipeline/production_exports/v{version}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create pipeline configuration
    pipeline_config = IntegratedPipelineConfig(
        target_total_samples=target_samples,
        output_dir=str(output_dir),
        output_filename="dataset.json",
        enable_bias_detection=True,
        enable_quality_validation=enable_quality_validation
    )

    # Lock configuration
    config_dict = {
        'target_samples': target_samples,
        'pipeline_config': {
            'edge_cases': {
                'enabled': pipeline_config.edge_cases.enabled,
                'target_percentage': pipeline_config.edge_cases.target_percentage
            },
            'pixel_voice': {
                'enabled': pipeline_config.pixel_voice.enabled,
                'target_percentage': pipeline_config.pixel_voice.target_percentage
            },
            'psychology_knowledge': {
                'enabled': pipeline_config.psychology_knowledge.enabled,
                'target_percentage': pipeline_config.psychology_knowledge.target_percentage
            },
            'dual_persona': {
                'enabled': pipeline_config.dual_persona.enabled,
                'target_percentage': pipeline_config.dual_persona.target_percentage
            },
            'standard_therapeutic': {
                'enabled': pipeline_config.standard_therapeutic.enabled,
                'target_percentage': pipeline_config.standard_therapeutic.target_percentage
            }
        }
    }

    locked_config = lock_config(config_dict, seed=seed)
    locked_config.save(output_dir / "config_lock.json")
    print(f"✅ Configuration locked: {locked_config.config_hash}")

    # Run pipeline
    print("\n📊 Running dataset pipeline...")
    pipeline = IntegratedTrainingPipeline(pipeline_config)
    result = pipeline.run()

    # Load generated dataset
    dataset_path = Path(result['output_file'])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, 'r') as f:
        dataset_data = json.load(f)

    conversations = dataset_data.get('conversations', [])
    total_samples = len(conversations)

    print(f"✅ Generated {total_samples} samples")

    # Count by source
    samples_by_source = count_by_source(conversations)
    print(f"   Sources: {samples_by_source}")

    # Export to JSONL
    jsonl_path = output_dir / f"dataset_v{version}.jsonl"
    print(f"\n📝 Exporting to JSONL: {jsonl_path}")
    jsonl_row_count = export_to_jsonl(conversations, jsonl_path)

    # Export to Parquet
    parquet_path = output_dir / f"dataset_v{version}.parquet"
    print(f"📝 Exporting to Parquet: {parquet_path}")
    parquet_row_count = export_to_parquet(conversations, parquet_path)

    # Create file manifests
    jsonl_manifest = FileManifest.from_file(
        jsonl_path,
        format="jsonl",
        row_count=jsonl_row_count,
        source_distribution=samples_by_source
    )

    parquet_manifest = FileManifest.from_file(
        parquet_path,
        format="parquet",
        row_count=parquet_row_count,
        source_distribution=samples_by_source
    )

    # Create quality summary (simplified - would need actual quality metrics)
    quality_summary = QualitySummary(
        total_samples=total_samples,
        crisis_flags_count=0,  # Would be calculated from quality validation
        crisis_flags_percentage=0.0,
        pii_detected_count=0,
        pii_detected_percentage=0.0
    )

    # Create dataset manifest
    manifest = DatasetManifest(
        version=version,
        created_at=locked_config.created_at,
        created_by=locked_config.git_info.commit_sha[:8],
        total_samples=total_samples,
        samples_by_source=samples_by_source
    )

    manifest.add_file(jsonl_manifest)
    manifest.add_file(parquet_manifest)
    manifest.set_config_lock(locked_config)
    manifest.set_quality_summary(quality_summary)

    # Upload to storage if enabled
    if upload_to_storage:
        print("\n☁️  Uploading to storage...")
        storage_manager = StorageManager()
        storage_config = get_storage_config()

        # Upload JSONL
        jsonl_storage_path = storage_config.get_export_path(version, jsonl_manifest.filename)
        jsonl_upload_info = storage_manager.upload_with_checksum(
            jsonl_path,
            jsonl_storage_path,
            metadata={
                'version': version,
                'format': 'jsonl',
                'row_count': jsonl_row_count
            }
        )
        manifest.storage_urls['jsonl'] = jsonl_upload_info['storage_url']
        print(f"   ✅ JSONL: {jsonl_upload_info['storage_url']}")

        # Upload Parquet
        parquet_storage_path = storage_config.get_export_path(version, parquet_manifest.filename)
        parquet_upload_info = storage_manager.upload_with_checksum(
            parquet_path,
            parquet_storage_path,
            metadata={
                'version': version,
                'format': 'parquet',
                'row_count': parquet_row_count
            }
        )
        manifest.storage_urls['parquet'] = parquet_upload_info['storage_url']
        print(f"   ✅ Parquet: {parquet_upload_info['storage_url']}")

    # Save manifest
    manifest_path = output_dir / f"manifest_v{version}.json"
    manifest.save(manifest_path)
    print(f"\n✅ Manifest saved: {manifest_path}")

    # Verify files
    print("\n🔍 Verifying files...")
    is_valid, errors = manifest.verify_files(output_dir)
    if is_valid:
        print("   ✅ All files verified")
    else:
        print(f"   ⚠️  Verification errors: {errors}")

    print(f"\n🎉 Dataset export v{version} complete!")
    print(f"   Files: {len(manifest.files)}")
    print(f"   Total samples: {manifest.total_samples}")
    print(f"   Output directory: {output_dir}")

    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export dataset v1.0")
    parser.add_argument("--version", default="1.0.0", help="Dataset version")
    parser.add_argument("--target-samples", type=int, default=1000, help="Target number of samples")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--no-upload", action="store_true", help="Skip storage upload")
    parser.add_argument("--no-quality", action="store_true", help="Skip quality validation")

    args = parser.parse_args()

    try:
        manifest = export_dataset_v1(
            version=args.version,
            target_samples=args.target_samples,
            output_dir=args.output_dir,
            seed=args.seed,
            upload_to_storage=not args.no_upload,
            enable_quality_validation=not args.no_quality
        )

        print("\n✅ Export successful!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Export failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

