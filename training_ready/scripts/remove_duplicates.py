#!/usr/bin/env python3
"""
Remove Duplicates - Deduplicate datasets based on full scan results
Removes duplicate entries while preserving one canonical copy
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, TypedDict, Set, DefaultDict, Optional
from collections import defaultdict
from datetime import datetime

# Add project root to path
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]
sys.path.insert(0, str(project_root))

from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader
import logging

# Suppress verbose warnings
logging.getLogger('ai.training_ready.utils.s3_dataset_loader').setLevel(logging.ERROR)

DEFAULT_S3_BUCKET = 'pixel-data'
DEDUP_REPORT_PATH = project_root / "ai/training_ready/data/full_deduplication_report.json"


class DeduplicationPlan(TypedDict):
    """Plan for removing duplicates"""
    file_path: str
    entries_to_remove: List[int]  # Indices to remove
    entries_to_keep: int  # Count of entries to keep
    duplicates_removed: int


def load_deduplication_report(report_path: Path) -> Dict[str, Any]:
    """Load the full deduplication report"""
    if not report_path.exists():
        raise FileNotFoundError(f"Deduplication report not found: {report_path}")

    with open(report_path, 'r') as f:
        return json.load(f)


def analyze_duplicate_groups(
    duplicate_groups: Dict[str, Any],
    keep_strategy: str = 'first_dataset'
) -> Dict[str, List[DeduplicationPlan]]:
    """
    Analyze duplicate groups and create deduplication plan.

    Args:
        duplicate_groups: Duplicate groups from report
        keep_strategy: How to choose which copy to keep
            - 'first_dataset': Keep first dataset's copy
            - 'priority_order': Keep based on dataset priority
            - 'largest_file': Keep from largest file

    Returns:
        Dict mapping file path to list of deduplication plans
    """
    # Priority order for keeping duplicates (lower = higher priority)
    dataset_priority = {
        'phase_1_priority_conversations': 1,
        'phase_2_professional_datasets': 2,
        'phase_3_cot_reasoning': 3,
        'phase_4_reddit_mental_health': 4,
        'priority_complete_fixed': 5,
        'professional_complete_integration': 6,
        'professional_datasets_final': 7,
        'soulchat_complete': 8,
    }

    # Group duplicates by file
    file_plans: DefaultDict[str, DefaultDict[int, bool]] = defaultdict(lambda: defaultdict(bool))

    for hash_val, group in duplicate_groups.items():
        entries = group['entries']
        datasets = group['datasets']

        if len(entries) <= 1:
            continue

        # Determine which entry to keep
        if keep_strategy == 'priority_order':
            # Sort by priority, keep lowest priority (highest priority dataset)
            sorted_entries = sorted(
                entries,
                key=lambda e: dataset_priority.get(e['dataset'], 999)
            )
            keep_entry = sorted_entries[0]
            remove_entries = sorted_entries[1:]
        elif keep_strategy == 'first_dataset':
            # Keep first entry, remove rest
            keep_entry = entries[0]
            remove_entries = entries[1:]
        else:
            # Default: keep first
            keep_entry = entries[0]
            remove_entries = entries[1:]

        # Mark entries for removal
        for entry in remove_entries:
            file_key = entry['source_path']
            index = entry['entry_index']
            file_plans[file_key][index] = True

    # Convert to DeduplicationPlan format
    plans_by_file: Dict[str, DeduplicationPlan] = {}

    for file_path, indices_to_remove in file_plans.items():
        plans_by_file[file_path] = {
            'file_path': file_path,
            'entries_to_remove': sorted(indices_to_remove.keys()),
            'entries_to_keep': 0,  # Will be calculated
            'duplicates_removed': len(indices_to_remove)
        }

    # Group by category for reporting
    plans_by_category: DefaultDict[str, List[DeduplicationPlan]] = defaultdict(list)
    for plan in plans_by_file.values():
        # Extract category from path
        path_parts = plan['file_path'].split('/')
        category = 'unknown'
        for part in path_parts:
            if part in dataset_priority:
                category = part
                break
        plans_by_category[category].append(plan)

    return dict(plans_by_category)


def load_file_entries(loader: S3DatasetLoader, s3_path: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load all entries from a file.

    Returns:
        (entries, file_format) where file_format is 'json' or 'jsonl'
    """
    if s3_path.endswith('.json'):
        data = loader.load_json(s3_path)

        if isinstance(data, dict):
            if 'conversations' in data:
                return data['conversations'], 'json'
            elif 'data' in data:
                return data['data'], 'json'
            else:
                return [data], 'json'
        elif isinstance(data, list):
            return data, 'json'
        else:
            return [], 'json'

    elif s3_path.endswith('.jsonl'):
        entries = []
        try:
            for line in loader.stream_jsonl(s3_path):
                if line:
                    entries.append(line)
        except (UnicodeDecodeError, Exception) as e:
            print(f"     ⚠️  Encoding/streaming error: {e}")
            # Try alternative loading method
            try:
                bucket, key = loader._parse_s3_path(s3_path)
                response = loader.s3_client.get_object(Bucket=bucket, Key=key)
                content = response['Body'].read()
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        text = content.decode(encoding)
                        for line in text.splitlines():
                            line = line.strip()
                            if line:
                                entries.append(json.loads(line))
                        break
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
            except Exception as e2:
                print(f"     ❌ Failed to load file: {e2}")
        return entries, 'jsonl'

    return [], 'unknown'


def remove_duplicates_from_file(
    loader: S3DatasetLoader,
    plan: DeduplicationPlan,
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    Remove duplicate entries from a file based on plan.

    Returns:
        Result dict with stats
    """
    s3_path = plan['file_path']
    indices_to_remove = set(plan['entries_to_remove'])

    print(f"\n  📝 Processing: {Path(s3_path).name}")
    print(f"     Removing {len(indices_to_remove)} duplicate entries...")

    # Load file
    entries, file_format = load_file_entries(loader, s3_path)

    if not entries:
        return {
            'success': False,
            'error': 'No entries loaded',
            'removed': 0,
            'kept': 0
        }

    original_count = len(entries)

    # Remove duplicates (in reverse order to maintain indices)
    kept_entries = [
        entry for idx, entry in enumerate(entries)
        if idx not in indices_to_remove
    ]

    removed_count = original_count - len(kept_entries)

    if dry_run:
        print(f"     [DRY RUN] Would remove {removed_count} entries, keep {len(kept_entries)}")
        return {
            'success': True,
            'dry_run': True,
            'removed': removed_count,
            'kept': len(kept_entries),
            'original': original_count
        }

    # Save back to S3
    try:
        bucket, key = loader._parse_s3_path(s3_path)

        if file_format == 'json':
            # Determine structure
            original_data = loader.load_json(s3_path)
            if isinstance(original_data, dict):
                if 'conversations' in original_data:
                    original_data['conversations'] = kept_entries
                elif 'data' in original_data:
                    original_data['data'] = kept_entries
                else:
                    original_data = kept_entries[0] if len(kept_entries) == 1 else kept_entries

                body = json.dumps(original_data, indent=2, ensure_ascii=False).encode('utf-8')
            else:
                body = json.dumps(kept_entries, indent=2, ensure_ascii=False).encode('utf-8')
        else:  # jsonl
            body = '\n'.join(json.dumps(entry, ensure_ascii=False) for entry in kept_entries).encode('utf-8')

        loader.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType='application/json' if file_format == 'json' else 'application/x-ndjson'
        )

        print(f"     ✅ Removed {removed_count} duplicates, kept {len(kept_entries)} entries")

        return {
            'success': True,
            'dry_run': False,
            'removed': removed_count,
            'kept': len(kept_entries),
            'original': original_count
        }

    except Exception as e:
        print(f"     ❌ Error: {e}")
        return {
            'success': False,
            'error': str(e),
            'removed': 0,
            'kept': original_count
        }


def print_deduplication_summary(plans_by_category: Dict[str, List[DeduplicationPlan]]) -> None:
    """Print summary of deduplication plan"""
    print("\n" + "=" * 80)
    print("📋 DEDUPLICATION PLAN")
    print("=" * 80)

    total_files = 0
    total_duplicates = 0

    for category, plans in sorted(plans_by_category.items()):
        category_duplicates = sum(p['duplicates_removed'] for p in plans)
        total_files += len(plans)
        total_duplicates += category_duplicates

        print(f"\n📁 {category}:")
        print(f"   Files to deduplicate: {len(plans)}")
        print(f"   Duplicates to remove: {category_duplicates:,}")

        # Show top files
        top_files = sorted(plans, key=lambda p: p['duplicates_removed'], reverse=True)[:5]
        for plan in top_files:
            filename = Path(plan['file_path']).name
            print(f"     - {filename}: {plan['duplicates_removed']} duplicates")

    print(f"\n📊 Total:")
    print(f"   Files: {total_files}")
    print(f"   Duplicates to remove: {total_duplicates:,}")


def main() -> None:
    """Main deduplication removal function"""
    import argparse

    parser = argparse.ArgumentParser(description='Remove duplicates from datasets')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without actually removing'
    )
    parser.add_argument(
        '--category',
        type=str,
        help='Only process specific category (e.g., phase_1_priority_conversations)'
    )
    parser.add_argument(
        '--keep-strategy',
        choices=['first_dataset', 'priority_order'],
        default='priority_order',
        help='Strategy for choosing which duplicate to keep'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Skip confirmation prompt (use with caution!)'
    )

    args = parser.parse_args()

    print("🗑️  Dataset Deduplication Removal")
    print("=" * 80)

    if args.dry_run:
        print("⚠️  DRY RUN MODE - No files will be modified")
    else:
        print("⚠️  LIVE MODE - Files will be modified in S3!")

    # Load deduplication report
    print(f"\n📋 Loading deduplication report...")
    report = load_deduplication_report(DEDUP_REPORT_PATH)

    duplicate_groups = report.get('duplicate_groups', {})
    print(f"   Found {len(duplicate_groups)} duplicate groups")

    # Create deduplication plan
    print(f"\n📊 Creating deduplication plan (strategy: {args.keep_strategy})...")
    plans_by_category = analyze_duplicate_groups(duplicate_groups, args.keep_strategy)

    # Filter by category if specified
    if args.category:
        if args.category in plans_by_category:
            plans_by_category = {args.category: plans_by_category[args.category]}
            print(f"   Filtered to category: {args.category}")
        else:
            print(f"❌ Category '{args.category}' not found in plans")
            return

    # Print summary
    print_deduplication_summary(plans_by_category)

    # Confirm before proceeding
    if not args.dry_run and not args.confirm:
        print("\n" + "=" * 80)
        response = input("⚠️  Proceed with deduplication? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Cancelled")
            return

    # Initialize S3 loader
    loader = S3DatasetLoader(bucket=DEFAULT_S3_BUCKET)

    # Process files
    print("\n🔧 Processing files...")
    results = []

    for category, plans in sorted(plans_by_category.items()):
        print(f"\n📁 Category: {category}")

        for plan in plans:
            result = remove_duplicates_from_file(loader, plan, dry_run=args.dry_run)
            result['category'] = category
            result['file'] = Path(plan['file_path']).name
            results.append(result)

    # Print final summary
    print("\n" + "=" * 80)
    print("📊 DEDUPLICATION RESULTS")
    print("=" * 80)

    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]

    total_removed = sum(r.get('removed', 0) for r in successful)
    total_kept = sum(r.get('kept', 0) for r in successful)

    print(f"\n✅ Successful: {len(successful)} files")
    print(f"❌ Failed: {len(failed)} files")
    print(f"📊 Total duplicates removed: {total_removed:,}")
    print(f"📊 Total entries kept: {total_kept:,}")

    if failed:
        print(f"\n❌ Failed files:")
        for result in failed:
            print(f"   - {result.get('file')}: {result.get('error', 'Unknown error')}")

    # Save results
    results_path = project_root / "ai/training_ready/data/deduplication_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'dry_run': args.dry_run,
            'keep_strategy': args.keep_strategy,
            'category_filter': args.category,
            'results': results,
            'summary': {
                'successful': len(successful),
                'failed': len(failed),
                'total_removed': total_removed,
                'total_kept': total_kept
            }
        }, f, indent=2)

    print(f"\n💾 Results saved to: {results_path}")
    print("\n✅ Deduplication complete!")


if __name__ == '__main__':
    main()
