#!/usr/bin/env python3
"""
Fix Encoding Issues - Detect and fix encoding problems in S3 datasets
Converts files to UTF-8 encoding
"""

import json
import sys
import chardet
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from io import BytesIO
from datetime import datetime

# Add project root to path
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]
sys.path.insert(0, str(project_root))

from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader
import logging

logging.getLogger('ai.training_ready.utils.s3_dataset_loader').setLevel(logging.ERROR)

DEFAULT_S3_BUCKET = 'pixel-data'


def detect_encoding(content: bytes) -> Tuple[str, float]:
    """
    Detect encoding of content.

    Returns:
        (encoding, confidence)
    """
    result = chardet.detect(content)
    return result.get('encoding', 'utf-8'), result.get('confidence', 0.0)


def try_decode(content: bytes, encodings: List[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to decode content with various encodings.

    Returns:
        (decoded_text, encoding_used) or (None, None) if all fail
    """
    if encodings is None:
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'windows-1252', 'utf-16']

    for encoding in encodings:
        try:
            text = content.decode(encoding)
            return text, encoding
        except (UnicodeDecodeError, LookupError):
            continue

    return None, None


def fix_jsonl_file(loader: S3DatasetLoader, s3_path: str, dry_run: bool = True) -> Dict[str, Any]:
    """
    Fix encoding issues in a JSONL file.

    Returns:
        Result dict with stats
    """
    bucket, key = loader._parse_s3_path(s3_path)
    filename = Path(key).name

    print(f"\n  📝 Processing: {filename}")

    try:
        # Download file
        response = loader.s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read()

        # Detect encoding
        detected_encoding, confidence = detect_encoding(content)
        print(f"     Detected encoding: {detected_encoding} (confidence: {confidence:.2%})")

        # Try to decode
        text, encoding_used = try_decode(content)

        if text is None:
            return {
                'success': False,
                'error': 'Could not decode with any known encoding',
                'detected_encoding': detected_encoding
            }

        if encoding_used != 'utf-8':
            print(f"     ⚠️  File is {encoding_used}, converting to UTF-8...")
        else:
            # Check if it's actually valid UTF-8
            try:
                content.decode('utf-8')
                print("     ✅ Already UTF-8")
                return {
                    'success': True,
                    'skipped': True,
                    'reason': 'Already UTF-8',
                    'encoding': 'utf-8'
                }
            except UnicodeDecodeError:
                print(f"     ⚠️  UTF-8 detection failed, but content decoded as {encoding_used}")

        # Parse JSONL lines
        lines = text.splitlines()
        valid_entries = []
        invalid_lines = []

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                valid_entries.append(entry)
            except json.JSONDecodeError as e:
                invalid_lines.append({
                    'line': line_num,
                    'error': str(e),
                    'preview': line[:100]
                })
                print(f"     ⚠️  Line {line_num}: JSON decode error - {e}")

        if invalid_lines:
            print(f"     ⚠️  {len(invalid_lines)} invalid JSON lines found")

        if dry_run:
            print(f"     [DRY RUN] Would convert {len(valid_entries)} entries to UTF-8")
            return {
                'success': True,
                'dry_run': True,
                'original_encoding': encoding_used,
                'entries_count': len(valid_entries),
                'invalid_lines': len(invalid_lines)
            }

        # Re-encode as UTF-8 JSONL
        utf8_content = '\n'.join(
            json.dumps(entry, ensure_ascii=False) for entry in valid_entries
        ).encode('utf-8')

        # Upload back to S3
        loader.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=utf8_content,
            ContentType='application/x-ndjson'
        )

        print(f"     ✅ Converted {len(valid_entries)} entries to UTF-8")

        return {
            'success': True,
            'dry_run': False,
            'original_encoding': encoding_used,
            'entries_count': len(valid_entries),
            'invalid_lines': len(invalid_lines),
            'bytes_before': len(content),
            'bytes_after': len(utf8_content)
        }

    except Exception as e:
        print(f"     ❌ Error: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def fix_json_file(loader: S3DatasetLoader, s3_path: str, dry_run: bool = True) -> Dict[str, Any]:
    """
    Fix encoding issues in a JSON file.

    Returns:
        Result dict with stats
    """
    bucket, key = loader._parse_s3_path(s3_path)
    filename = Path(key).name

    print(f"\n  📝 Processing: {filename}")

    try:
        # Download file
        response = loader.s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read()

        # Detect encoding
        detected_encoding, confidence = detect_encoding(content)
        print(f"     Detected encoding: {detected_encoding} (confidence: {confidence:.2%})")

        # Try to decode
        text, encoding_used = try_decode(content)

        if text is None:
            return {
                'success': False,
                'error': 'Could not decode with any known encoding',
                'detected_encoding': detected_encoding
            }

        if encoding_used != 'utf-8':
            print(f"     ⚠️  File is {encoding_used}, converting to UTF-8...")
        else:
            try:
                content.decode('utf-8')
                print("     ✅ Already UTF-8")
                return {
                    'success': True,
                    'skipped': True,
                    'reason': 'Already UTF-8',
                    'encoding': 'utf-8'
                }
            except UnicodeDecodeError:
                print(f"     ⚠️  UTF-8 detection failed, but content decoded as {encoding_used}")

        # Parse JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'error': f'JSON decode error: {e}',
                'encoding': encoding_used
            }

        if dry_run:
            entries_count = len(data) if isinstance(data, list) else 1
            print("     [DRY RUN] Would convert to UTF-8")
            return {
                'success': True,
                'dry_run': True,
                'original_encoding': encoding_used,
                'entries_count': entries_count
            }

        # Re-encode as UTF-8 JSON
        utf8_content = json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')

        # Upload back to S3
        loader.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=utf8_content,
            ContentType='application/json'
        )

        entries_count = len(data) if isinstance(data, list) else 1
        print(f"     ✅ Converted to UTF-8 ({entries_count} entries)")

        return {
            'success': True,
            'dry_run': False,
            'original_encoding': encoding_used,
            'entries_count': entries_count,
            'bytes_before': len(content),
            'bytes_after': len(utf8_content)
        }

    except Exception as e:
        print(f"     ❌ Error: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def find_files_with_encoding_issues(loader: S3DatasetLoader, manifest_path: Path) -> List[Dict[str, Any]]:
    """Find files that likely have encoding issues based on previous scan results"""
    # Files known to have encoding issues from deduplication scan
    problematic_files = [
        'datasets/gdrive/processed/phase_1_priority_conversations/task_5_1_priority_1/priority_1_conversations.jsonl',
        'datasets/gdrive/processed/phase_1_priority_conversations/task_5_2_priority_2/priority_2_conversations.jsonl',
        'datasets/gdrive/processed/phase_1_priority_conversations/task_5_3_priority_3/priority_3_conversations.jsonl',
        'datasets/gdrive/processed/phase_1_priority_conversations/task_5_6_unified_priority/unified_priority_conversations.jsonl',
        'datasets/gdrive/processed/phase_2_professional_datasets/task_5_9_soulchat/soulchat_2_0_conversations.jsonl',
        'datasets/gdrive/processed/phase_2_professional_datasets/task_5_11_llama3_mental_counseling/llama3_mental_counseling_conversations.jsonl',
        'datasets/gdrive/processed/phase_3_cot_reasoning/task_5_25_tot_reasoning/tot_reasoning_conversations.jsonl',
        'datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_27_condition_specific/condition_specific_conversations.jsonl',
        'datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_28_specialized_populations/specialized_populations_conversations.jsonl',
        'datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_29_temporal_analysis/temporal_analysis_conversations.jsonl',
        'datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_29_temporal_analysis/temporal_analysis_data.jsonl',
        'datasets/gdrive/processed/professional_datasets_final/soulchat_2_0_complete_no_limits.jsonl',
        'datasets/gdrive/processed/soulchat_complete/soulchat_2_0_complete_no_limits.jsonl',
    ]

    # Load manifest to verify files exist
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    files_to_fix = []
    processed_cats = manifest.get('categories', {}).get('gdrive', {}).get('processed', {})

    # Build lookup of all files
    all_files = {}
    for category, files_info in processed_cats.items():
        if isinstance(files_info, dict) and 'objects' in files_info:
            for obj in files_info['objects']:
                key = obj['key']
                if key.endswith(('.json', '.jsonl')):
                    all_files[key] = {
                        'key': key,
                        'category': category,
                        'size': obj['size']
                    }

    # Find problematic files
    for file_path in problematic_files:
        if file_path in all_files:
            files_to_fix.append(all_files[file_path])
        elif file_path.replace('datasets/', '') in all_files:
            # Try without datasets/ prefix
            alt_path = file_path.replace('datasets/', '')
            files_to_fix.append(all_files[alt_path])

    return files_to_fix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fix encoding issues in S3 datasets')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be fixed without actually fixing',
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Fix specific file (S3 key path)',
    )
    parser.add_argument(
        '--category',
        type=str,
        help='Fix all files in a category',
    )
    parser.add_argument(
        '--all-problematic',
        action='store_true',
        help='Fix all known problematic files from scan',
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Skip confirmation prompt',
    )

    return parser.parse_args()


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    with open(manifest_path, 'r') as f:
        return json.load(f)


def collect_files_to_fix(
    *,
    args: argparse.Namespace,
    loader: S3DatasetLoader,
    manifest_path: Path,
) -> List[Dict[str, Any]]:
    if args.file:
        # Single file
        return [{'key': args.file, 'category': 'manual', 'size': 0}]

    if args.category:
        manifest = load_manifest(manifest_path)
        processed_cats = manifest.get('categories', {}).get('gdrive', {}).get('processed', {})
        files_to_fix: List[Dict[str, Any]] = []

        if args.category in processed_cats:
            files_info = processed_cats[args.category]
            if isinstance(files_info, dict) and 'objects' in files_info:
                for obj in files_info['objects']:
                    key = obj['key']
                    if key.endswith(('.json', '.jsonl')):
                        files_to_fix.append({
                            'key': key,
                            'category': args.category,
                            'size': obj['size'],
                        })

        return files_to_fix

    if args.all_problematic:
        return find_files_with_encoding_issues(loader, manifest_path)

    return []


def confirm_proceed(*, dry_run: bool, confirm: bool) -> bool:
    if dry_run or confirm:
        return True

    print("\n" + "=" * 80)
    response = input("⚠️  Proceed with encoding fixes? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Cancelled")
        return False

    return True


def process_files(
    *,
    files_to_fix: List[Dict[str, Any]],
    loader: S3DatasetLoader,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for file_info in files_to_fix:
        s3_path = f"s3://{loader.bucket}/{file_info['key']}"

        if file_info['key'].endswith('.jsonl'):
            result = fix_jsonl_file(loader, s3_path, dry_run=dry_run)
        elif file_info['key'].endswith('.json'):
            result = fix_json_file(loader, s3_path, dry_run=dry_run)
        else:
            result = {'success': False, 'error': 'Unknown file type'}

        result['file'] = Path(file_info['key']).name
        result['category'] = file_info['category']
        results.append(result)

    return results


def print_results(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("📊 ENCODING FIX RESULTS")
    print("=" * 80)

    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]
    skipped = [r for r in successful if r.get('skipped')]
    fixed = [r for r in successful if not r.get('skipped')]

    print(f"\n✅ Fixed: {len(fixed)} files")
    print(f"⏭️  Skipped (already UTF-8): {len(skipped)} files")
    print(f"❌ Failed: {len(failed)} files")

    if fixed:
        print("\n📊 Fixed files:")
        for result in fixed:
            encoding = result.get('original_encoding', 'unknown')
            entries = result.get('entries_count', 0)
            print(f"   - {result.get('file')}: {encoding} → UTF-8 ({entries} entries)")

    if failed:
        print("\n❌ Failed files:")
        for result in failed:
            print(f"   - {result.get('file')}: {result.get('error', 'Unknown error')}")


def save_results(
    *,
    project_root: Path,
    dry_run: bool,
    results: List[Dict[str, Any]],
) -> Path:
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]
    skipped = [r for r in successful if r.get('skipped')]
    fixed = [r for r in successful if not r.get('skipped')]

    results_path = project_root / "ai/training_ready/data/encoding_fix_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'dry_run': dry_run,
            'results': results,
            'summary': {
                'total': len(results),
                'fixed': len(fixed),
                'skipped': len(skipped),
                'failed': len(failed),
            },
        }, f, indent=2)

    return results_path


def main() -> None:
    """Main encoding fix function"""
    args = parse_args()

    print("🔧 Dataset Encoding Fix")
    print("=" * 80)

    if args.dry_run:
        print("⚠️  DRY RUN MODE - No files will be modified")
    else:
        print("⚠️  LIVE MODE - Files will be converted to UTF-8 in S3!")

    # Initialize S3 loader
    loader = S3DatasetLoader(bucket=DEFAULT_S3_BUCKET)

    # Determine files to fix
    manifest_path = project_root / "ai/training_ready/data/s3_manifest.json"
    files_to_fix = collect_files_to_fix(
        args=args,
        loader=loader,
        manifest_path=manifest_path,
    )

    if not (args.file or args.category or args.all_problematic):
        print("❌ Specify --file, --category, or --all-problematic")
        return

    if not files_to_fix:
        print("❌ No files found to fix")
        return

    print(f"\n📋 Found {len(files_to_fix)} files to check/fix")

    # Confirm before proceeding
    if not confirm_proceed(dry_run=args.dry_run, confirm=args.confirm):
        return

    # Process files
    print("\n🔧 Processing files...")
    results = process_files(files_to_fix=files_to_fix, loader=loader, dry_run=args.dry_run)

    # Print summary
    print_results(results)

    # Save results
    results_path = save_results(project_root=project_root, dry_run=args.dry_run, results=results)
    print(f"\n💾 Results saved to: {results_path}")
    print("\n✅ Encoding fix complete!")


if __name__ == '__main__':
    main()
