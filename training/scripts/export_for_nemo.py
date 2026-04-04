#!/usr/bin/env python3
"""
Export Release 0 datasets for NVIDIA NeMo ingestion.

Creates:
1. JSONL files in NeMo format (text completion)
2. CSV files for human review
3. HTML report for each stage

Usage:
    python -m ai.training.scripts.export_for_nemo --release-version v2026-04-03
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_release_data(release_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load compiled export from release directory."""
    export_file = release_dir / 'compiled_export.jsonl'
    if not export_file.exists():
        raise FileNotFoundError(f"Export file not found: {export_file}")

    stages: Dict[str, List[Dict[str, Any]]] = {
        'stage1_foundation': [],
        'stage2_therapeutic_expertise': [],
        'stage4_voice_persona': [],
    }

    with open(export_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                stage = record.get('metadata', {}).get('stage', 'unknown')
                if stage in stages:
                    stages[stage].append(record)

    logger.info(f"Loaded {sum(len(v) for v in stages.values())} total records")
    return stages


def convert_to_nemo_format(record: Dict[str, Any]) -> Dict[str, str]:
    """Convert record to NeMo text completion format."""
    messages = record.get('messages', [])

    # Extract actual conversation from nested JSON structure
    conversation_text = ""
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        # Content might be a JSON string - parse it
        if content.startswith('{') and content.endswith('}'):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and 'messages' in parsed:
                    # Extract from nested structure
                    inner_messages = parsed.get('messages', [])
                    parts = []
                    for inner_msg in inner_messages:
                        inner_role = inner_msg.get('role', '')
                        inner_content = inner_msg.get('content', '')
                        parts.append(f"{inner_role}: {inner_content}")
                    conversation_text = "\n".join(parts)
                    break
            except json.JSONDecodeError:
                pass

    # Fallback: use content directly if not nested JSON
    if not conversation_text:
        parts = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role and content:
                parts.append(f"{role}: {content}")
        conversation_text = "\n".join(parts)

    # Final fallback: use text field
    if not conversation_text:
        conversation_text = record.get('text', '')

    return {
        'text': conversation_text,
    }


def export_neomo_jsonl(stages: Dict[str, List[Dict[str, Any]]], output_dir: Path) -> Dict[str, Path]:
    """Export to NeMo-compatible JSONL format."""
    output_files = {}

    for stage_name, records in stages.items():
        if not records:
            continue

        output_file = output_dir / f"{stage_name}_nemo.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            for record in records:
                nemo_record = convert_to_nemo_format(record)
                f.write(json.dumps(nemo_record, ensure_ascii=False) + '\n')

        output_files[stage_name] = output_file
        logger.info(f"Exported {len(records)} records to {output_file}")

    return output_files


def export_human_readable_csv(stages: Dict[str, List[Dict[str, Any]]], output_dir: Path) -> Dict[str, Path]:
    """Export human-readable CSV for review."""
    import csv
    output_files = {}

    for stage_name, records in stages.items():
        if not records:
            continue

        output_file = output_dir / f"{stage_name}_review.csv"

        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['#', 'User Input', 'Assistant Response', 'Source'])

            for i, record in enumerate(records):
                messages = record.get('messages', [])
                user_text = ""
                assistant_text = ""
                for msg in messages:
                    if msg.get('role') == 'user':
                        user_text = msg.get('content', '')
                    elif msg.get('role') == 'assistant':
                        assistant_text = msg.get('content', '')

                source = record.get('metadata', {}).get('source', 'unknown')
                writer.writerow([i, user_text, assistant_text, source])

        output_files[stage_name] = output_file
        logger.info(f"Exported {len(records)} records to {output_file}")

    return output_files


def export_html_report(stages: Dict[str, List[Dict[str, Any]]], output_dir: Path, release_version: str) -> Path:
    """Export HTML report for human review."""
    output_file = output_dir / 'release_0_review.html'

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Release 0 Human QA Review - {release_version}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        .stage {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .stage h2 {{ margin-top: 0; color: #666; }}
        .record {{ background: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 3px; }}
        .user {{ color: #0066cc; font-weight: bold; }}
        .assistant {{ color: #009900; }}
        .meta {{ color: #888; font-size: 0.9em; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f0f0f0; }}
    </style>
</head>
<body>
    <h1>Release 0 Human QA Review</h1>
    <p><strong>Release:</strong> {release_version}</p>
    <p><strong>Generated:</strong> {datetime.now(timezone.utc).isoformat()}</p>
    <p><strong>Total Records:</strong> {sum(len(v) for v in stages.values())}</p>
"""

    for stage_name, records in stages.items():
        if not records:
            continue

        stage_display = stage_name.replace('_', ' ').title()
        html_content += f"""
    <div class="stage">
        <h2>{stage_display} ({len(records)} records)</h2>
        <table>
        <tr><th>#</th><th>User</th><th>Assistant</th></tr>
"""
        for i, record in enumerate(records[:20]):  # First 20 for review
            messages = record.get('messages', [])
            user_text = ""
            assistant_text = ""
            for msg in messages:
                if msg.get('role') == 'user':
                    user_text = msg.get('content', '')[:200]  # Truncate for display
                elif msg.get('role') == 'assistant':
                    assistant_text = msg.get('content', '')[:200]

            html_content += f"""
        <tr>
            <td>{i}</td>
            <td class="user">{user_text}...</td>
            <td class="assistant">{assistant_text}...</td>
        </tr>
"""
        html_content += """
        </table>
        <p><em>Showing first 20 records. See CSV/JSONL for full data.</em></p>
    </div>
"""

    html_content += """
</body>
</html>
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"HTML report saved to {output_file}")
    return output_file


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Export Release 0 for NeMo + Human Review')
    parser.add_argument(
        '--release-version',
        type=str,
        default='v2026-04-03',
        help='Release version'
    )
    parser.add_argument(
        '--release-dir',
        type=str,
        default='ai/data/releases/v2026-04-03',
        help='Release directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='ai/data/nemo_export',
        help='Output directory for NeMo export'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Release 0: NeMo Export + Human Review")
    print("=" * 60)
    print(f"Release: {args.release_version}")
    print(f"Output: {args.output_dir}")
    print()

    release_dir = Path(args.release_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading release data...")
    stages = load_release_data(release_dir)

    # Export NeMo JSONL
    logger.info("Exporting NeMo JSONL...")
    nemo_files = export_neomo_jsonl(stages, output_dir)

    # Export human-readable CSV
    logger.info("Exporting human-readable CSV...")
    csv_files = export_human_readable_csv(stages, output_dir)

    # Export HTML report
    logger.info("Exporting HTML report...")
    html_file = export_html_report(stages, output_dir, args.release_version)

    # Summary
    print()
    print("=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print()
    print("NeMo Format (JSONL):")
    for stage, path in nemo_files.items():
        print(f"  {stage}: {path}")
    print()
    print("Human Review (CSV):")
    for stage, path in csv_files.items():
        print(f"  {stage}: {path}")
    print()
    print(f"HTML Report: {html_file}")
    print()
    print("Next Steps:")
    print("  1. Open HTML report for human review")
    print("  2. Use NeMo JSONL files for training:")
    print("     python train.py --data_dir ai/data/nemo_export")
    print("=" * 60)


if __name__ == '__main__':
    main()
