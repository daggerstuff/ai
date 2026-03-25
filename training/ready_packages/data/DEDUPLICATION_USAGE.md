# Deduplication Removal Script Usage

## Overview

The `remove_duplicates.py` script removes duplicate entries from datasets based on the full deduplication scan results. It preserves one canonical copy of each duplicate while removing the rest.

## Usage

### Basic Commands

```bash
# Dry run - see what would be removed (SAFE, no changes)
uv run python ai/training_ready/scripts/remove_duplicates.py --dry-run

# Process specific category only
uv run python ai/training_ready/scripts/remove_duplicates.py --dry-run --category phase_1_priority_conversations

# Live run with confirmation prompt
uv run python ai/training_ready/scripts/remove_duplicates.py --keep-strategy priority_order

# Live run without confirmation (use with caution!)
uv run python ai/training_ready/scripts/remove_duplicates.py --keep-strategy priority_order --confirm
```

### Options

- `--dry-run`: Preview what would be removed without making changes (recommended first step)
- `--category CATEGORY`: Only process a specific category (e.g., `phase_1_priority_conversations`)
- `--keep-strategy STRATEGY`: How to choose which duplicate to keep
  - `priority_order` (default): Keep duplicate from highest priority dataset
  - `first_dataset`: Keep first occurrence
- `--confirm`: Skip confirmation prompt (dangerous - only use when certain)

### Keep Strategy Details

#### `priority_order` (Recommended)
Keeps duplicates based on dataset priority:
1. `phase_1_priority_conversations` (highest)
2. `phase_2_professional_datasets`
3. `phase_3_cot_reasoning`
4. `phase_4_reddit_mental_health`
5. Others (lower priority)

**Use case**: When you want to preserve entries from more important/curated datasets.

#### `first_dataset`
Keeps the first occurrence found in the duplicate group.

**Use case**: When order doesn't matter, just want to remove duplicates.

## Workflow

### Step 1: Review the Plan (Dry Run)

```bash
uv run python ai/training_ready/scripts/remove_duplicates.py --dry-run
```

This shows:
- Which files will be deduplicated
- How many duplicates will be removed from each file
- Total impact

### Step 2: Test on One Category

```bash
# Test on a single category first
uv run python ai/training_ready/scripts/remove_duplicates.py \
  --dry-run \
  --category phase_1_priority_conversations \
  --keep-strategy priority_order
```

Review the output to ensure it looks correct.

### Step 3: Execute (if satisfied)

```bash
# Run for real (will prompt for confirmation)
uv run python ai/training_ready/scripts/remove_duplicates.py \
  --category phase_1_priority_conversations \
  --keep-strategy priority_order
```

### Step 4: Verify Results

Check the results file:
```bash
cat ai/training_ready/data/deduplication_results.json
```

## Example Output

```
📋 DEDUPLICATION PLAN
================================================================================

📁 phase_1_priority_conversations:
   Files to deduplicate: 1
   Duplicates to remove: 53
     - priority_1_conversations.jsonl: 53 duplicates

📊 Total:
   Files: 1
   Duplicates to remove: 53
```

## Important Notes

1. **Always start with `--dry-run`** to preview changes
2. **Test on one category first** before processing all
3. **Backup important data** before running live (though S3 has versioning)
4. **Encoding issues**: Some files have encoding problems - the script handles these gracefully but may skip problematic entries
5. **Large files**: Very large files may timeout - consider processing in batches

## Current Limitations

- Only processes first 100 duplicate groups from report (to limit report size)
- Encoding issues in some JSONL files may cause some entries to be skipped
- Large files may timeout during processing

## Results

Results are saved to:
- `ai/training_ready/data/deduplication_results.json`

This includes:
- Timestamp
- Files processed
- Duplicates removed
- Entries kept
- Any errors encountered

## Safety Features

1. **Dry run mode** - preview without changes
2. **Confirmation prompt** - requires explicit "yes" to proceed
3. **Category filtering** - process one category at a time
4. **Error handling** - continues processing even if some files fail
5. **Detailed logging** - all actions are logged

## Troubleshooting

### Encoding Errors
If you see encoding errors:
- The script will try alternative encodings automatically
- Some entries may be skipped if encoding can't be resolved
- Check the results file for details

### Connection Timeouts
For large files:
- Process smaller categories first
- Consider processing during off-peak hours
- Check S3 connection stability

### No Duplicates Found
If no duplicates are found:
- Check that `full_deduplication_report.json` exists
- Verify the report contains duplicate groups
- Ensure the category name matches exactly
