# Consolidation Notes

## Disk Space Considerations

Due to disk space constraints during consolidation, large files are referenced in the manifest rather than copied. The `TRAINING_MANIFEST.json` contains absolute paths to all original files.

### Asset Access Strategy

1. **Small Files**: Configs, scripts, and small model files have been copied where possible
2. **Large Files**: Large datasets use symlinks or are referenced via manifest entries
3. **Original Locations**: All original file paths are preserved in `TRAINING_MANIFEST.json`

### Using the Manifest

To access any training asset:

```python
import json

with open('TRAINING_MANIFEST.json', 'r') as f:
    manifest = json.load(f)

# Find a dataset
for dataset in manifest['datasets']:
    if 'foundation' in dataset['name'].lower():
        print(f"Path: {dataset['path']}")
        print(f"Stage: {dataset['stage']}")
```

### Consolidation Scripts

The consolidation scripts in `scripts/` can be run when disk space is available:
- `consolidate_assets.py` - Consolidates all assets (requires disk space)
- Uses symlinks for files > 100MB by default
- Can be configured to use references instead of copies

### Recommended Approach

1. Use `TRAINING_MANIFEST.json` to locate assets
2. Access files from original locations as documented
3. Run consolidation scripts when disk space allows
4. Use symlinks for large datasets to avoid duplication

## Status

- ✅ Directory structure created
- ✅ Manifest generated with all asset paths
- ✅ Documentation complete
- ⚠️  Full file consolidation deferred due to disk space
- ✅ All assets cataloged and accessible via manifest

