 # Training Data Migration Plan

## Quick Start

```bash
# 1. Preview what will happen (dry-run)
cd /home/vivi/pixelated/ai
./scripts/migrate_training_data.sh --dry-run

# 2. Execute migration
./scripts/migrate_training_data.sh

# 3. Validate
./scripts/validate_migration.sh
```

**Status**: ✅ Ready to execute | **Risk**: LOW | **Time**: ~1 hour

---

## What's Happening

Merging 7 scattered directories into 1 unified structure:

| Source | Files | Size | Purpose |
|--------|-------|------|---------|
| `training` | 1,508 | 175M | Main training infrastructure |
| `training_corpus` | 1,015 | 243M | Corpus building tools |
| `training_data` | 6 | 72K | External datasets |
| `training_data_consolidated` | 111 | 82M | **Stage splits (CRITICAL)** |
| `training_ready/*` | 3 | 2.2M | Placeholders |
| **Total** | **2,643** | **~495M** | |

### Target Structure

All data moves to `training_data_unified/`:
```
training_data_unified/
├── raw/              # Raw transcripts
├── processed/        # Stage splits (train/val/test)
├── generated/        # Synthetic data
├── configs/          # Configurations
├── scripts/          # Processing scripts
├── models/           # Model architectures
├── tools/            # Utilities
└── docs/             # Documentation
```

---

## Safety Features

- ✅ **Full backup** created first (compressed tar.gz)
- ✅ **Dry-run mode** to preview changes
- ✅ **Symlink handling** skips broken links
- ✅ **Incremental validation** after each step
- ✅ **Auto-generated rollback** script
- ✅ **Copy-only** operations (originals preserved)

---

## Pre-Flight Checklist

- [x] Disk space verified (118GB available, need ~1GB)
- [x] Permissions verified (all writable)
- [x] Critical files verified (7/7 present)
- [x] Symlinks detected (2 broken, will skip)
- [ ] Dry-run executed
- [ ] Team notified

---

## Execution Steps

### Step 1: Dry Run (Recommended First)
```bash
./scripts/migrate_training_data.sh --dry-run
```
Shows what would happen without making changes.

### Step 2: Execute Migration
```bash
./scripts/migrate_training_data.sh
```

The script will:
1. Create compressed backup at `/tmp/training_backup_TIMESTAMP/`
2. Generate rollback script automatically
3. Create new directory structure
4. Migrate processed data (stage splits - PRIORITY)
5. Migrate raw transcripts
6. Migrate configs, scripts, models, tools
7. Generate documentation
8. Validate migration

### Step 3: Validate
```bash
./scripts/validate_migration.sh
```

Checks:
- File counts match
- Critical files present
- Stage splits intact
- No data loss

---

## Critical Data Protection

### Highest Priority (Stage Splits)
These are migrated FIRST and validated:
- `stage1_foundation/{train,val,test}.jsonl`
- `stage2_therapeutic_expertise/{train,val,test}.jsonl`
- `stage3_edge_stress_test/{train,val,test}.jsonl`
- `stage4_voice_persona/{train,val,test}.jsonl`

### High Priority
- YouTube transcripts (91+ files)
- Training configurations
- Processing scripts
- Model architectures

---

## Known Issues

### Symlinks (RESOLVED)
**Issue**: 2 broken symlinks in `training/defense_mechanisms/data/`
- `test.json` → broken (Windows reparse tag)
- `train.json` → broken (Windows reparse tag)

**Solution**: Script automatically detects and skips symlinks

**Impact**: None (files are inaccessible anyway)

---

## Rollback Plan

If anything goes wrong:

### Option 1: Auto-Generated Rollback (Easiest)
```bash
/tmp/training_backup_TIMESTAMP/rollback.sh
```

### Option 2: Manual Rollback
```bash
# Remove new structure
rm -rf ai/training_data_unified

# Restore from backup
mv /tmp/training_backup_*/training ai/
mv /tmp/training_backup_*/training_corpus ai/
mv /tmp/training_backup_*/training_data ai/
mv /tmp/training_backup_*/training_data_consolidated ai/
mv /tmp/training_backup_*/training_ready ai/
```

---

## Post-Migration

### Immediate (After Migration)
- [ ] Run validation script
- [ ] Verify critical files present
- [ ] Test key workflows
- [ ] Update team documentation

### Validation Period (2 weeks)
- [ ] Monitor for issues
- [ ] Mark old directories as DEPRECATED
- [ ] Archive or remove old directories

---

## Success Criteria

- [x] Plan documented
- [x] Scripts created and tested
- [x] Pre-migration validation complete
- [ ] Migration executed
- [ ] Post-migration validation passes
- [ ] Critical files verified
- [ ] Team notified
- [ ] Old directories deprecated (after 2 weeks)

---

## Timeline

| Phase | Duration | When |
|-------|----------|------|
| Preparation | 5 min | Before migration |
| Dry Run | 5 min | Before execution |
| Migration | 10-30 min | Execution window |
| Validation | 15-30 min | Immediately after |
| **Total Active Work** | **~1 hour** | |
| Validation Period | 2 weeks | Post-migration |

---

## Support

### Files
- `TRAINING_DATA_MIGRATION.md` - This document
- `scripts/migrate_training_data.sh` - Migration script
- `scripts/validate_migration.sh` - Validation script

### Key Commands
```bash
# Preview
./scripts/migrate_training_data.sh --dry-run

# Execute
./scripts/migrate_training_data.sh

# Validate
./scripts/validate_migration.sh

# Rollback (if needed)
/tmp/training_backup_*/rollback.sh
```

---

## FAQ

**Q: How long will this take?**  
A: About 1 hour total (10-30 min execution + 15-30 min validation)

**Q: Will I lose any data?**  
A: No. All operations are copy-only. Full backup created first.

**Q: What if something goes wrong?**  
A: Run the auto-generated rollback script. Full backup is available.

**Q: Do I need to stop anything?**  
A: No downtime required. This is a read-only migration.

**Q: What happens to old directories?**  
A: They remain untouched. Mark as DEPRECATED after 2-week validation.

---

**Ready to proceed?** → Run `./scripts/migrate_training_data.sh --dry-run`

---

**Last Updated**: 2026-05-09  
**Status**: ✅ READY FOR EXECUTION  
**Risk Level**: LOW  
**Confidence**: HIGH
