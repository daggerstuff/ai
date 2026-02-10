# CPTSD Dataset Expansion Progress Summary

**Generated**: 2026-02-10T20:19:00Z
**Task**: Build CPTSD Dataset
**Status**: In Progress

## Current State

### Existing Dataset

- **Current Examples**: 91 CPTSD training examples
- **Format**: ChatML with metadata
- **Source**: Tim Fletcher transcripts
- **Issues Identified**:
  - Underutilized chunking (1 chunk per file vs 6-chunk capacity)
  - Missing CPTSD-specific features (topic tagging, crisis detection,
    emotional flashback detection)
  - Limited topic coverage (missing: emotional flashbacks, triggers,
    hypervigilance, dissociation, inner child)

### Available Sources Discovered

#### 1. Heidi Priebe (2 files, 88.55 KB)

- "10 'Survival Lies' You May Tell If You Have CPTSD.txt" (49.32 KB)
- "CPTSD： Breaking The Toxic Shame⧸Procrastination Cycle With Self-Compassion.txt"
  (39.23 KB)
- **Topics**: Survival lies, toxic shame, procrastination, self-compassion

#### 2. Patrick Teahan (1 file, 38.73 KB)

- "9 Random Examples of Shame from PTSD & CPTSD.txt" (38.73 KB)
- **Topics**: Shame, PTSD, CPTSD, shame examples

#### 3. Tim Fletcher Raw Files (8 files, 285.41 KB)

- Multiple complex trauma transcripts with #complextrauma tag
- **Topics**: Big T vs Little T trauma, betrayal trauma, depletion, self-care,
  healing, recovery, letting go, sexuality, needs, trauma bonding

#### 4. Processed Transcripts (50 files, 1.14 MB)

- Already cleaned complex trauma transcripts from pixelated-v2 dataset
- **Topics**: Anger, shame, codependency, attachment issues, betrayal trauma,
  brain fog, emotional dysregulation, boundaries, narcissistic abuse,
  procrastination, toxic shame, trauma bonding, trust, commitment,
  relationships, survival mode, and more

#### 5. Existing CPTSD Dataset (1 file, 4.48 MB)

- Current 91 examples in ChatML format
- Source: Tim Fletcher transcripts

## Topic Coverage Analysis

### Covered Topics

- ✅ Shame cycles (Heidi Priebe, Patrick Teahan, Tim Fletcher)
- ✅ Trauma bonding (Tim Fletcher)
- ✅ Betrayal trauma (Tim Fletcher)
- ✅ Survival mode (Tim Fletcher processed)
- ✅ Emotional dysregulation (Tim Fletcher processed)
- ✅ Reparenting (Tim Fletcher)
- ✅ Codependency (Tim Fletcher processed)
- ✅ Narcissistic abuse (Tim Fletcher processed)
- ✅ Boundaries (Tim Fletcher processed)
- ✅ Procrastination (Heidi Priebe, Tim Fletcher processed)
- ✅ Toxic shame (Heidi Priebe, Patrick Teahan, Tim Fletcher processed)

### Missing Topics

- ❌ Emotional flashbacks
- ❌ Triggers
- ❌ Hypervigilance
- ❌ Dissociation
- ❌ Inner child

## Next Steps

### Immediate Actions

1. **Leverage existing 50 processed transcripts** (1.14 MB) - These are already
   cleaned and ready for processing
2. **Create voice profiles** for Heidi Priebe and Patrick Teahan based on their
   content themes
3. **Enhance chunking strategy** to 6 chunks per file for better context
   preservation
4. **Add CPTSD-specific topic tagging** to metadata (emotional flashbacks,
   triggers, shame cycles, etc.)
5. **Implement crisis signal detection** for CPTSD content
6. **Generate synthetic examples** for missing topics (emotional flashbacks,
   triggers, hypervigilance, dissociation, inner child)

### Processing Pipeline

1. Analyze existing 50 processed transcripts for CPTSD topics
2. Create enhanced processing script with:
   - 6-chunk strategy per file
   - CPTSD-specific topic tagging
   - Crisis signal detection
   - Emotional flashback and triggers detection
   - Improved PII redaction patterns
3. Process all sources with enhanced script
4. Generate synthetic CPTSD dialogue scenarios for missing topics
5. Create CPTSD-specific therapeutic response examples
6. Add CPTSD recovery journey progression examples (5 stages)
7. Validate all CPTSD examples against therapeutic standards
8. Test dataset for crisis signal handling
9. Verify emotional score normalization (0-1 range)
10. Run quality validation pipeline on expanded dataset

## Target Metrics

- **Current Examples**: 91
- **Target Examples**: 500+
- **Current Sources**: 1 (Tim Fletcher)
- **Target Sources**: 4+ (Tim Fletcher, Heidi Priebe, Patrick Teahan, Pete Walker)
- **Current Topics**: ~11 covered
- **Target Topics**: 16+ (all CPTSD topics covered)

## Files Created

1. `ai/training/ready_packages/data/cptsd_source_inventory.json`
   - Comprehensive source inventory
2. `ai/training/ready_packages/scripts/download_cptsd_sources.py`
   - Download script (S3 access not working)
3. `ai/training/ready_packages/data/cptsd_sources/download_catalog.json`
   - Download catalog (empty due to S3 issues)

## Challenges Encountered

1. **S3 Download Issues**: ovhai CLI not returning data for specific prefixes
2. **Unicode Encoding**: S3 manifest has Unicode characters in filenames that
   don't match expected patterns
3. **File Access**: Cannot directly access S3 files for download

## Workaround Strategy

Since S3 download is not working, the strategy is to:

1. Focus on the 50 already-processed complex trauma transcripts (1.14 MB)
2. Create voice profiles based on content analysis
3. Generate synthetic examples for missing topics
4. Enhance the existing 91 examples with improved metadata
5. Process the 50 processed transcripts with enhanced chunking strategy

## Estimated Impact

With the 50 processed transcripts + 91 existing examples + synthetic generation
for missing topics, we can reach the target of 500+ CPTSD training examples with
comprehensive topic coverage.
