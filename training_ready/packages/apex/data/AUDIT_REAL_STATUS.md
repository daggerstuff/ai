# Real Audit: Final Training Dataset Plan - Verified Status

**Date**: 2025-12-13  
**Method**: Code inspection, file existence checks, implementation verification  
**Warning**: Previous status documents may have been overly optimistic

---

## 🔍 Audit Methodology

For each claimed "completed" item, I:
1. Verified file existence
2. Read actual implementation code
3. Checked if implementation matches plan requirements
4. Identified gaps, incomplete implementations, and unsafe practices

---

## 1. inventory-map

**Claimed**: ✅ Completed  
**Verified**: ✅ **ACTUALLY COMPLETE**

**Evidence**:
- ✅ `data/dataset_coverage_report.json` exists with real data (19,323 S3 objects indexed)
- ✅ `data/dataset_routing_config.json` exists with family mappings
- ✅ `scripts/dataset_inventory_audit.py` exists (340 lines)
- ✅ Coverage report shows 14 families mapped with status (present/partial/missing)

**Status**: **VERIFIED COMPLETE**

---

## 2. define-contract

**Claimed**: ✅ Completed  
**Verified**: ✅ **ACTUALLY COMPLETE**

**Evidence**:
- ✅ `docs/FINAL_DATASET_CONTRACT.md` exists (291 lines, comprehensive)
- ✅ Defines ChatML schema with required metadata fields
- ✅ Defines manifest schema with shards, splits, provenance
- ✅ Defines split rules (90/5/5) with hard holdout families
- ✅ Defines provenance tracking requirements
- ✅ Defines PII handling rules
- ✅ Defines content hash algorithm

**Status**: **VERIFIED COMPLETE**

---

## 3. dedup-leakage

**Claimed**: ✅ Completed  
**Verified**: ⚠️ **PARTIALLY COMPLETE - PLACEHOLDER IMPLEMENTATION**

**Evidence**:
- ✅ `scripts/enhanced_deduplication.py` exists (330 lines)
- ✅ Exact duplicate detection implemented (content hash)
- ✅ Split leakage prevention implemented
- ✅ Holdout family isolation checks implemented
- ⚠️ **CRITICAL ISSUE**: Near-duplicate detection uses **word overlap**, NOT semantic similarity
  - Code: `compute_simple_similarity()` uses Jaccard similarity on word sets
  - Comment in code: `"For production, use sentence-transformers"`
  - Plan requirement: "semantic/approx similarity pass"
  - **This is a placeholder, not production-ready**

**Status**: **INCOMPLETE - Needs real semantic similarity (sentence-transformers/embeddings)**

---

## 4. generators-missing

**Claimed**: ✅ Completed  
**Verified**: ⚠️ **MOSTLY COMPLETE - Missing Preference Pairs**

**Evidence**:
- ✅ `scripts/extract_long_running_therapy.py` exists (309 lines) - **ACTUALLY IMPLEMENTED**
- ✅ Edge case generator: Available in S3 (33 objects found)
- ✅ Edge case synthetic: Generated locally (`data/generated/edge_case_synthetic.jsonl`)
- ✅ Sarcasm: Available in S3 (1 object, marked partial)
- ✅ Roleplay/simulator: Available in S3 (2 objects, marked partial)
- ❌ **MISSING**: Preference pair generation script
  - Plan requires: "generate/curate preference pairs from roleplay/simulator + adversarial prompts"
  - Curriculum references: `dpo_preference`, `roleplay_simulator_preferences`, `edge_case_adversarial`
  - **No script found** to generate these preference pairs

**Status**: **INCOMPLETE - Preference pair generation missing**

---

## 5. compile-export

**Claimed**: ✅ Completed  
**Verified**: ⚠️ **UNSAFE IMPLEMENTATION - No Upload Verification**

**Evidence**:
- ✅ `scripts/compile_final_dataset.py` exists (1489 lines)
- ✅ `_upload_to_s3()` method exists
- ✅ Uploads shards, compiled export, and manifest to S3
- ✅ Removes local copies after upload
- ❌ **CRITICAL SAFETY ISSUE**: **No verification before deletion**
  - Code uploads, then immediately deletes local file
  - No `head_object()` check to verify upload succeeded
  - If upload silently fails, local file is still deleted
  - Compare to `upload_local_datasets_to_s3.py` which DOES verify:
    ```python
    s3_client.head_object(Bucket=bucket, Key=s3_key)  # Verification
    local_path.unlink()  # Only after verification
    ```

**Status**: **UNSAFE - Needs upload verification before local file deletion**

---

## 6. curriculum

**Claimed**: ✅ Completed  
**Verified**: ✅ **ACTUALLY COMPLETE**

**Evidence**:
- ✅ `configs/training_curriculum_2025.json` exists (176 lines, complete)
- ✅ `docs/TRAINING_CURRICULUM_2025.md` exists (347 lines, comprehensive)
- ✅ Phase A (pretraining) defined with datasets and params
- ✅ Phase B (7-stage SFT curriculum) fully defined
- ✅ Phase C (preference alignment) defined with algorithm options
- ✅ All dataset-to-phase routing specified
- ✅ Training order and dependencies defined

**Status**: **VERIFIED COMPLETE**

---

## 7. verify

**Claimed**: ✅ Completed  
**Verified**: ✅ **ACTUALLY COMPLETE**

**Evidence**:
- ✅ `scripts/verify_final_dataset.py` exists (419 lines)
- ✅ Coverage gate implemented
- ✅ Leakage gate implemented (checks violations file)
- ✅ Distribution gate implemented
- ✅ PII gate implemented
- ✅ Provenance gate implemented
- ✅ Hash gate implemented
- ✅ Split gate implemented
- ✅ Stats gate implemented

**Status**: **VERIFIED COMPLETE**

---

## 🚨 Critical Issues Found

### Issue 1: Unsafe S3 Upload (compile_final_dataset.py)
**Severity**: HIGH  
**Risk**: Data loss if upload fails silently  
**Location**: `scripts/compile_final_dataset.py` lines 1128-1136, 1157, 1255

**Problem**:
```python
def _upload_to_s3(self, local_path: Path, s3_key: str) -> None:
    self.s3_client.upload_file(str(local_path), self.s3_bucket, s3_key)
    # No verification!
    # Later: shard_path.unlink()  # Deletes without checking if upload succeeded
```

**Fix Required**: Add verification like `upload_local_datasets_to_s3.py`:
```python
s3_client.upload_file(...)
s3_client.head_object(Bucket=bucket, Key=s3_key)  # Verify
local_path.unlink()  # Only then delete
```

### Issue 2: Placeholder Near-Duplicate Detection
**Severity**: MEDIUM  
**Risk**: May miss semantic duplicates, causing split leakage  
**Location**: `scripts/enhanced_deduplication.py` lines 75-89

**Problem**:
- Uses word overlap (Jaccard similarity) instead of semantic similarity
- Comment says "For production, use sentence-transformers"
- Plan requires "semantic/approx similarity pass"

**Fix Required**: Implement real semantic similarity using sentence-transformers or embeddings

### Issue 3: Missing Preference Pair Generator
**Severity**: MEDIUM  
**Risk**: Phase C training cannot proceed without preference data  
**Location**: No script exists

**Problem**:
- Curriculum references `dpo_preference`, `roleplay_simulator_preferences`, `edge_case_adversarial`
- Plan requires: "generate/curate preference pairs from roleplay/simulator + adversarial prompts"
- No script found to generate these

**Fix Required**: Create preference pair generation script

---

## 📊 Real Completion Status

| Todo | Claimed | Actual | Notes |
|------|---------|--------|-------|
| inventory-map | ✅ | ✅ | Verified complete |
| define-contract | ✅ | ✅ | Verified complete |
| dedup-leakage | ✅ | ⚠️ | Placeholder implementation |
| generators-missing | ✅ | ⚠️ | Missing preference pairs |
| compile-export | ✅ | ⚠️ | Unsafe (no verification) |
| curriculum | ✅ | ✅ | Verified complete |
| verify | ✅ | ✅ | Verified complete |

**Real Completion**: ~70% (not 95%)

---

## ✅ What Actually Works

1. **Inventory & Coverage**: Fully functional, real data
2. **Contract Definition**: Comprehensive and complete
3. **Curriculum**: Complete 3-phase training plan
4. **Verification Gates**: All gates implemented
5. **Long-Running Therapy Extraction**: Actually implemented (found script)

---

## ❌ What Needs Work

1. **S3 Upload Safety**: Add verification before deletion (HIGH PRIORITY)
2. **Semantic Similarity**: Replace word overlap with real embeddings (MEDIUM PRIORITY)
3. **Preference Pairs**: Create generator script (MEDIUM PRIORITY)

---

## 🎯 Recommendations

1. **Immediate**: Fix `compile_final_dataset.py` to verify uploads before deletion
2. **Short-term**: Implement real semantic similarity for near-duplicate detection
3. **Short-term**: Create preference pair generation script for Phase C
4. **Documentation**: Update plan status to reflect actual implementation state

---

## 📝 Files That Actually Exist vs Plan References

| Plan Reference | Actual Location | Status |
|----------------|-----------------|--------|
| `scripts/full_deduplication_scan.py` | ❌ Deleted | Replaced by `enhanced_deduplication.py` |
| `pipelines/integrated_training_pipeline.py` | `packages/velocity/data_pipeline/` | Different location |
| `extract_long_running_therapy.py` | ✅ `scripts/extract_long_running_therapy.py` | Actually exists! |
| Preference pair generator | ❌ Not found | Missing |

---

**Conclusion**: The plan is more complete than initially thought (long-running therapy extraction exists), but has critical safety issues (unsafe S3 deletion) and incomplete implementations (semantic similarity, preference pairs).
