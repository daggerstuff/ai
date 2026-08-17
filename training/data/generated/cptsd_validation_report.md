# CPTSD Dataset Validation Report

**Generated:** 2026-08-17  
**Source:** 233 transcript files from 93 YouTube channels (`~/joiner2/transcripts/transcripts/`)  
**Builder:** `ai/training/scripts/build_cptsd_dataset_from_transcripts.py`

## Summary

| Metric | Value |
|--------|-------|
| Raw examples generated | 3,036 |
| Non-CPTSD filtered out (no topics) | 1,275 |
| **Final CPTSD examples** | **1,761** |
| Authors represented | 5 (tim_fletcher, patrick_teahan, heidi_priebe, crappy_childhood_fairy, doc_snipes) |
| PII status | All scrubbed (0 email/phone/SSN/URL leaks detected) |

## Train/Val/Test Split

Split by source file (no leakage across splits), seed=42.

| Split | Count | % | Topics | Stages | Crisis |
|-------|-------|---|--------|--------|--------|
| Train | 1,406 | 79.8% | 10/10 | 5/5 | 135 |
| Val | 173 | 9.8% | 9/10 | 5/5 | 14 |
| Test | 182 | 10.3% | 10/10 | 5/5 | 12 |

## Topic Coverage

| Topic | Count | % | Assessment |
|-------|-------|---|------------|
| shame_toxic_shame | 671 | 38.1% | Well covered |
| recovery_stages | 513 | 29.1% | Well covered |
| triggers_trauma_responses | 492 | 27.9% | Well covered |
| boundary_setting | 394 | 22.4% | Well covered |
| survival_responses | 215 | 12.2% | Adequate |
| healing_milestones | 157 | 8.9% | Adequate |
| self_compassion | 113 | 6.4% | Adequate |
| inner_child_work | 111 | 6.3% | Adequate |
| emotional_regulation | 85 | 4.8% | Underrepresented — needs synthetic augmentation |
| emotional_flashbacks | 18 | 1.0% | Critical — needs synthetic augmentation |

## Recovery Stage Distribution

| Stage | Count | % |
|-------|-------|---|
| stabilization | 612 | 34.8% |
| awareness | 557 | 31.6% |
| integration | 383 | 21.8% |
| processing | 139 | 7.9% |
| thriving | 70 | 4.0% |

577 records originally had `None` recovery stage; fixed via topic-to-stage inference mapping.

## Crisis Detection

| Severity | Count |
|----------|-------|
| MEDIUM | 126 |
| HIGH | 35 |
| **Total** | **161** (9.1% of dataset) |

Crisis detection working correctly — flagging records with crisis language for safety gating.

## PII Redaction

- All 3,036 records marked `pii_status: scrubbed`
- Post-hoc regex scan of assistant content: **0 email, 0 phone, 0 SSN, 0 URL hits**
- Builder uses regex-based PII redaction (Presidio not available in current environment)

## Issues & Next Steps

1. **`emotional_flashbacks` (18) and `emotional_regulation` (85)** — critically underrepresented. Synthetic dialogue generation needed (task `cptsd-synthetic`).
2. **`processing` (7.9%) and `thriving` (4.0%)** — underrepresented recovery stages. Synthetic augmentation should target these.
3. **Val split missing `emotional_flashbacks`** — only 18 examples total; synthetic generation will fix this.
4. **Voice profile assignment** — `voice_profile_used` reports fallback for all records; author-specific system prompts exist but the metadata flag is not correctly propagated.

## Files

| File | Description |
|------|-------------|
| `cptsd_transcripts.jsonl` | Raw builder output (3,036 records) |
| `cptsd_filtered.jsonl` | Filtered to CPTSD-topic records (1,761) |
| `cptsd_final.jsonl` | Recovery stages fixed (1,761, no None) |
| `cptsd_train.jsonl` | Train split (1,406) |
| `cptsd_val.jsonl` | Val split (173) |
| `cptsd_test.jsonl` | Test split (182) |
| `cptsd_transcripts_stats.json` | Builder run statistics |
