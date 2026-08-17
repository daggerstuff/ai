# CPTSD Dataset Validation Report

**Generated:** 2026-08-17 (updated)  
**Source:** 233 transcript files from 93 YouTube channels (`~/joiner2/transcripts/transcripts/`)  
**Builder:** `ai/training/scripts/build_cptsd_dataset_from_transcripts.py`

## Summary

| Metric | Value |
|--------|-------|
| Raw examples generated | 3,036 |
| Non-CPTSD filtered out (no topics) | 526 |
| **Final CPTSD examples** | **2,510** |
| Authors represented | 5+ (tim_fletcher, patrick_teahan, heidi_priebe, crappy_childhood_fairy, doc_snipes, DoctorRamani, etc.) |
| PII status | All scrubbed (0 email/phone/SSN/URL leaks detected) |

### Tagger Fix History

| Round | Kept | Dropped | False Negatives | Action |
|-------|------|---------|-----------------|--------|
| Original | 1,761 | 1,275 | 648 (51%) | — |
| After pattern expansion 1 | 2,480 | 556 | 67 (12%) | Added 137 patterns |
| After pattern expansion 2 | 2,510 | 526 | 56 (11%) | Added 31 patterns |
| Remaining 526 drops | — | — | 470 genuinely non-CPTSD, 56 edge cases | Stopped (diminishing returns) |

**Root cause:** Original detection patterns missed core CPTSD terms: "trauma" (294 hits), "narcissist" (306), "abuse" (160), "complex trauma" (137), "childhood" (68), "scapegoat" (56), "gaslighting" (18), etc. Expanded from ~131 to ~299 total patterns across 10 topics.

## Train/Val/Test Split

Split by source file (no leakage across splits), seed=42.

| Split | Count | % | Topics | Stages | Crisis |
|-------|-------|---|--------|--------|--------|
| Train | 2,001 | 79.7% | 10/10 | 5/5 | 151 |
| Val | 275 | 11.0% | 10/10 | 5/5 | 28 |
| Test | 234 | 9.3% | 10/10 | 5/5 | 13 |

## Topic Coverage

| Topic | Count | % | Assessment |
|-------|-------|---|------------|
| triggers_trauma_responses | 1,304 | 52.0% | Well covered |
| boundary_setting | 930 | 37.1% | Well covered |
| shame_toxic_shame | 817 | 32.5% | Well covered |
| survival_responses | 755 | 30.1% | Well covered |
| recovery_stages | 702 | 28.0% | Well covered |
| inner_child_work | 484 | 19.3% | Well covered |
| emotional_regulation | 387 | 15.4% | Adequate |
| self_compassion | 214 | 8.5% | Adequate |
| healing_milestones | 196 | 7.8% | Adequate |
| emotional_flashbacks | 30 | 1.2% | Critical — needs synthetic augmentation |

## Recovery Stage Distribution

| Stage | Count | % |
|-------|-------|---|
| stabilization | 978 | 39.0% |
| awareness | 649 | 25.9% |
| integration | 565 | 22.5% |
| processing | 210 | 8.4% |
| thriving | 108 | 4.3% |

926 records originally had `None` recovery stage; fixed via topic-to-stage inference mapping.

## Crisis Detection

| Severity | Count |
|----------|-------|
| MEDIUM | 151 |
| HIGH | 41 |
| **Total** | **192** (7.7% of dataset) |

## PII Redaction

- All 3,036 records marked `pii_status: scrubbed`
- Post-hoc regex scan of assistant content: **0 email, 0 phone, 0 SSN, 0 URL hits**
- Builder uses regex-based PII redaction (Presidio not available in current environment)

## Remaining Drops (526)

470 (89.4%) are genuinely non-CPTSD content:
- German documentaries (WDR, ARTE, SWR Doku, Y-Kollektiv, ZDF)
- Church sermons (New Life Covenant)
- Comedy (LastWeekTonight)
- General wellness (Jay Shetty, Diary of a CEO, Chris Williamson)

56 (10.6%) are edge cases with CPTSD terms in non-CPTSD contexts:
- Domestic violence statistics (Kerry McAvoy)
- Couples therapy (Couples Therapy Official)
- German crime documentaries (Kaltblütig)
- Tim Fletcher biblical/theological tangents

## Issues & Next Steps

1. **`emotional_flashbacks` (30)** — still critically underrepresented. Synthetic dialogue generation needed (task `cptsd-synthetic`).
2. **`processing` (8.4%) and `thriving` (4.3%)** — underrepresented recovery stages. Synthetic augmentation should target these.
3. **`self_compassion` (8.5%) and `healing_milestones` (7.8%)** — could use more examples.

## Files

| File | Description |
|------|-------------|
| `cptsd_transcripts.jsonl` | Raw builder output (3,036 records) |
| `cptsd_filtered.jsonl` | Filtered to CPTSD-topic records (2,510) |
| `cptsd_final.jsonl` | Recovery stages fixed (2,510, no None) |
| `cptsd_train.jsonl` | Train split (2,001) |
| `cptsd_val.jsonl` | Val split (275) |
| `cptsd_test.jsonl` | Test split (234) |
| `cptsd_transcripts_stats.json` | Builder run statistics |
