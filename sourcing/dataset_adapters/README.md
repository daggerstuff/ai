# Dataset Adapters — Clinical AI Training Data Ingestion

## Overview

This directory contains 22 dataset adapters that download, extract, and convert clinical AI datasets into standardized ChatML JSONL format for LLM training.

## Architecture

```
dataset_adapters/
  base_adapter.py       # BaseDatasetAdapter ABC (download → extract → convert → validate → save)
  adapter_factory.py    # @register_adapter decorator + get_adapter() factory
  <name>_adapter.py × 22  # One file per dataset
```

## Standardized Output Schema

Each adapter outputs JSONL with this schema:

```json
{
  "messages": [{"role": "system|user|assistant", "content": "..."}],
  "source": "dataset_name",
  "task_type": "symptom_classification|severity_estimation|therapy_response_generation|risk_assessment|empathy_scoring|dpo_preference|adversarial_safety",
  "diagnostic_tag": "string|null",
  "demographic_tags": ["age_18_25", "gender_female", ...],
  "linguistic_style": "formal|informal|mixed",
  "clinical_reviewed": false,
  "provenance": {
    "source_url": "...",
    "access_method": "direct_download|github|huggingface|kaggle|request",
    "original_format": "csv|json|sharegpt|sql|csv_audio|hf_dataset",
    "transformations": ["download", "extract", "convert_to_chatml", "validate"],
    "extracted_at": "ISO timestamp"
  }
}
```

## Datasets (22 total)

### Phase 1: Therapy & Adversarial (10 adapters, 104 tests)
| Adapter | Dataset | Task Type | Access |
|---------|---------|-----------|--------|
| esconv | ESConv | therapy_response_generation | GitHub |
| hope | HOPE | therapy_response_generation | GitHub (clone) |
| mitags | MI-TAGS | therapy_response_generation | GitHub (clone) |
| memo | MEMO | therapy_response_generation | Request |
| mit_psychosis | MIT ai-psychosis | adversarial_safety | GitHub |
| vera_mh | VERA-MH | adversarial_safety | GitHub |
| sim_vail | SIM-VAIL | adversarial_safety | GitHub |
| empath | EMPATH | empathy_scoring | GitHub |
| clinical_redteam | Clinical Red Teaming | adversarial_safety | GitHub |
| psydial | PsyDial | therapy_response_generation | GitHub |

### Phase 2: Clinical Pathology & Longitudinal (9 adapters, 73 tests)
| Adapter | Dataset | Task Type | Access |
|---------|---------|-----------|--------|
| clpsych | CLPsych Shared Tasks | risk_assessment | Request |
| erisk | eRisk (CLEF) | severity_estimation | Request |
| personalitydbench | PersonalityDBench | symptom_classification | Request |
| annomi | AnnoMI | therapy_response_generation | GitHub |
| ml_bpd | machine_learning_BPD | severity_estimation | GitHub |
| bopd | BoPD | symptom_classification | GitHub (clone) |
| dmtcorpus | DMTCorpus | therapy_response_generation | Request |
| mhsafeeval | MHSafeEval | adversarial_safety | HuggingFace |
| crisis_benchmark | Crisis Benchmark | risk_assessment | HuggingFace |

### Phase 3: Higher-Effort Access (3 adapters, 24 tests)
| Adapter | Dataset | Task Type | Access |
|---------|---------|-----------|--------|
| daic_woz | DAIC-WOZ | severity_estimation | Request |
| bbrd | BBRD | symptom_classification | Request (CC BY-NC) |
| reddit_mental_nlp | Mental Disorders Reddit NLP | symptom_classification | Kaggle |

## Usage

### Download all datasets
```bash
uv run python -m ai.sourcing.scripts.download_all --output-dir ai/data/raw
```

### Download one dataset
```bash
uv run python -m ai.sourcing.scripts.download_all --dataset esconv
```

### List available adapters
```bash
uv run python -m ai.sourcing.scripts.download_all --list
```

### Run tests
```bash
uv run python -m pytest ai/sourcing/tests/ -q
```

## Pipeline Integration

Adapters output to `ai/data/raw/<dataset>/<dataset>.jsonl`. The `unified_dataset_pipeline.py` ingests this directory alongside existing sources (YouTube, journals, training-ready). The existing pipeline then handles:
- PII scrubbing (`pii_scrubber.py`)
- Clinical validity scoring (`clinical_validity_scorer.py`)
- Merging & deduplication (`merge_final_dataset.py`)
- Train/val/test splits

## Notes

- **Request-based adapters** (MEMO, CLPsych, eRisk, PersonalityDBench, DMTCorpus, DAIC-WOZ, BBRD) create a README.txt with manual download instructions. Place data files in the raw directory after obtaining access.
- **Git clone adapters** (HOPE, MI-TAGS, BoPD) clone the full repo and extract data files from it.
- **HuggingFace adapters** (MHSafeEval, Crisis Benchmark) create README instructions for `datasets` library export.
- **Kaggle adapter** (Reddit Mental NLP) attempts Kaggle API download, falls back to manual instructions.
