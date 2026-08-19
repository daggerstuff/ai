# HANDOFF — PR #526 Remaining P1 Correctness Fixes

**Branch:** `feature/sync-local-pipeline-work` in `daggerstuff/ai`
**PR:** #526 ("feat: sync local pipeline work to staging (99 files, junk excluded)")
**Last commit:** `bb3b4fcf` — acquisition_rubric fixes
**Date:** 2026-08-18

## Context

PR #526 received a full Sourcery/Cubic AI review with 40 issues across 99 files. The review body was saved to `/tmp/pr526_review_body.txt`. Issues were classified P0 (critical) through P1 (correctness) and are being fixed in order.

## Completed Work

### P0 fixes (committed in `49031b28`)
1. **`test_normalization_pipeline.py:68`** — SyntaxError: deleted first empty duplicate `test_normalization_pipeline_writes_duplicate_evidence` definition
2. **`normalization_pipeline.py`** — P1 data-loss: per-file `.rejected.tmp.jsonl` aggregated into final `reject_path` as JSONL
3. **`analytics_dashboard.py:193-208`** — P1 data-loss: `export_report` now writes actual JSON/CSV from `self._report_data`
4. **`crisis_intervention_detector.py`** — P1 safety: tightened keyword patterns, added `NEGATION_TERMS` and `_is_negated()`
5. **`privacy_content_gates.py`** — P0 Gate 4 override: `PrivacyContentReport.passed` now preserves mandatory `BLOCK` from Gates 0–3
6. **`transcript_quality_pipeline.py`** — P0 crisis-detection no-op: `detect_crisis_narratives` now raises `NotImplementedError`

### P1 correctness batch 1 (committed in `49031b28`)
7. **`slicer_s3_streaming.py`** — `rclone cat` exit-code check, malformed JSON tracking, conditional `downstream_ready`
8. **`slicer_fast.py`** — exit-code check, conditional `downstream_ready`
9. **`data_splitter.py`** — per-ratio non-negative validation
10. **`stage_slicer_enhanced.py`** — removed duplicate docstring/dead code, replaced hard-coded `safety_scores.append(1.0)` with heuristic, gated output on `validation_result.passed`, `downstream_ready` defaults to `False`

### P1 correctness batch 2 (committed in `bb3b4fcf`)
11. **`acquisition_rubric.py`** (both `core/pipelines/` and `pkg_mera/core/pipelines/`):
    - Exception-eligible licenses (`cc-by-nc-4.0`, `cc-by-nc-sa-4.0`) now block intake until explicit exception grant
    - `promote()` verifies `pilot.source_id` and `curation_exit.source_id` match `intake.source_id`
    - CLI `score` subparser generates distinct flags instead of four duplicate `--data-structure-quality`
    - Added 2 source_id mismatch tests; 35 tests pass

## Remaining P1 Correctness Issues (10 tasks)

### 1. `p1_clean_pii` — `core/pipelines/processing/clean.py:168`
**Issue:** PII required-column exemption. Columns marked as "required" may be exempted from PII checking, allowing PII to slip through if a column happens to be in the required list.

### 2. `p1_ears_gate` — `core/pipelines/ears_compliance_gate.py:56`
**Issue:** All non-empty payloads pass the EARS compliance gate regardless of actual content. The gate should validate structure/content, not just non-emptiness.

### 3. `p1_pii_scrubber` — `core/pipelines/processing/pii_scrubber.py`
Three sub-issues:
- **`:148`** — Unsalted MD5 hash for PII pseudonymization (reversible via rainbow tables)
- **`:205`** — `ent.confidence` AttributeError (entity object may not have `confidence` attribute)
- **`:280`** — Overlapping spans discarded silently instead of being resolved/merged

### 4. `p1_crisis_expansion` — `core/pipelines/processing/crisis_expansion.py:477`
**Issue:** Negated crisis expansions are flagged. Similar to the crisis_intervention_detector fix — needs negation context checking. Pattern: apply the same `NEGATION_TERMS` / `_is_negated()` approach used in `crisis_intervention_detector.py`.

### 5. `p1_clinical_validator` — `core/pipelines/clinical_accuracy_validator.py:17`
**Issue:** Keyword-heuristic stub. The clinical accuracy validator is a keyword-matching stub rather than a real clinical validation. This is a design-level issue — for now, document the limitation and add a `NotImplementedError` guard or warning if the stub is being used in production paths. Ask user before implementing a real validator.

### 6. `p1_packaging` — `core/pipelines/packaging.py`
Two sub-issues:
- **`:111`** — Stale promotion token. Promotion tokens may persist past their intended lifecycle.
- **`:220`** — Package directory collision. No check for existing package directories before writing.

### 7. `p1_orchestrator_hash` — `core/pipelines/pipeline_orchestrator.py:123`
**Issue:** Input hash is computed after the input data has been mutated, making the hash meaningless for integrity verification. Fix: compute hash before any processing/modification.

### 8. `p1_pattern_analyzer` — `core/pipelines/conversation_quality_pattern_analyzer.py:933`
**Issue:** Overlap/entity limits defined in config but never enforced. The analyzer has configurable limits for conversation overlap and entity counts but doesn't actually apply them.

### 9. `p1_reprioritization` — `core/pipelines/reprioritization_engine.py:302`
**Issue:** Feedback points are not recorded. The reprioritization engine should log/store feedback points for audit trail but currently discards them.

### 10. `p1_verify_all` — Final verification
Run the full test suite (`python -m pytest --tb=short`) to verify all P1 fixes pass. Known pre-existing failures (should NOT be fixed, they're unrelated):
- 3 `test_human_review_queue.py` failures (`ModuleNotFoundError: No module named 'ai.training_corpus.rewrite_contracts'`)
- 2 `test_privacy_content_gates.py` failures (missing spaCy `en_core_web_sm` PII tier classification)

## Key Conventions

- **Two copies of files:** Many pipeline files exist in both `core/pipelines/` and `pkg_mera/core/pipelines/`. Fix both unless they differ.
- **`GateResult` dataclass:** fields are `gate: str`, `decision: GateDecision`, `details: str` (NOT `reason`).
- **Test runner:** `python -m pytest` (use `uv` for Python per AGENTS.md).
- **Anti-suppression:** No `# noqa`, `# type: ignore`, `@ts-ignore`, etc. Fix the issue, never hide it.
- **Verify before commit:** Run targeted pytest for each file before committing.
- **Commit style:** `fix:` prefix, bullet list of changes, `Co-Authored-By: Mastra Code (cloudflare-workers-ai/@cf/zai-org/glm-5.2) <noreply@mastra.ai>`

## File Locations for Remaining Issues

All files are under `/home/vivi/pixelated/ai/`:
```
core/pipelines/processing/clean.py
core/pipelines/ears_compliance_gate.py
core/pipelines/processing/pii_scrubber.py
core/pipelines/processing/crisis_expansion.py
core/pipelines/clinical_accuracy_validator.py
core/pipelines/packaging.py
core/pipelines/pipeline_orchestrator.py
core/pipelines/conversation_quality_pattern_analyzer.py
core/pipelines/reprioritization_engine.py
```

## Review Body

Full review text saved at `/tmp/pr526_review_body.txt` on this machine. Search for the file:line references above to get exact review comments.
