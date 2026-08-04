# PAL Framework Implementation Plan

> Status: 5 of 6 phases implemented. Last update: 2026-08-03.

## Goal

Replicate the **Persona-Aware Alignment (PAL)** framework (arxiv:2511.10215v1, Li et al.) over the `Meddies/meddies-persona-vie` Vietnamese patient-persona dataset. Final form: a model that **strictly adheres to a complex patient persona** (demographic, socioeconomic, clinical) instead of producing generic next-token responses.

## Context & Constraints

* **Dataset:** ~150k Vietnamese patient-persona records from `Meddies/meddies-persona-vie`.
* **Stack:** HF `transformers` + `trl` + `peft` (QLoRA). No Lightning.
* **Pipeline:** SFT (mixed-task: persona selection + persona-conditioned dialogue) → DPO (preference pairs) → eval (paper §C.score).
* **Adaptation, not pure replication:** the paper uses PersonaChat (English, no medical framing); this plan applies the same two-stage SFT + DPO + C.score structure to Vietnamese medical-persona dialogue. The mechanics are the paper's; the data domain is ours.

## From the paper (kept verbatim)

* **Two-stage SFT.** Stage 1 = dialogue-informed persona selection from a list of distractors. Stage 2 = persona-conditioned dialogue generation.
* **DPO preference pairs.** Chosen = persona-adherent response. Rejected = persona-violating response.
* **C.score evaluation (§4).** Per-pair NLI label mapped to {-1, 0, +1}: entailment ⇒ persona-consistent, neutral ⇒ 0, contradiction ⇒ -1. Aggregate = mean over the dialogue.

## Our adaptations (intentional divergences — document these when reading)

* **Domain:** PersonaChat → Vietnamese medical dialogue (Meddies personas).
* **Persona source:** dataset ships Vietnamese enum values; we translate to English NL strings (paper specifies English NL personas).
* **DPO rejected-source.** Paper uses a persona-blind model's response as the rejected; we currently synthesize a jargon-style rejection (see Phase 3 caveats).
* **Backbone.** Paper's pipeline specifics are replicated structurally; backbone model choice is project-wide, not paper-pinned.

---

## Phase 1 — Persona Data Pipeline ✅

Translate dense Vietnamese Meddies JSON into fluid English NL persona strings.

* [x] **1.1 NL formatter** — `ai/training_corpus/wrapper/pal_framework/meddies_to_pal.py` (`format_persona`). Reads fixture-shape `{demographics, healthcare_behavior}`; emits a single NL paragraph.
* [x] **1.2 Schema adapter** — `ai/training_corpus/wrapper/pal_framework/meddies_adapter.py`. Maps real `Meddies/meddies-persona-vie` records (Vietnamese enum values, `demographics.province`, `healthcare_behavior.health_literacy_level`, `healthcare_seeking_pattern`) → fixture shape used by `meddies_to_pal`. Vietnamese → English translation tables for gender / health literacy / healthcare-seeking preference. CLI: `python meddies_adapter.py in.jsonl out.jsonl`.
* [x] **1.3 Tests** — `ai/training_corpus/wrapper/pal_framework/test_meddies_to_pal.py` (not `tests/utils/...` — that path in the prior plan was wrong). Verifies no JSON/brackets/quotes leak into NL output, fixture shapes, edge cases.

## Phase 2 — Mixed-task SFT ✅

Two SFT tasks per the paper, joined into one JSONL.

* [x] **2.1 Persona selection (Task 1)** — `ai/training_corpus/wrapper/pal_framework/generate_selection_dataset.py`. Builds prompts that ask the model to pick the correct persona from 3–4 distractors given a dialogue. PIX-4072.
* [x] **2.2 Persona-conditioned dialogue (Task 2)** — `ai/training_corpus/wrapper/pal_framework/generate_sft_dialogue.py`. Builds `messages`-format ChatML records; assistant turn = persona-adherent response. PIX-4073.
* [x] **2.3 Unified 10k JSONL** — `ai/training_corpus/wrapper/pal_framework/build_unified_sft.py`. Combines selection + dialogue into one mixed-task file with `task_type` metadata. ChatML-validated.

## Phase 3 — DPO Preference Pairs ✅ (with one open caveat)

* [x] **3.1 Generator** — `ai/training_corpus/wrapper/pal_framework/generate_dpo_pairs.py`. Emits TRL conversational DPO records: `{prompt, chosen: [{role,content}], rejected: [{role,content}]}`. PIX-4074.
* [x] **3.2 Linter** — `ai/training_corpus/wrapper/pal_framework/lint_dpo_dataset.py`. Validates TRL DPO conversational schema. PIX-4075.
* [x] **3.3 Trainer wiring** — `ai/training/dpo_trainer.py` (`load_preference_dataset`) coerces conversational message-list `chosen`/`rejected` to the standard string form the rest of the trainer expects. String-form records pass through. Logged as `"Coerced %d conversational (message-list) records to string form"` so silent schema mismatches surface. PIX-4076.
* [x] **3.4 Trainer tests** — `training/tests/test_dpo_trainer.py`: 16 tests including conversational-only, mixed-schema, missing-assistant-turn-skipped.
* [x] **3.5 Persona-blind rejected source** — `ai/training_corpus/wrapper/pal_framework/meddies_synthesizer.py` `_persona_blind_prompt` + `_llm_rejected_persona_blind`. When an `llm_client` is provided, the rejected side is now a real base-model roll-out with **no persona conditioning** (PAL paper definition), replacing the synthetic `_REJECTED_SYSTEM` jargon path. The offline rule-based fallback (`_synthesize_rejected_response_rule`) is kept for no-API-key runs so CI is unaffected. `_call_rejected_with_dialogue` dispatches by signature so old single-arg `rejected_fn`s still work. Tests in `test_meddies_synthesizer_llm.py` cover the persona-blind prompt (no persona leakage), the empty-LLM fallback, the dispatch by arity, the chosen/rejected split end-to-end through `build_dpo_input`.

> **Old caveat (closed):** DPO rejected-source was synthetic jargon. Now matches the paper: persona-blind base-model roll-out when an LLM client is wired, with the offline rule-based fallback for no-API-key runs.

### ⚠️ Safety Filter Status — DISABLED

The DPO trainer (`ai/training/dpo_trainer.py`) has its safety filter **disabled** at three locations:

* **Line 82** — docstring: `SAFETY FILTERING DISABLED PER USER REQUEST - ALL CONTENT ALLOWED`
* **Lines 125–135** — `load_preference_dataset()`: the `safety_checker.is_unsafe()` calls for both `chosen` and `rejected` responses are commented out. All content passes through regardless of safety classification.
* **Line 213** — `main()` entry point: comment `SAFETY FILTER DISABLED PER USER REQUEST - NO SAFETY CHECKER USED`; no safety checker is instantiated or passed to `load_preference_dataset`.

**What the filter does when enabled:** It would call `safety_checker.is_unsafe()` on each chosen/rejected response and skip any pair where either side is flagged, logging the skip. This prevents the model from being trained on unsafe content (e.g., harmful medical advice, self-harm references).

**Why it needs re-enabling:** The PAL framework trains on Vietnamese medical-patient dialogue — a domain where unsafe model outputs could cause real harm. The filter was disabled per user request for therapeutic training on difficult conversations, but should be re-enabled before any production training run. Tracked as PIX-4224.

## Phase 4 — Select-then-Generate Inference ✅

* [x] **4.1 Inference wrapper** — `ai/training_corpus/wrapper/pal_framework/inference_wrapper.py`. Two-stage inference: (1) select persona from dialogue, (2) condition generation on the selected persona. 2.0s latency budget. PIX-4077.

## Phase 5 — Evaluation (paper §C.score) ✅

> **This phase was missing from the original plan.** It was added in 2026-07 after audit.

* [x] **5.1 NLI persona-consistency eval** — `ai/training/pal_persona_consistency_eval.py`. CrossEncoder backend (default: `cross-encoder/nli-deberta-v3-base`) with a deterministic heuristic fallback for offline / CI runs and as an auto-fallback when the CrossEncoder fails to load (e.g., missing model weights, no network). Reports per-example label/score and aggregate C.score = mean over the dialogue. CLI: `python pal_persona_consistency_eval.py input.jsonl --output report.json`. Returns exit code 2 if the heuristic ran without `--force-heuristic` so operators don't mistake offline numbers for real results. PIX-4078.
* [x] **5.2 Tests** — `training/tests/test_pal_persona_consistency_eval.py`: 32 tests covering heuristic prediction rules (low-literacy + jargon → contradiction; preference echoed → entailment; location echoed → entailment; neutral default), `score_example` / `score_pairs` / `score_records`, `_extract_persona_response` for both `{persona, response}` and SFT `{messages, metadata.persona_string}` schemas, `_aggregate`, JSONL reader (blank lines, malformed), CLI (write report, missing input, `--limit`, model flag). PIX-4227.

## Phase 6 — End-to-end smoke ✅ (complete)

* [x] **6.1 Run SFT + DPO on a small subset.** Confirm checkpoint shape, inference latency, and that the eval C.score moves (or at least doesn't crater) when persona conditioning is on vs. off. PIX-4227.
  > **Test infrastructure verified 2026-08-03 (PIX-4223):** `test_dpo_trainer.py` (16 tests) and `test_dry_run_sft_overfit.py` (14 tests) both pass. C.score eval verified end-to-end with 5 synthetic PAL records (c_score=0.4, exit code 0 with `--force-heuristic`).
  > **Smoke run completed 2026-08-04 (PIX-4227):** `run_phase6_smoke.py` ran on the `layup` GPU host (Tesla V100S-PCIE-32GB) on a tiny model (`sshleifer/tiny-gpt2`) over 16 SFT + 20 DPO synthetic PAL records. Results: SFT checkpoint ✅ saves adapter_config.json + adapter_model.safetensors (loss=10.828 elapsed=1.61s); DPO checkpoint ✅ saves adapter_config.json + adapter_model.safetensors (loss=0.693 elapsed=1.27s); inference within budget ✅ (mean 0.067s, max 0.114s vs. 120s CPU-smoke budget; production AC retains 2s on A100 per PIX-4077); C.score ✅ computed persona ON (0.0) vs OFF (0.0), heuristic backend, 4 scored records each. Persona-conditioning delta is 0 on the tiny untrained smoke model — expected for a smoke without full alignment training; the contract is that the harness runs end-to-end and the metric is computable, which it is. Results JSON at `training_corpus/wrapper/pal_framework/phase6_smoke_results.json`. All 377 PAL tests still pass post-smoke.

## File map

| Phase | File | Purpose |
|---|---|---|
| 1.1 | `training_corpus/wrapper/pal_framework/meddies_to_pal.py` | Fixture → NL persona |
| 1.2 | `training_corpus/wrapper/pal_framework/meddies_adapter.py` | Real Meddies → fixture |
| 1.3 | `training_corpus/wrapper/pal_framework/test_meddies_to_pal.py` | NL formatting tests |
| 2.1 | `training_corpus/wrapper/pal_framework/generate_selection_dataset.py` | SFT Task 1 |
| 2.2 | `training_corpus/wrapper/pal_framework/generate_sft_dialogue.py` | SFT Task 2 |
| 2.3 | `training_corpus/wrapper/pal_framework/build_unified_sft.py` | Unified mixed-task JSONL |
| 3.1 | `training_corpus/wrapper/pal_framework/generate_dpo_pairs.py` | DPO pair generation |
| 3.2 | `training_corpus/wrapper/pal_framework/lint_dpo_dataset.py` | DPO schema lint |
| 3.3 | `training/dpo_trainer.py` | Trainer loader (schema coerce) |
| 3.4 | `training/tests/test_dpo_trainer.py` | Trainer tests |
| 3.5 | `training_corpus/wrapper/pal_framework/meddies_synthesizer.py` (`_persona_blind_prompt`, `_llm_rejected_persona_blind`) | Persona-blind rejected source |
| 4.1 | `training_corpus/wrapper/pal_framework/inference_wrapper.py` | Select-then-Generate inference |
| 5.1 | `training/pal_persona_consistency_eval.py` | C.score NLI eval |
| 5.2 | `training/tests/test_pal_persona_consistency_eval.py` | Eval tests |
| — | `training/pal_dataloader.py` (271 lines) | PAL-aware SFT/DPO dataloaders for HuggingFace Trainer |
| — | `api/pal_inference_service.py` (364 lines) | FastAPI microservice for two-stage PAL inference |
| — | `training/tests/test_pal_dataloader.py` | PAL dataloader tests |
| — | `training/tests/test_dry_run_sft_overfit.py` | SFT dry-run overfit test infra (14 tests) |
| — | `training_corpus/wrapper/pal_framework/test_meddies_synthesizer_llm.py` | Persona-blind rejected source tests |
| 6.1 | `training_corpus/wrapper/pal_framework/run_phase6_smoke.py` | E2E smoke: SFT + DPO checkpoint save + inference + C.score (run on layup V100 GPU) |
| 6.1 | `training_corpus/wrapper/pal_framework/phase6_smoke_results.json` | Smoke run results (PASS 2026-08-04) |

## References

* **Paper:** [PAL: Persona-Aware Alignment](https://arxiv.org/abs/2511.10215v1) (Li et al., Nov 2025)
* **Dataset:** `Meddies/meddies-persona-vie` (HuggingFace)
* **Linear issues:** PIX-4070 (data), PIX-4072 (SFT-1), PIX-4073 (SFT-2), PIX-4074 (DPO gen), PIX-4075 (DPO lint), PIX-4076 (DPO trainer), PIX-4077 (inference), PIX-4078 (eval — verified 2026-08-03: Bias Detection Engine Overhaul, status Done), PIX-4227 (Phase 5.2 + 6.1 tracking).
