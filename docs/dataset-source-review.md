# Dataset source inventory — human review (PROPOSALS, not verdicts)

> **Status: AWAITING HUMAN SIGN-OFF.**
>
> The following admissibility calls were previously hardcoded in
> `ai/training_corpus/source_inventory.py` and acted as the de-facto ingest gate
> for the training corpus builder (`builder.py` skips any source where
> `inventory_decision != "keep"`). That is a human/dataset-owner decision, not a
> code decision. The verdicts have been removed from the code and recorded here
> as **proposals** for review. Until a human signs off below, every source stays
> `defer` / `unknown` and the builder ingests nothing.

## How to apply a decision

For each row a human reviewer accepts, edit the source's entry in the dataset
registry (or a per-source overrides file) to set the fields the builder reads:

- `inventory_decision` — `keep` | `defer` | `reject`
- `rights_status` — `cleared` | `review_required` | `restricted` | `unknown`
- `license_status` — human-determined license classification
- `allowed_lanes` — subset of `simulation`, `policy`, `evaluator`, `benchmark`
- `default_lane` — one of the allowed lanes, or `null`

Until then `source_inventory._build_source` emits neutral defaults
(`defer` / `unknown` / empty lanes / `not_eligible`) for every source.

## Group defaults (previously `_GROUP_DEFAULT_DECISIONS`)

| Group | Proposed decision | Proposed rights | Proposed lanes | Proposed default lane | Rationale |
|---|---|---|---|---|---|
| `cot_reasoning` | defer | review_required | evaluator, benchmark | evaluator | Keep out of simulation training unless experiments show no rubric leakage. |
| `professional_therapeutic` | keep | review_required | simulation, evaluator, benchmark | simulation | High-trust simulation foundation pending rights verification per source. |
| `therapeutic` | reject | restricted | — | — | Compiled training outputs are release artifacts, not admissible source inputs. |
| `voice_persona` | defer | restricted | simulation, benchmark | simulation | Transcript-derived identity material may only be used after archetype extraction. Direct celebrity or educator mimicry is out of scope. |
| `wendy_curated_sets` | keep | review_required | simulation, evaluator, benchmark | simulation | Curated priority conversations fit the foundation lane if provenance is confirmed. |

## Training V3 by stage (previously `_training_v3_decision`)

| Stage | Proposed decision | Proposed rights | Proposed lanes | Proposed default lane | Rationale |
|---|---|---|---|---|---|
| `stage1_foundation` | keep | review_required | simulation, benchmark | simulation | Fresh stage-native foundation sources remain eligible for client simulation. |
| `stage2_specialist` / `stage2_therapeutic_expertise` | keep | review_required | simulation, evaluator, benchmark | simulation | Specialist material supports difficult-domain simulation and rubric coverage. |
| `stage3_edge_stress_test` | keep | review_required | simulation, policy, benchmark | simulation | Stage-native severe scenarios remain eligible for controlled edge-case experimentation. |
| `stage4_voice` / `stage4_voice_persona` | keep | review_required | simulation, benchmark | simulation | Only stage-native persona corpora are eligible. Do not reuse transcript-derived identity exports in this lane. |
| (other) | defer | unknown | — | — | Training V3 source requires manual stage review. |

## Edge-case sources (previously `_edge_case_decision`)

| Dataset name pattern | Proposed decision | Proposed rights | Proposed lanes | Proposed default lane | Rationale |
|---|---|---|---|---|---|
| `*_seed_benchmark` | keep | review_required | benchmark | benchmark | Synthesized benchmark seeds remain eligible as benchmark-only holdout assets. |
| `edge_case_generator*` / `edge_simulation*` / `edge_benchmark*` | keep | review_required | simulation, policy, benchmark | simulation | Preserve severe scenarios as tagged training assets, not runtime bypasses. |
| `safety_dpo_pairs*_merged` / `edge_policy*_merged` | keep | review_required | simulation, benchmark | simulation | Experiment overlay merges policy transforms into simulation for comparison only. |
| `safety_dpo_pairs*` / `edge_policy*` (non-merged) | keep | review_required | policy, benchmark | policy | Use preference pairs only in the policy lane, never as blended simulation data. |
| `scenario_prompt_library*` | defer | review_required | evaluator, benchmark | benchmark | Prompt seeds are suitable for scenario design, not direct training examples. |
| (any other edge_case_sources) | reject | restricted | — | — | Raw forum and bundle sources fail provenance and contamination requirements. |

## Supplementary (previously `_supplementary_decision`)

| Dataset name pattern | Proposed decision | Proposed rights | Proposed lanes | Proposed default lane | Rationale |
|---|---|---|---|---|---|
| `*_seed_evaluator` | keep | review_required | evaluator, benchmark | evaluator | Synthesized evaluator seeds remain eligible for rubric and benchmark enrichment. |
| `psychology_10k*` / `evaluator_psychology*` | keep | review_required | evaluator, benchmark | evaluator | Knowledge assets belong in evaluator and benchmark enrichment, not simulation. |
| `legacy_compiled_dataset_csv` | reject | restricted | — | — | Legacy compiled mixtures violate the fresh-namespace rewrite boundary. |
| `psychology_10k` / `academic_psychology_books` / `research_instruments` | keep | review_required | evaluator, benchmark | evaluator | Knowledge assets belong in evaluator and benchmark enrichment, not simulation. |
| (other supplementary) | defer | review_required | evaluator, benchmark | evaluator | Consolidated research assets need per-source provenance review before release. |

## Previously inferred license classifications (previously `_license_status`)

These were code-inferred; a human reviewer should confirm the actual license
per source rather than relying on the inference.

| Condition | Previously inferred license_status |
|---|---|
| `professional_therapeutic` with "licensed" in focus | licensed |
| `professional_therapeutic` or `wendy_curated_sets` | review_required |
| `training_v3` | derived_internal |
| `edge_case_sources` starting `safety_dpo_pairs`/`edge_policy` | review_required |
| other `edge_case_sources` | internal_synthetic |
| `voice_persona` | restricted_identity_source |
| `supplementary` == `legacy_compiled_dataset_csv` | prohibited_legacy_mix |
| other `supplementary` | review_required |
| path starts `s3://pixel-data/datasets/consolidated` | compiled_derivative |

## Sign-off

Reviewer: _<name>_
Date: _<YYYY-MM-DD>_

Accepted decisions: _list source_ids + the fields to set_
Rejected / changed decisions: _list source_ids + the corrected fields_
