# Persona-Aware Alignment Framework (PAL) Implementation Plan

## Goal
Implement the PAL (Persona-Aware Alignment) framework as described in arxiv:2511.10215v1, using the `Meddies/meddies-persona-vie` dataset as our foundational persona repository. This will train our models to strictly adhere to patient personas across demographic, socioeconomic, and clinical dimensions rather than regressing to generic responses.

## Constraints & Context
- **Base Personas:** 150,000 synthetic Vietnamese patient records from `Meddies/meddies-persona-vie`.
- **Primary Objective:** Moving beyond Next-Token Prediction (NTP) to explicit Persona Alignment via DPO (Direct Preference Optimization).
- **Target Files:** New scripts in `ai/training_corpus/pal_framework/`.

## Phase 1: Meddies Schema to Natural Language (Prompt Construction)
The PAL framework requires personas to be represented in natural language strings.
- **Action:** Build `meddies_to_pal.py` to extract dense Meddies fields (Demographics, Healthcare Behavior, LLM-facing fields) and format them into coherent paragraph descriptions (e.g., "This patient is a 45-year-old female from Hanoi with low health literacy who prefers traditional medicine...").
- **Verification:** `pytest tests/utils/test_pal_persona_formatting.py` ensuring no JSON/dict structures leak into the prompt output.

## Phase 2: Persona-aware Learning (SFT Stage)
Implement the mixed-task SFT pipeline (Task 1 + Task 2) from the PAL paper.
- **Task 1: Dialogue-Informed Persona Selection:** 
  - Generate synthetic dialogue using a specific persona.
  - Create prompts that ask the model to select the correct persona from a list of distractors.
- **Task 2: Persona-Enhanced Dialogue Generation:**
  - Standard prompt: "Given this persona: [Meddies NL string] and this dialogue history, generate the next response."
- **Verification:** Run a dry-run generation of 10,000 mixed-task JSONL records and validate formatting against ChatML.

## Phase 3: Persona Alignment (DPO Stage)
Construct preference pairs to explicitly penalize out-of-character generations.
- **Action:** Build `generate_dpo_pairs.py`.
- **Chosen (y_w):** The generated response that adheres to the persona (e.g., uses appropriate dialect, respects low health literacy).
- **Rejected (y_l):** A contrasting response that violates the persona (e.g., uses high medical jargon for a low-literacy patient, ignores a cultural health bias).
- **Verification:** Ensure dataset complies with standard HuggingFace `trl` DPO formatting (prompt, chosen, rejected).

## Phase 4: Integration & Training
- Integrate the generated datasets into the main Lightning/Torch training pipeline.
- Define a "Select then Generate" inference wrapper for production use.
