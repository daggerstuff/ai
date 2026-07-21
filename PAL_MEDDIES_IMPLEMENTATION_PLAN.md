# 🚀 PAL Framework Implementation Plan (v2)

## 🎯 Goal
Implement the **Persona-Aware Alignment (PAL)** framework (arxiv:2511.10215v1) using the `Meddies/meddies-persona-vie` dataset as our foundational persona repository. The ultimate objective is to train our models to **strictly adhere to complex patient personas** across demographic, socioeconomic, and clinical dimensions, moving beyond generic Next-Token Prediction (NTP) to explicit Persona Alignment via Direct Preference Optimization (DPO).

---

## 📋 Context & Constraints
* **Dataset:** 150,000 synthetic Vietnamese patient records from `Meddies/meddies-persona-vie`.
* **Objective:** Replace generic model responses with highly specific, persona-aligned dialogue.
* **Core Mechanisms:** Supervised Fine-Tuning (SFT) followed by Direct Preference Optimization (DPO).

---

## 🛠️ Implementation Phases

### 🟩 Phase 1: Natural Language Persona Prompting (COMPLETED ✅)
*We need to translate dense JSON Meddies records into fluid, natural language strings that the LLM can interpret organically.*

* **[x] Task 1.1: Build Extraction Pipeline**
  * Created `ai/training_corpus/pal_framework/meddies_to_pal.py`.
  * Extracts demographics, healthcare behavior, and LLM-facing fields.
  * Formats into a clean paragraph (e.g., *"This patient is a 45-year-old female from Hanoi with low health literacy..."*).
* **[x] Task 1.2: Validation & Testing**
  * Created `tests/utils/test_pal_persona_formatting.py`.
  * Validated that absolutely **no JSON formatting, brackets, or quotes leak** into the natural language output.

### 🟨 Phase 2: Persona-Aware Learning (SFT Stage)
*Implement the mixed-task Supervised Fine-Tuning pipeline defined in the PAL paper.*

* **[ ] Task 2.1: Dialogue-Informed Persona Selection (Task 1)**
  * **Objective:** Train the model to deduce a persona from dialogue.
  * **Action:** Generate synthetic dialogue using a specific persona. Create prompts asking the model to select the correct persona from a list of 3-4 distractors.
  * **File:** `ai/training_corpus/pal_framework/generate_selection_dataset.py`
* **[ ] Task 2.2: Persona-Enhanced Dialogue Generation (Task 2)**
  * **Objective:** Train the model to roleplay the persona.
  * **Action:** Construct standard prompts: *"Given this persona: [Meddies NL string] and this dialogue history, generate the next response."*
  * **File:** `ai/training_corpus/pal_framework/generate_sft_dialogue.py`
* **[ ] Task 2.3: Dry-Run SFT Validation**
  * Generate a unified 10,000-record mixed-task JSONL file.
  * Validate output against strict ChatML formatting requirements.

### 🟧 Phase 3: Persona Alignment (DPO Stage)
*Construct preference pairs to explicitly penalize out-of-character or generic generations.*

* **[ ] Task 3.1: Construct Preference Pairs**
  * **Action:** Build `ai/training_corpus/pal_framework/generate_dpo_pairs.py`.
  * **Chosen ($y_w$):** The generated response that adheres to the persona (e.g., uses appropriate dialect, respects low health literacy).
  * **Rejected ($y_l$):** A contrasting response that violates the persona (e.g., uses high medical jargon for a low-literacy patient, ignores a cultural health bias).
* **[ ] Task 3.2: Formatting & Linting**
  * Ensure the dataset strictly complies with the standard HuggingFace `trl` DPO format (`prompt`, `chosen`, `rejected`).

### 🟥 Phase 4: Integration & Training
*Wire the datasets into the main Lightning/Torch pipeline for actual training.*

* **[ ] Task 4.1: Model Pipeline Integration**
  * Wire the generated SFT and DPO datasets into the existing Torch training loop.
* **[ ] Task 4.2: Inference Wrapper**
  * Define a "Select then Generate" inference wrapper for production use, forcing the model to select a persona before generating the final response.

---

## 🔗 Links & Resources
* **Paper Reference:** [PAL: Persona-Aware Alignment](https://arxiv.org/abs/2511.10215v1)
* **Dataset Repo:** `Meddies/meddies-persona-vie`
* **Target Directory:** `ai/training_corpus/pal_framework/`
