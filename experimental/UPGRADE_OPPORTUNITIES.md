# Upgrade Opportunities & Future Expansions

Now that the core **Supervised Fine-Tuning (SFT)** pipeline is complete, here are the most impactful areas for expansion to reach "State of the Art" performance.

## 1. DPO/RLHF Pipeline (Alignment)
**The "Good vs. Bad" Update**
Currently, we generate instructions (SFT). To make the model truly prefer "safe/empathetic" responses over "cold/clinical" ones, we need **Direct Preference Optimization (DPO)**.
*   **Upgrade**: Update `DatasetSynthesizer` to generate *pairs* of responses:
    *   `chosen`: Empathetic, validated response (Tim Fletcher style).
    *   `rejected`: Dismissive, overly clinical, or advice-giving response.
*   **Goal**: Train the model to *reject* harmful behaviors implicitly.

## 2. "TherapyBench" Evaluation Suite
**The "Standardized Test" Update**
We have `ClinicalValidator` for individual items, but we lack a holistic benchmark.
*   **Upgrade**: Create a static set of 500 "Golden Questions" (e.g., "I feel suicidal," "My partner hit me").
*   **Action**: build a script that runs the model against these 500 prompts and uses GPT-4 to grade them on Empathy, Safety, and Reflection.
*   **Metric**: "Empathy Score" (0-100).

## 3. RAG-Augmented Inference
**The "Open Book" Update**
Don't just rely on the model's weights.
*   **Upgrade**: Index the `academic_literature` (ArXiv papers) and `stage1_foundation` data into a VectorDB (Chroma/Pinecone).
*   **Action**: Before the bot answers, it queries the database for "CBT protocols for Panic Attacks" and uses that context.
*   **Result**: Reduced hallucination, citation-backed therapy.

## 4. Multi-Modal "Presence"
**The "Visual/Audio" Update**
Therapy is 90% non-verbal.
*   **Upgrade**: If we can source video data (not just transcripts), we can train a multi-modal adapter (like LLaVA) to recognize facial expressions (tears, anger).
*   **Audio**: Train a VITS/Tortoise TTS model specifically on the **Tim Fletcher** audio samples to give the bot his exact voice.

## 5. "Agentic" Therapist
**The "Proactive" Update**
Turn the model into an Agent with memory.
*   **Upgrade**: Give the model a `MemoryLayer` (Summarization of past sessions).
*   **Action**: "You mentioned your mother last week; how is that going?"
*   **Tech**: LangGraph or vector memory integration.
