# Kaggle NLP Dataset Research Summary
## Project Nightmare Fuel — Phase 1.2 — PIX-4237

**Goal:** Map Kaggle NLP datasets targeting BPD splitting and CPTSD addiction clinical edge cases.
**Date:** 2026-08-05
**Status:** Research complete — no datasets downloaded, mapping + summary only.

---

## Top 10 Most Relevant Datasets

Ranked by composite relevance to Nightmare Fuel generator seed categories (BPD, CPTSD, addiction, trauma).

### 1. BPD and Behaviour Reddit Dataset (BBRD) — Lancaster University
- **URL:** https://research.lancaster-university.uk/en/datasets/bpd-and-behaviour-reddit-dataset-bbrd/
- **Format:** CSV (gated, sign Data Usage Agreement)
- **Size:** 68,590 posts from 992 unique users; ~17,000 manually annotated
- **License:** CC BY (non-commercial research, DUA required)
- **Relevance:** BPD=10
- **Contents:** r/BPD and r/BorderlinePDisorder posts, Oct 2011–Dec 2023. Trained-rater annotations for suicidality (attempts/ideation, recent/past), self-harm, medication, therapy, substance use, impulsive behaviors, psychotic symptoms, intense emotions.
- **Preprocessing:** Text pre-cleaned. Use behavior annotation columns as supervised labels for BPD splitting cycles (idealization→devaluation). Link posts per user_id for temporal Idealization/Devaluation pattern extraction.
- **Maps to:** BPD splitting generator seed (PRIMARY). Intense Emotions + Impulsive Behaviors columns directly encode splitting-adjacent phenomena.

### 2. Mental Disorders Identification (Reddit NLP) — Kaggle
- **URL:** https://www.kaggle.com/datasets/kamaruladha/mental-disorders-identification-reddit-nlp
- **Format:** CSV
- **Size:** ~701,809 rows
- **License:** Kaggle terms (public)
- **Relevance:** BPD=10, CPTSD=8
- **Contents:** Reddit posts categorized by subreddit origin: BPD, Anxiety, Depression, Bipolar, Mental Illness, Schizophrenia. `combined_text` field (title + body).
- **Preprocessing:** Combine title+body → `combined_text`. Remove nulls. Clean punctuation/stopwords, lemmatize, tokenize. Word2Vec or BERT embeddings.
- **Maps to:** BPD (direct subreddit label). CPTSD proxy via mental illness class (no explicit CPTSD subreddit, but PTSD-adjacent content mixed in). Largest single BPD-labeled Kaggle corpus.

### 3. Mental Health Condition Classification — Kaggle
- **URL:** https://www.kaggle.com/datasets/haideradnan77/mental-health-condition-classification
- **Format:** CSV
- **Size:** 103,488 rows, 46.53 MB
- **License:** Kaggle terms (public)
- **Relevance:** BPD=9
- **Contents:** 7-class: Anxiety, Normal, Depression, Stress, Personality Disorder, Bipolar, Suicidal. First-person reflections, emotional descriptions, symptom narratives.
- **Preprocessing:** Lowercase, strip special chars/URLs, lemmatize. Balance "Personality Disorder" class (only 17% of rows). BERT/DistilBERT fine-tuning.
- **Maps to:** BPD via "Personality Disorder" label (closest Kaggle proxy — captures unstable moods/behaviors characteristic of BPD splitting).

### 4. Sentiment Analysis for Mental Health — Kaggle
- **URL:** https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
- **Format:** CSV
- **Size:** 51,074 rows, ~30 MB
- **License:** CC BY-SA 4.0
- **Relevance:** BPD=9
- **Contents:** Amalgamation of 8 Kaggle sources. 7-class: Normal, Depression, Suicide, Anxiety, Stress, Bipolar, Personality Disorder.
- **Preprocessing:** LabelEncoder for 7-class target. Clean text. BERT tokenizer max_length=128. Stratified 80/20 split.
- **Maps to:** BPD via "Personality Disorder" label. CC BY-SA 4.0 license allows redistribution — strongest licensing posture of any Kaggle BPD proxy.

### 5. Reddit Mental Health Dataset (Low et al.) — Zenodo (non-Kaggle, included due to high relevance)
- **URL:** https://zenodo.org/records/3941387
- **Format:** CSV per subreddit per timeframe
- **Size:** 826,961 unique users across 28 subreddits, 2018–2020
- **License:** CC BY (research)
- **Relevance:** BPD=10, CPTSD=10, addiction=8
- **Contents:** r/bpd, r/ptsd, r/addiction, r/alcoholism directly represented. Pre-extracted features: TF-IDF 256, LIWC, readability indices, sentiment, and totals for stress/isolation/substance_use/suicidality.
- **Preprocessing:** Features already extracted. Directly use r/bpd + r/ptsd + r/addiction CSVs. Join LIWC + tfidf_256 for downstream models.
- **Maps to:** All four seed categories. Only public dataset with explicit r/ptsd + r/addiction co-occurrence data usable for CPTSD+addiction comorbidity edge cases.

### 6. Expert-Annotated Reddit Posts on Psychological Abuse — UCL
- **URL:** https://doi.org/10.5522/04/31587925
- **Format:** CSV (4 annotator files + aggregate OR/AND)
- **Size:** 1,500 Reddit posts
- **License:** Open access (UCL)
- **Relevance:** CPTSD=10, trauma=9
- **Contents:** Posts from r/abusiverelationships, r/domesticviolence, r/emotionalabuse (2021). 6 non-mutually-exclusive psychological abuse categories: Rules/control, Justifying/minimizing/denying, Threats/intimidation, Shaming/degrading, Isolation, Surveillance/monitoring.
- **Preprocessing:** Use AND labels (2+ annotators) for higher precision. LLaMA-3 70B few-shot baseline F1 available.
- **Maps to:** CPTSD etiology — coercive control is a primary CPTSD origin. Direct trauma seed material.

### 7. ACE-NLP (Adverse Childhood Experiences) — KnowLab
- **URL:** https://github.com/knowlab/ace-nlp
- **Format:** JSON
- **Size:** 322 ACE concepts (UMLS CUIs); 780+ documents w/ suicide concept
- **License:** MIT (open)
- **Relevance:** CPTSD=9, trauma=9
- **Contents:** NLP resource for identifying Adverse Childhood Experiences from free-text. Uses Reddit Mental Health Corpus. Concept embeddings (50d autoencoder) and document vectors ready for ML.
- **Preprocessing:** JSON. Concept vectors + document vectors ML-ready. Autoencoder-compressed 50d embeddings available.
- **Maps to:** CPTSD roots (adverse childhood experiences are canonical CPTSD etiology).

### 8. Reddit-Impacts (Substance Use NER) — SMM4H 2024
- **URL:** https://github.com/Yao-Ge-1218AM/Reddit-Impacts-Dataset
- **Format:** JSON/CSV (NER annotations)
- **Size:** 1,380 posts, 30 entity types
- **License:** SMM4H 2024 Shared Task (research access)
- **Relevance:** addiction=9
- **Contents:** NER for clinical and social effects of substance use from Reddit. 14 opioid-related subreddits. 30 entity types. Train 843 / Val 259 / Test 278.
- **Preprocessing:** Span-level annotations. BERT-based fine-tuning. 23% of posts contain annotated impact entities.
- **Maps to:** addiction seed category (clinical impacts of substance use).

### 9. Depression Detection using Sentiment Analysis — Kaggle
- **URL:** https://www.kaggle.com/datasets/szegeelim/mental-health
- **Format:** CSV (Combined Data.csv)
- **Size:** ~53,000 rows, 31.47 MB
- **License:** Kaggle terms (public)
- **Relevance:** BPD=8
- **Contents:** 7-class (Depression, Suicidal, Anxiety, Stress, Bipolar, Personality Disorder, Normal). Aggregated from Reddit, Twitter, social media.
- **Preprocessing:** Lowercase, remove punctuation/stopwords, lemmatize. Multi-class. ~2,652 samples per class (balanced).
- **Maps to:** BPD via Personality Disorder class. Aggregate source diversity complements other BPD proxy datasets.

### 10. MentalHealthBERT Fine-Tuned Model — Kaggle Models
- **URL:** https://www.kaggle.com/models/priyangshumukherjee/mental-health-bert-fine-tunes
- **Format:** Model weights (BERT-base)
- **Size:** BERT-base
- **License:** Kaggle terms (public)
- **Relevance:** BPD=8, CPTSD=8
- **Contents:** Fine-tuned mental/mental-bert-base-uncased on 4-class (Anxiety, Depression, Normal, Suicidal). 89.7% accuracy. Pre-trained on Reddit mental health conversations.
- **Preprocessing:** Use as base model for domain-adaptive pretraining on BPD/CPTSD corpus, then fine-tune on edge case labels.
- **Maps to:** Transfer learning backbone for all four seed categories.

---

## Recommended Preprocessing Pipeline (All Datasets)

1. **Text normalization:** lowercase, strip URLs/mentions, remove special chars (preserve punctuation signal for BPD emotional intensity).
2. **Tokenization:** BERT tokenizer (MentalBERT preferred) for transformer models; whitespace + lemmatization for classical ML.
3. **Label alignment:** Map dataset-specific labels → unified seed categories:
   - `Personality Disorder` → BPD splitting (proxy)
   - `r/bpd` subreddit → BPD splitting (direct)
   - `r/ptsd` subreddit → CPTSD (proxy; no explicit CPTSD label exists on Kaggle)
   - `r/addiction` + `r/alcoholism` + substance NER entities → addiction
   - ACE concepts + psychological abuse categories → trauma
4. **Class imbalance handling:** Personality Disorder label is minority class in all Kaggle datasets (17% in Mental Health Condition Classification). Use SMOTE-Tomek or weighted sampling.
5. **Domain-adaptive pretraining:** Continue pretraining MentalBERT on r/bpd + r/ptsd + r/addiction combined corpus before supervised fine-tuning.
6. **Temporal pattern extraction (BBRD only):** Group posts by `user_id`, order by `post_date`, extract idealization/devaluation cycles from "intense_emotions" + "impulsive_behaviors" annotation sequences.

## Dataset-to-Generator-Seed-Category Mapping

| Dataset | BPD | CPTSD | Addiction | Trauma |
|---|---|---|---|---|
| BBRD (Lancaster) | ✅ PRIMARY | — | ✅ col | — |
| Mental Disorders Reddit NLP | ✅ direct | proxy | — | — |
| Mental Health Condition Classification | ✅ proxy | — | — | — |
| Sentiment Analysis for Mental Health | ✅ proxy | — | — | — |
| Reddit Mental Health (Zenodo Low et al.) | ✅ direct | ✅ direct | ✅ direct | — |
| UCL Psychological Abuse | — | ✅ PRIMARY | — | ✅ PRIMARY |
| ACE-NLP | — | ✅ roots | — | ✅ roots |
| Reddit-Impacts NER | — | — | ✅ PRIMARY | — |
| Depression Detection (Kaggle) | ✅ proxy | — | — | — |
| MentalHealthBERT | backbone | backbone | backbone | backbone |

## Gaps and Limitations

- **No explicit CPTSD label** exists in any Kaggle dataset — CPTSD is clinically distinct from PTSD (ICD-11 vs DSM-5). All CPTSD mapping uses PTSD subreddit + ACE + psychological abuse proxies.
- **BPD "splitting" is never explicitly labeled** — BBRD's `intense_emotions` + `impulsive_behaviors` annotations are the closest behavioral proxy, but require downstream inference to derive idealization/devaluation cycles.
- **No CPTSD+addiction comorbidity dataset** — must be synthesized by joining r/ptsd + r/addiction rows from Low et al. Zenodo set or by co-training on separately labeled corpora.
- **Three Kaggle datasets require manual inspection** (Mental Health Corpus, Bhavik Jikadara, Reddit-Based Sentiment) — Kaggle CAPTCHA/errors blocked automated label inspection during this research pass.
- **Gated datasets (BBRD, OP-Reddit-Post, TraumaNarratives, REDOSE)** require institutional affiliation or data usage agreements — allow lead time for access.
