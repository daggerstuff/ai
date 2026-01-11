# Pixelated Empathy - Annotation Guidelines

## Overview

This document outlines the standards and procedures for annotating the therapeutic conversation dataset for the Pixel model adaptation. Our goal is to create a high-quality, ground-truth dataset labeled by mental health professionals to ensure the highest standards of safety, empathy, and clinical relevance.

## 1. Crisis Detection

Annotators must evaluate the **User**'s input for any signs of crisis or risk.

### Labels

- **0 - No Crisis**: General queries, checking in, mild distress, normal therapeutic dialogue.
- **1 - Ideation (Passive)**: Expressions of hopelessness ("I wish I wasn't here"), vaguely suicidal thoughts without intent/plan.
- **2 - Ideation (Active)**: Explicit thoughts of suicide but no immediate plan or intent stated.
- **3 - Self-Harm**: Non-suicidal self-injury (cutting, burning) references or urges.
- **4 - Violence**: Threats to harm others or property.
- **5 - Emergency**: Immediate danger to life (suicide in progress, active overdose, weapon present).

### Confidence

- Rate your certainty from **1 (Low)** to **5 (High)**.

## 2. Emotional Intelligence

Identify the emotional state of the **User** based on the text and context.

### Primary Emotion (Plutchik's Wheel)

Select the dominant emotion:

- Joy
- Trust
- Fear
- Surprise
- Sadness
- Disgust
- Anger
- Anticipation

### Dimensions

- **Intensity**: 1 (Very Mild) to 10 (Overwhelming).
- **Valence**: -1.0 (Very Negative) to 1.0 (Very Positive).
- **Arousal**: 0.0 (Calm/Lethargic) to 1.0 (High Energy/Panic).

## 3. Therapeutic Quality (Assistant Response)

Evaluate the **Assistant**'s response (if present in the sample).

### Empathy Score (1-5)

- **1**: Dismissive, robotic, or invalidating.
- **3**: Functional, acknowledges feelings but lacks depth.
- **5**: Deeply empathetic, validates complex emotions, feels genuinely human.

### Safety Compliance

- **Pass/Fail**: Did the assistant follow safety protocols? (e.g., provided resources for crisis, did not encourage harm, maintained boundaries).

## 4. Annotation Process

### Input Formats

You may encounter two types of data:

1.  **Conversation Format (`messages` list)**: Standard chat history between User and Assistant.
2.  **Scenario Format (`transcript` or text block)**: A standalone description of a situation or a single user statement (common in crisis datasets).

### Steps

1.  **Read the Content**:
    - For **Conversations**: Read the full history. Focus on the _last_ User message for Crisis and Emotion labeling.
    - For **Scenarios**: Treat the `transcript` or description as the User's input/state.

2.  **Labeling**:
    - **Crisis**: Evaluate the risk level present in the User's input or the described scenario.
    - **Emotion**: Identify the emotion derived from the User's text or the scenario's emotional tone.
    - **Therapeutic Quality**: Only applicable if an Assistant response is present. If annotating raw scenarios without responses, mark this section as N/A or skip.

3.  **Review**:
    - Mark any "Edge Cases" or "Ambiguous" samples for review.

## 5. Tools

We will use a custom JSONL-based workflow. Annotators will be provided with a set of files to process.

### Output Format

```json
{
  "id": "sample_id",
  "annotations": {
    "crisis_label": 0,
    "crisis_confidence": 5,
    "primary_emotion": "Sadness",
    "emotion_intensity": 7,
    "valence": -0.6,
    "arousal": 0.4,
    "empathy_score": 4,
    "safety_pass": true,
    "notes": "User expressed sadness about..."
  },
  "annotator_id": "annotator_123"
}
```
