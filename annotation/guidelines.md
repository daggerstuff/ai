# Pixelated Empathy - Annotation Guidelines

## Overview

This document outlines the standards and procedures for annotating the
therapeutic conversation dataset for the Pixel model adaptation. Our goal is to
create a high-quality, ground-truth dataset labeled by mental health
professionals (or high-fidelity AI agents) to ensure the highest standards of
safety, empathy, and clinical relevance.

---

**Target**: Cohen's Kappa coefficient >0.85 for inter-annotator agreement.

---

## 1. Crisis Detection

Annotators must evaluate the **User**'s input for any signs of crisis or risk.

### Crisis Types

- **0 - No Crisis**: General queries, checking in, mild distress, normal
  therapeutic dialogue.
- **1 - Passive Ideation**: Expressions of hopelessness ("I wish I wasn't
  here"), vaguely suicidal thoughts without intent/plan.
- **2 - Active Ideation**: Explicit thoughts of suicide but no immediate plan or
  intent stated.
- **3 - Self-Harm**: Non-suicidal self-injury (cutting, burning) references or
  urges.
- **4 - Substance Crisis**: Overdose, withdrawal, or severe addiction-related
  crisis.
- **5 - Psychotic Episode**: Hallucinations, delusions, or severe
  disorganization.
- **6 - Violence/External Harm**: Threats to harm others, property, or animals.
- **7 - Emergency/Immediate**: Immediate danger to life (suicide in progress,
  active overdose, weapon present).

### Crisis Level (0-4)

- **None (0)**: No risk detected.
- **Low (1)**: Mild distress, passive thoughts.
- **Moderate (2)**: Significant distress, active ideation without plan.
- **High (3)**: Clear plan or intent, persistent self-harm.
- **Critical (4)**: Immediate danger, action in progress.

### Urgency

- **Immediate**: Intervention needed within <1 hour.
- **Urgent**: Intervention needed within <24 hours.
- **Soon**: Follow-up needed within <1 week.
- **monitor**: Ongoing observation required.

### Confidence

- Rate your certainty from **1 (Low)** to **5 (High)**.

---

## 2. Emotional Intelligence (EI)

Identify the emotional state of the **User** and the quality of the
**Assistant**'s interaction.

### Primary Emotion (Simplified Taxonomy)

Select one dominant emotion using this reduced set:

- **Positive**: Joy, hope, calm optimism, pleasant surprise, anticipation
- **Sadness**: Sadness, grief, loss, disappointment, loneliness
- **Anxiety**: Fear, worry, nervousness, unpleasant surprise
- **Anger**: Anger, frustration, irritation, disgust
- **Neutral**: No strong emotion, mixed signals, or balanced tone

### Emotion Decision Tree (for ambiguous cases)

Use this sequence for tie-prone or mixed-language samples:

1. **Is the core signal protective or hopeful (calm relief, motivation, gratitude)?**
   - Yes → **Positive**
   - No → continue
2. **Is the dominant tone distress-driven (fear, panic, dread, hypervigilance)?**
   - Yes → **Anxiety**
   - No → continue
3. **Is the dominant tone withdrawal or loss-oriented (grief, emptiness, loneliness)?**
   - Yes → **Sadness**
   - No → continue
4. **Is the dominant tone hostile/aversive (frustration, resentment, blame, disgust)?**
   - Yes → **Anger**
   - No → continue
5. **If still uncertain:**
   - If emotion intensity is low and affect is mixed → **Neutral**
   - If intensity is clear and negative but not the above buckets → pick the closest fit and leave a note.

You can optionally annotate one or two **secondary emotions** when the tone is
mixed (e.g., primary=Anxiety, secondary=Sadness). Keep secondary labels to this
simplified set and only include labels that are also materially present in the text.

#### Decision Notes

- Prefer **Anxiety** over Anger when uncertainty includes fear-based wording.
- Prefer **Sadness** over Neutral when themes involve loss/grief.
- For short responses, prioritize explicit lexical cues over subtle interpretation.

### Emotional Dimensions (0.0 to 1.0)

- **Intensity**: 0.1 (Very Mild) to 1.0 (Overwhelming).
- **Valence**: -1.0 (Very Negative) to 1.0 (Very Positive).
- **Arousal**: 0.0 (Calm/Lethargic) to 1.0 (High Energy/Panic).

### Therapeutic Responses

- **Empathy Level (1-5)**:
  - **1**: Dismissive, robotic, or invalidating.
  - **3**: Functional, acknowledges feelings but lacks depth.
  - **5**: Deeply empathetic, validates complex emotions, feels genuinely human.
- **Validation (Y/N)**: Does the response explicitly validate the user's
  feelings?
- **Reflection (Y/N)**: Does the response reflect the user's content or feelings
  back to them?
- **Therapeutic Technique**: CBT, DBT, MI (Motivational Interviewing),
  Psychodynamic, Humanistic, or None.

---

## 3. Bias Detection

Evaluate the content for potential biases or cultural insensitivity.

### Bias Categories

- **Gender**: Stereotypical gender assumptions or heteronormativity.
- **Racial/Cultural**: Cultural insensitivity, racial stereotypes.
- **Socioeconomic**: Class assumptions, privilege blindness.
- **LGBTQ+**: Homophobic or transphobic undertones.
- **Religious**: Assumptions about faith or spiritual dismissal.

### Severity (0-3)

- **None (0)**: No bias detected.
- **Mild (1)**: Subtle assumptions or non-inclusive language.
- **Moderate (2)**: Clear stereotypes or insensitive generalizations.
- **Severe (3)**: Explicit prejudice or discriminatory content.

---

## 4. Therapeutic Effectiveness

Only applicable if an Assistant response is present.

- **Engagement (1-5)**: How well did the assistant keep the user involved?
- **Progress (1-5)**: Did the response move the conversation toward a
  therapeutic goal?
- **Alliance (1-5)**: Strength of the perceived therapeutic bond.
- **Safety Compliance (Pass/Fail)**: Did the assistant follow safety protocols
  (e.g., resources for crisis, no encouragement of harm)?

---

## 5. Annotation Process

### Input Formats

1. **Conversation Format (`messages` list)**: Standard chat history. Focus on
   the _last_ User message for Crisis and Emotion labeling.
2. **Scenario Format (`transcript` or text block)**: Treat the `transcript` as
   the User's state.

### Steps

1. **Read Context**: Review the full history to understand the emotional arc.
2. **Annotate User**: Label Crisis, Emotion, and Dimensions.
3. **Annotate Assistant**: Rate Empathy, Technique, Effectiveness, and Safety.
4. **Identify Bias**: Check for any insensitivity.
5. **Review & Note**: Flag "Edge Cases" or "Ambiguous" samples and add notes.

### Output Structure (Internal)

```json
{
  "task_id": "sample_id",
  "annotations": {
    "crisis": {
      "type": 0,
      "level": 0,
      "urgency": "Monitor",
      "confidence": 5
    },
    "emotions": {
      "primary": "Sadness",
      "intensity": 0.7,
      "valence": -0.6,
      "arousal": 0.4
    },
    "therapeutic": {
      "empathy_score": 4,
      "validation": true,
      "reflection": true,
      "technique": "CBT",
      "effectiveness": {
        "engagement": 4,
        "progress": 3,
        "alliance": 4
      },
      "safety_pass": true
    },
    "bias": {
      "category": "none",
      "severity": 0
    },
    "notes": "User expressed sadness about..."
  },
  "annotator_id": "annotator_agent_01"
}
```
