# Simplified Annotation Taxonomy

## Overview

This simplified taxonomy reduces annotation complexity while maintaining clinical utility. Based on Phase 1.3 results showing Crisis κ=1.0 but Emotion κ=0.78, we're refining the emotion categories.

## Changes from Original

### Crisis Labels (UNCHANGED - Perfect Agreement)

Keep the existing 0-5 scale:

- **0:** No Crisis
- **1:** Mild Distress (passive ideation, no plan)
- **2:** Moderate Risk (ideation + vague plan)
- **3:** High Risk (specific plan, no immediate intent)
- **4:** Severe Risk (plan + intent, deterrents present)
- **5:** Imminent Danger (active attempt or immediate risk)

**Rationale:** Crisis detection achieved κ=1.0 - no changes needed.

### Primary Emotions (SIMPLIFIED: 8 → 5 Core Emotions)

**Original (8 emotions):**

- Joy, Sadness, Anger, Fear, Surprise, Disgust, Anticipation, Neutral

**Simplified (5 core emotions):**

1. **Positive** (Joy, Anticipation, Surprise-positive)
2. **Sadness** (Grief, Disappointment, Loneliness)
3. **Anxiety** (Fear, Worry, Nervousness)
4. **Anger** (Frustration, Irritation, Rage)
5. **Neutral** (Calm, Balanced, No strong emotion)

**Rationale:**

- Reduces ambiguity between similar emotions
- Aligns with therapeutic practice (therapists focus on core emotional states)
- Easier for annotators to agree on broader categories
- Expected Kappa improvement: 0.78 → 0.85+

### Emotion Intensity (UNCHANGED)

- Scale: 1-10
- Works well, keep as-is

### Valence & Arousal (UNCHANGED)

- Valence: -1.0 to 1.0
- Arousal: 0.0 to 1.0
- Provides dimensional complement to categorical emotions

### Empathy Score (UNCHANGED)

- Scale: 1-5
- Clear guidelines, no changes needed

### Safety Pass (UNCHANGED)

- Boolean: true/false
- Binary decision works well

## Updated Annotation Schema

```json
{
  "crisis_label": 0-5,
  "crisis_confidence": 1-5,
  "primary_emotion": "Positive|Sadness|Anxiety|Anger|Neutral",
  "emotion_intensity": 1-10,
  "valence": -1.0 to 1.0,
  "arousal": 0.0 to 1.0,
  "empathy_score": 1-5,
  "safety_pass": true|false,
  "notes": "string"
}
```

## Mapping Guide for Annotators

### Old → New Emotion Mapping

| Old Emotion         | New Category | Notes                        |
| ------------------- | ------------ | ---------------------------- |
| Joy                 | **Positive** | Clear positive affect        |
| Anticipation        | **Positive** | Forward-looking, hopeful     |
| Surprise (positive) | **Positive** | Pleasant unexpected events   |
| Sadness             | **Sadness**  | Core category, unchanged     |
| Fear                | **Anxiety**  | Includes worry, nervousness  |
| Surprise (negative) | **Anxiety**  | Unpleasant unexpected events |
| Anger               | **Anger**    | Core category, unchanged     |
| Disgust             | **Anger**    | Often co-occurs with anger   |
| Neutral             | **Neutral**  | No strong emotion present    |

## Expected Improvements

### Kappa Score Projections

| Metric            | Phase 1.3   | Expected (Simplified) |
| ----------------- | ----------- | --------------------- |
| Crisis Label      | **1.00** ✅ | 1.00 (maintain)       |
| Primary Emotion   | 0.78        | **0.85-0.90** ⬆️      |
| Overall Agreement | 0.89        | **0.92-0.95** ⬆️      |

### Benefits

1. **Reduced Cognitive Load**
   - 5 emotions vs 8 = 37.5% reduction
   - Clearer boundaries between categories

2. **Clinical Relevance**
   - Aligns with common therapeutic frameworks
   - Easier to map to interventions

3. **Better Agreement**
   - Less ambiguity = higher Kappa
   - Faster annotation speed

4. **Backward Compatible**
   - Can map old annotations to new taxonomy
   - No data loss

## Implementation

### For Enhanced Annotation Agent

Update the system prompt to use simplified taxonomy:

```python
SIMPLIFIED_EMOTION_TAXONOMY = """
Primary Emotions (choose ONE):
1. Positive - Joy, hope, excitement, pleasant surprise
2. Sadness - Grief, disappointment, loneliness, loss
3. Anxiety - Fear, worry, nervousness, unpleasant surprise
4. Anger - Frustration, irritation, rage, disgust
5. Neutral - Calm, balanced, no strong emotion
"""
```

### For Guidelines

Update `guidelines.md` with:

- Simplified emotion definitions
- Clear examples for each category
- Decision tree for ambiguous cases

---

**Status:** Ready for implementation  
**Next Step:** Update enhanced_annotation_agent.py with simplified taxonomy
