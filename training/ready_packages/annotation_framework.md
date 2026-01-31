# Pixel Model Training - Annotation Framework

## Overview

This document outlines the comprehensive annotation framework for creating high-quality labeled therapeutic conversation datasets for Pixel model training, targeting >0.85 Cohen's Kappa coefficient for inter-annotator agreement.

## Annotation Categories

### 1. Crisis Detection Labels

**Primary Crisis Types**:
- **Immediate Harm**: Suicidal ideation with plan/intent, self-harm in progress
- **Self-Harm**: Non-suicidal self-injury, cutting behaviors
- **Substance Crisis**: Overdose, withdrawal, addiction crisis
- **Panic/Anxiety**: Severe panic attacks, agoraphobia, severe anxiety
- **Psychotic Episodes**: Hallucinations, delusions, severe disorganization
- **Passive Ideation**: Passive suicidal thoughts without plan/intent

**Labeling Criteria**:
- **Crisis Level**: None (0), Low (1), Moderate (2), High (3), Critical (4)
- **Urgency**: Immediate (<1hr), Urgent (<24hr), Soon (<1week), Monitor (ongoing)
- **Intervention Type**: Safety planning, medication, hospitalization, therapy

### 2. Emotional Intelligence Annotations

**Emotional Dimensions**:
- **Valence**: Positive (1.0) to Negative (-1.0)
- **Arousal**: High (1.0) to Low (-1.0) 
- **Dominance**: High (1.0) to Low (-1.0)

**Specific Emotions**:
- Happiness, Sadness, Anger, Fear, Surprise, Disgust, Calm, Neutral

**Therapeutic Responses**:
- **Empathy Level**: 1-5 scale (1=minimal, 5=exceptional)
- **Validation**: Present/Absent
- **Reflection**: Present/Absent
- **Question Type**: Open/Closed/Leading
- **Therapeutic Technique**: CBT, DBT, MI, Psychodynamic, Humanistic

### 3. Bias Detection Labels

**Bias Categories**:
- **Gender Bias**: Stereotypical gender assumptions
- **Racial Bias**: Cultural insensitivity, racial stereotypes
- **Cultural Bias**: Ethnocentrism, cultural assumptions
- **Socioeconomic Bias**: Class assumptions, privilege blindness
- **LGBTQ+ Bias**: Heteronormative assumptions, transphobia
- **Religious Bias**: Faith assumptions, spiritual insensitivity

**Severity Scale**:
- None (0), Mild (1), Moderate (2), Severe (3)

### 4. Therapeutic Effectiveness

**Outcome Measures**:
- **Engagement**: Patient participation level (1-5)
- **Progress**: Therapeutic goal advancement (1-5)
- **Alliance**: Therapeutic relationship strength (1-5)
- **Safety**: Risk assessment accuracy (1-5)

## Annotation Guidelines

### Quality Standards

1. **Consistency**: Use standardized definitions and examples
2. **Objectivity**: Base labels on observable evidence
3. **Completeness**: Annotate all relevant aspects
4. **Reliability**: Achieve >0.85 inter-annotator agreement

### Annotation Process

1. **Double-Blind**: Two annotators per sample
2. **Calibration**: Weekly agreement checks
3. **Conflict Resolution**: Third annotator for disagreements
4. **Quality Control**: Random sample re-annotation

### Training Requirements

**Annotator Qualifications**:
- Licensed mental health professional (LCSW, LPC, LMFT, Psychologist)
- Minimum 2 years clinical experience
- Crisis intervention training
- Cultural competency certification

**Training Components**:
- 8-hour annotation training workshop
- Practice annotation with feedback
- Inter-annotator calibration sessions
- Ongoing quality monitoring

## Annotation Platform

### Tool Requirements

- **Interface**: Web-based annotation tool
- **Features**: Multi-label support, confidence scoring, notes
- **Quality Control**: Agreement tracking, conflict flagging
- **Data Security**: HIPAA compliant, encrypted storage

### Sample Annotation Workflow

1. **Load Sample**: Display conversation transcript
2. **Crisis Assessment**: Rate crisis level and type
3. **Emotional Analysis**: Annotate emotions and EI metrics
4. **Bias Check**: Identify potential bias instances
5. **Effectiveness Rating**: Score therapeutic outcomes
6. **Confidence**: Rate annotation confidence (1-5)
7. **Notes**: Add clarifications or edge cases

## Quality Assurance

### Inter-Annotator Agreement

**Calculation Method**: Cohen's Kappa coefficient
**Target**: >0.85 (substantial to almost perfect agreement)
**Monitoring**: Weekly agreement reports
**Action**: Re-training if <0.85

### Conflict Resolution

1. **Automatic Flagging**: Disagreements >1 point on scales
2. **Discussion**: Annotators review conflicting labels
3. **Expert Review**: Third annotator for unresolved conflicts
4. **Documentation**: Record resolution rationale

### Quality Metrics

- **Coverage**: 100% of samples annotated
- **Accuracy**: >90% confidence scores
- **Consistency**: <5% conflicts requiring resolution
- **Timeliness**: Complete within 2 weeks

## Data Management

### Storage Structure

```
/annotation_data/
├── raw_samples/
├── annotations/
│   ├── annotator_1/
│   ├── annotator_2/
│   └── consensus/
├── quality_reports/
└── final_dataset/
```

### Security Requirements

- **Encryption**: AES-256 at rest, TLS 1.3 in transit
- **Access Control**: Role-based permissions
- **Audit Trail**: All actions logged
- **Data Retention**: 7 years per HIPAA requirements

## Success Criteria

### Quantitative Targets

- **Kappa Coefficient**: >0.85
- **Annotation Coverage**: 100%
- **Quality Score**: >90% average confidence
- **Conflict Rate**: <5%
- **Timeline**: 2 weeks maximum

### Qualitative Standards

- **Clinical Accuracy**: Expert-validated labels
- **Cultural Sensitivity**: Diverse perspective inclusion
- **Ethical Compliance**: HIPAA and professional standards
- **Reproducibility**: Clear documentation and procedures

## Implementation Timeline

### Week 1: Setup and Training
- Day 1-2: Platform setup and testing
- Day 3-4: Annotator recruitment and screening
- Day 5-7: Training and calibration sessions

### Week 2: Annotation and Quality Control
- Day 8-10: Primary annotation phase
- Day 11-12: Quality checks and conflict resolution
- Day 13-14: Final validation and dataset compilation

---

**Status**: Framework ready for implementation
**Next Step**: Begin annotator recruitment and platform setup