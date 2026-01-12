# Agent-Based Annotation Architecture

## Inspired by NVIDIA AI Blueprints

> **Mission**: Build empathy-driven annotation agents that prioritize psychological safety, accuracy, and inter-annotator reliability.

---

## 🎯 Core Principles from NVIDIA Blueprints

### 1. **Multi-Agent Orchestration**

- **Specialized Agents**: Each agent has a distinct role and expertise
- **Tool Integration**: Agents use specific tools for their domain
- **Evaluation Framework**: Built-in profiling and optimization (NeMo Agent Toolkit pattern)

### 2. **Fast Reasoning with Quality**

- **Low Latency**: Optimize for speed without sacrificing accuracy
- **Confidence Scoring**: All outputs include confidence metrics
- **Iterative Refinement**: Agents can review and improve their work

### 3. **Guardrails & Safety**

- **Content Safety**: Validate all outputs for appropriateness
- **Topic Control**: Keep agents focused on their domain
- **Context-Aware**: Understand the therapeutic context

---

## 🤖 Agent Ecosystem

### **Primary Annotation Agents**

#### 1. **Dr. A - Conservative Clinical Expert**

**Persona**: Risk-averse, trauma-informed, prioritizes safety
**Specialization**: Crisis detection, safety assessment
**Tools**:

- Crisis severity classifier
- Safety risk evaluator
- Trauma indicator detector

**Reasoning Pattern**:

```
1. Scan for crisis indicators
2. Assess immediate safety concerns
3. Evaluate trauma markers
4. Generate conservative labels
5. Document reasoning with high confidence thresholds
```

#### 2. **Dr. B - Pragmatic Research Analyst**

**Persona**: Evidence-based, balanced, research-focused
**Specialization**: Emotional analysis, empathy scoring
**Tools**:

- Emotion taxonomy mapper
- Empathy quality assessor
- Therapeutic alliance evaluator

**Reasoning Pattern**:

```
1. Identify primary emotions
2. Measure intensity and valence
3. Assess therapist empathy
4. Evaluate conversation quality
5. Generate balanced annotations
```

### **Supporting Agents**

#### 3. **Consensus Orchestrator**

**Role**: Coordinate multi-agent annotation and resolve conflicts
**Capabilities**:

- Aggregate annotations from Dr. A and Dr. B
- Identify disagreements
- Calculate inter-annotator agreement (Kappa)
- Flag items needing human review

#### 4. **Quality Assurance Agent**

**Role**: Validate annotation quality and consistency
**Capabilities**:

- Check annotation completeness
- Verify score ranges (0-1 normalization)
- Detect outliers
- Ensure guideline compliance

#### 5. **Insight Synthesizer**

**Role**: Generate meta-analysis and recommendations
**Capabilities**:

- Identify annotation patterns
- Suggest guideline improvements
- Recommend training data additions
- Generate quality reports

---

## 🔧 Tool Integration

### **Retrieval Tools**

- **Guideline Retriever**: Fetch relevant annotation guidelines
- **Example Retriever**: Find similar annotated examples
- **Context Retriever**: Pull conversation history

### **Analysis Tools**

- **Emotion Classifier**: Multi-label emotion detection
- **Crisis Detector**: Specialized crisis signal identification
- **Empathy Scorer**: Therapeutic empathy measurement

### **Validation Tools**

- **Range Validator**: Ensure scores are within bounds
- **Consistency Checker**: Cross-validate related fields
- **Safety Filter**: Flag potentially harmful content

---

## 📊 Workflow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Conversation Data                  │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼────┐            ┌────▼────┐
    │  Dr. A  │            │  Dr. B  │
    │ Agent   │            │ Agent   │
    └────┬────┘            └────┬────┘
         │                       │
         │    ┌─────────────┐   │
         └────►  Consensus  ◄───┘
              │ Orchestrator│
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │   Quality   │
              │  Assurance  │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │   Insight   │
              │ Synthesizer │
              └──────┬──────┘
                     │
         ┌───────────▼────────────┐
         │  Final Annotations +   │
         │  Quality Metrics +     │
         │  Recommendations       │
         └────────────────────────┘
```

---

## 🎨 Implementation Strategy

### Phase 1: Core Agent Framework

1. **Agent Base Class**: Common interface for all agents
2. **Tool Registry**: Centralized tool management
3. **Prompt Templates**: Structured prompts for each agent
4. **Response Parsing**: Robust JSON extraction

### Phase 2: Specialized Agents

1. **Dr. A Implementation**: Conservative crisis-focused agent
2. **Dr. B Implementation**: Balanced emotion-focused agent
3. **Persona Validation**: Ensure distinct annotation patterns

### Phase 3: Orchestration Layer

1. **Consensus Building**: Aggregate multi-agent outputs
2. **Conflict Resolution**: Handle disagreements systematically
3. **Quality Metrics**: Calculate Kappa, accuracy, consistency

### Phase 4: Advanced Features

1. **Adaptive Learning**: Improve from human feedback
2. **Batch Optimization**: Parallel processing
3. **Real-time Monitoring**: Track agent performance
4. **Explainability**: Document reasoning chains

---

## 🔬 Evaluation Framework

### **Agent Performance Metrics**

- **Accuracy**: Agreement with gold standard
- **Consistency**: Self-agreement over time
- **Speed**: Annotations per second
- **Confidence Calibration**: Confidence vs. accuracy correlation

### **Inter-Agent Metrics**

- **Cohen's Kappa**: Pairwise agreement
- **Fleiss' Kappa**: Multi-rater agreement
- **Disagreement Patterns**: Where agents diverge
- **Consensus Rate**: Percentage of automatic agreement

### **Quality Indicators**

- **Completeness**: All required fields populated
- **Validity**: Scores within acceptable ranges
- **Coherence**: Related fields are consistent
- **Explainability**: Clear reasoning provided

---

## 🛡️ Safety & Ethics

### **Guardrails**

1. **Content Safety**: Filter harmful outputs
2. **Bias Detection**: Monitor for systematic biases
3. **Privacy Protection**: Redact PII in examples
4. **Crisis Escalation**: Flag high-risk cases for human review

### **Ethical Considerations**

- **Transparency**: Document agent decisions
- **Accountability**: Human oversight required
- **Fairness**: Regular bias audits
- **Consent**: Respect data usage agreements

---

## 🚀 Getting Started

### **Quick Start**

```bash
# 1. Set up environment
uv sync

# 2. Configure agents
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.nvidia.com/v1"  # Optional

# 3. Run multi-agent annotation
uv run python scripts/run_multi_agent_annotation.py \
  --input batches/batch_real_001.jsonl \
  --output results/multi_agent_real_001.jsonl \
  --model nvidia/nemotron-3-nano-30b-a3b

# 4. Calculate agreement
uv run python scripts/calculate_multi_agent_kappa.py \
  --input results/multi_agent_real_001.jsonl
```

### **Configuration**

```python
# config/agent_config.yaml
agents:
  dr_a:
    model: "nvidia/nemotron-3-nano-30b-a3b"
    temperature: 0.1
    max_tokens: 1000
    tools: ["crisis_detector", "safety_assessor"]

  dr_b:
    model: "nvidia/nemotron-3-nano-30b-a3b"
    temperature: 0.2
    max_tokens: 1000
    tools: ["emotion_classifier", "empathy_scorer"]

orchestration:
  consensus_threshold: 0.8
  require_human_review: true
  kappa_target: 0.85
```

---

## 📚 References

### **NVIDIA Blueprints**

- [AI-Q Enterprise Research](https://build.nvidia.com/nvidia/aiq)
- [Ambient Healthcare Agents](https://build.nvidia.com/nvidia/ambient-healthcare-agents)
- [Digital Twins for AI Factories](https://build.nvidia.com/nvidia/digital-twins-for-ai-factories)

### **Key Technologies**

- **NeMo Agent Toolkit**: Agent evaluation and profiling
- **NeMo Retriever**: Advanced semantic search
- **NeMo Guardrails**: Safety and topic control
- **Llama Nemotron**: Fast reasoning capabilities

### **Research Foundations**

- Multi-agent systems for annotation
- Inter-rater reliability in clinical settings
- Emotion AI and affective computing
- Therapeutic alliance measurement

---

_Building annotation agents that understand the nuances of human empathy._
