# NeMo Gym Integration Design (Pixelated Empathy)

**Status**: Design document – no direct runtime dependency on NeMo Gym code.  
**Goal**: Define how a Pixelated Empathy–specific RL environment would look,
how it connects to existing training infrastructure, and how Nemotron 3 can
act as a teacher/reward model without breaking the S3-first architecture.

---

## 1. High-Level Architecture

We follow the patterns from the official NeMo Gym repository
(`NVIDIA-NeMo/Gym`):

- **Resource servers** encapsulate task logic, tools, and verifiers.
- **Agents** (policies) interact with resource servers to generate rollouts.
- **Rollouts** are stored as JSONL, suitable for RL algorithms like GRPO.

For Pixelated Empathy we introduce:

- A **TherapeuticSession resource server** that:
  - Wraps the existing inference service (`packages/velocity/training_scripts/inference_service.py`)
    or an OVH/Lightning deployment.
  - Encodes environment state as: conversation history, scenario metadata,
    and target competencies (safety, empathy, bias constraints).
  - Emits **verifiable rewards** based on transcript analysis.
- One or more **RL jobs** that:
  - Use either Nemotron 3 Nano or the Pixelated Empathy model as the policy.
  - Store rollouts in S3 under `rl/` prefixes for later training.

Data flow (conceptual):

```text
NeMo Gym (TherapeuticSession env)
    ↕ HTTP / gRPC / internal Python calls
Pixelated Inference Service (FastAPI / Lightning on H100/OVH)
    ↕
S3 (rl/pixelated/...)  ← rollouts, rewards, configs
    ↕
MoE / Apex Training Scripts (train_moe_h100.py, train_enhanced.py)
```

---

## 2. Environment Definition: `TherapeuticSession`

### 2.1 State Representation

Each environment episode represents a single therapeutic scenario:

- **Context**:
  - `scenario_id`: identifier for the scenario template.
  - `difficulty`: e.g. beginner / intermediate / advanced.
  - `risk_level`: e.g. crisis / subclinical / edge case.
  - `client_profile`: tags for CPTSD, addiction, cultural factors, etc.
- **Dialogue history**:
  - `messages`: Chat-style list of `{role, content}` objects.
  - `turn_index`: current turn number.
- **Targets / constraints**:
  - `expected_outcomes`: high-level goals (de-escalation, validation, psychoeducation).
  - `hard_constraints`: non-negotiables (e.g., crisis protocol adherence).

State is serialized as JSON and passed between:

- NeMo Gym server
- Pixelated inference service
- Reward verifiers

### 2.2 Actions

An action is a **single assistant turn**:

- The policy (Pixelated model or Nemotron 3 Nano) receives the state and
  outputs one assistant message (`{role: "assistant", content: "..."}`).
- The environment appends this to the `messages` list and optionally updates
  internal hidden state (e.g. client mood trajectory).

### 2.3 Episode Termination

An episode terminates when:

- A maximum number of turns is reached.
- A “session end” condition is triggered (e.g. client stabilized, hard
  safety violation, or explicit end-of-session marker).

At termination, the environment:

- Packages the full transcript.
- Computes rewards and auxiliary metrics.
- Writes a structured rollout record.

---

## 3. Reward Design (Verifiers)

Reward is decomposed into several components, all computed from transcripts
and scenario metadata:

1. **Safety** (hard constraint):
   - Deduct or fail if:
     - Crisis protocol is violated.
     - Encouragement of self-harm or other obvious red flags.
   - Sources:
     - Existing safety/bias validators (Python services).
     - Optional Nemotron 3–powered LLM-as-a-judge (non-clinical layer).

2. **Empathy & Therapeutic Skill**:
   - Reward high use of:
     - Reflective listening.
     - Validation and normalization.
     - Open questions instead of interrogation.
   - Penalize:
     - Dismissiveness or minimizing responses.
   - Implemented as a scorer script that:
     - Parses the transcript.
     - Computes numeric scores per competency.

3. **Bias / Cultural Competency**:
   - Penalize biased language or neglect of key cultural context.
   - Reward correct adaptation to client identity (as defined by scenario).

4. **Task Completion / Adherence to Instructions**:
   - If a scenario has explicit tasks (e.g. “safety plan by end of session”),
     reward execution and clarity of outcomes.

The reward vector can be stored as:

```json
{
  "total_reward": 0.73,
  "components": {
    "safety": 1.0,
    "empathy": 0.8,
    "bias": 0.6,
    "task_completion": 0.5
  }
}
```

This is compatible with GRPO-style pipelines used in Nemotron RL recipes.

---

## 4. Rollout Format & S3 Storage

Following Nemotron RL datasets on Hugging Face, we target a JSONL-based
rollout format:

Each line in `s3://pixelated-training-data/rl/pixelated/therapeutic_session/{run_id}.jsonl`
contains:

```json
{
  "run_id": "2025-12-15T12-00-00Z_h100_experiment_01",
  "episode_index": 3,
  "scenario": {
    "scenario_id": "cptsd_breakup_001",
    "difficulty": "intermediate",
    "risk_level": "moderate"
  },
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    "..."
  ],
  "policy": {
    "name": "pixelated-moe-2025-12",
    "base_model": "LatitudeGames/Wayfarer-2-12B",
    "nemotron_teacher": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
  },
  "reward": {
    "total_reward": 0.73,
    "components": {
      "safety": 1.0,
      "empathy": 0.8,
      "bias": 0.6,
      "task_completion": 0.5
    }
  },
  "metadata": {
    "env_version": "therapeutic_session_v1",
    "generator": "nemo_gym",
    "created_at": "..."
  }
}
```

This can be consumed later by:

- Dedicated RL trainers (GRPO/ORPO/SimPO style).
- Offline evaluators and dashboards.

---

## 5. Connecting to Existing Training Scripts

### 5.1 Collection Jobs (Gym ↔ Pixelated)

We expect separate **collection jobs** that:

1. Start NeMo Gym with the `TherapeuticSession` resource server.
2. Point the environment at a Pixelated inference endpoint:
   - Either local (FastAPI service from `inference_service.py`).
   - Or OVH/Lightning deployment (via HTTPS).
3. Use NeMo Gym tooling (`ng_collect_rollouts` or similar) to:
   - Generate rollouts.
   - Store them to local disk.
4. Mirror rollouts to S3 under `rl/pixelated/...` using the existing
   S3 utilities or a small upload script.

Pixelated codebase remains the **owner** of S3 paths and RL configuration;
NeMo Gym is purely an external environment/collector.

### 5.2 RL Fine-Tuning

Experimental RL jobs (e.g. using `train_moe_h100.py`) can add support for:

- Loading rollout JSONL files from S3 via `S3DatasetLoader`.
- Converting them into:
  - GRPO-style trajectories, or
  - Preference pairs (for ORPO/SimPO), using reward or LLM-as-judge labels.

These jobs should:

- Keep RL configuration **optional** and gated by flags/config files.
- Maintain strict separation between:
  - Core SFT curriculum.
  - RL-based alignment experiments.

---

## 6. Nemotron 3 as Teacher / Reward Model

Nemotron 3 Nano can participate in two ways:

1. **Policy / Co-Policy**:
   - Use Nemotron 3 as the acting policy inside the environment to:
     - Generate idealized trajectories for specific scenarios.
     - Produce “gold” trajectories stored in S3 as teacher data.
   - Pixelated models can then be fine-tuned to imitate or compete with
     these trajectories using standard distillation or preference learning.

2. **Reward Model / LLM-as-a-Judge**:
   - Use Nemotron 3 in analysis mode:
     - Pass full transcript + scenario goal.
     - Ask for a structured rubric score (safety, empathy, adherence, etc.).
   - Combine this with rule-based verifiers to compute a final reward.
   - Store Nemotron’s rubric scores explicitly in `reward.components`.

In both roles:

- Nemotron-derived judgments are **advisory**, not authoritative.
- Final training decisions are grounded in Pixelated’s own ethical and
  clinical guidelines.

---

## 7. Safety & Compliance Considerations

- NeMo Gym environments must **never** emit or require real PHI; all
  scenarios are synthetic or drawn from de-identified corpora already
  in the training pipeline.
- All RL rollouts must be stored under clear S3 prefixes that separate:
  - Synthetic RL data.
  - Real or de-identified supervised data.
- Any use of Nemotron 3 for rewards must:
  - Avoid encoding clinical decisions that contradict licensed practice.
  - Be reviewed by domain experts before entering main training loops.

---

## 8. Next Steps (Implementation Checklist)

1. **Prototype resource server**:
   - Implement a `TherapeuticSession` server in a separate repo that follows
     NeMo Gym’s `resources_servers/` patterns.
   - Wire it to call Pixelated’s inference service.
2. **Define rubrics & verifiers**:
   - Translate existing evaluation criteria (empathy, safety, bias) into
     machine-checkable rubrics and scoring functions.
3. **S3 rollout writer**:
   - Implement a small bridge (can live alongside this repo) that:
     - Receives rollouts from Gym.
     - Writes JSONL to `rl/pixelated/...` via `S3DatasetLoader`.
4. **Experimental RL training config**:
   - Add an H100-focused config variant to `train_moe_h100.py` that:
     - Consumes RL rollouts.
     - Performs GRPO/ORPO/SimPO-style updates.
   - Keep this experiment behind a clear flag / config so the main SFT
     path remains unchanged.

This design keeps NeMo Gym and Nemotron 3 as **well-defined external
systems** while respecting the S3-first, safety-first architecture of
Pixelated Empathy.

