# AGENTS.md

> **AI Coding Assistant Instructions (AI Sub-Repo)** - This document guides AI
> tools (GitHub Copilot, Cursor, Claude, Gemini, etc.) on how to work with the
> **`ai/`** repository.
>
> This repository is a standalone git repo, but it is an integral component of
> **`pixelated`** and **Pixelated Empathy** as a whole.
>
> 🎭 _"We don't just process conversations. We understand them."_

---

## Project Overview

**What this repo is**: The Pixelated Empathy AI / ML codebase: data sourcing
pipelines, dataset generation, training, evaluation, safety/quality gates, and
inference services.

**Primary goals**:

- Produce **reproducible** datasets and model artifacts
- Enforce **safety, privacy, and crisis-aware behavior** in all AI outputs
- Support **production-grade** inference and monitoring

**Tech stack (AI repo)**:

- **Language**: Python (>= 3.11)
- **Package manager**: **uv** (do not use pip/conda/poetry)
- **API**: FastAPI + Uvicorn
- **ML**: PyTorch, Transformers, Datasets, PEFT/TRL, sentence-transformers
- **Audio**: openai-whisper / faster-whisper, librosa, pydub
- **Memory**: Zep (`zep-cloud`)
- **Quality**: Ruff, Black, Pytest, Pytest-Cov, NBQA

---

## ⛔ ABSOLUTE PROHIBITION: No Stubs or Filler

**Every implementation MUST be complete and production-ready.**

- ❌ No `pass`, `...`, `TODO`, `NotImplementedError`, `# FIXME`
- ❌ No placeholder returns (`return True`, `return []`, hardcoded dummies)
- ❌ No mock implementations disguised as real code
- ✅ If it can't be fully implemented, it must not be committed

---

## ⛔ ABSOLUTE PROHIBITION: No Ignore Comments to Silence Warnings

**Warnings are signals, not noise. Fix the root cause, don't silence it.**

- ❌ No `# noqa`, `# type: ignore`, or similar suppression comments
- ❌ No linter bypass comments to avoid addressing real issues
- ✅ Refactor and fix code to resolve the underlying issue
- ✅ If a warning is truly a false positive, document _why_ with a detailed explanation

---

## Repo Layout (Authoritative)

This repo is large; when in doubt, prefer these entry points and domains:

- **`.agent/steering/`**: **MANDATORY START POINT.** High-signal
  operational and architectural intent. Read ALL files here first.
- **`.agent/knowledge/`**: Technical and operational notes (including runbook-level context).
- **`sourcing/`**: raw data ingestion
- **`pipelines/`**: dataset transformation and orchestration
- **`training/`**: training scripts/configs and packaging
- **`evals/`** and **`tests/`**: evaluation harnesses and test suites
- **`api/`** and **`main.py`**: service entry points
- **`safety/`** and **`security/`**: safety filters, policy, remediation, audits
- **`docs/`**: AI-specific documentation

Also see: `ARCHITECTURE.md` for the consolidation plan and canonical responsibilities.

---

## Quick Start (uv)

**Install dependencies** (from the `ai/` repo root):

```bash
uv install
```

**Run a script**:

```bash
uv run python <script>.py
```

**Run the FastAPI service** (if applicable for your change):

```bash
uv run uvicorn main:app --reload
```

---

## Engineering Conventions (AI Repo)

- **Type safety**:
  - Prefer explicit types and dataclasses / Pydantic models where appropriate.
  - Avoid `Any` unless there is no reasonable alternative.
- **Determinism**:
  - Make randomness explicit (seed where needed).
  - Log configuration inputs that materially affect outputs (datasets,
    sampling, model versions).
- **I/O discipline**:
  - Never implicitly write large artifacts into the repo.
  - Keep data/model outputs under the appropriate data/artifacts directories
    and ensure `.gitignore` rules are respected.
- **Notebooks**:
  - Treat notebooks as analysis artifacts.
  - Prefer moving reusable logic into importable modules; keep notebooks thin.

---

## 🔒 Security, Privacy, and Psychological Safety

This platform handles sensitive mental health data and safety-critical outputs.
In this repo, that means:

### Zero-Leak Policy

- **Never expose** API keys, tokens, secrets, PII/PHI, or raw private transcripts.
- **Never commit** `.env` files or secrets.
- If you add new configuration:
  - Use environment variables.
  - Update `.env.example` (never real secrets).

### Crisis and Harm Signals

- Do not ignore potential self-harm / crisis indicators.
- Safety-related changes should be conservative and include tests.

### Data Handling Rules

- Minimize retention of raw user text/audio.
- Prefer derived/aggregated features when possible.
- Ensure any exports/redactions are explicit and documented.

### Reporting

If you discover a security vulnerability, follow `SECURITY.md` (do not file
public issues).

---

## Testing & Quality Gates

**Preferred workflow** (from repo root):

```bash
uv run pytest
```

**Coverage** (if configured by the change):

```bash
uv run pytest --cov
```

**Lint/format** (repo uses Ruff + Black):

```bash
uv run ruff check .
uv run black .
```

**Notebooks** (repo uses NBQA; run only when you touched notebooks):

```bash
uv run nbqa ruff .
uv run nbqa black .
```

---

## Critical Rules (Non-Negotiable)

```text
❌ NEVER use pip, conda, or poetry (use uv)
❌ NEVER commit .env files or secrets
❌ NEVER introduce stubs, placeholders, or unfinished implementations
❌ NEVER ignore safety or crisis-related edge cases

✅ ALWAYS validate inputs at module boundaries (API, CLI, pipeline stage)
✅ ALWAYS keep runs reproducible (configs + seeds)
✅ ALWAYS add/adjust tests for behavior-changing edits
✅ ALWAYS run lint + tests before committing
```

---

## Interaction Protocol (Hooks)

- **Thread Start**: Check `supermemory` (project: `pixelated`) and
  `.ralph/progress.txt` for context on the current/upcoming task.
- **Thread End**: Log the completed task/milestone in `supermemory`
  (project: `pixelated`) and update `.ralph/progress.txt`.

---

## Mission Reminder

> **We don't just process conversations. We understand them.**

Every decision in this repo should prioritize:

- 🛡️ **Psychological Safety**
- 🔐 **Privacy & Confidentiality**
- 🧠 **Ethical AI Practices**
- 💜 **Genuine Human Connection**

---

## ⚠️ CRITICAL: PR Creation Rules for Jules

Based on experience from pixelated repo where 60 massive PRs blocked each other.

### RULE 1: MAXIMUM 30 FILES PER PR
- **Hard limit: 30 files maximum**
- **Ideal size: 5-15 files**
- If your change touches more than 30 files, SPLIT IT into multiple PRs
- **Why**: We had 60 PRs all blocked because they touched the same files

### RULE 2: ONE TASK = ONE PR
- Each PR should implement ONE specific feature or fix
- Good examples:
  - "Add unit tests for CheckpointManager"
  - "Fix S3 credential handling in data pipeline"
  - "Optimize voice pipeline caching"
- Bad examples (DO NOT DO):
  - "Fix everything" (857 files across all modules)
  - "Update configs and add tests and fix linting"

### RULE 3: CHECK BEFORE CREATING
Before creating ANY PR:
1. Check open PRs: `gh pr list --state open | wc -l`
2. If >5 open PRs: STOP and wait
3. Target branch: `master`

### RULE 4: DESCRIPTIVE BRANCH NAMES
```
[scope]/[action]-[component]-[id]
```
Examples:
- `fix/checkpoint-manager-memory-leak-abc123`
- `feat/add-s3-retry-logic-def456`
- `test/coverage-for-voice-pipeline-ghi789`

### Forbidden Actions
- ❌ PRs touching >30 files
- ❌ Multiple unrelated changes in one PR
- ❌ PRs when >5 are already open
- ❌ Vague branch names like `fix/things`

### Auto-Abort Conditions
- >30 files changed
- >5 open PRs exist
- Cannot explain change in one sentence

---

## Verification Checklist

Before submitting:
- [ ] PR touches ≤30 files
- [ ] PR focuses on ONE change
- [ ] Branch name is descriptive
- [ ] Open PR count < 5
