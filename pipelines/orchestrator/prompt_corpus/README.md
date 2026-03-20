# Prompt Corpus

This directory provides category-scoped therapeutic dialogue prompt templates used by the Stage 3 edge stress test pipeline.

## Purpose

- Mirror and version scenario prompt assets in-repo for deterministic training runs.
- Seed prompt generation with category-specific templates derived from the psychology knowledge base taxonomy.
- Provide stable prompt inputs for MTGC-05 and downstream freshness checks.

## Source Anchors

Category names are aligned with the enhanced psychology knowledge-base summary:

- therapeutic_modalities
- developmental_psychology
- trauma_psychology
- contemplative_psychology
- family_systems
- body_psychology
- well_being_psychology
- postmodern_approaches
- emotion_psychology
- psychology_book_reference
- therapeutic_conversation_example

### Book-Anchored Categories (MTGC-05 Addition)

These categories are grounded in specific therapeutic texts from `ai/docs/Books/`:

- **internal_family_systems** — Schwartz, R.C. *Internal Family Systems Therapy*
- **complex_ptsd_recovery** — Walker, P. *Complex PTSD: From Surviving to Thriving*
- **brain_energy_metabolism** — Palmer, C.M. *Brain Energy: A Revolutionary Breakthrough in Understanding Mental Health*

## Template Format

Each category file includes reusable prompt skeletons with placeholders:

- {{client_profile}}
- {{presenting_problem}}
- {{risk_context}}
- {{therapy_goal}}
- {{book_anchor}}
- {{session_constraints}}

Use these placeholders with dataset-specific context when generating synthetic dialogue examples.

## Metadata Contract (Parser-Ready)

Every template section must include a `Template Metadata` YAML block with the fields below:

- `template_id`: Stable ID (`<category>.tN`)
- `category`: Category slug matching filename
- `version`: Template version string
- `safety_level`: `standard` or `elevated`
- `placeholders`: Required placeholder variables
- `output_format`: Expected response shape (e.g., `structured_sections`, `dialogue_turns`)
- `tags`: Short semantic labels

Example:

```yaml
template_id: trauma_psychology.t1
category: trauma_psychology
version: "1.0"
safety_level: elevated
placeholders:
 - client_profile
 - presenting_problem
 - risk_context
output_format: structured_sections
tags:
 - trauma_informed
 - safety
 - grounding
```
