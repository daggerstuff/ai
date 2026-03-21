# developmental_psychology

## Template 1: Developmental Lens Formulation

### Template Metadata

```yaml
template_id: developmental_psychology.t1
category: developmental_psychology
version: "1.0"
safety_level: standard
placeholders:
 - client_profile
 - presenting_problem
 - therapy_goal
output_format: structured_sections
tags:
 - developmental_tasks
 - case_formulation
```

Create a concise formulation using a developmental psychology lens.

- Client age/stage: {{client_profile}}
- Core conflict: {{presenting_problem}}
- Immediate objective: {{therapy_goal}}

Output:

1. Stage-relevant developmental task
2. How the presenting problem maps to that task
3. One therapist intervention question

## Template 2: Reframe by Developmental Need

### Template Metadata

```yaml
template_id: developmental_psychology.t2
category: developmental_psychology
version: "1.0"
safety_level: standard
placeholders:
 - presenting_problem
output_format: short_reframe
tags:
 - unmet_needs
 - collaborative_next_step
```

Given {{presenting_problem}}, produce a reframe that centers unmet developmental needs while keeping clinical tone.

Constraints:

- Do not pathologize.
- Keep to 3-5 sentences.
- End with one collaborative next step.
