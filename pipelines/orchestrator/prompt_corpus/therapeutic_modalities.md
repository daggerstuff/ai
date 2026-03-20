# therapeutic_modalities

## Template 1: Modality Match and Rationale

### Template Metadata

```yaml
template_id: therapeutic_modalities.t1
category: therapeutic_modalities
version: "1.0"
safety_level: standard
placeholders:
 - client_profile
 - presenting_problem
 - therapy_goal
 - risk_context
 - book_anchor
output_format: structured_sections
tags:
 - modality_selection
 - formulation
```

Design a therapeutic response for:

- Client profile: {{client_profile}}
- Presenting problem: {{presenting_problem}}
- Session goal: {{therapy_goal}}
- Risk context: {{risk_context}}
- Preferred modality anchor: {{book_anchor}}

Requirements:

1. Name the chosen modality and why it fits.
2. Provide one intervention and one reflection statement.
3. Include a safe next-session bridge.

## Template 2: Compare Two Modalities

### Template Metadata

```yaml
template_id: therapeutic_modalities.t2
category: therapeutic_modalities
version: "1.0"
safety_level: standard
placeholders:
 - presenting_problem
 - risk_context
output_format: side_by_side_comparison
tags:
 - cbt
 - dbt
 - differential_planning
```

Given the case below, compare CBT and DBT style responses.

- Case: {{presenting_problem}}
- Current emotional state: {{risk_context}}

Output:

- CBT-style intervention (2-4 lines)
- DBT-style intervention (2-4 lines)
- Short recommendation on which to prioritize now and why

## Template 3: Stage-3 Stress Prompt

### Template Metadata

```yaml
template_id: therapeutic_modalities.t3
category: therapeutic_modalities
version: "1.0"
safety_level: elevated
placeholders:
 - therapy_goal
output_format: dialogue_starter
tags:
 - stress_test
 - alliance_repair
```

Generate a difficult-client dialogue starter where the therapist must stay grounded while applying {{therapy_goal}}.

Constraints:

- Include emotional escalation without graphic details.
- Therapist must validate emotion before strategy.
- Add one rupture-and-repair opportunity.
