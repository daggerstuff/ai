# trauma_psychology

## Template 1: Trauma-Informed Opening

### Template Metadata

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

Generate a trauma-informed opening response.

- Client context: {{client_profile}}
- Trigger narrative: {{presenting_problem}}
- Safety context: {{risk_context}}

Requirements:

1. Prioritize safety and agency language.
2. Include one grounding suggestion.
3. Avoid certainty claims and re-traumatizing detail.

## Template 2: Window of Tolerance Check

### Template Metadata

```yaml
template_id: trauma_psychology.t2
category: trauma_psychology
version: "1.0"
safety_level: elevated
placeholders:
 - therapy_goal
output_format: structured_sections
tags:
 - regulation
 - window_of_tolerance
```

Create a therapist prompt that assesses arousal level and keeps the client in the window of tolerance.

Output sections:

- Regulation check question
- Stabilization micro-intervention
- Gentle transition toward {{therapy_goal}}
