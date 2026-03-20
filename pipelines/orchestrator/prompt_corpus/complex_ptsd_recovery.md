# complex_ptsd_recovery

## Template 1: Emotional Flashback Recognition

### Template Metadata

```yaml
template_id: complex_ptsd_recovery.t1
category: complex_ptsd_recovery
version: "1.0"
safety_level: elevated
placeholders:
  - client_profile
  - presenting_problem
  - risk_context
output_format: structured_sections
tags:
  - cptsd
  - emotional_flashback
  - grounding
```

Generate a dialogue helping a client recognize and respond to an emotional flashback.

- Client context: {{client_profile}}
  - Trigger situation: {{presenting_problem}}
  - Current regulation state: {{risk_context}}

Requirements:

1. Therapist names the flashback without pathologizing.
2. Offer one grounding technique (sensory or breath-based).
3. Include a "this is then, this is now" distinction statement.
4. Avoid asking for flashback details—focus on present safety.

## Template 2: 4F Response Identification

### Template Metadata

```yaml
template_id: complex_ptsd_recovery.t2
category: complex_ptsd_recovery
version: "1.0"
safety_level: standard
placeholders:
  - presenting_problem
  - therapy_goal
output_format: structured_sections
tags:
  - cptsd
  - fight_flight_freeze_fawn
  - survival_responses
```

Help the client identify which of the 4F responses is activated and validate its protective function.

- Situation: {{presenting_problem}}
  - Therapeutic goal: {{therapy_goal}}

Output:

1. Gentle inquiry about the survival response
2. Validation of its historical necessity
3. One question about what safety would look like now

## Template 3: Self-Compassion for the Inner Critic

### Template Metadata

```yaml
template_id: complex_ptsd_recovery.t3
category: complex_ptsd_recovery
version: "1.0"
safety_level: standard
placeholders:
  - client_profile
  - therapy_goal
output_format: dialogue_turns
tags:
  - cptsd
  - inner_criticism
  - self_reparenting
```

Generate a dialogue addressing the inner critic as a protective part from childhood.

- Client context: {{client_profile}}
  - Target shift: {{therapy_goal}}

Requirements:

1. Frame the critic as a younger part trying to prevent shame.
2. Offer a compassionate response from the client's present-day Self.
3. Include one gentle reframe of a self-critical statement.
