# internal_family_systems

## Template 1: Parts Identification Dialogue

### Template Metadata

```yaml
template_id: internal_family_systems.t1
category: internal_family_systems
version: "1.0"
safety_level: standard
placeholders:
  - client_profile
  - presenting_problem
  - therapy_goal
output_format: dialogue_turns
tags:
  - ifs
  - parts_work
  - self_leadership
```

Generate an IFS-informed dialogue where the therapist helps the client identify and differentiate parts.

- Client context: {{client_profile}}
- Presenting issue: {{presenting_problem}}
  - Session goal: {{therapy_goal}}

Requirements:

1. Therapist demonstrates curiosity toward all parts without judgment.
2. Include one question inviting the client to notice which part is active.
3. Model Self-energy language: calm, curious, compassionate.
4. End with a gentle invitation to get to know the part, not change it.

## Template 2: Protector Acknowledgment

### Template Metadata

```yaml
template_id: internal_family_systems.t2
category: internal_family_systems
version: "1.0"
safety_level: standard
placeholders:
  - presenting_problem
  - risk_context
output_format: structured_sections
tags:
  - ifs
  - protectors
  - safety_negotiation
```

Create a dialogue where the therapist acknowledges a protective part's positive intention.

- Client's protective response: {{presenting_problem}}
  - Safety context: {{risk_context}}

Output:

1. Validation statement for the protector's role
2. Inquiry about what the protector is trying to prevent
3. Invitation for collaboration between Self and protector

## Template 3: Unblending Exercise

### Template Metadata

```yaml
template_id: internal_family_systems.t3
category: internal_family_systems
version: "1.0"
safety_level: elevated
placeholders:
  - therapy_goal
output_format: guided_exercise
tags:
  - ifs
  - unblending
  - self_differentiation
```

Generate a guided unblending exercise for a client fused with an anxious part.

Focus: {{therapy_goal}}

Constraints:

- Use "some part of you" language rather than "you."
- Include grounding breath cues between steps.
- Avoid pushing for immediate insight; prioritize safety.
