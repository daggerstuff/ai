# brain_energy_metabolism

## Template 1: Metabolic Framing Question

### Template Metadata

```yaml
template_id: brain_energy_metabolism.t1
category: brain_energy_metabolism
version: "1.0"
safety_level: standard
placeholders:
  - client_profile
  - presenting_problem
  - therapy_goal
output_format: structured_sections
tags:
  - metabolic_mental_health
  - lifestyle_psychiatry
  - holistic
```

Generate a therapist response that gently introduces metabolic considerations alongside traditional therapy.

- Client context: {{client_profile}}
  - Presenting concern: {{presenting_problem}}
  - Session focus: {{therapy_goal}}

Requirements:

1. Acknowledge psychological factors first.
2. Introduce one metabolic question (sleep, nutrition, movement) without medical advice.
3. Frame as "worth exploring with your care team."
4. Maintain therapeutic alliance—avoid prescriptive language.

## Template 2: Energy and Mood Connection

### Template Metadata

```yaml
template_id: brain_energy_metabolism.t2
category: brain_energy_metabolism
version: "1.0"
safety_level: standard
placeholders:
  - presenting_problem
output_format: psychoeducation_snippet
tags:
  - metabolic_mental_health
  - psychoeducation
  - lifestyle
```

Create a brief psychoeducation piece connecting brain energy to the client's experience.

Client's report: {{presenting_problem}}

Output:

1. One sentence normalizing the brain-body connection
2. One open question about energy patterns
3. One non-prescriptive lifestyle curiosity (not recommendation)

## Template 3: Collaborative Treatment Team Discussion

### Template Metadata

```yaml
template_id: brain_energy_metabolism.t3
category: brain_energy_metabolism
version: "1.0"
safety_level: standard
placeholders:
  - therapy_goal
  - risk_context
output_format: dialogue_starter
tags:
  - metabolic_mental_health
  - integrated_care
  - collaboration
```

Generate a dialogue where the therapist discusses coordinating with medical providers about metabolic factors.

Therapeutic context: {{therapy_goal}}
  - Client stability: {{risk_context}}

Requirements:

1. Emphasize client's right to explore options.
2. Suggest questions for the client to ask their medical provider.
3. Avoid diagnosing or prescribing—stay in therapeutic scope.
