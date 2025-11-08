# Edge Case Generator Integration

## Overview

The Edge Case Generator uses NVIDIA NeMo Data Designer to generate challenging, rare, and edge case scenarios for therapeutic training. It integrates seamlessly with the existing scenario generation system in Pixelated Empathy.

## Features

- **10 Edge Case Types**: Crisis, cultural complexity, comorbidity, boundary violations, trauma disclosure, substance abuse, ethical dilemmas, rare diagnoses, multi-generational families, and systemic oppression
- **Difficulty Levels**: Supports beginner, intermediate, and advanced difficulty levels
- **Scalable Generation**: Generate single or multiple edge case types at scale
- **API Integration**: Formatted output compatible with existing scenario generation API

## Edge Case Types

1. **CRISIS** - Suicidal ideation, self-harm, substance overdose, domestic violence, psychotic episodes
2. **CULTURAL_COMPLEXITY** - Language barriers, immigration status, cultural conflicts, stigma
3. **COMORBIDITY** - Multiple co-occurring mental health conditions
4. **BOUNDARY_VIOLATION** - Dual relationships, gift offering, social media requests, physical boundary testing
5. **TRAUMA_DISCLOSURE** - Childhood abuse, sexual assault, domestic violence, combat trauma
6. **SUBSTANCE_ABUSE** - Alcohol, opioids, stimulants, polysubstance use
7. **ETHICAL_DILEMMA** - Confidentiality breaches, mandatory reporting, competence boundaries
8. **RARE_DIAGNOSIS** - Dissociative identity disorder, factitious disorder, selective mutism
9. **MULTI_GENERATIONAL** - Extended families, generational conflicts, cultural values
10. **SYSTEMIC_OPPRESSION** - Racism, sexism, classism, ableism, intersectional oppression

## Usage

### Basic Usage

```python
from ai.data_designer.edge_case_generator import EdgeCaseGenerator, EdgeCaseType

# Initialize generator
generator = EdgeCaseGenerator()

# Generate crisis scenarios
result = generator.generate_edge_case_dataset(
    edge_case_type=EdgeCaseType.CRISIS,
    num_samples=10,
    difficulty_level="advanced",
)

# Access generated data
scenarios = result['data']
print(f"Generated {len(scenarios)} crisis scenarios")
```

### Multiple Edge Case Types

```python
# Generate multiple types
result = generator.generate_multi_edge_case_dataset(
    edge_case_types=[
        EdgeCaseType.CRISIS,
        EdgeCaseType.CULTURAL_COMPLEXITY,
        EdgeCaseType.BOUNDARY_VIOLATION,
    ],
    num_samples_per_type=20,
    difficulty_level="advanced",
)

print(f"Generated {result['total_samples']} total scenarios")
```

### API Integration

```python
from ai.data_designer.edge_case_api import EdgeCaseAPI

# Initialize API
api = EdgeCaseAPI()

# Generate scenarios formatted for scenario API
result = api.generate_scenario(
    edge_case_type="crisis",
    num_samples=10,
    difficulty_level="advanced",
)

# Access formatted scenarios
scenarios = result['scenarios']
for scenario in scenarios:
    print(f"ID: {scenario['id']}")
    print(f"Title: {scenario['title']}")
    print(f"Challenge Level: {scenario['challengeLevel']}")
```

## Integration with Scenario Generation API

The edge case generator produces scenarios compatible with the existing scenario generation API format:

```typescript
interface EdgeCaseScenario {
  id: string
  title: string
  description: string
  edge_case_type: string
  difficulty_level: string
  clientProfile: {
    age: number
    gender: string
    ethnicity: string
    background: string
  }
  edgeCaseDetails: {
    // Type-specific details
    crisis_type?: string
    crisis_severity?: string
    // ... other fields based on edge case type
  }
  challengeLevel: string
  estimatedDuration: string
  metadata: {
    generatedAt: string
    source: string
  }
}
```

## Example: Crisis Scenario

```python
from ai.data_designer.edge_case_api import EdgeCaseAPI

api = EdgeCaseAPI()
result = api.generate_scenario("crisis", num_samples=5)

# Example output:
# {
#   "id": "edge_case_crisis_0001",
#   "title": "Crisis Intervention Scenario: Age 42",
#   "description": "Crisis scenario involving suicidal_ideation with severity level high",
#   "edge_case_type": "crisis",
#   "difficulty_level": "advanced",
#   "clientProfile": {
#     "age": 42,
#     "gender": "non-binary",
#     "ethnicity": "Mixed/Other"
#   },
#   "edgeCaseDetails": {
#     "crisis_type": "suicidal_ideation",
#     "crisis_severity": "high",
#     "suicidal_ideation_present": "yes",
#     "immediate_risk_score": 8.5
#   },
#   "challengeLevel": "advanced",
#   "estimatedDuration": "60-90 minutes"
# }
```

## Files

- `edge_case_generator.py` - Core edge case generation logic
- `edge_case_api.py` - API interface for scenario integration
- `edge_case_example.py` - Example usage scripts

## Testing

Run the example script to test:

```bash
uv run python ai/data_designer/edge_case_example.py
```

## Integration Points

1. **Scenario Generation API** (`src/pages/api/psychology/generate-scenario.ts`)
   - Can call edge case API to generate edge case scenarios
   - Edge cases formatted to match existing scenario format

2. **Training Pipeline**
   - Generate edge case datasets for training data augmentation
   - Focus on rare but critical scenarios

3. **Bias Detection**
   - Use edge cases to test bias detection systems
   - Generate scenarios with protected attributes

## Next Steps

1. **API Endpoint**: Create TypeScript API endpoint that calls the Python edge case generator
2. **UI Integration**: Add edge case generation to the scenario generation UI
3. **Training Integration**: Integrate edge cases into the training data pipeline
4. **Quality Metrics**: Add quality metrics for generated edge cases

## Notes

- Edge cases are generated using synthetic data from NeMo Data Designer
- All scenarios are designed for training purposes
- Edge cases focus on challenging, rare scenarios that require specialized skills
- Generated data maintains privacy and does not contain real patient information

