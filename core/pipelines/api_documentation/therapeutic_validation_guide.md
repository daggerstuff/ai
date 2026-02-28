
# Therapeutic Conversation Validation Integration

## Overview
The validation API provides comprehensive assessment of therapeutic conversations using multiple validation layers.

## Validation Levels
1. **Basic**: Quality score and tier assessment
2. **Standard**: Includes safety and coherence validation
3. **Comprehensive**: Full multi-tier validation with recommendations
4. **Clinical**: Includes DSM-5 accuracy and effectiveness prediction

## Integration Steps

### 1. Prepare Conversation Data
```python
conversation = {
    "id": "unique_conversation_id",
    "content": "Full conversation text",
    "turns": [
        {"speaker": "user", "text": "User message"},
        {"speaker": "therapist", "text": "Therapist response"}
    ],
    "metadata": {
        "source": "professional",  # priority, professional, cot, reddit, research
        "condition": "anxiety",    # mental health condition
        "approach": "CBT",         # therapeutic approach
        "timestamp": "2025-08-10T07:30:00Z"
    }
}
```

### 2. Submit for Validation
```python
response = requests.post(
    "https://api.pixelatedempathy.com/api/v1/validate/conversation",
    headers={"Authorization": "Bearer your_api_key"},
    json={
        "conversation": conversation,
        "validation_level": "comprehensive"
    }
)
```

### 3. Process Results
```python
if response.status_code == 200:
    result = response.json()
    
    # Overall quality assessment
    quality_score = result["overall_quality_score"]
    tier = result["tier_assessment"]
    
    # Detailed validation results
    validations = result["validation_results"]
    safety_score = validations["safety_ethics"]["score"]
    effectiveness = validations["effectiveness_prediction"]["score"]
    
    # Improvement recommendations
    recommendations = result["recommendations"]
```

## Best Practices
- Validate conversations before adding to training datasets
- Use comprehensive validation for clinical applications
- Implement retry logic for rate limit handling
- Cache validation results to avoid duplicate requests
        