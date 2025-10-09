# Crisis Detection System Documentation

## Overview

The Pixelated Empathy Crisis Detection System is an enterprise-grade safety system designed to identify and respond to mental health crises in therapeutic conversations. This system provides real-time crisis detection, automated escalation protocols, and comprehensive audit logging for clinical compliance.

**⚠️ CRITICAL SAFETY NOTICE**: This system is designed to assist, not replace, human clinical judgment. All crisis detections require human verification and appropriate clinical response.

## System Architecture

### Core Components

1. **Crisis Detection Engine** (`crisis_intervention_detector.py`)
   - Real-time analysis of conversation content
   - 10 crisis type classifications
   - Confidence-based severity scoring
   - Multi-format input support

2. **Escalation Protocol System**
   - Automated emergency response workflows
   - Multi-level escalation (Emergency, Critical, Elevated, Routine)
   - Emergency services integration
   - Response time validation

3. **Monitoring & Alerting**
   - Real-time system health monitoring
   - 7 alert types for system failures
   - Performance degradation detection
   - Automatic failure recovery

4. **Audit Logging System**
   - Comprehensive event logging (8 event types)
   - JSON-structured audit trails
   - Content integrity verification
   - Compliance-ready documentation

## Crisis Types Detected

### 1. Suicide Ideation (Emergency/Critical)
**Indicators**: Death wishes, suicide plans, goodbye messages, giving away possessions
**Patterns**: "I want to die", "end it all", "better off without me", "suicide plan"
**Response**: Immediate emergency services contact, safety planning

### 2. Self-Harm (Critical/Elevated)
**Indicators**: Cutting, burning, self-injury tools, escalation patterns
**Patterns**: "cut myself", "hurt myself", "razor blade", "burning"
**Response**: Crisis hotline referral, immediate clinical assessment

### 3. Violence Threats (Emergency/Critical)
**Indicators**: Threats to others, weapon access, specific targets, planning
**Patterns**: "kill them", "hurt someone", "weapon", "revenge"
**Response**: Law enforcement notification, threat assessment

### 4. Psychotic Episodes (Emergency/Critical)
**Indicators**: Command hallucinations, reality loss, paranoid delusions
**Patterns**: "voices telling me", "not real", "they're watching", "command"
**Response**: Emergency psychiatric evaluation, medication review

### 5. Substance Overdose (Emergency)
**Indicators**: Overdose mentions, drug combinations, lethal amounts
**Patterns**: "overdose", "too many pills", "mixing drugs", "lethal dose"
**Response**: Emergency medical services, poison control

### 6. Domestic Violence (Critical/Elevated)
**Indicators**: Abuse threats, escalation, children in danger, immediate harm
**Patterns**: "going to hurt me", "hit me", "children scared", "escalating"
**Response**: Safety planning, domestic violence resources

### 7. Child Abuse (Emergency/Critical)
**Indicators**: Child harm, abuse reports, neglect, immediate danger
**Patterns**: "hurt the child", "abuse", "neglect", "child in danger"
**Response**: Child protective services, mandatory reporting

### 8. Severe Depression (Critical/Elevated)
**Indicators**: Hopelessness, worthlessness, isolation, despair
**Patterns**: "no hope", "worthless", "can't go on", "nothing matters"
**Response**: Intensive therapy referral, medication evaluation

### 9. Panic Attacks (Elevated/Routine)
**Indicators**: Severe anxiety, physical symptoms, fear of dying
**Patterns**: "can't breathe", "heart racing", "going to die", "panic"
**Response**: Grounding techniques, anxiety management

### 10. Eating Disorders (Critical/Elevated)
**Indicators**: Severe restriction, purging, medical complications
**Patterns**: "starving myself", "purging", "medical emergency", "eating disorder"
**Response**: Medical evaluation, eating disorder specialist

## Severity Levels & Escalation

### Emergency (Confidence ≥ 0.8)
- **Response Time**: Immediate (0-5 minutes)
- **Actions**: Emergency services contact, supervisor alert, safety planning, documentation
- **Contacts**: 911, supervisor, crisis coordinator
- **Examples**: Active suicide attempt, violence in progress, overdose

### Critical (Confidence ≥ 0.6)
- **Response Time**: Immediate (0-15 minutes)
- **Actions**: Crisis hotline, immediate referral, supervisor alert, documentation
- **Contacts**: 988 Suicide & Crisis Lifeline, supervisor, crisis team
- **Examples**: Suicide ideation with plan, self-harm escalation, psychotic episode

### Elevated (Confidence ≥ 0.4)
- **Response Time**: Within 1 hour
- **Actions**: Safety planning, follow-up scheduling, documentation
- **Contacts**: Supervisor, case manager
- **Examples**: Depression with hopelessness, anxiety crisis, relationship crisis

### Routine (Confidence < 0.4)
- **Response Time**: Standard workflow
- **Actions**: Documentation only
- **Contacts**: None
- **Examples**: General stress, mild anxiety, normal therapeutic content

## Usage Guide

### Basic Implementation

```python
from crisis_intervention_detector import CrisisInterventionDetector

# Initialize detector
detector = CrisisInterventionDetector()

# Analyze conversation
conversation = [
    {"role": "user", "content": "I've been thinking about ending it all"},
    {"role": "assistant", "content": "I'm concerned about what you're saying..."}
]

result = detector.analyze_conversation(conversation)

# Check results
if result['crisis_detected']:
    print(f"Crisis Type: {result['crisis_type']}")
    print(f"Severity: {result['severity_level']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Escalation Required: {result['escalation_required']}")
```

### Input Formats Supported

#### Format 1: Messages Array
```python
conversation = [
    {"role": "user", "content": "I want to hurt myself"},
    {"role": "assistant", "content": "I'm here to help"}
]
```

#### Format 2: Content String
```python
conversation = "I've been having thoughts of suicide lately"
```

#### Format 3: Turns Array
```python
conversation = [
    {"speaker": "client", "text": "I can't take it anymore"},
    {"speaker": "therapist", "text": "Tell me more about that"}
]
```

### Response Handling

```python
def handle_crisis_response(result):
    if result['severity_level'] == 'emergency':
        # Immediate emergency response
        contact_emergency_services()
        notify_supervisor()
        create_safety_plan()
        
    elif result['severity_level'] == 'critical':
        # Crisis intervention
        contact_crisis_hotline()
        schedule_immediate_appointment()
        notify_supervisor()
        
    elif result['severity_level'] == 'elevated':
        # Enhanced support
        create_safety_plan()
        schedule_follow_up()
        document_concerns()
        
    # Always log the event
    log_crisis_event(result)
```

## Safety Protocols

### Clinical Guidelines

1. **Human Verification Required**
   - All crisis detections must be reviewed by qualified clinical staff
   - Automated responses supplement, never replace, human judgment
   - Clinical override capabilities must be available

2. **Response Time Requirements**
   - Emergency: 0-5 minutes maximum
   - Critical: 0-15 minutes maximum
   - Elevated: Within 1 hour
   - All response times are monitored and logged

3. **Documentation Standards**
   - All crisis events must be documented in clinical records
   - Audit trail must be maintained for legal compliance
   - Patient confidentiality must be preserved in all logging

4. **Escalation Procedures**
   - Clear chain of command for crisis response
   - 24/7 supervisor availability for emergency situations
   - Backup procedures for system failures

### Technical Safety Measures

1. **System Monitoring**
   - Real-time monitoring of detection accuracy
   - Automatic alerts for system failures
   - Performance degradation detection
   - Redundant backup systems

2. **Data Security**
   - Encrypted storage of all crisis data
   - Access controls and audit logging
   - HIPAA compliance for all data handling
   - Secure transmission protocols

3. **Quality Assurance**
   - Regular testing of crisis detection accuracy
   - Continuous monitoring of false positive/negative rates
   - Periodic review of escalation protocols
   - Staff training on system limitations

## Configuration

### Detection Thresholds
```python
CRISIS_THRESHOLDS = {
    'emergency': 0.8,    # Immediate emergency response
    'critical': 0.6,     # Crisis intervention required
    'elevated': 0.4,     # Enhanced support needed
    'routine': 0.0       # Standard therapeutic response
}
```

### Response Time Limits
```python
RESPONSE_TIME_LIMITS = {
    'emergency': 300,    # 5 minutes (seconds)
    'critical': 900,     # 15 minutes (seconds)
    'elevated': 3600,    # 1 hour (seconds)
    'routine': None      # No time limit
}
```

### Monitoring Settings
```python
MONITORING_CONFIG = {
    'check_interval': 30,        # seconds
    'response_time_threshold': 1000,  # milliseconds
    'detection_rate_threshold': 0.8,  # 80%
    'error_rate_threshold': 0.05,     # 5%
    'escalation_timeout': 60          # seconds
}
```

## API Reference

### CrisisInterventionDetector Class

#### Methods

**`analyze_conversation(conversation)`**
- **Input**: Conversation in supported format
- **Output**: Crisis detection result dictionary
- **Returns**: 
  ```python
  {
      'crisis_detected': bool,
      'crisis_type': str,
      'severity_level': str,
      'confidence': float,
      'escalation_required': bool,
      'indicators': list,
      'risk_factors': list,
      'protective_factors': list,
      'response_time_ms': float
  }
  ```

**`get_escalation_protocol(severity_level)`**
- **Input**: Severity level string
- **Output**: Escalation protocol dictionary
- **Returns**: Actions, contacts, and response time requirements

**`log_crisis_event(result, conversation_id=None)`**
- **Input**: Crisis detection result and optional conversation ID
- **Output**: Log entry confirmation
- **Purpose**: Creates audit trail entry for compliance

### Monitoring Functions

**`get_system_health()`**
- **Output**: Current system health metrics
- **Includes**: Response times, detection rates, error rates, active alerts

**`get_active_alerts()`**
- **Output**: List of current system alerts
- **Includes**: Alert type, severity, timestamp, description

## Troubleshooting

### Common Issues

1. **Low Detection Accuracy**
   - Check pattern matching configuration
   - Verify input format compatibility
   - Review confidence threshold settings
   - Validate training data quality

2. **Slow Response Times**
   - Monitor system resource usage
   - Check database connection performance
   - Verify network connectivity
   - Review concurrent request handling

3. **False Positives**
   - Adjust confidence thresholds
   - Review pattern matching rules
   - Implement context-aware filtering
   - Add protective factor detection

4. **False Negatives**
   - Expand crisis pattern coverage
   - Lower confidence thresholds carefully
   - Improve multi-turn conversation analysis
   - Enhance context understanding

### System Alerts

1. **Detection Failure Alert**
   - **Cause**: Crisis keywords present but no crisis detected
   - **Action**: Review pattern matching, check system logs
   - **Severity**: CRITICAL

2. **Performance Degradation Alert**
   - **Cause**: Response times exceeding thresholds
   - **Action**: Check system resources, optimize queries
   - **Severity**: MEDIUM

3. **Escalation Failure Alert**
   - **Cause**: Escalation protocol not triggered properly
   - **Action**: Verify escalation configuration, check contacts
   - **Severity**: HIGH

## Compliance & Legal

### HIPAA Compliance
- All patient data encrypted at rest and in transit
- Access controls and audit logging implemented
- Business associate agreements required for third parties
- Data retention policies enforced

### Clinical Documentation
- All crisis events documented in patient records
- Audit trail maintained for legal compliance
- Clinical oversight and review procedures
- Quality assurance and improvement processes

### Liability Considerations
- System designed to assist, not replace clinical judgment
- Human verification required for all crisis responses
- Clear documentation of system limitations
- Regular training on proper system usage

## Support & Maintenance

### Technical Support
- **Emergency Support**: 24/7 for system failures affecting patient safety
- **Standard Support**: Business hours for configuration and usage questions
- **Documentation**: Comprehensive guides and API reference available

### System Updates
- Regular updates for pattern improvements
- Security patches applied immediately
- Feature updates with clinical review
- Backward compatibility maintained

### Training Requirements
- Initial training for all clinical staff
- Ongoing education on system capabilities and limitations
- Regular competency assessments
- Documentation of training completion

---

**Version**: 1.0  
**Last Updated**: August 10, 2025  
**Next Review**: September 10, 2025  

**Contact**: Pixelated Empathy Technical Team  
**Emergency Support**: Available 24/7 for safety-critical issues
