# Compliance and Security Module

This module provides comprehensive compliance checking, privacy verification, HIPAA validation, audit logging, and data encryption capabilities for the journal dataset research system.

## Overview

The compliance module implements all requirements from Task 9, including:

1. **License Compatibility Checker** (9.1)
2. **Privacy Verification** (9.2)
3. **HIPAA Compliance Validation** (9.3)
4. **Audit Logging** (9.4)
5. **Data Encryption** (9.5)

## Components

### License Checker (`license_checker.py`)

- Supports common open-source licenses (MIT, Apache, BSD, GPL, LGPL, AGPL)
- Supports Creative Commons licenses (CC-BY, CC-BY-SA, CC-BY-NC, CC0, etc.)
- Detects AI training restrictions
- Verifies commercial use compatibility
- Flags incompatible licenses for review

**Usage:**
```python
from ai.sourcing.journal.compliance import LicenseChecker

checker = LicenseChecker()
result = checker.check_license("MIT License")
print(result.is_usable())  # True
```

### Privacy Verifier (`privacy_verifier.py`)

- Detects PII in dataset samples (email, phone, SSN, credit card, addresses, medical IDs)
- Assesses anonymization quality
- Evaluates re-identification risks
- Generates privacy assessment reports

**Usage:**
```python
from ai.sourcing.journal.compliance import PrivacyVerifier

verifier = PrivacyVerifier()
assessment = verifier.verify_privacy(
    source_id="dataset-1",
    dataset_sample="Sample text from dataset..."
)
print(assessment.pii_detected)  # True/False
print(assessment.compliance_score)  # 0.0-1.0
```

### HIPAA Validator (`hipaa_validator.py`)

- Detects PHI in datasets
- Validates encryption requirements (at rest and in transit)
- Checks access control implementation
- Validates audit logging completeness
- Provides HIPAA compliance checklist

**Usage:**
```python
from ai.sourcing.journal.compliance import HIPAAValidator

validator = HIPAAValidator()
result = validator.validate_hipaa_compliance(
    source_id="dataset-1",
    dataset_description="Patient therapy session data",
    encryption_status={"at_rest": True, "in_transit": True},
    access_control_status={"implemented": True},
    audit_logging_status=True,
)
print(result.is_compliant())  # True/False
```

### Audit Logger (`audit_logger.py`)

- Comprehensive audit logging for all dataset access and modifications
- Tamper-proof log storage with cryptographic hash chaining
- Query capabilities for log analysis
- Log integrity verification

**Usage:**
```python
from ai.sourcing.journal.compliance import AuditLogger, AuditEventType

logger = AuditLogger(log_directory="./logs/audit")
logger.log_dataset_access(
    source_id="dataset-1",
    user_id="user-1",
    action="download",
    outcome="success",
)

# Query logs
entries = logger.query_logs(source_id="dataset-1")

# Verify integrity
results = logger.verify_log_integrity()
```

### Encryption Manager (`encryption_manager.py`)

- Encryption for datasets at rest (file encryption)
- Secure key management
- Encryption for sensitive configuration data
- Support for symmetric (Fernet) and asymmetric (RSA) encryption

**Usage:**
```python
from ai.sourcing.journal.compliance import EncryptionManager

manager = EncryptionManager(key_directory="./keys")

# Encrypt file
encrypted_file = manager.encrypt_file("dataset.csv")

# Decrypt file
decrypted_file = manager.decrypt_file(encrypted_file)

# Encrypt string
encrypted = manager.encrypt_string("sensitive data")
decrypted = manager.decrypt_string(encrypted)
```

### Compliance Checker (`compliance_checker.py`)

Orchestrates all compliance checks and generates comprehensive compliance reports.

**Usage:**
```python
from ai.sourcing.journal.compliance import ComplianceChecker, AuditLogger, EncryptionManager
from ai.sourcing.journal.models.dataset_models import DatasetSource

# Initialize components
audit_logger = AuditLogger()
encryption_manager = EncryptionManager()
checker = ComplianceChecker(
    audit_logger=audit_logger,
    encryption_manager=encryption_manager,
)

# Check compliance
result = checker.check_compliance(
    source=dataset_source,
    dataset_sample="Sample text...",
    license_text="MIT License",
)

print(result.is_compliant())  # True/False
print(result.overall_compliance_score)  # 0.0-1.0
```

## Integration with Evaluation Engine

The compliance module is integrated with the evaluation engine to enhance ethical accessibility assessment:

```python
from ai.sourcing.journal.compliance import ComplianceChecker
from ai.sourcing.journal.evaluation import DatasetEvaluationEngine

# Initialize with compliance checker
compliance_checker = ComplianceChecker()
evaluation_engine = DatasetEvaluationEngine(
    compliance_checker=compliance_checker,
)

# Evaluate dataset (compliance checks run automatically)
evaluation = evaluation_engine.evaluate_dataset(source)
print(evaluation.compliance_status)  # compliant, partially_compliant, non_compliant
print(evaluation.compliance_score)  # 0.0-1.0
```

## Data Model Updates

The `DatasetEvaluation` and `AcquiredDataset` models have been updated to include compliance information:

- `compliance_checked`: Whether compliance checks were performed
- `compliance_status`: Overall compliance status
- `compliance_score`: Compliance score (0.0-1.0)
- `license_compatible`: License compatibility status
- `privacy_compliant`: Privacy compliance status
- `hipaa_compliant`: HIPAA compliance status
- `encrypted`: Whether dataset is encrypted

## Testing

Comprehensive tests are available in `tests/test_compliance.py`:

```bash
pytest ai/sourcing/journal/tests/test_compliance.py -v
```

## Requirements Met

- ✅ **6.1**: Verify dataset license permits AI training use
- ✅ **6.2**: Verify dataset license permits commercial use
- ✅ **6.3**: Assess anonymization standards for patient privacy
- ✅ **6.4**: Document usage restrictions or attribution requirements
- ✅ **6.5**: Flag datasets requiring IRB approval
- ✅ **6.6**: Ensure all acquired datasets comply with HIPAA privacy standards
- ✅ **6.7**: Reject datasets with inadequate privacy protections or incompatible licenses

## Security Considerations

- Master keys are stored securely with restrictive file permissions (0o600)
- Audit logs use cryptographic hash chaining for tamper detection
- Encryption uses industry-standard algorithms (Fernet, RSA)
- All compliance checks are logged for audit trails

## Future Enhancements

- Support for additional license types
- Enhanced PII detection with machine learning
- Integration with external compliance databases
- Automated compliance reporting
- Real-time compliance monitoring

