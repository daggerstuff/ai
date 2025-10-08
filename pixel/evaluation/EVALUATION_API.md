# Pixel Evaluation Framework – API & Workflow Documentation

## Overview

The Pixel Evaluation Framework provides a modular, production-grade system for evaluating conversational AI models across multiple clinical and behavioral domains. It is designed for extensibility, security, compliance, and high-throughput operation.

---

## Core Concepts

- **BaseEvaluator**: Abstract base class providing standardized error handling, HIPAA/PII-safe logging, audit/event tracking, metric monitoring, and compliance hooks.
- **Domain Evaluators**: Subclasses of `BaseEvaluator` implementing domain-specific evaluation logic (e.g., EQ, Empathy, Clinical Accuracy, Persona Switching, Conversational Quality, Therapeutic Appropriateness).
- **Batch & Async Support**: All evaluators support batch and asynchronous evaluation for scalable, high-throughput workflows.

---

## API Summary

### BaseEvaluator

```python
class BaseEvaluator(ABC):
    def evaluate(self, conversation: Any) -> dict[str, float]:
        """Run all evaluations and aggregate results."""

    def batch_evaluate(self, conversations: list[Any]) -> list[dict[str, float]]:
        """Batch process multiple conversations for evaluation."""

    async def async_batch_evaluate(self, conversations: list[Any]) -> list[dict[str, float]]:
        """Asynchronously batch process multiple conversations for evaluation."""

    def audit_log(self, action: str, status: str, context: dict | None = None):
        """Audit log for evaluation actions (context is sanitized for PII)."""

    def track_event(self, event: str, details: dict | None = None):
        """Track workflow or reporting events (details are sanitized for PII)."""

    def monitor_metric(self, metric_name: str, value: float, **kwargs):
        """Hook for monitoring/metrics system."""

    def sanitize_context(self, context: dict | None) -> dict:
        """Remove or mask PII/sensitive data from context before logging or auditing."""

    def encrypt_data(self, data: Any) -> Any:
        """Stub for encrypting sensitive data before storage/transmission."""

    def verify_integrity(self, data: Any) -> bool:
        """Stub for verifying data integrity."""
```

---

## Usage Example

```python
from ai.pixel.evaluation.eq_evaluator import EQEvaluator

evaluator = EQEvaluator()
single_result = evaluator.evaluate(conversation)

batch_results = evaluator.batch_evaluate([conv1, conv2, conv3])

import asyncio
async def run_async_batch():
    results = await evaluator.async_batch_evaluate([conv1, conv2, conv3])
```

---

## Security, Privacy, and Compliance

- **No PII in Logs**: All logging and audit methods use `sanitize_context` to strip or mask PII/sensitive data.
- **Encryption/Integrity**: Stubs are provided for encryption and integrity checks; extend as needed for your deployment.
- **HIPAA/PII-Safe Logging**: Use a HIPAA-compliant logger if available.
- **No Raw Data Storage**: Subclasses must not log or store raw conversation data or PII.

---

## Extending the Framework

- **Add a New Evaluator**: Subclass `BaseEvaluator` and implement the `evaluate` method.
- **Add New Metrics**: Use `monitor_metric` to report new metrics.
- **Customize Compliance**: Override `sanitize_context`, `encrypt_data`, or `verify_integrity` as needed.

---

## Workflow

1. Instantiate the appropriate evaluator.
2. Call `evaluate` for single conversations, or `batch_evaluate`/`async_batch_evaluate` for high-throughput workflows.
3. All actions are logged/audited with sanitized context for compliance.
4. Extend or override compliance hooks as your deployment requires.

---

## File Structure

- `base_evaluator.py`: Core abstract class and compliance logic.
- `*_evaluator.py`: Domain-specific evaluators.
- `test_evaluators.py`: Comprehensive test suite.
- `EVALUATION_API.md`: This documentation.

---

## Contact

For questions or contributions, see the project repository or contact the maintainers.
