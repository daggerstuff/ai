import contextlib
from typing import Any


def detect_bias(text: str, bias_engine: Any) -> dict[str, float]:
    """Detect bias using the provided bias_engine."""
    if bias_engine is None:
        return {}
    return bias_engine.analyze(text)


def check_pii(text: str, privacy_engine: Any) -> list[str]:
    """Check for PII using the provided privacy_engine."""
    if privacy_engine is None:
        return []
    return privacy_engine.check_pii(text)


def check_phi(text: str, privacy_engine: Any) -> list[str]:
    """Check for PHI using the provided privacy_engine."""
    if privacy_engine is None:
        return []
    return privacy_engine.check_phi(text)


def log_audit_event(event_type: str, text: str, logger: Any) -> None:
    """Log an audit event using the provided logger."""
    if logger is not None:
        with contextlib.suppress(Exception):
            logger.log_event(event_type, text)


def run_bias_privacy_pipeline(
    text: str,
    bias_engine: Any | None = None,
    privacy_engine: Any | None = None,
    audit_logger: Any | None = None,
) -> dict[str, Any]:
    """Run bias and privacy checks, and log audit event."""
    if text is None:
        raise ValueError("Input text cannot be None")
    bias = detect_bias(text, bias_engine) if bias_engine else {}
    pii = check_pii(text, privacy_engine) if privacy_engine else []
    phi = check_phi(text, privacy_engine) if privacy_engine else []
    if audit_logger is not None:
        with contextlib.suppress(Exception):
            audit_logger.log_event("pipeline_run", text)
    return {"bias": bias, "pii": pii, "phi": phi}
