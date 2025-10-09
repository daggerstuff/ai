"""
Test suite for ai.safety.bias_privacy_hooks

Covers:
- Bias detection (positive, negative, edge)
- Privacy compliance (PII, PHI, audit)
- Audit logging (success, error)
- Error handling (exceptions, invalid input)
- Integration with pipeline hooks

London School TDD: All dependencies are mocked/stubbed.
"""

import pytest
from unittest import mock
from typing import Any

# TDD Anchor: Import the module under test (will fail if not implemented)
import ai.safety.bias_privacy_hooks as hooks

@pytest.fixture
def dummy_text():
    return "The patient is calm and cooperative."

@pytest.fixture
def dummy_pii_text():
    return "Patient John Doe, SSN: 123-45-6789, is present."

@pytest.fixture
def dummy_bias_text():
    return "He is aggressive because of his background."

@pytest.fixture
def mock_audit_logger():
    return mock.Mock()

@pytest.fixture
def mock_bias_engine():
    engine = mock.Mock()
    engine.analyze.return_value = {"gender": 0.1, "race": 0.05}
    return engine

@pytest.fixture
def mock_privacy_engine():
    engine = mock.Mock()
    engine.check_pii.return_value = []
    engine.check_phi.return_value = []
    return engine

def test_bias_detection_positive(dummy_text, mock_bias_engine):
    """Given neutral text, When bias detection runs, Then no bias is detected."""
    result = hooks.detect_bias(dummy_text, bias_engine=mock_bias_engine)
    assert isinstance(result, dict)
    assert all(score < 0.2 for score in result.values())

def test_bias_detection_negative(dummy_bias_text, mock_bias_engine):
    """Given biased text, When bias detection runs, Then bias is detected above threshold."""
    mock_bias_engine.analyze.return_value = {"gender": 0.8, "race": 0.05}
    result = hooks.detect_bias(dummy_bias_text, bias_engine=mock_bias_engine)
    assert result["gender"] > 0.5

def test_privacy_check_no_pii(dummy_text, mock_privacy_engine):
    """Given text with no PII, When privacy check runs, Then no PII is found."""
    pii = hooks.check_pii(dummy_text, privacy_engine=mock_privacy_engine)
    assert pii == []

def test_privacy_check_with_pii(dummy_pii_text, mock_privacy_engine):
    """Given text with PII, When privacy check runs, Then PII is detected."""
    mock_privacy_engine.check_pii.return_value = ["John Doe", "123-45-6789"]
    pii = hooks.check_pii(dummy_pii_text, privacy_engine=mock_privacy_engine)
    assert "John Doe" in pii
    assert "123-45-6789" in pii

def test_audit_logging_success(dummy_text, mock_audit_logger):
    """Given valid input, When audit logging runs, Then log entry is created."""
    hooks.log_audit_event("bias_check", dummy_text, logger=mock_audit_logger)
    mock_audit_logger.log_event.assert_called_once()
    args, kwargs = mock_audit_logger.log_event.call_args
    assert args[0] == "bias_check"
    assert dummy_text in args

def test_audit_logging_error(dummy_text, mock_audit_logger):
    """Given logger that raises, When audit logging runs, Then error is handled gracefully."""
    mock_audit_logger.log_event.side_effect = Exception("Logging failed")
    # Should not raise
    hooks.log_audit_event("privacy_check", dummy_text, logger=mock_audit_logger)

def test_integration_bias_privacy(dummy_bias_text, mock_bias_engine, mock_privacy_engine):
    """Given biased text with PII, When full pipeline runs, Then both bias and PII are detected."""
    mock_bias_engine.analyze.return_value = {"gender": 0.7}
    mock_privacy_engine.check_pii.return_value = ["Jane Smith"]
    result = hooks.run_bias_privacy_pipeline(
        dummy_bias_text,
        bias_engine=mock_bias_engine,
        privacy_engine=mock_privacy_engine,
    )
    assert result["bias"]["gender"] > 0.5
    assert "Jane Smith" in result["pii"]

def test_error_on_invalid_input():
    """Given invalid input, When pipeline runs, Then ValueError is raised."""
    with pytest.raises(ValueError):
        hooks.run_bias_privacy_pipeline(None)

def test_privacy_check_phi(dummy_text, mock_privacy_engine):
    """Given text, When PHI check runs, Then PHI list is returned."""
    mock_privacy_engine.check_phi.return_value = ["Diagnosis: Depression"]
    phi = hooks.check_phi(dummy_text, privacy_engine=mock_privacy_engine)
    assert "Diagnosis: Depression" in phi

def test_pipeline_audit_logging(dummy_text, mock_audit_logger, mock_bias_engine, mock_privacy_engine):
    """Given valid input, When pipeline runs with audit logger, Then audit log is written."""
    hooks.run_bias_privacy_pipeline(
        dummy_text,
        bias_engine=mock_bias_engine,
        privacy_engine=mock_privacy_engine,
        audit_logger=mock_audit_logger,
    )
    mock_audit_logger.log_event.assert_called()