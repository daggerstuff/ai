"""Tests for DeepRareConfig environment-based configuration."""

from __future__ import annotations

import os

import pytest

from ai.pkg_mera.platform.deep_rare.config import (
    DeepRareConfig,
    FeatureFlags,
    LoggingConfig,
    PerformanceConfig,
    SafetyThresholds,
)


class TestSafetyThresholds:
    def test_defaults(self):
        s = SafetyThresholds()
        assert s.min_confidence_to_confirm == 0.85
        assert s.min_confidence_to_eliminate == 0.001
        assert s.max_elimination_per_iteration == 3
        assert s.protect_life_threatening is True
        assert s.min_evidence_count == 2
        assert s.min_symptom_match == 1

    def test_frozen(self):
        s = SafetyThresholds()
        with pytest.raises(AttributeError):
            setattr(s, "min_confidence_to_confirm", 0.5)


class TestFeatureFlags:
    def test_defaults(self):
        f = FeatureFlags()
        assert f.enable_safety_gates is True
        assert f.enable_audit_trail is True
        assert f.enable_red_flag_detection is True
        assert f.enable_parallel_agents is False
        assert f.enable_evaluation is True


class TestPerformanceConfig:
    def test_defaults(self):
        p = PerformanceConfig()
        assert p.max_iterations == 10
        assert p.convergence_window == 3
        assert p.pruning_threshold == 0.01
        assert p.timeout_seconds == 60.0
        assert p.agent_timeout_seconds == 15.0
        assert p.max_hypotheses == 20


class TestDeepRareConfig:
    def test_for_testing(self):
        config = DeepRareConfig.for_testing()
        assert config.environment == "testing"
        assert config.safety.min_confidence_to_confirm == 0.5
        assert config.performance.max_iterations == 5
        assert config.performance.timeout_seconds == 5.0

    def test_for_development(self):
        config = DeepRareConfig.for_development()
        assert config.environment == "development"
        assert config.features.enable_safety_gates is True

    def test_from_env_defaults(self, monkeypatch):
        for key in list(os.environ.keys()):
            if key.startswith("DEEPRARE_"):
                monkeypatch.delenv(key, raising=False)
        config = DeepRareConfig.from_env()
        assert config.environment == "production"

    def test_from_env_overrides(self, monkeypatch):
        monkeypatch.setenv("DEEPRARE_ENVIRONMENT", "staging")
        config = DeepRareConfig.from_env()
        assert config.environment == "staging"

    def test_to_dict(self):
        config = DeepRareConfig.for_testing()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "safety" in d
        assert "features" in d
        assert "performance" in d
        assert "logging" in d

    def test_validate_valid(self):
        config = DeepRareConfig.for_testing()
        issues = config.validate()
        assert isinstance(issues, list)

    def test_logging_config_defaults(self):
        lc = LoggingConfig()
        assert lc.log_level == "INFO"
        assert lc.enable_structured_logging is True
        assert lc.log_safety_events is True
        assert lc.log_format == "json"
