"""Environment-based configuration for DeepRare diagnostic pipeline.

Provides configurable safety thresholds, feature flags, and runtime
parameters loaded from environment variables with secure defaults.

Usage:
    config = DeepRareConfig.from_env()
    pipeline = RareDiseasePipeline(config=config.to_pipeline_config())
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Configuration Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafetyThresholds:
    """Clinical safety threshold configuration.

    These thresholds control when hypotheses can be eliminated, confirmed,
    or require additional evidence. They are safety-critical and should
    only be lowered after clinical validation.
    """

    # Minimum confidence (0-1) to confirm a diagnosis
    min_confidence_to_confirm: float = 0.85

    # Maximum posterior probability to allow elimination
    # A disease with posterior > this cannot be eliminated
    min_confidence_to_eliminate: float = 0.001

    # Maximum hypotheses eliminated per iteration (prevents over-pruning)
    max_elimination_per_iteration: int = 3

    # Protect life-threatening conditions from premature elimination
    protect_life_threatening: bool = True

    # Minimum evidence count before a hypothesis can be confirmed
    min_evidence_count: int = 2

    # Minimum number of matching symptoms to generate a hypothesis
    min_symptom_match: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_confidence_to_confirm": self.min_confidence_to_confirm,
            "min_confidence_to_eliminate": self.min_confidence_to_eliminate,
            "max_elimination_per_iteration": self.max_elimination_per_iteration,
            "protect_life_threatening": self.protect_life_threatening,
            "min_evidence_count": self.min_evidence_count,
            "min_symptom_match": self.min_symptom_match,
        }


@dataclass(frozen=True)
class FeatureFlags:
    """Feature flags for enabling/disabling pipeline components."""

    enable_safety_gates: bool = True
    enable_audit_trail: bool = True
    enable_red_flag_detection: bool = True
    enable_parallel_agents: bool = False  # Sequential by default for safety
    enable_evaluation: bool = True
    enable_single_agent_baseline: bool = True
    enable_literature_matching: bool = True
    enable_test_interpretation: bool = True
    enable_convergence_check: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_safety_gates": self.enable_safety_gates,
            "enable_audit_trail": self.enable_audit_trail,
            "enable_red_flag_detection": self.enable_red_flag_detection,
            "enable_parallel_agents": self.enable_parallel_agents,
            "enable_evaluation": self.enable_evaluation,
            "enable_single_agent_baseline": self.enable_single_agent_baseline,
            "enable_literature_matching": self.enable_literature_matching,
            "enable_test_interpretation": self.enable_test_interpretation,
            "enable_convergence_check": self.enable_convergence_check,
        }


@dataclass(frozen=True)
class PerformanceConfig:
    """Performance and runtime configuration."""

    max_iterations: int = 10
    convergence_window: int = 3
    pruning_threshold: float = 0.01
    timeout_seconds: float = 60.0
    agent_timeout_seconds: float = 15.0
    max_hypotheses: int = 20

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "convergence_window": self.convergence_window,
            "pruning_threshold": self.pruning_threshold,
            "timeout_seconds": self.timeout_seconds,
            "agent_timeout_seconds": self.agent_timeout_seconds,
            "max_hypotheses": self.max_hypotheses,
        }


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration for observability."""

    log_level: str = "INFO"
    enable_structured_logging: bool = True
    log_safety_events: bool = True
    log_agent_decisions: bool = True
    log_evidence_accumulation: bool = False  # Verbose
    log_format: str = "json"  # "json" or "text"

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_level": self.log_level,
            "enable_structured_logging": self.enable_structured_logging,
            "log_safety_events": self.log_safety_events,
            "log_agent_decisions": self.log_agent_decisions,
            "log_evidence_accumulation": self.log_evidence_accumulation,
            "log_format": self.log_format,
        }


@dataclass(frozen=True)
class DeepRareConfig:
    """Top-level configuration for the DeepRare pipeline.

    Combines safety thresholds, feature flags, performance, and logging
    configuration. Can be loaded from environment variables or constructed
    directly for testing.
    """

    safety: SafetyThresholds = field(default_factory=SafetyThresholds)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Build metadata
    version: str = "1.0.0"
    environment: str = "production"  # "production", "staging", "development", "testing"

    @classmethod
    def from_env(cls) -> DeepRareConfig:
        """Load configuration from environment variables.

        Environment variables follow the pattern:
            DEEPRARE_SAFETY_MIN_CONFIRM=0.90
            DEEPRARE_SAFETY_MIN_ELIMINATE=0.001
            DEEPRARE_FEATURES_ENABLE_PARALLEL=true
            DEEPRARE_PERF_MAX_ITERATIONS=15
            DEEPRARE_LOG_LEVEL=DEBUG
            DEEPRARE_ENVIRONMENT=staging
        """

        def _get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key)
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes", "on")

        def _get_float(key: str, default: float) -> float:
            val = os.environ.get(key)
            return float(val) if val is not None else default

        def _get_int(key: str, default: int) -> int:
            val = os.environ.get(key)
            return int(val) if val is not None else default

        def _get_str(key: str, default: str) -> str:
            return os.environ.get(key, default)

        safety = SafetyThresholds(
            min_confidence_to_confirm=_get_float("DEEPRARE_SAFETY_MIN_CONFIRM", 0.85),
            min_confidence_to_eliminate=_get_float("DEEPRARE_SAFETY_MIN_ELIMINATE", 0.001),
            max_elimination_per_iteration=_get_int("DEEPRARE_SAFETY_MAX_ELIM_PER_ITER", 3),
            protect_life_threatening=_get_bool("DEEPRARE_SAFETY_PROTECT_LIFETHREAT", True),
            min_evidence_count=_get_int("DEEPRARE_SAFETY_MIN_EVIDENCE", 2),
            min_symptom_match=_get_int("DEEPRARE_SAFETY_MIN_SYMPTOM_MATCH", 1),
        )

        features = FeatureFlags(
            enable_safety_gates=_get_bool("DEEPRARE_FEATURES_SAFETY_GATES", True),
            enable_audit_trail=_get_bool("DEEPRARE_FEATURES_AUDIT_TRAIL", True),
            enable_red_flag_detection=_get_bool("DEEPRARE_FEATURES_RED_FLAG", True),
            enable_parallel_agents=_get_bool("DEEPRARE_FEATURES_PARALLEL_AGENTS", False),
            enable_evaluation=_get_bool("DEEPRARE_FEATURES_EVALUATION", True),
            enable_single_agent_baseline=_get_bool("DEEPRARE_FEATURES_BASELINE", True),
            enable_literature_matching=_get_bool("DEEPRARE_FEATURES_LITERATURE", True),
            enable_test_interpretation=_get_bool("DEEPRARE_FEATURES_TEST_INTERP", True),
            enable_convergence_check=_get_bool("DEEPRARE_FEATURES_CONVERGENCE", True),
        )

        performance = PerformanceConfig(
            max_iterations=_get_int("DEEPRARE_PERF_MAX_ITERATIONS", 10),
            convergence_window=_get_int("DEEPRARE_PERF_CONVERGENCE_WINDOW", 3),
            pruning_threshold=_get_float("DEEPRARE_PERF_PRUNING_THRESHOLD", 0.01),
            timeout_seconds=_get_float("DEEPRARE_PERF_TIMEOUT_SECONDS", 60.0),
            agent_timeout_seconds=_get_float("DEEPRARE_PERF_AGENT_TIMEOUT", 15.0),
            max_hypotheses=_get_int("DEEPRARE_PERF_MAX_HYPOTHESES", 20),
        )

        logging_cfg = LoggingConfig(
            log_level=_get_str("DEEPRARE_LOG_LEVEL", "INFO"),
            enable_structured_logging=_get_bool("DEEPRARE_LOG_STRUCTURED", True),
            log_safety_events=_get_bool("DEEPRARE_LOG_SAFETY_EVENTS", True),
            log_agent_decisions=_get_bool("DEEPRARE_LOG_AGENT_DECISIONS", True),
            log_evidence_accumulation=_get_bool("DEEPRARE_LOG_EVIDENCE", False),
            log_format=_get_str("DEEPRARE_LOG_FORMAT", "json"),
        )

        environment = _get_str("DEEPRARE_ENVIRONMENT", "production")

        return cls(
            safety=safety,
            features=features,
            performance=performance,
            logging=logging_cfg,
            environment=environment,
        )

    @classmethod
    def for_testing(cls) -> DeepRareConfig:
        """Create a configuration optimized for testing."""
        return cls(
            safety=SafetyThresholds(
                min_confidence_to_confirm=0.5,  # Lower for test speed
                min_confidence_to_eliminate=0.01,
                max_elimination_per_iteration=5,
                min_evidence_count=1,
                min_symptom_match=1,
            ),
            features=FeatureFlags(
                enable_parallel_agents=False,
                enable_evaluation=True,
                enable_single_agent_baseline=True,
            ),
            performance=PerformanceConfig(
                max_iterations=5,
                convergence_window=2,
                pruning_threshold=0.05,
                timeout_seconds=5.0,
                agent_timeout_seconds=3.0,
                max_hypotheses=10,
            ),
            logging=LoggingConfig(
                log_level="DEBUG",
                log_evidence_accumulation=True,
            ),
            environment="testing",
        )

    @classmethod
    def for_development(cls) -> DeepRareConfig:
        """Create a configuration for development with relaxed safety."""
        return cls(
            safety=SafetyThresholds(
                min_confidence_to_confirm=0.7,
            ),
            features=FeatureFlags(
                enable_parallel_agents=True,
            ),
            performance=PerformanceConfig(
                max_iterations=15,
                timeout_seconds=120.0,
            ),
            logging=LoggingConfig(
                log_level="DEBUG",
                log_evidence_accumulation=True,
            ),
            environment="development",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "environment": self.environment,
            "safety": self.safety.to_dict(),
            "features": self.features.to_dict(),
            "performance": self.performance.to_dict(),
            "logging": self.logging.to_dict(),
        }

    def validate(self) -> list[str]:
        """Validate configuration values. Returns list of error messages."""
        errors: list[str] = []

        if not 0.0 < self.safety.min_confidence_to_confirm <= 1.0:
            errors.append("min_confidence_to_confirm must be in (0, 1]")
        if not 0.0 <= self.safety.min_confidence_to_eliminate < 1.0:
            errors.append("min_confidence_to_eliminate must be in [0, 1)")
        if self.safety.max_elimination_per_iteration < 1:
            errors.append("max_elimination_per_iteration must be >= 1")
        if self.performance.max_iterations < 1:
            errors.append("max_iterations must be >= 1")
        if self.performance.convergence_window < 1:
            errors.append("convergence_window must be >= 1")
        if not 0.0 < self.performance.pruning_threshold < 1.0:
            errors.append("pruning_threshold must be in (0, 1)")
        if self.performance.timeout_seconds <= 0:
            errors.append("timeout_seconds must be > 0")

        # Safety cross-checks
        if self.safety.min_confidence_to_confirm <= self.safety.min_confidence_to_eliminate:
            errors.append("min_confidence_to_confirm must be > min_confidence_to_eliminate")

        return errors
