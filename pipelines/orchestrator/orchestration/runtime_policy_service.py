"""Runtime policy and operational freshness helpers for dataset assembly flows."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from ai.pipelines.orchestrator.orchestration.integration_config_resolver import (
    IntegrationConfigResolver,
)


class RuntimeConfigProtocol(Protocol):
    asana_project_gid: str | None
    asana_section_gid: str | None
    asana_dataset_section_gid: str | None
    asana_parent_task_gid: str | None
    enable_asana_sync: bool
    enable_beads_sync: bool
    enable_jira_sync: bool
    enable_linear_sync: bool
    tracker_sync_state_output_path: str
    stage_distribution: dict[str, float]


class WarningSinkProtocol(Protocol):
    warnings: list[str]


@dataclass(frozen=True)
class StagePolicyBundle:
    stage_distribution: dict[str, float]
    quality_profiles: dict[str, dict[str, Any]]
    drift_waivers: dict[str, dict[str, Any]]


class RuntimePolicyService:
    """Own manifest-backed policy loading, env hydration, and ops freshness checks."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        warning_sink: WarningSinkProtocol,
        default_stage_distribution: dict[str, float],
        default_stage_drift_tolerance: float,
        integration_config_resolver: IntegrationConfigResolver | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.warning_sink = warning_sink
        self.default_stage_distribution = dict(default_stage_distribution)
        self.default_stage_drift_tolerance = default_stage_drift_tolerance
        self.integration_config_resolver = (
            integration_config_resolver or IntegrationConfigResolver()
        )

    def hydrate_tracker_config(self, config: RuntimeConfigProtocol) -> None:
        """Load tracker-related runtime config from environment when unset."""
        config.asana_project_gid = self._resolve_gid(
            current_value=config.asana_project_gid,
            env_names=("ASANA_PROJECT_GID",),
            resolver=self.integration_config_resolver.resolve_training_asana_project_gid,
            fallback_env_names=("ASANA_PROJECT_ID",),
        )
        config.asana_section_gid = self._resolve_gid(
            current_value=config.asana_section_gid,
            env_names=("ASANA_SECTION_GID",),
        )
        config.asana_dataset_section_gid = self._resolve_gid(
            current_value=config.asana_dataset_section_gid,
            env_names=("ASANA_DATASET_SECTION_GID",),
        )
        config.asana_parent_task_gid = self._resolve_gid(
            current_value=config.asana_parent_task_gid,
            env_names=("ASANA_PARENT_TASK_GID",),
        )

        config.enable_asana_sync = self._env_flag(
            "ENABLE_ASANA_SYNC", config.enable_asana_sync
        )
        config.enable_beads_sync = self._env_flag(
            "ENABLE_BEADS_SYNC", config.enable_beads_sync
        )
        config.enable_jira_sync = self._env_flag(
            "ENABLE_JIRA_SYNC", config.enable_jira_sync
        )
        config.enable_linear_sync = self._env_flag(
            "ENABLE_LINEAR_SYNC", config.enable_linear_sync
        )

        tracker_sync_state_path = os.getenv("TRACKER_SYNC_STATE_PATH")
        if tracker_sync_state_path:
            config.tracker_sync_state_output_path = tracker_sync_state_path

    def load_stage_policy(self) -> StagePolicyBundle:
        """Load stage distribution, quality profiles, and drift waivers."""
        if not self.manifest_path.exists():
            return StagePolicyBundle(
                stage_distribution=dict(self.default_stage_distribution),
                quality_profiles={},
                drift_waivers={},
            )

        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._warn(f"Failed to load stage policy manifest: {exc}")
            return StagePolicyBundle(
                stage_distribution=dict(self.default_stage_distribution),
                quality_profiles={},
                drift_waivers={},
            )

        manifest_stages = manifest.get("stages", {})
        stage_distribution = self._load_stage_distribution(manifest_stages)
        quality_profiles = self._load_stage_quality_profiles(manifest_stages)
        drift_waivers = self._load_stage_drift_waivers(
            manifest.get("stage_drift_waivers", {})
        )

        return StagePolicyBundle(
            stage_distribution=stage_distribution,
            quality_profiles=quality_profiles,
            drift_waivers=drift_waivers,
        )

    def resolve_stage_drift_tolerance(
        self,
        stage: str,
        *,
        drift_waivers: dict[str, dict[str, Any]],
    ) -> tuple[float, bool]:
        """Resolve stage drift tolerance, applying valid waivers when available."""
        waiver = drift_waivers.get(stage, {})
        if not isinstance(waiver, dict):
            return self.default_stage_drift_tolerance, False

        max_drift = waiver.get("max_drift")
        if not isinstance(max_drift, (int, float)):
            return self.default_stage_drift_tolerance, False

        expires_at = self._parse_iso_timestamp(waiver.get("expires_at"))
        if expires_at is not None and datetime.now(timezone.utc) > expires_at:
            self._warn(
                "Drift waiver for stage "
                f"'{stage}' is expired at {expires_at.isoformat()}"
            )
            return self.default_stage_drift_tolerance, False

        return float(max_drift), True

    def collect_ops_freshness(self) -> dict[str, Any]:
        """Collect freshness status for inventory, prompt mirror, and voice exports."""
        threshold_hours = self._resolve_ops_freshness_threshold_hours()
        now = datetime.now(timezone.utc)
        targets = self._resolve_ops_freshness_targets()

        checks: dict[str, Any] = {}
        all_fresh = True
        for key, path in targets.items():
            check = self._build_ops_freshness_check(
                path=path,
                checked_at=now,
                threshold_hours=threshold_hours,
            )
            checks[key] = check
            if not check["fresh"]:
                all_fresh = False

        return {
            "checked_at": now.isoformat(),
            "threshold_hours": threshold_hours,
            "all_fresh": all_fresh,
            "checks": checks,
        }

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _resolve_ops_freshness_threshold_hours() -> float:
        threshold_hours_raw = os.getenv("TRAINING_OPS_FRESHNESS_HOURS", "24").strip()
        try:
            return float(threshold_hours_raw)
        except ValueError:
            return 24.0

    @staticmethod
    def _resolve_ops_freshness_targets() -> dict[str, Path]:
        return {
            "inventory": Path(
                os.getenv(
                    "TRAINING_OPS_INVENTORY_PATH",
                    "ai/training_data_consolidated/final/MASTER_STAGE_MANIFEST.json",
                )
            ),
            "prompt_mirror": Path(
                os.getenv(
                    "TRAINING_OPS_PROMPT_MIRROR_PATH",
                    "ai/pipelines/orchestrator/prompt_corpus",
                )
            ),
            "voice_export": Path(
                os.getenv(
                    "TRAINING_OPS_VOICE_EXPORT_PATH",
                    "ai/training_data_consolidated/transcripts",
                )
            ),
        }

    @staticmethod
    def _build_ops_freshness_check(
        *,
        path: Path,
        checked_at: datetime,
        threshold_hours: float,
    ) -> dict[str, Any]:
        if not path.exists():
            return {
                "path": str(path),
                "exists": False,
                "fresh": False,
                "reason": "missing",
            }

        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        age_hours = (checked_at - modified_at).total_seconds() / 3600.0
        fresh = age_hours <= threshold_hours
        return {
            "path": str(path),
            "exists": True,
            "fresh": fresh,
            "modified_at": modified_at.isoformat(),
            "age_hours": round(age_hours, 3),
            "threshold_hours": threshold_hours,
        }

    def _resolve_gid(
        self,
        *,
        current_value: str | None,
        env_names: tuple[str, ...],
        resolver: Callable[[], str | None] | None = None,
        fallback_env_names: tuple[str, ...] = (),
    ) -> str | None:
        normalized_current = self._normalize_gid(current_value)
        if normalized_current is not None:
            return normalized_current

        for env_name in env_names:
            normalized_env = self._normalize_gid(os.getenv(env_name))
            if normalized_env is not None:
                return normalized_env

        if resolver is None:
            return self._resolve_gid_from_env(fallback_env_names)

        normalized_resolved = self._normalize_gid(resolver())
        if normalized_resolved is not None:
            return normalized_resolved
        return self._resolve_gid_from_env(fallback_env_names)

    def _resolve_gid_from_env(self, env_names: tuple[str, ...]) -> str | None:
        for env_name in env_names:
            normalized_env = self._normalize_gid(os.getenv(env_name))
            if normalized_env is not None:
                return normalized_env
        return None

    @staticmethod
    def _normalize_gid(value: str | None) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().strip("'\"")
        if not normalized or not normalized.isdigit():
            return None
        return normalized

    def _load_stage_distribution(
        self, manifest_stages: Any
    ) -> dict[str, float]:
        if not isinstance(manifest_stages, dict):
            return dict(self.default_stage_distribution)

        stage_distribution: dict[str, float] = {}
        for stage_id, stage_config in manifest_stages.items():
            if not isinstance(stage_config, dict):
                continue
            target = stage_config.get("target_percentage")
            if isinstance(target, (int, float)) and target > 0:
                stage_distribution[stage_id] = float(target)

        total = sum(stage_distribution.values())
        if stage_distribution and abs(total - 1.0) < 1e-6:
            return stage_distribution
        if stage_distribution:
            self._warn(
                "Ignoring manifest stage distribution "
                f"(sum {total:.4f} != 1.0): {stage_distribution}"
            )
        return dict(self.default_stage_distribution)

    @staticmethod
    def _load_stage_quality_profiles(
        manifest_stages: Any,
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(manifest_stages, dict):
            return {}

        profiles: dict[str, dict[str, Any]] = {}
        for stage_id, stage_config in manifest_stages.items():
            if not isinstance(stage_config, dict):
                continue
            quality_profile = stage_config.get("quality_profile", {})
            if isinstance(quality_profile, dict):
                profiles[stage_id] = quality_profile
        return profiles

    @staticmethod
    def _load_stage_drift_waivers(
        waivers_payload: Any,
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(waivers_payload, dict):
            return {}

        parsed: dict[str, dict[str, Any]] = {}
        for stage, waiver in waivers_payload.items():
            if not isinstance(waiver, dict):
                continue
            max_drift = waiver.get("max_drift")
            if not isinstance(max_drift, (int, float)):
                continue
            parsed[stage] = {
                "max_drift": float(max_drift),
                "reason": waiver.get("reason", ""),
                "approved_by": waiver.get("approved_by", ""),
                "expires_at": waiver.get("expires_at"),
            }
        return parsed

    @staticmethod
    def _parse_iso_timestamp(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        normalized = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _warn(self, message: str) -> None:
        self.warning_sink.warnings.append(message)


__all__ = ["RuntimePolicyService", "StagePolicyBundle"]
