#!/usr/bin/env python3
"""
Training Configuration Profiles
Maps stage configs and dataset profiles into concrete training data selections.
Ensures default/prod profiles do not silently include edge/red-team profiles.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set, Iterable, Union
from enum import Enum
from pathlib import Path

from ..configs.stages import (
    StageConfig,
    get_stage_config,
    get_all_stages,
    STAGE1_ID,
    STAGE2_ID,
    STAGE3_ID,
    STAGE4_ID,
)
from ..types.edge_categories import EdgeCategory, EdgeProfile
from ..utils.logger import get_logger

logger = get_logger("dataset_pipeline.training.config_profiles")


class TrainingProfile(Enum):
    """Named training profiles that map to stages and dataset types"""
    FOUNDATION = "foundation"  # Stage 1: Foundation & Rapport
    REASONING = "reasoning"  # Stage 2: Therapeutic Expertise & Reasoning
    EDGE_CRISIS = "edge_crisis"  # Stage 3: Edge Stress Test & Scenario Bank
    VOICE_PERSONA = "voice_persona"  # Stage 4: Voice, Persona & Delivery
    PRODUCTION = "production"  # General-purpose production training (no edge)
    RESEARCH = "research"  # Research/red-team profile (includes edge)


@dataclass
class ProfileConfig:
    """Configuration for a training profile"""
    profile_name: str
    stage_ids: List[str]  # Which stages to include
    allow_edge_profiles: bool  # Whether edge/red-team datasets are allowed
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# Predefined profile configurations
PROFILE_CONFIGS: Dict[str, ProfileConfig] = {
    TrainingProfile.FOUNDATION.value: ProfileConfig(
        profile_name=TrainingProfile.FOUNDATION.value,
        stage_ids=[STAGE1_ID],
        allow_edge_profiles=False,
        description="Foundation & Rapport training (Stage 1 only, no edge cases)",
    ),
    TrainingProfile.REASONING.value: ProfileConfig(
        profile_name=TrainingProfile.REASONING.value,
        stage_ids=[STAGE2_ID],
        allow_edge_profiles=False,
        description="Therapeutic Expertise & Reasoning training (Stage 2 only, no edge cases)",
    ),
    TrainingProfile.EDGE_CRISIS.value: ProfileConfig(
        profile_name=TrainingProfile.EDGE_CRISIS.value,
        stage_ids=[STAGE3_ID],
        allow_edge_profiles=True,
        description="Edge Stress Test & Scenario Bank (Stage 3, edge cases allowed)",
    ),
    TrainingProfile.VOICE_PERSONA.value: ProfileConfig(
        profile_name=TrainingProfile.VOICE_PERSONA.value,
        stage_ids=[STAGE4_ID],
        allow_edge_profiles=False,
        description="Voice, Persona & Delivery training (Stage 4 only, no edge cases)",
    ),
    TrainingProfile.PRODUCTION.value: ProfileConfig(
        profile_name=TrainingProfile.PRODUCTION.value,
        stage_ids=[STAGE1_ID, STAGE2_ID, STAGE4_ID],  # Explicitly exclude Stage 3
        allow_edge_profiles=False,
        description="General-purpose production training (Stages 1, 2, 4 - no edge cases)",
    ),
    TrainingProfile.RESEARCH.value: ProfileConfig(
        profile_name=TrainingProfile.RESEARCH.value,
        stage_ids=[STAGE1_ID, STAGE2_ID, STAGE3_ID, STAGE4_ID],  # All stages
        allow_edge_profiles=True,
        description="Research/red-team profile (all stages, edge cases allowed)",
    ),
}


class TrainingDataSelector:
    """
    Profile-aware data selector that ensures edge profiles are only used
    in appropriate training configurations.
    """

    def __init__(self, manifest_path: Optional[Union[str, Path]] = None):
        """
        Initialize the training data selector.

        Args:
            manifest_path: Optional path to dataset manifest
        """
        self.manifest_path = Path(manifest_path) if manifest_path else None

    def select_data(
        self,
        profile_name: str,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> Iterable[Dict[str, Any]]:
        """
        Select training data based on profile configuration.

        Args:
            profile_name: Name of the training profile
            manifest: Optional dataset manifest (if None, loads from manifest_path)

        Yields:
            Training examples matching the profile
        """
        # Get profile config
        if profile_name not in PROFILE_CONFIGS:
            raise ValueError(
                f"Unknown profile: {profile_name}. "
                f"Available profiles: {', '.join(PROFILE_CONFIGS.keys())}"
            )

        profile_config = PROFILE_CONFIGS[profile_name]

        logger.info(
            f"Selecting data for profile '{profile_name}': "
            f"stages={profile_config.stage_ids}, "
            f"allow_edge={profile_config.allow_edge_profiles}"
        )

        # Load manifest if not provided
        if manifest is None:
            manifest = self._load_manifest()

        # Select examples based on profile
        for example in self._iterate_examples(manifest):
            # Check stage
            example_stage = example.get("metadata", {}).get("stage")
            if example_stage not in profile_config.stage_ids:
                continue

            # Check edge profile if not allowed
            if not profile_config.allow_edge_profiles:
                if self._is_edge_example(example):
                    logger.warning(
                        f"Skipping edge example in non-edge profile '{profile_name}': "
                        f"{example.get('id', 'unknown')}"
                    )
                    continue

            yield example

    def _is_edge_example(self, example: Dict[str, Any]) -> bool:
        """Check if an example is an edge/red-team example"""
        metadata = example.get("metadata", {})

        # Check for edge profile metadata
        if "edge_profile" in metadata:
            return True

        # Check for edge category
        if "edge_category" in metadata:
            return True

        # Check for stage 3 (edge stress test)
        if metadata.get("stage") == STAGE3_ID:
            return True

        # Check for crisis intensity flags
        if metadata.get("crisis_intensity") in ["very_high", "extreme"]:
            return True

        return False

    def _load_manifest(self) -> Dict[str, Any]:
        """Load dataset manifest"""
        if not self.manifest_path or not self.manifest_path.exists():
            logger.warning(f"Manifest not found at {self.manifest_path}, returning empty manifest")
            return {"examples": []}

        import json
        with open(self.manifest_path, 'r') as f:
            return json.load(f)

    def _iterate_examples(self, manifest: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        """Iterate over examples in manifest"""
        examples = manifest.get("examples", [])
        if not examples:
            # Try alternative manifest structures
            examples = manifest.get("data", [])
            if not examples and isinstance(manifest, list):
                examples = manifest

        for example in examples:
            yield example

    def assert_no_edge_in_profile(
        self,
        profile_name: str,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Assert that a profile does not contain edge examples.
        Raises ValueError if edge examples are found.

        Args:
            profile_name: Name of the profile to check
            manifest: Optional dataset manifest
        """
        if profile_name not in PROFILE_CONFIGS:
            raise ValueError(f"Unknown profile: {profile_name}")

        profile_config = PROFILE_CONFIGS[profile_name]

        if profile_config.allow_edge_profiles:
            logger.info(f"Profile '{profile_name}' allows edge profiles, skipping assertion")
            return

        # Load manifest if not provided
        if manifest is None:
            manifest = self._load_manifest()

        # Check for edge examples
        edge_examples = []
        for example in self._iterate_examples(manifest):
            example_stage = example.get("metadata", {}).get("stage")
            if example_stage in profile_config.stage_ids:
                if self._is_edge_example(example):
                    edge_examples.append(example.get("id", "unknown"))

        if edge_examples:
            raise ValueError(
                f"Profile '{profile_name}' contains {len(edge_examples)} edge examples: "
                f"{edge_examples[:5]}{'...' if len(edge_examples) > 5 else ''}. "
                f"This profile does not allow edge/red-team data."
            )

        logger.info(f"Profile '{profile_name}' validated: no edge examples found")

    def get_profile_stats(
        self,
        profile_name: str,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get statistics for a profile.

        Args:
            profile_name: Name of the profile
            manifest: Optional dataset manifest

        Returns:
            Statistics dictionary
        """
        if profile_name not in PROFILE_CONFIGS:
            raise ValueError(f"Unknown profile: {profile_name}")

        profile_config = PROFILE_CONFIGS[profile_name]

        # Load manifest if not provided
        if manifest is None:
            manifest = self._load_manifest()

        stats = {
            "profile_name": profile_name,
            "stages": profile_config.stage_ids,
            "allow_edge_profiles": profile_config.allow_edge_profiles,
            "total_examples": 0,
            "by_stage": {},
            "edge_examples": 0,
            "non_edge_examples": 0,
        }

        for example in self.select_data(profile_name, manifest):
            stats["total_examples"] += 1

            example_stage = example.get("metadata", {}).get("stage", "unknown")
            stats["by_stage"][example_stage] = stats["by_stage"].get(example_stage, 0) + 1

            if self._is_edge_example(example):
                stats["edge_examples"] += 1
            else:
                stats["non_edge_examples"] += 1

        return stats


def get_profile_config(profile_name: str) -> ProfileConfig:
    """Get configuration for a training profile"""
    if profile_name not in PROFILE_CONFIGS:
        raise ValueError(
            f"Unknown profile: {profile_name}. "
            f"Available: {', '.join(PROFILE_CONFIGS.keys())}"
        )
    return PROFILE_CONFIGS[profile_name]


def list_profiles() -> List[str]:
    """List all available training profiles"""
    return list(PROFILE_CONFIGS.keys())


def validate_profile_config(profile_name: str) -> tuple[bool, Optional[str]]:
    """
    Validate that a profile configuration is correct.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if profile_name not in PROFILE_CONFIGS:
        return False, f"Unknown profile: {profile_name}"

    profile_config = PROFILE_CONFIGS[profile_name]

    # Validate stage IDs
    all_stage_ids = {STAGE1_ID, STAGE2_ID, STAGE3_ID, STAGE4_ID}
    for stage_id in profile_config.stage_ids:
        if stage_id not in all_stage_ids:
            return False, f"Invalid stage ID in profile: {stage_id}"

    # Validate production profile doesn't allow edge
    if profile_name == TrainingProfile.PRODUCTION.value:
        if profile_config.allow_edge_profiles:
            return False, "Production profile must not allow edge profiles"
        if STAGE3_ID in profile_config.stage_ids:
            return False, "Production profile must not include Stage 3 (edge stress test)"

    return True, None

