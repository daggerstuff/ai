"""
Integration module for NeMo Data Designer with existing Pixelated Empathy systems.

This module provides integration points between the NeMo Data Designer service
and existing systems like bias detection and dataset pipeline.
"""

import logging
from typing import Any, Optional

from ai.pipelines.design.service import NeMoDataDesignerService
from ai.pipelines.design.config import DataDesignerConfig

logger = logging.getLogger(__name__)


class BiasDetectionIntegration:
    """Integration with bias detection system."""

    def __init__(self, designer_service: Optional[NeMoDataDesignerService] = None):
        """
        Initialize bias detection integration.

        Args:
            designer_service: NeMo Data Designer service instance. If None, creates new.
        """
        self.designer_service = designer_service or NeMoDataDesignerService()

    def generate_bias_test_dataset(
        self,
        num_samples: int = 1000,
        protected_attributes: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Generate dataset specifically formatted for bias detection testing.

        Args:
            num_samples: Number of samples to generate
            protected_attributes: List of protected attributes to include

        Returns:
            Dictionary with dataset and metadata formatted for bias detection
        """
        if protected_attributes is None:
            protected_attributes = ["gender", "ethnicity", "age_group"]

        # Generate dataset
        result = self.designer_service.generate_bias_detection_dataset(
            num_samples=num_samples,
            protected_attributes=protected_attributes,
        )

        # Format for bias detection service
        formatted_data = {
            "dataset": result["data"],
            "metadata": {
                "num_samples": result["num_samples"],
                "protected_attributes": result["protected_attributes"],
                "generation_time": result["generation_time"],
                "source": "nemo_data_designer",
            },
        }

        logger.info(
            f"Generated bias test dataset with {num_samples} samples "
            f"and protected attributes: {protected_attributes}",
        )

        return formatted_data

    def generate_fairness_analysis_dataset(
        self,
        num_samples: int = 2000,
        outcome_variables: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Generate dataset for fairness analysis with multiple outcome variables.

        Args:
            num_samples: Number of samples to generate
            outcome_variables: List of outcome variable names

        Returns:
            Dictionary with dataset formatted for fairness analysis
        """
        if outcome_variables is None:
            outcome_variables = [
                "treatment_response",
                "session_attendance_rate",
                "therapist_rating",
            ]

        # Generate dataset with all protected attributes
        result = self.designer_service.generate_bias_detection_dataset(
            num_samples=num_samples,
            protected_attributes=["gender", "ethnicity", "age_group"],
        )

        formatted_data = {
            "dataset": result["data"],
            "metadata": {
                "num_samples": result["num_samples"],
                "protected_attributes": result["protected_attributes"],
                "outcome_variables": outcome_variables,
                "generation_time": result["generation_time"],
                "source": "nemo_data_designer",
                "analysis_type": "fairness",
            },
        }

        logger.info(
            f"Generated fairness analysis dataset with {num_samples} samples "
            f"and {len(outcome_variables)} outcome variables",
        )

        return formatted_data


class DatasetPipelineIntegration:
    """Integration with dataset pipeline system."""

    def __init__(self, designer_service: Optional[NeMoDataDesignerService] = None):
        """
        Initialize dataset pipeline integration.

        Args:
            designer_service: NeMo Data Designer service instance. If None, creates new.
        """
        self.designer_service = designer_service or NeMoDataDesignerService()

    def generate_training_dataset(
        self,
        num_samples: int = 5000,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """
        Generate dataset formatted for training pipeline.

        Args:
            num_samples: Number of samples to generate
            include_metadata: Whether to include metadata columns

        Returns:
            Dictionary with dataset formatted for training pipeline
        """
        # Generate therapeutic dataset
        result = self.designer_service.generate_therapeutic_dataset(
            num_samples=num_samples,
            include_demographics=True,
            include_symptoms=True,
            include_treatments=True,
            include_outcomes=True,
        )

        formatted_data = {
            "data": result["data"],
            "metadata": {
                "num_samples": result["num_samples"],
                "columns": result["columns"],
                "generation_time": result["generation_time"],
                "source": "nemo_data_designer",
                "pipeline_format": "training",
            },
        }

        logger.info(
            f"Generated training dataset with {num_samples} samples "
            f"and {len(result['columns'])} columns",
        )

        return formatted_data

    def augment_existing_dataset(
        self,
        existing_dataset: dict[str, Any],
        augmentation_factor: float = 0.5,
    ) -> dict[str, Any]:
        """
        Augment existing dataset with synthetic data.

        Args:
            existing_dataset: Existing dataset to augment
            augmentation_factor: Factor to determine how many samples to add
                                 (0.5 = add 50% more samples)

        Returns:
            Dictionary with augmented dataset
        """
        # Determine number of samples to generate
        original_count = len(existing_dataset.get("data", []))
        num_new_samples = int(original_count * augmentation_factor)

        logger.info(
            f"Augmenting dataset: {original_count} existing samples, "
            f"generating {num_new_samples} new samples",
        )

        # Generate synthetic data matching the structure of existing dataset
        result = self.designer_service.generate_therapeutic_dataset(
            num_samples=num_new_samples,
        )

        # Combine datasets
        augmented_data = {
            "data": existing_dataset.get("data", []) + result["data"],
            "metadata": {
                "original_samples": original_count,
                "augmented_samples": num_new_samples,
                "total_samples": original_count + num_new_samples,
                "augmentation_factor": augmentation_factor,
                "source": "nemo_data_designer_augmentation",
            },
        }

        logger.info(
            f"Dataset augmentation complete: {augmented_data['metadata']['total_samples']} total samples",
        )

        return augmented_data


class TherapeuticDatasetIntegration:
    """Integration for therapeutic dataset generation."""

    def __init__(self, designer_service: Optional[NeMoDataDesignerService] = None):
        """
        Initialize therapeutic dataset integration.

        Args:
            designer_service: NeMo Data Designer service instance. If None, creates new.
        """
        self.designer_service = designer_service or NeMoDataDesignerService()

    def generate_conversation_dataset(
        self,
        num_conversations: int = 100,
        avg_turns_per_conversation: int = 8,
    ) -> dict[str, Any]:
        """
        Generate dataset for therapeutic conversation training.

        Args:
            num_conversations: Number of conversations to generate
            avg_turns_per_conversation: Average number of turns per conversation

        Returns:
            Dictionary with conversation dataset
        """
        # Generate base therapeutic data
        total_samples = num_conversations * avg_turns_per_conversation
        result = self.designer_service.generate_therapeutic_dataset(
            num_samples=total_samples,
            include_demographics=True,
            include_symptoms=True,
            include_treatments=True,
            include_outcomes=False,  # Outcomes not needed for conversation data
        )

        # Format for conversation dataset
        conversation_data = {
            "conversations": [],
            "metadata": {
                "num_conversations": num_conversations,
                "avg_turns_per_conversation": avg_turns_per_conversation,
                "total_samples": total_samples,
                "source": "nemo_data_designer",
                "dataset_type": "therapeutic_conversations",
            },
        }

        # Group samples into conversations
        samples = result["data"]
        samples_per_conversation = total_samples // num_conversations

        for i in range(num_conversations):
            start_idx = i * samples_per_conversation
            end_idx = start_idx + samples_per_conversation
            conversation_samples = samples[start_idx:end_idx]

            conversation = {
                "conversation_id": f"conv_{i+1:04d}",
                "turns": conversation_samples,
                "metadata": {
                    "num_turns": len(conversation_samples),
                },
            }
            conversation_data["conversations"].append(conversation)

        logger.info(
            f"Generated {num_conversations} conversations with "
            f"{avg_turns_per_conversation} average turns",
        )

        return conversation_data

