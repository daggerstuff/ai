"""
Evaluation Score Filter

Filters datasets based on evaluation scores from journal research system.
Uses evaluation scores to prioritize and filter datasets before training.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class EvaluationScoreFilter:
    """
    Filter datasets based on evaluation scores.
    
    Uses evaluation scores from journal research system to:
    1. Filter datasets by quality thresholds
    2. Prioritize high-scoring datasets
    3. Map evaluation scores to quality thresholds
    """

    def __init__(
        self,
        min_overall_score: float = 7.0,
        min_therapeutic_relevance: int = 7,
        min_data_structure_quality: int = 6,
        min_training_integration: int = 6,
        min_ethical_accessibility: int = 7,
        priority_threshold: float = 8.5,
    ):
        """
        Initialize evaluation score filter.
        
        Args:
            min_overall_score: Minimum overall score (0-10) to include dataset
            min_therapeutic_relevance: Minimum therapeutic relevance score (1-10)
            min_data_structure_quality: Minimum data structure quality score (1-10)
            min_training_integration: Minimum training integration score (1-10)
            min_ethical_accessibility: Minimum ethical accessibility score (1-10)
            priority_threshold: Overall score threshold for high priority (0-10)
        """
        self.min_overall_score = min_overall_score
        self.min_therapeutic_relevance = min_therapeutic_relevance
        self.min_data_structure_quality = min_data_structure_quality
        self.min_training_integration = min_training_integration
        self.min_ethical_accessibility = min_ethical_accessibility
        self.priority_threshold = priority_threshold
        
        logger.info(
            f"Initialized EvaluationScoreFilter: "
            f"min_overall={min_overall_score}, "
            f"priority_threshold={priority_threshold}"
        )

    def should_include_dataset(
        self, evaluation_score: Optional[float], evaluation_details: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, str]:
        """
        Determine if dataset should be included based on evaluation score.
        
        Args:
            evaluation_score: Overall evaluation score (0-10) or None
            evaluation_details: Optional detailed evaluation scores dictionary with keys:
                - therapeutic_relevance (int 1-10)
                - data_structure_quality (int 1-10)
                - training_integration (int 1-10)
                - ethical_accessibility (int 1-10)
        
        Returns:
            Tuple of (should_include: bool, reason: str)
        """
        # If no evaluation score, include by default (let other filters handle it)
        if evaluation_score is None:
            return True, "No evaluation score available, including by default"
        
        # Check overall score threshold
        if evaluation_score < self.min_overall_score:
            return False, (
                f"Overall score {evaluation_score:.2f} below minimum "
                f"{self.min_overall_score:.2f}"
            )
        
        # Check detailed scores if provided
        if evaluation_details:
            therapeutic = evaluation_details.get("therapeutic_relevance")
            if therapeutic is not None and therapeutic < self.min_therapeutic_relevance:
                return False, (
                    f"Therapeutic relevance {therapeutic} below minimum "
                    f"{self.min_therapeutic_relevance}"
                )
            
            data_structure = evaluation_details.get("data_structure_quality")
            if data_structure is not None and data_structure < self.min_data_structure_quality:
                return False, (
                    f"Data structure quality {data_structure} below minimum "
                    f"{self.min_data_structure_quality}"
                )
            
            training_integration = evaluation_details.get("training_integration")
            if training_integration is not None and training_integration < self.min_training_integration:
                return False, (
                    f"Training integration {training_integration} below minimum "
                    f"{self.min_training_integration}"
                )
            
            ethical = evaluation_details.get("ethical_accessibility")
            if ethical is not None and ethical < self.min_ethical_accessibility:
                return False, (
                    f"Ethical accessibility {ethical} below minimum "
                    f"{self.min_ethical_accessibility}"
                )
        
        return True, f"Evaluation score {evaluation_score:.2f} meets all thresholds"

    def get_priority(self, evaluation_score: Optional[float]) -> str:
        """
        Get priority level based on evaluation score.
        
        Args:
            evaluation_score: Overall evaluation score (0-10) or None
        
        Returns:
            Priority level: "high", "medium", "low", or "unknown"
        """
        if evaluation_score is None:
            return "unknown"
        
        if evaluation_score >= self.priority_threshold:
            return "high"
        elif evaluation_score >= self.min_overall_score:
            return "medium"
        else:
            return "low"

    def map_to_quality_threshold(self, evaluation_score: Optional[float]) -> float:
        """
        Map evaluation score to pipeline quality threshold.
        
        Maps evaluation score (0-10) to quality threshold (0-1) for pipeline.
        
        Args:
            evaluation_score: Overall evaluation score (0-10) or None
        
        Returns:
            Quality threshold (0-1) for pipeline
        """
        if evaluation_score is None:
            # Default quality threshold if no evaluation
            return 0.7
        
        # Map 0-10 scale to 0-1 scale
        # Score 7.0 -> 0.7, Score 10.0 -> 1.0, Score 0.0 -> 0.0
        quality_threshold = evaluation_score / 10.0
        
        # Ensure minimum quality threshold
        quality_threshold = max(quality_threshold, 0.5)
        
        return quality_threshold

    def filter_datasets(
        self,
        datasets: list[Dict[str, Any]],
        evaluation_scores: Optional[Dict[str, float]] = None,
        evaluation_details: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        """
        Filter datasets based on evaluation scores.
        
        Args:
            datasets: List of dataset dictionaries with 'source_id' or 'id' key
            evaluation_scores: Optional dict mapping source_id to overall score
            evaluation_details: Optional dict mapping source_id to detailed scores
        
        Returns:
            Tuple of (included_datasets, excluded_datasets)
        """
        included = []
        excluded = []
        
        for dataset in datasets:
            source_id = dataset.get("source_id") or dataset.get("id", "")
            if not source_id:
                logger.warning("Dataset missing source_id, including by default")
                included.append(dataset)
                continue
            
            # Get evaluation score
            score = None
            if evaluation_scores:
                score = evaluation_scores.get(source_id)
            
            # Get evaluation details
            details = None
            if evaluation_details:
                details = evaluation_details.get(source_id)
            
            # Check if should include
            should_include, reason = self.should_include_dataset(score, details)
            
            if should_include:
                # Add priority and quality threshold to dataset metadata
                priority = self.get_priority(score)
                quality_threshold = self.map_to_quality_threshold(score)
                
                dataset_metadata = dataset.get("metadata", {})
                dataset_metadata.update({
                    "evaluation_score": score,
                    "evaluation_priority": priority,
                    "quality_threshold": quality_threshold,
                    "evaluation_details": details,
                })
                dataset["metadata"] = dataset_metadata
                
                included.append(dataset)
                logger.debug(f"Including dataset {source_id}: {reason}")
            else:
                excluded.append(dataset)
                logger.debug(f"Excluding dataset {source_id}: {reason}")
        
        logger.info(
            f"Filtered {len(datasets)} datasets: "
            f"{len(included)} included, {len(excluded)} excluded"
        )
        
        return included, excluded


