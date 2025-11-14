"""
Tier 3 Chain-of-Thought Reasoning Dataset Loader

Loads CoT reasoning datasets for advanced therapeutic reasoning patterns.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from conversation_schema import Conversation

from ai.dataset_pipeline.ingestion.tier_loaders.base_tier_loader import (
    BaseTierLoader,
)

logger = logging.getLogger(__name__)


class Tier3CoTLoader(BaseTierLoader):
    """
    Loader for Tier 3: Chain-of-Thought Reasoning Datasets.
    
    Tier 3 datasets are:
    - Advanced therapeutic reasoning patterns
    - Chain-of-thought reasoning examples
    - Quality threshold: 90%
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        quality_threshold: float = 0.90,
    ):
        """
        Initialize Tier 3 CoT loader.
        
        Args:
            base_path: Optional base path to datasets directory
            quality_threshold: Quality threshold for Tier 3 (default: 0.90 = 90%)
        """
        super().__init__(tier=3, quality_threshold=quality_threshold, base_path=base_path)
        self.base_path = Path(base_path) if base_path else Path("ai/datasets")
        
        # Known Tier 3 CoT datasets
        self.dataset_paths = {
            "cot_clinical_diagnosis": self.base_path / "CoT_Reasoning_Clinical_Diagnosis_Mental_Health",
            "cot_neurodivergent": self.base_path / "CoT_Neurodivergent_vs_Neurotypical_Interactions",
            "cot_heartbreak": self.base_path / "CoT_Heartbreak_and_Breakups",
            "cot_mens_mental_health": self.base_path / "CoT_Reasoning_Mens_Mental_Health",
            "cot_legal": self.base_path / "CoT_Legal_Issues_And_Laws",
            "cot_philosophical": self.base_path / "CoT_Philosophical_Understanding",
            "cot_rare_diseases": self.base_path / "CoT_Rare-Diseases_And_Health-Conditions",
            "cot_temporal": self.base_path / "CoT_Temporal_Reasoning_Dataset",
            "cot_scientific": self.base_path / "CoT_Reasoning_Scientific_Discovery_and_Research",
            "cot_cultural": self.base_path / "CoT-Reasoning_Cultural_Nuances",
        }
        
        logger.info(
            f"Initialized Tier3CoTLoader: quality_threshold={quality_threshold}"
        )

    def load_datasets(self) -> Dict[str, List[Conversation]]:
        """
        Load all Tier 3 CoT reasoning datasets.
        
        Returns:
            Dictionary mapping dataset name to list of conversations
        """
        datasets = {}
        
        for dataset_name, dataset_path in self.dataset_paths.items():
            if not dataset_path.exists():
                logger.warning(f"Tier 3 dataset not found: {dataset_path}")
                continue
            
            logger.info(f"Loading Tier 3 CoT dataset: {dataset_name}")
            
            try:
                conversations = self._load_dataset_directory(dataset_path, dataset_name)
                
                # Add tier metadata with CoT-specific info
                self.add_tier_metadata(conversations, {
                    "dataset_name": dataset_name,
                    "source": f"tier3_cot_{dataset_name}",
                    "reasoning_type": "chain_of_thought",
                })
                
                datasets[dataset_name] = conversations
                logger.info(
                    f"Loaded {len(conversations)} conversations from {dataset_name}"
                )
                
            except Exception as e:
                logger.error(
                    f"Error loading Tier 3 dataset {dataset_name}: {e}",
                    exc_info=True,
                )
                continue
        
        total_conversations = sum(len(convs) for convs in datasets.values())
        logger.info(
            f"Tier 3 loading complete: {len(datasets)} datasets, "
            f"{total_conversations} total conversations"
        )
        
        return datasets

    def _load_dataset_directory(
        self, dataset_path: Path, dataset_name: str
    ) -> List[Conversation]:
        """
        Load conversations from a CoT dataset directory.
        
        Args:
            dataset_path: Path to dataset directory
            dataset_name: Name of the dataset
        
        Returns:
            List of Conversation objects
        """
        conversations = []
        
        # Look for JSONL files
        jsonl_files = list(dataset_path.rglob("*.jsonl"))
        if not jsonl_files:
            # Try JSON files
            jsonl_files = list(dataset_path.rglob("*.json"))
        
        for jsonl_file in jsonl_files:
            file_conversations = self.load_jsonl_file(jsonl_file)
            conversations.extend(file_conversations)
        
        return conversations


