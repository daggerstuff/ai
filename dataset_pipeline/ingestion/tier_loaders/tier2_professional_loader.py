"""
Tier 2 Professional Therapeutic Dataset Loader

Loads professional therapeutic datasets (Psych8k, mental_health_counseling, etc.)
with clinical-grade quality validation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from conversation_schema import Conversation

from ai.dataset_pipeline.ingestion.tier_loaders.base_tier_loader import (
    BaseTierLoader,
)

logger = logging.getLogger(__name__)


class Tier2ProfessionalLoader(BaseTierLoader):
    """
    Loader for Tier 2: Professional Therapeutic Datasets.
    
    Tier 2 datasets are:
    - Clinical-grade conversation data
    - Professional therapy sessions
    - Quality threshold: 95%
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        quality_threshold: float = 0.95,
    ):
        """
        Initialize Tier 2 professional loader.
        
        Args:
            base_path: Optional base path to datasets directory
            quality_threshold: Quality threshold for Tier 2 (default: 0.95 = 95%)
        """
        super().__init__(tier=2, quality_threshold=quality_threshold, base_path=base_path)
        self.base_path = Path(base_path) if base_path else Path("ai/datasets")
        
        # Known Tier 2 datasets
        self.dataset_paths = {
            "psych8k": self.base_path / "Psych8k",
            "mental_health_counseling": self.base_path / "mental_health_counseling_conversations",
            "soulchat": self.base_path / "SoulChat2.0",
            "counsel_chat": self.base_path / "counsel-chat",
            "llama3_mental": self.base_path / "LLAMA3_Mental_Counseling_Data",
            "therapist_sft": self.base_path / "therapist-sft-format",
            "neuro_qa": self.base_path / "neuro_qa_SFT_Trainer",
        }
        
        logger.info(
            f"Initialized Tier2ProfessionalLoader: quality_threshold={quality_threshold}"
        )

    def load_datasets(self) -> Dict[str, List[Conversation]]:
        """
        Load all Tier 2 professional datasets.
        
        Returns:
            Dictionary mapping dataset name to list of conversations
        """
        datasets = {}
        
        for dataset_name, dataset_path in self.dataset_paths.items():
            if not dataset_path.exists():
                logger.warning(f"Tier 2 dataset not found: {dataset_path}")
                continue
            
            logger.info(f"Loading Tier 2 dataset: {dataset_name}")
            
            try:
                conversations = self._load_dataset_directory(dataset_path, dataset_name)
                
                # Add tier metadata
                self.add_tier_metadata(conversations, {
                    "dataset_name": dataset_name,
                    "source": f"tier2_{dataset_name}",
                })
                
                datasets[dataset_name] = conversations
                logger.info(
                    f"Loaded {len(conversations)} conversations from {dataset_name}"
                )
                
            except Exception as e:
                logger.error(
                    f"Error loading Tier 2 dataset {dataset_name}: {e}",
                    exc_info=True,
                )
                continue
        
        total_conversations = sum(len(convs) for convs in datasets.values())
        logger.info(
            f"Tier 2 loading complete: {len(datasets)} datasets, "
            f"{total_conversations} total conversations"
        )
        
        return datasets

    def _load_dataset_directory(
        self, dataset_path: Path, dataset_name: str
    ) -> List[Conversation]:
        """
        Load conversations from a dataset directory.
        
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


