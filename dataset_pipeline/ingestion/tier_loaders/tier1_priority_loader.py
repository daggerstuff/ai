"""
Tier 1 Priority Dataset Loader

Loads curated priority datasets (priority_1-5_FINAL.jsonl) with highest quality validation.
Tier 1 datasets are production-ready and used as gold standard for quality validation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from conversation_schema import Conversation, Message

from ai.dataset_pipeline.ingestion.tier_loaders.base_tier_loader import (
    BaseTierLoader,
)

logger = logging.getLogger(__name__)


class Tier1PriorityLoader(BaseTierLoader):
    """
    Loader for Tier 1: Curated Priority Datasets.
    
    Tier 1 datasets are:
    - Highest quality, production-ready data
    - Pre-curated and validated
    - Used as gold standard for quality validation
    - Quality threshold: 99%
    """

    def __init__(
        self,
        base_path: Path = Path("ai/datasets/datasets-wendy"),
        quality_threshold: float = 0.99,
    ):
        """
        Initialize Tier 1 priority loader.
        
        Args:
            base_path: Base path to datasets-wendy directory
            quality_threshold: Quality threshold for Tier 1 (default: 0.99 = 99%)
        """
        super().__init__(tier=1, quality_threshold=quality_threshold, base_path=base_path)
        self.base_path = Path(base_path)
        
        logger.info(
            f"Initialized Tier1PriorityLoader: base_path={base_path}, "
            f"quality_threshold={quality_threshold}"
        )

    def load_datasets(
        self, priority_numbers: Optional[List[int]] = None
    ) -> Dict[str, List[Conversation]]:
        """
        Load priority datasets (priority_1-5_FINAL.jsonl).
        
        Args:
            priority_numbers: Optional list of priority numbers to load (1-5).
                             If None, loads all available priority datasets.
        
        Returns:
            Dictionary mapping priority number to list of conversations
        """
        return self.load_priority_datasets(priority_numbers)
    
    def load_priority_datasets(
        self, priority_numbers: Optional[List[int]] = None
    ) -> Dict[str, List[Conversation]]:
        """
        Load priority datasets (priority_1-5_FINAL.jsonl).
        
        Args:
            priority_numbers: Optional list of priority numbers to load (1-5).
                             If None, loads all available priority datasets.
        
        Returns:
            Dictionary mapping priority number to list of conversations
        """
        if priority_numbers is None:
            priority_numbers = [1, 2, 3, 4, 5]
        
        datasets = {}
        
        for priority_num in priority_numbers:
            priority_file = self.base_path / f"priority_{priority_num}_FINAL.jsonl"
            summary_file = self.base_path / f"priority_{priority_num}_summary.json"
            
            if not priority_file.exists():
                logger.warning(
                    f"Priority {priority_num} file not found: {priority_file}"
                )
                continue
            
            logger.info(f"Loading priority_{priority_num}_FINAL.jsonl")
            
            try:
                conversations = self._load_priority_file(priority_file, priority_num)
                
                # Load summary if available
                metadata = {}
                if summary_file.exists():
                    metadata = self._load_summary(summary_file)
                
                # Add tier metadata to all conversations
                self.add_tier_metadata(conversations, {
                    "priority_number": priority_num,
                    "source": f"priority_{priority_num}",
                    **metadata,
                })
                
                datasets[f"priority_{priority_num}"] = conversations
                logger.info(
                    f"Loaded {len(conversations)} conversations from priority_{priority_num}"
                )
                
            except Exception as e:
                logger.error(
                    f"Error loading priority_{priority_num}: {e}",
                    exc_info=True,
                )
                continue
        
        total_conversations = sum(len(convs) for convs in datasets.values())
        logger.info(
            f"Tier 1 loading complete: {len(datasets)} datasets, "
            f"{total_conversations} total conversations"
        )
        
        return datasets

    def _load_priority_file(
        self, file_path: Path, priority_num: int
    ) -> List[Conversation]:
        """
        Load conversations from a priority JSONL file.
        
        Args:
            file_path: Path to priority JSONL file
            priority_num: Priority number for metadata
        
        Returns:
            List of Conversation objects
        """
        conversations = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        conversation = self._convert_to_conversation(data, priority_num)
                        if conversation:
                            conversations.append(conversation)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to parse line {line_num} in {file_path}: {e}"
                        )
                        continue
                    except Exception as e:
                        logger.warning(
                            f"Error converting line {line_num} in {file_path}: {e}"
                        )
                        continue
        
        except Exception as e:
            logger.error(f"Error loading priority file {file_path}: {e}", exc_info=True)
            raise
        
        return conversations

    def _convert_to_conversation(
        self, data: Dict[str, Any], priority_num: Optional[int] = None
    ) -> Optional[Conversation]:
        """
        Convert data dictionary to Conversation object.
        
        Args:
            data: Data dictionary from JSONL file
            priority_num: Optional priority number for metadata
        
        Returns:
            Conversation object or None if conversion fails
        """
        source_name = f"tier1_priority_{priority_num}" if priority_num else "tier1_priority"
        conv = super()._convert_to_conversation(data, source_name)
        
        if conv and priority_num:
            conv.metadata["priority_number"] = priority_num
        
        return conv

    def _load_summary(self, summary_file: Path) -> Dict[str, Any]:
        """
        Load summary.json file for metadata.
        
        Args:
            summary_file: Path to summary.json file
        
        Returns:
            Dictionary with summary metadata
        """
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load summary file {summary_file}: {e}")
            return {}


