"""
Base Tier Loader

Base class for tier-specific dataset loaders with common functionality.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from conversation_schema import Conversation, Message

logger = logging.getLogger(__name__)


class BaseTierLoader(ABC):
    """
    Base class for tier-specific dataset loaders.
    
    Provides common functionality:
    - JSONL file loading
    - Conversation conversion
    - Tier metadata management
    - Quality threshold handling
    """

    def __init__(
        self,
        tier: int,
        quality_threshold: float,
        base_path: Optional[Path] = None,
    ):
        """
        Initialize base tier loader.
        
        Args:
            tier: Tier number (1-6)
            quality_threshold: Quality threshold for this tier (0.0-1.0)
            base_path: Optional base path for datasets
        """
        self.tier = tier
        self.quality_threshold = quality_threshold
        self.base_path = Path(base_path) if base_path else None
        
        logger.info(
            f"Initialized Tier{tier}Loader: quality_threshold={quality_threshold}"
        )

    @abstractmethod
    def load_datasets(self) -> Dict[str, List[Conversation]]:
        """
        Load datasets for this tier.
        
        Returns:
            Dictionary mapping dataset name to list of conversations
        """
        pass

    def load_jsonl_file(self, file_path: Path) -> List[Conversation]:
        """
        Load conversations from a JSONL file.
        
        Args:
            file_path: Path to JSONL file
        
        Returns:
            List of Conversation objects
        """
        conversations = []
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return conversations
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        conversation = self._convert_to_conversation(data)
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
            logger.error(f"Error loading JSONL file {file_path}: {e}", exc_info=True)
            raise
        
        return conversations

    def _convert_to_conversation(
        self, data: Dict[str, Any], source_name: Optional[str] = None
    ) -> Optional[Conversation]:
        """
        Convert data dictionary to Conversation object.
        
        Args:
            data: Data dictionary from JSONL file
            source_name: Optional source name for metadata
        
        Returns:
            Conversation object or None if conversion fails
        """
        try:
            messages = []
            
            # Try various message formats
            if "messages" in data:
                for msg_data in data["messages"]:
                    role = msg_data.get("role", "user")
                    content = msg_data.get("content", "")
                    if content:
                        messages.append(Message(role=role, content=content))
            
            elif "conversation" in data:
                conv_data = data["conversation"]
                if isinstance(conv_data, list):
                    for msg_data in conv_data:
                        if isinstance(msg_data, dict):
                            role = msg_data.get("role", msg_data.get("speaker", "user"))
                            content = msg_data.get("content", msg_data.get("text", ""))
                            if content:
                                messages.append(Message(role=role, content=content))
            
            elif "user" in data and "assistant" in data:
                messages.append(Message(role="user", content=str(data["user"])))
                messages.append(Message(role="assistant", content=str(data["assistant"])))
            
            elif "question" in data and "answer" in data:
                messages.append(Message(role="user", content=str(data["question"])))
                messages.append(Message(role="assistant", content=str(data["answer"])))
            
            if not messages:
                return None
            
            source = source_name or data.get("source", f"tier{self.tier}")
            
            conversation = Conversation(
                conversation_id=data.get("id", data.get("conversation_id", "")),
                source=source,
                messages=messages,
                metadata={
                    "tier": self.tier,
                    "quality_threshold": self.quality_threshold,
                    "original_data": {
                        k: v
                        for k, v in data.items()
                        if k
                        not in [
                            "messages",
                            "conversation",
                            "user",
                            "assistant",
                            "question",
                            "answer",
                        ]
                    },
                },
            )
            
            return conversation
            
        except Exception as e:
            logger.warning(f"Error converting data to conversation: {e}")
            return None

    def add_tier_metadata(
        self, conversations: List[Conversation], additional_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add tier metadata to conversations.
        
        Args:
            conversations: List of conversations to update
            additional_metadata: Optional additional metadata to add
        """
        for conv in conversations:
            conv.metadata.update({
                "tier": self.tier,
                "quality_threshold": self.quality_threshold,
            })
            if additional_metadata:
                conv.metadata.update(additional_metadata)

    def get_quality_threshold(self) -> float:
        """Get quality threshold for this tier."""
        return self.quality_threshold

    def get_tier(self) -> int:
        """Get tier number."""
        return self.tier


