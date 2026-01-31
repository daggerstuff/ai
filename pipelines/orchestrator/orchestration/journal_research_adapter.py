"""
Journal Research Adapter

Adapter layer between journal dataset research system and training pipeline orchestrator.
Converts journal research datasets to pipeline format and integrates them into the training pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from ai.pipelines.orchestrator.storage_config import get_dataset_pipeline_output_root

from ai.sourcing.journal.integration.pipeline_integration_service import (
    PipelineIntegrationService,
)
from ai.sourcing.journal.models.dataset_models import (
    AcquiredDataset,
    IntegrationPlan,
)

# Import conversation schema - adjust path based on where this file is located
import sys
from pathlib import Path

# Add parent directory to path to find conversation_schema
adapter_path = Path(__file__).parent
pipeline_root = adapter_path.parent.parent
sys.path.insert(0, str(pipeline_root))

try:
    from schemas.conversation_schema import Conversation, Message
except ImportError:
    # Fallback: try relative import
    try:
        from ai.pipelines.orchestrator.schemas.conversation_schema import Conversation, Message
    except ImportError:
        # Last resort: try direct import
        from conversation_schema import Conversation, Message

logger = logging.getLogger(__name__)


class JournalResearchAdapter:
    """
    Adapter for integrating journal research datasets into the training pipeline.

    This adapter:
    1. Takes AcquiredDataset and IntegrationPlan from journal research system
    2. Uses PipelineIntegrationService to convert to training format
    3. Loads converted conversations into pipeline orchestrator format
    4. Provides progress tracking for integration operations
    """

    def __init__(
        self,
        integration_service: Optional[PipelineIntegrationService] = None,
        output_directory: Path = get_dataset_pipeline_output_root()
        / "processed"
        / "journal_research",
    ):
        """
        Initialize the journal research adapter.

        Args:
            integration_service: PipelineIntegrationService instance (creates new if None)
            output_directory: Directory for converted datasets
        """
        self.integration_service = (
            integration_service or PipelineIntegrationService()
        )
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Track integration progress
        self.integration_progress: dict[str, dict[str, Any]] = {}

        logger.info(
            f"Initialized JournalResearchAdapter with output directory: {output_directory}"
        )

    def integrate_dataset(
        self,
        dataset: AcquiredDataset,
        integration_plan: IntegrationPlan,
        existing_dataset_path: Optional[Path] = None,
        target_format: str = "chatml",
    ) -> dict[str, Any]:
        """
        Integrate a journal research dataset into the training pipeline.

        Args:
            dataset: Acquired dataset from journal research system
            integration_plan: Integration plan for the dataset
            existing_dataset_path: Path to existing dataset for merging (optional)
            target_format: Target format ("chatml" or "conversation_record")

        Returns:
            Dictionary with integration results and converted dataset path
        """
        logger.info(f"Integrating journal research dataset: {dataset.source_id}")

        # Initialize progress tracking
        self.integration_progress[dataset.source_id] = {
            "status": "in_progress",
            "source_id": dataset.source_id,
            "conversion": None,
            "validation": None,
            "merge": None,
            "quality_check": None,
            "conversations_loaded": 0,
            "errors": [],
        }

        try:
            # Determine output path
            output_filename = f"{dataset.source_id}_integrated.jsonl"
            output_path = self.output_directory / output_filename

            # Convert existing_dataset_path to string if provided
            existing_path_str = (
                str(existing_dataset_path) if existing_dataset_path else None
            )

            # Call integration service
            integration_result = self.integration_service.integrate_dataset(
                dataset=dataset,
                integration_plan=integration_plan,
                existing_dataset_path=existing_path_str,
                output_path=str(output_path),
                target_format=target_format,
                validate=True,
                merge=existing_path_str is not None,
                quality_check=True,
            )

            # Update progress tracking
            self.integration_progress[dataset.source_id].update({
                "status": "completed" if integration_result.get("success") else "failed",
                "conversion": integration_result.get("conversion"),
                "validation": integration_result.get("validation"),
                "merge": integration_result.get("merge"),
                "quality_check": integration_result.get("quality_check"),
                "output_path": str(output_path),
            })

            if integration_result.get("success"):
                # Load conversations from converted dataset
                conversations = self._load_conversations_from_file(
                    output_path, target_format
                )
                self.integration_progress[dataset.source_id]["conversations_loaded"] = (
                    len(conversations)
                )

                logger.info(
                    f"Successfully integrated dataset {dataset.source_id}: "
                    f"{len(conversations)} conversations loaded"
                )
            else:
                error_msg = integration_result.get("error", "Unknown error")
                self.integration_progress[dataset.source_id]["errors"].append(error_msg)
                logger.error(f"Integration failed for {dataset.source_id}: {error_msg}")

            return {
                **integration_result,
                "conversations": conversations if integration_result.get("success") else [],
                "output_path": str(output_path),
            }

        except Exception as e:
            logger.error(f"Error integrating dataset {dataset.source_id}: {e}", exc_info=True)
            self.integration_progress[dataset.source_id].update({
                "status": "failed",
                "errors": [str(e)],
            })
            raise

    def _load_conversations_from_file(
        self, file_path: Path, target_format: str
    ) -> list[Conversation]:
        """
        Load conversations from converted dataset file.

        Args:
            file_path: Path to converted dataset file
            target_format: Format of the file ("chatml" or "conversation_record")

        Returns:
            List of Conversation objects
        """
        conversations = []

        if not file_path.exists():
            logger.warning(f"Converted dataset file not found: {file_path}")
            return conversations

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)

                        # Convert based on target format
                        if target_format == "chatml":
                            conversation = self._convert_chatml_to_conversation(data)
                        elif target_format == "conversation_record":
                            conversation = self._convert_conversation_record_to_conversation(data)
                        else:
                            logger.warning(
                                f"Unknown target format: {target_format}, "
                                f"attempting generic conversion"
                            )
                            conversation = self._convert_generic_to_conversation(data)

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

            logger.info(f"Loaded {len(conversations)} conversations from {file_path}")

        except Exception as e:
            logger.error(f"Error loading conversations from {file_path}: {e}", exc_info=True)
            raise

        return conversations

    def _convert_chatml_to_conversation(self, data: dict[str, Any]) -> Optional[Conversation]:
        """Convert ChatML format to Conversation."""
        try:
            messages = []

            # ChatML format typically has "messages" array with role/content
            if "messages" in data:
                for msg_data in data["messages"]:
                    role = msg_data.get("role", "user")
                    content = msg_data.get("content", "")
                    if content:
                        messages.append(Message(role=role, content=content))

            # Alternative: direct role/content at top level
            elif "role" in data and "content" in data:
                messages.append(
                    Message(role=data["role"], content=data["content"])
                )

            if not messages:
                return None

            conversation = Conversation(
                conversation_id=data.get("id", data.get("conversation_id", "")),
                source=data.get("source", "journal_research"),
                messages=messages,
                metadata={
                    "format": "chatml",
                    "source_id": data.get("source_id"),
                    **data.get("metadata", {}),
                },
            )

            return conversation

        except Exception as e:
            logger.warning(f"Error converting ChatML to Conversation: {e}")
            return None

    def _convert_conversation_record_to_conversation(
        self, data: dict[str, Any]
    ) -> Optional[Conversation]:
        """Convert ConversationRecord format to Conversation."""
        try:
            messages = []

            # ConversationRecord format
            if "conversation" in data:
                conv_data = data["conversation"]
                if "messages" in conv_data:
                    for msg_data in conv_data["messages"]:
                        role = msg_data.get("role", "user")
                        content = msg_data.get("content", "")
                        if content:
                            messages.append(Message(role=role, content=content))

            if not messages:
                return None

            conversation = Conversation(
                conversation_id=data.get("conversation_id", ""),
                source=data.get("source", "journal_research"),
                messages=messages,
                metadata={
                    "format": "conversation_record",
                    "source_id": data.get("source_id"),
                    **data.get("metadata", {}),
                },
            )

            return conversation

        except Exception as e:
            logger.warning(f"Error converting ConversationRecord to Conversation: {e}")
            return None

    def _convert_generic_to_conversation(self, data: dict[str, Any]) -> Optional[Conversation]:
        """Generic conversion attempt for unknown formats."""
        try:
            messages = []

            # Try to find messages in various possible locations
            if "messages" in data:
                msg_list = data["messages"]
            elif "conversation" in data and "messages" in data["conversation"]:
                msg_list = data["conversation"]["messages"]
            elif "dialogue" in data:
                msg_list = data["dialogue"]
            else:
                # Try to construct from available fields
                if "user" in data and "assistant" in data:
                    messages.append(Message(role="user", content=str(data["user"])))
                    messages.append(Message(role="assistant", content=str(data["assistant"])))
                elif "question" in data and "answer" in data:
                    messages.append(Message(role="user", content=str(data["question"])))
                    messages.append(Message(role="assistant", content=str(data["answer"])))
                else:
                    return None

                conversation = Conversation(
                    conversation_id=data.get("id", data.get("conversation_id", "")),
                    source=data.get("source", "journal_research"),
                    messages=messages,
                    metadata={
                        "format": "generic",
                        "source_id": data.get("source_id"),
                        **data.get("metadata", {}),
                    },
                )
                return conversation

            # Process message list
            for msg_data in msg_list:
                if isinstance(msg_data, dict):
                    role = msg_data.get("role", msg_data.get("speaker", "user"))
                    content = msg_data.get("content", msg_data.get("text", ""))
                    if content:
                        messages.append(Message(role=role, content=content))
                elif isinstance(msg_data, str):
                    # Assume alternating user/assistant if just strings
                    role = "user" if len(messages) % 2 == 0 else "assistant"
                    messages.append(Message(role=role, content=msg_data))

            if not messages:
                return None

            conversation = Conversation(
                conversation_id=data.get("id", data.get("conversation_id", "")),
                source=data.get("source", "journal_research"),
                messages=messages,
                metadata={
                    "format": "generic",
                    "source_id": data.get("source_id"),
                    **data.get("metadata", {}),
                },
            )

            return conversation

        except Exception as e:
            logger.warning(f"Error in generic conversion: {e}")
            return None

    def get_integration_status(self, source_id: str) -> Optional[dict[str, Any]]:
        """
        Get integration status for a dataset.

        Args:
            source_id: Source ID of the dataset

        Returns:
            Integration status dictionary or None if not found
        """
        return self.integration_progress.get(source_id)

    def list_integrated_datasets(self) -> list[str]:
        """
        List all integrated dataset source IDs.

        Returns:
            List of source IDs
        """
        return list(self.integration_progress.keys())

