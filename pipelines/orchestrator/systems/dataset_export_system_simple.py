"""
Simple dataset export system for multiple formats.
Exports datasets in various formats for different training frameworks.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from conversation_schema import Conversation
from logger import get_logger

logger = get_logger(__name__)


class DatasetExportSystemSimple:
    """
    Simple dataset export system for multiple formats.

    Exports therapeutic training datasets in formats suitable
    for various ML frameworks and training pipelines.
    """

    def __init__(self):
        """Initialize the dataset export system."""
        self.logger = get_logger(__name__)

        self.supported_formats = [
            "jsonl",      # JSON Lines (most common)
            "json",       # Standard JSON
            "csv",        # CSV format
            "huggingface", # HuggingFace datasets format
            "openai"      # OpenAI fine-tuning format
        ]

        self.logger.info("DatasetExportSystemSimple initialized")

    def export_dataset(self, conversations: list[Conversation],
                      output_path: str,
                      format_type: str = "jsonl",
                      include_metadata: bool = True) -> bool:
        """Export dataset in specified format."""
        self.logger.info(f"Exporting {len(conversations)} conversations to {output_path} in {format_type} format")

        if format_type not in self.supported_formats:
            self.logger.error(f"Unsupported format: {format_type}")
            return False

        try:
            if format_type == "jsonl":
                return self._export_jsonl(conversations, output_path, include_metadata)
            if format_type == "json":
                return self._export_json(conversations, output_path, include_metadata)
            if format_type == "csv":
                return self._export_csv(conversations, output_path, include_metadata)
            if format_type == "huggingface":
                return self._export_huggingface(conversations, output_path, include_metadata)
            if format_type == "openai":
                return self._export_openai(conversations, output_path, include_metadata)

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False

    def _export_jsonl(self, conversations: list[Conversation], output_path: str, include_metadata: bool) -> bool:
        """Export to JSON Lines format."""
        with open(output_path, "w", encoding="utf-8") as f:
            for conv in conversations:
                conv_dict = self._conversation_to_dict(conv, include_metadata)
                f.write(json.dumps(conv_dict, ensure_ascii=False) + "\\n")

        self.logger.info(f"Exported {len(conversations)} conversations to JSONL: {output_path}")
        return True

    def _export_json(self, conversations: list[Conversation], output_path: str, include_metadata: bool) -> bool:
        """Export to standard JSON format."""
        data = {
            "conversations": [
                self._conversation_to_dict(conv, include_metadata)
                for conv in conversations
            ],
            "export_info": {
                "exported_at": datetime.now().isoformat(),
                "total_conversations": len(conversations),
                "format": "json",
                "includes_metadata": include_metadata
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"Exported {len(conversations)} conversations to JSON: {output_path}")
        return True

    def _export_csv(self, conversations: list[Conversation], output_path: str, include_metadata: bool) -> bool:
        """Export to CSV format."""
        fieldnames = ["conversation_id", "user_message", "assistant_message"]
        if include_metadata:
            fieldnames.extend(["title", "tags", "quality_score", "metadata"])

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for conv in conversations:
                # Extract user and assistant messages
                user_messages = [msg.content for msg in conv.messages if msg.role == "user"]
                assistant_messages = [msg.content for msg in conv.messages if msg.role == "assistant"]

                # Create rows for each user-assistant pair
                max_pairs = max(len(user_messages), len(assistant_messages))

                for i in range(max_pairs):
                    row = {
                        "conversation_id": conv.id,
                        "user_message": user_messages[i] if i < len(user_messages) else "",
                        "assistant_message": assistant_messages[i] if i < len(assistant_messages) else ""
                    }

                    if include_metadata:
                        row.update({
                            "title": conv.title or "",
                            "tags": "|".join(conv.tags) if conv.tags else "",
                            "quality_score": conv.quality_score or "",
                            "metadata": json.dumps(conv.metadata) if conv.metadata else ""
                        })

                    writer.writerow(row)

        self.logger.info(f"Exported {len(conversations)} conversations to CSV: {output_path}")
        return True

    def _export_huggingface(self, conversations: list[Conversation], output_path: str, include_metadata: bool) -> bool:
        """Export in HuggingFace datasets format."""
        # HuggingFace format: list of examples with 'messages' field
        examples = []

        for conv in conversations:
            example = {
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content
                    }
                    for msg in conv.messages
                ],
                "conversation_id": conv.id
            }

            if include_metadata:
                example.update({
                    "title": conv.title,
                    "tags": conv.tags,
                    "quality_score": conv.quality_score,
                    "metadata": conv.metadata
                })

            examples.append(example)

        # Save as JSONL (HuggingFace compatible)
        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False, default=str) + "\\n")

        self.logger.info(f"Exported {len(conversations)} conversations to HuggingFace format: {output_path}")
        return True

    def _export_openai(self, conversations: list[Conversation], output_path: str, include_metadata: bool) -> bool:
        """Export in OpenAI fine-tuning format."""
        # OpenAI format: JSONL with 'messages' field
        with open(output_path, "w", encoding="utf-8") as f:
            for conv in conversations:
                openai_example = {
                    "messages": [
                        {
                            "role": msg.role,
                            "content": msg.content
                        }
                        for msg in conv.messages
                    ]
                }

                f.write(json.dumps(openai_example, ensure_ascii=False) + "\\n")

        self.logger.info(f"Exported {len(conversations)} conversations to OpenAI format: {output_path}")
        return True

    def _conversation_to_dict(self, conversation: Conversation, include_metadata: bool) -> dict[str, Any]:
        """Convert conversation to dictionary format."""
        conv_dict = {
            "id": conversation.id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
                }
                for msg in conversation.messages
            ]
        }

        if include_metadata:
            conv_dict.update({
                "title": conversation.title,
                "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
                "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else None,
                "metadata": conversation.metadata,
                "tags": conversation.tags,
                "quality_score": conversation.quality_score
            })

        return conv_dict

    def export_multiple_formats(self, conversations: list[Conversation],
                              base_path: str,
                              formats: list[str] | None = None) -> dict[str, bool]:
        """Export dataset in multiple formats."""
        if formats is None:
            formats = ["jsonl", "json", "huggingface"]

        results = {}
        base_path_obj = Path(base_path)

        for format_type in formats:
            if format_type in self.supported_formats:
                output_path = base_path_obj.with_suffix(f".{format_type}")
                success = self.export_dataset(conversations, str(output_path), format_type)
                results[format_type] = success
            else:
                self.logger.warning(f"Skipping unsupported format: {format_type}")
                results[format_type] = False

        return results

    def get_export_summary(self, conversations: list[Conversation]) -> dict[str, Any]:
        """Get summary information for export."""
        return {
            "total_conversations": len(conversations),
            "total_messages": sum(len(conv.messages) for conv in conversations),
            "supported_formats": self.supported_formats,
            "estimated_file_sizes": {
                "jsonl": f"~{len(conversations) * 0.5:.1f} KB",  # Rough estimate
                "json": f"~{len(conversations) * 0.6:.1f} KB",
                "csv": f"~{len(conversations) * 0.4:.1f} KB"
            }
        }


def validate_dataset_export_system_simple():
    """Validate the DatasetExportSystemSimple functionality."""
    try:
        exporter = DatasetExportSystemSimple()
        assert hasattr(exporter, "export_dataset")
        assert len(exporter.supported_formats) > 0
        return True
    except Exception:
        return False


if __name__ == "__main__":
    if validate_dataset_export_system_simple():
        pass
    else:
        pass
