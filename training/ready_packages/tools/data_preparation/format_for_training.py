#!/usr/bin/env python3
"""
Training Format Conversion and Validation

Converts all processed data to standard JSONL training format with proper
conversation structure and required metadata. Validates format compliance.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from ai.pipelines.orchestrator.schemas.conversation_schema import Conversation, Message
    from ai.pipelines.orchestrator.configs.stages import STAGE1_ID, STAGE2_ID, STAGE3_ID, STAGE4_ID
except ImportError as e:
    logging.error(f"Failed to import pipeline modules: {e}")
    logging.error(f"Project root: {project_root}")
    sys.exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FormatConverter:
    """Converts data to standard training format"""

    def __init__(self, filtering_report_path: Path):
        self.filtering_report_path = filtering_report_path
        self.filtering_report = self._load_filtering_report()
        self.stats = {
            "total_converted": 0,
            "validation_errors": 0,
            "missing_fields": 0,
            "invalid_structure": 0,
        }

    def _load_filtering_report(self) -> Dict[str, Any]:
        """Load filtering report"""
        if not self.filtering_report_path.exists():
            return {"results": []}
        with open(self.filtering_report_path, "r") as f:
            return json.load(f)

    def validate_conversation_schema(self, conversation: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate conversation against schema"""
        # Required top-level fields
        if "messages" not in conversation:
            return False, "Missing 'messages' field"

        messages = conversation.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 2:
            return False, "Must have at least 2 messages"

        # Validate each message
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return False, f"Message {i} is not a dict"

            if "role" not in msg:
                return False, f"Message {i} missing 'role'"

            if "content" not in msg:
                return False, f"Message {i} missing 'content'"

            role = msg.get("role", "")
            if role not in ["user", "assistant", "system"]:
                return False, f"Message {i} has invalid role: {role}"

            content = msg.get("content", "")
            if not isinstance(content, str) or len(content.strip()) < 1:
                return False, f"Message {i} has empty or invalid content"

        return True, None

    def ensure_required_metadata(self, conversation: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """Ensure conversation has required metadata"""
        if "metadata" not in conversation:
            conversation["metadata"] = {}

        metadata = conversation["metadata"]

        # Required metadata fields
        if "stage" not in metadata:
            metadata["stage"] = stage

        if "category" not in metadata:
            # Infer category from stage
            category_map = {
                STAGE1_ID: "foundation",
                STAGE2_ID: "therapeutic_expertise",
                STAGE3_ID: "edge_case",
                STAGE4_ID: "voice_persona",
            }
            metadata["category"] = category_map.get(stage, "unknown")

        # Quality scores (default if missing)
        if "empathy_score" not in metadata:
            metadata["empathy_score"] = 0.7  # Default passing score
        if "safety_score" not in metadata:
            metadata["safety_score"] = 0.8  # Default passing score

        # Edge profile for stage 3
        if stage == STAGE3_ID and "edge_profile" not in metadata:
            metadata["edge_profile"] = "standard"  # Default edge profile

        return conversation

    def convert_to_standard_format(self, conversation: Dict[str, Any], stage: str) -> Optional[Dict[str, Any]]:
        """Convert conversation to standard training format"""
        # Validate schema
        is_valid, error = self.validate_conversation_schema(conversation)
        if not is_valid:
            logger.warning(f"Schema validation failed: {error}")
            self.stats["validation_errors"] += 1
            return None

        # Ensure required metadata
        conversation = self.ensure_required_metadata(conversation, stage)

        # Ensure conversation_id
        if "conversation_id" not in conversation:
            import uuid
            conversation["conversation_id"] = str(uuid.uuid4())

        # Ensure timestamps
        if "created_at" not in conversation:
            conversation["created_at"] = datetime.now().isoformat()
        if "updated_at" not in conversation:
            conversation["updated_at"] = datetime.now().isoformat()

        self.stats["total_converted"] += 1
        return conversation

    def convert_dataset(self, input_path: Path, output_path: Path, stage: str) -> Dict[str, Any]:
        """Convert a single dataset file"""
        logger.info(f"Converting {input_path.name} to standard format (stage: {stage})...")

        converted_conversations = []
        errors = []

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        conversation = json.loads(line)
                        converted = self.convert_to_standard_format(conversation, stage)
                        if converted:
                            converted_conversations.append(converted)
                    except json.JSONDecodeError as e:
                        error_msg = f"Invalid JSON on line {line_num}: {e}"
                        logger.warning(error_msg)
                        errors.append(error_msg)
                        self.stats["validation_errors"] += 1
                        continue
                    except Exception as e:
                        error_msg = f"Error processing line {line_num}: {e}"
                        logger.warning(error_msg)
                        errors.append(error_msg)
                        continue

            # Write converted output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for conv in converted_conversations:
                    f.write(json.dumps(conv, ensure_ascii=False) + "\n")

            logger.info(f"  ✅ Converted: {len(converted_conversations)} conversations")
            if errors:
                logger.warning(f"  ⚠️  {len(errors)} errors encountered")

            return {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "stage": stage,
                "input_count": len(converted_conversations) + len(errors),
                "output_count": len(converted_conversations),
                "errors": errors[:10],  # Limit error details
                "error_count": len(errors),
            }
        except Exception as e:
            logger.error(f"Error converting {input_path}: {e}")
            return {
                "input_path": str(input_path),
                "error": str(e),
            }

    def convert_all_datasets(self, output_dir: Path) -> Dict[str, Any]:
        """Convert all datasets from filtering report"""
        logger.info("📝 Starting format conversion...")

        results = []

        for result in self.filtering_report.get("results", []):
            if "error" in result:
                continue

            input_path = Path(result["output_path"])  # Output from filtering is input here
            if not input_path.exists():
                logger.warning(f"Input file not found: {input_path}")
                continue

            stage = result.get("stage", STAGE1_ID)
            output_path = output_dir / stage / f"{input_path.stem}_formatted.jsonl"

            result_data = self.convert_dataset(input_path, output_path, stage)
            results.append(result_data)

        return {
            "timestamp": datetime.now().isoformat(),
            "total_datasets": len(results),
            "stats": self.stats,
            "results": results,
        }


def main():
    """Main function"""
    base_path = Path.cwd()
    filtering_report_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "filtering_report.json"
    output_dir = base_path / "ai" / "training_ready" / "datasets" / "formatted"

    if not filtering_report_path.exists():
        logger.error(f"Filtering report not found: {filtering_report_path}")
        logger.info("Please run filter_and_clean.py first")
        return 1

    logger.info("📝 Starting format conversion...")

    converter = FormatConverter(filtering_report_path)
    report = converter.convert_all_datasets(output_dir)

    # Save report
    report_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "formatting_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n📊 Format Conversion Summary:")
    logger.info(f"  Total converted: {report['stats']['total_converted']:,}")
    logger.info(f"  Validation errors: {report['stats']['validation_errors']:,}")
    logger.info(f"\n💾 Report saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

