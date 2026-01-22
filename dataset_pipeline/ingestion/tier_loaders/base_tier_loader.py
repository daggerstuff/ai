"""
Base Tier Loader

Base class for tier-specific dataset loaders with common functionality.
"""

import json
import logging
import shutil
import subprocess
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
        dataset_registry_path: str = "ai/data/dataset_registry.json",
    ):
        """
        Initialize base tier loader.

        Args:
            tier: Tier number (1-6)
            quality_threshold: Quality threshold for this tier (0.0-1.0)
            base_path: Optional base path for datasets
            dataset_registry_path: Path to dataset_registry.json
        """
        self.tier = tier
        self.quality_threshold = quality_threshold
        self.base_path = Path(base_path) if base_path else None
        self.registry_path = Path(dataset_registry_path)

        # Load registry
        self.registry = self._load_registry()

        logger.info(
            f"Initialized Tier{tier}Loader: quality_threshold={quality_threshold}"
        )

    def _load_registry(self) -> Dict[str, Any]:
        """Load dataset registry."""
        if not self.registry_path.exists():
            logger.warning(f"Dataset registry not found at {self.registry_path}")
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
            return {}

    def _is_ovhai_available(self) -> bool:
        """Check if OVHAI CLI is available."""
        return shutil.which("ovhai") is not None

    def _is_s3_path(self, path_value: str | None) -> bool:
        """Check if a path is an S3 URI."""
        return bool(path_value and str(path_value).startswith("s3://"))

    def _ensure_dataset_locally(
        self,
        dataset_name: str,
        current_path: Path,
        registry_category: Optional[str] = None,
    ) -> Path:
        """
        Ensure dataset exists locally, download from S3 via OVHAI if needed.

        Args:
            dataset_name: Internal or registry name of the dataset
            current_path: Current predicted local path
            registry_category: Optional category in registry (e.g. 'cot_reasoning')

        Returns:
            The path where the dataset is located
        """
        if current_path.exists():
            return current_path

        # If we don't know the category, try all of them
        categories = (
            [registry_category]
            if registry_category
            else [
                "cot_reasoning",
                "professional_therapeutic",
                "wendy_curated_sets",
                "edge_case_sources",
                "supplementary",
                "voice_persona",
            ]
        )

        registry_entry = None
        datasets_in_reg = self.registry.get("datasets", {})

        for cat in categories:
            cat_data = datasets_in_reg.get(cat, {})
            # Also check top level categories
            if not cat_data and cat in self.registry:
                cat_data = self.registry[cat]

            if isinstance(cat_data, dict):
                # Try exact match or case-insensitive match
                registry_entry = cat_data.get(dataset_name)
                if not registry_entry:
                    # Search by lower case
                    for name, entry in cat_data.items():
                        if name.lower() == dataset_name.lower() or name.lower().replace(
                            "-", "_"
                        ) == dataset_name.lower().replace("-", "_"):
                            registry_entry = entry
                            break
            if registry_entry:
                break

        if not registry_entry or not isinstance(registry_entry, dict):
            return current_path

        s3_path = registry_entry.get("path", "")
        if not self._is_s3_path(s3_path):
            return current_path

        if not self._is_ovhai_available():
            logger.warning(
                f"ovhai not found. attempting fallback to boto3 for {dataset_name}..."
            )
            return self._download_with_boto3(s3_path, current_path)

        logger.info(f"Downloading {dataset_name} from S3 (ovhai): {s3_path}")

        # Determine if it's a file or directory
        is_file = (
            s3_path.endswith(".json")
            or s3_path.endswith(".jsonl")
            or s3_path.endswith(".csv")
        )

        target_dir = current_path if not is_file else current_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            # New syntax: ovhai data download <DATA_STORE> <CONTAINER> [OBJECTS]...
            # We assume DATA_STORE is 'GRA' (common default) or verify env
            # However, since we can't easily list datastores without auth, we'll try 'GRA'
            # and fallback if it fails.
            # Also, s3_path includes "s3://pixel-data/...", we need pixel-data and the key.

            parts = s3_path.replace("s3://", "").split("/", 1)
            bucket_name = parts[0]
            key_path = parts[1] if len(parts) > 1 else ""

            # Since ovhai data download requires container and object,
            # and potentially the datastore name (GRA).
            # Command: ovhai data download GRA pixel-data <key_path> --output <local_path>

            # NOTE: If we don't know the datastore, this is risky.
            # But 'pixel-data' is the container.

            # Try to run with GRA as store
            cmd = [
                "ovhai",
                "data",
                "download",
                "GRA",  # Assumed datastore
                bucket_name,
                key_path,
                "--output",
                str(current_path),
            ]

            # Use run() but catch auth errors specifically
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            if result.returncode != 0:
                # Check for "unrecognized subcommand" which implies wrong version/alias
                if "unrecognized subcommand" in result.stderr:
                    logger.warning(
                        "ovhai CLI syntax mismatch (unrecognized subcommand). "
                        "Used 'data download'. Falling back to boto3."
                    )
                else:
                    logger.warning(
                        f"ovhai download failed (code {result.returncode}): "
                        f"{result.stderr.strip()}. "
                        "This may be due to missing auth token or wrong region."
                    )

                # Check for specific 401/Auth error to inform user
                if "401" in result.stderr or "unauthorized" in result.stderr.lower():
                    logger.info("Tip: Ensure OVH_AI_TOKEN is set for ovhai CLI.")

                return self._download_with_boto3(s3_path, current_path)

            logger.info(f"Successfully downloaded {dataset_name} to {current_path}")
            return current_path

        except Exception as e:
            logger.error(f"Error during ovhai execution: {e}")
            return self._download_with_boto3(s3_path, current_path)

    def _download_with_boto3(self, s3_uri: str, local_path: Path) -> Path:
        """Download from S3 using boto3."""
        try:
            import boto3
            from ai.dataset_pipeline.storage_config import (
                get_storage_config,
            )

            config = get_storage_config()

            # Use credentials from config if available (which load from .env)
            s3_client = boto3.client(
                "s3",
                endpoint_url=config.s3_endpoint_url,
                aws_access_key_id=config.s3_access_key_id,
                aws_secret_access_key=config.s3_secret_access_key,
                # Default if missing
                region_name=config.s3_region or "us-east-1",
            )

            # Parse bucket and key from s3://bucket/key...
            parts = s3_uri.replace("s3://", "").split("/", 1)
            bucket_name = parts[0]
            key_path = parts[1] if len(parts) > 1 else ""

            # Check if it looks like a directory download (legacy check)
            # Tier loader logic assumes directories for some, files for others.
            # BaseTierLoader calls this with current_path as the target.

            # Simple single file download
            if not local_path.parent.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Boto3 downloading {key_path} from {bucket_name} to {local_path}"
            )
            s3_client.download_file(bucket_name, key_path, str(local_path))

            return local_path

        except Exception as e:
            logger.error(f"Boto3 download failed for {s3_uri}: {e}")
            return local_path

    def _load_dataset_directory(
        self, dataset_path: Path, dataset_name: str
    ) -> List[Conversation]:
        """
        Load conversations recursively from a dataset directory.

        Args:
            dataset_path: Path to dataset directory
            dataset_name: Name of the dataset for source metadata

        Returns:
            List of Conversation objects
        """
        conversations = []

        if not dataset_path.exists():
            return conversations

        # Look for JSONL and JSON files
        data_files = list(dataset_path.rglob("*.jsonl")) + list(
            dataset_path.rglob("*.json")
        )

        for data_file in data_files:
            try:
                if data_file.suffix == ".jsonl":
                    file_conversations = self.load_jsonl_file(data_file)
                else:
                    file_conversations = self.load_json_file(data_file)
                conversations.extend(file_conversations)
            except Exception as e:
                logger.warning(f"Failed to load file {data_file}: {e}")
                continue

        return conversations

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

    def load_json_file(self, file_path: Path) -> List[Conversation]:
        """
        Load conversations from a standard JSON file (list of objects).

        Args:
            file_path: Path to JSON file

        Returns:
            List of Conversation objects
        """
        conversations = []

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return conversations

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)

                if isinstance(data_list, dict):
                    # Handle single object dataset
                    conversation = self._convert_to_conversation(data_list)
                    if conversation:
                        conversations.append(conversation)
                elif isinstance(data_list, list):
                    for data in data_list:
                        conversation = self._convert_to_conversation(data)
                        if conversation:
                            conversations.append(conversation)
                else:
                    logger.warning(
                        f"Unexpected JSON format in {file_path}: expected list or dict"
                    )

        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}", exc_info=True)
            raise

        return conversations

    def _format_instruction(self, data: Dict[str, Any]) -> str:
        """Format Alpaca-style instruction and input."""
        instruction = str(data.get("instruction", ""))
        input_text = str(data.get("input", ""))
        if input_text:
            return f"{instruction}\n\n{input_text}"
        return instruction

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
                messages.append(
                    Message(role="assistant", content=str(data["assistant"]))
                )

            elif "question" in data and "answer" in data:
                messages.append(Message(role="user", content=str(data["question"])))
                messages.append(Message(role="assistant", content=str(data["answer"])))

            elif "instruction" in data and "response" in data:
                # Alpaca format variant 1
                messages.append(
                    Message(role="user", content=self._format_instruction(data))
                )
                messages.append(
                    Message(role="assistant", content=str(data["response"]))
                )

            elif "instruction" in data and "output" in data:
                # Alpaca format variant 2
                messages.append(
                    Message(role="user", content=self._format_instruction(data))
                )
                messages.append(Message(role="assistant", content=str(data["output"])))

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
        self,
        conversations: List[Conversation],
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add tier metadata to conversations.

        Args:
            conversations: List of conversations to update
            additional_metadata: Optional additional metadata to add
        """
        for conv in conversations:
            conv.metadata.update(
                {
                    "tier": self.tier,
                    "quality_threshold": self.quality_threshold,
                }
            )
            if additional_metadata:
                conv.metadata.update(additional_metadata)

    def get_quality_threshold(self) -> float:
        """Get quality threshold for this tier."""
        return self.quality_threshold

    def get_tier(self) -> int:
        """Get tier number."""
        return self.tier

    def load_sample(self, max_conversations: int = 100) -> List[Conversation]:
        """
        Load a sample of conversations for testing.
        Attempts to stream from S3 if available, otherwise falls back to local.

        Args:
            max_conversations: Maximum number of conversations to return

        Returns:
            List of Conversation objects
        """
        # Try to find a dataset to sample from
        sample_s3_path = self._get_sample_s3_key()

        if sample_s3_path:
            try:
                logger.info(f"Attempting to stream sample from S3: {sample_s3_path}")
                return self._stream_s3_sample(sample_s3_path, max_conversations)
            except Exception as e:
                logger.warning(f"S3 streaming failed, falling back to full load: {e}")

        # Fallback to loading all (which triggers download)
        sample = []
        all_datasets = self.load_datasets()

        for dataset_convs in all_datasets.values():
            sample.extend(dataset_convs[:max_conversations])
            if len(sample) >= max_conversations:
                break

        return sample[:max_conversations]

    def _get_sample_s3_key(self) -> Optional[str]:
        """
        Get a single S3 key to sample from.
        Can be overridden by subclasses or derived from registry.
        """
        # Dictionary of sample keys for known tiers
        # This prevents us from having to refactor every subclass immediately
        # These paths assume the standard bucket layout
        sample_keys = {
            1: "datasets/consolidated/datasets/priority_1_FINAL.jsonl",
            2: "datasets/consolidated/datasets/priority_2_FINAL.jsonl",
            3: "datasets/consolidated/datasets/priority_3_FINAL.jsonl",
            4: "datasets/consolidated/datasets/reddit_mental_health_filtered.json",
            5: "datasets/consolidated/datasets/research_datasets_filtered.json",
            6: "datasets/consolidated/datasets/Psychology-6K.json",
        }

        key_suffix = sample_keys.get(self.tier)
        if not key_suffix:
            return None

        # Try to match with registry if possible, otherwise use hardcoded guess
        return key_suffix

    def _stream_s3_sample(self, s3_key: str, limit: int) -> List[Conversation]:
        """Stream first N lines/rows from S3 object."""
        try:
            import csv
            import io

            import boto3
            from ai.dataset_pipeline.storage_config import (
                StorageBackend,
                get_storage_config,
            )
            from botocore.exceptions import ClientError
        except ImportError:
            logger.warning("boto3 not installed, cannot stream")
            raise

        config = get_storage_config()
        if config.backend != StorageBackend.S3 or not config.s3_bucket:
            raise ValueError("S3 storage not configured")

        s3 = boto3.client(
            "s3",
            endpoint_url=config.s3_endpoint_url,
            aws_access_key_id=config.s3_access_key_id,
            aws_secret_access_key=config.s3_secret_access_key,
            region_name=config.s3_region,
        )

        conversations = []
        try:
            # Range request for first 10MB should be enough for 100 samples
            # Note: CSVs need careful line handling if cutting byte stream
            resp = s3.get_object(
                Bucket=config.s3_bucket,
                Key=s3_key,
                Range="bytes=0-10485760",  # 10MB
            )

            content = resp["Body"].read().decode("utf-8", errors="ignore")
            lines = content.splitlines()

            # Process based on file extension
            if s3_key.endswith(".jsonl"):
                for line in lines:
                    if len(conversations) >= limit:
                        break
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        conv = self._convert_to_conversation(data)
                        if conv:
                            conversations.append(conv)
                    except Exception:
                        continue

            elif s3_key.endswith(".csv"):
                # Use io.StringIO to parse the CSV string
                # We need to ensure we don't have a partial last line
                if len(lines) > 1:  # Header + at least one line
                    # standard safety: drop the last line in case it's incomplete bytes
                    valid_lines = lines[:-1]
                    csv_io = io.StringIO("\n".join(valid_lines))
                    reader = csv.DictReader(csv_io)
                    dataset_name = Path(s3_key).stem

                    for row in reader:
                        if len(conversations) >= limit:
                            break
                        try:
                            # Basic CSV parsing logic replicated from Tier4
                            text = (
                                row.get("text")
                                or row.get("post")
                                or row.get("content")
                                or row.get("message")
                                or ""
                            )
                            if text:
                                conv = Conversation(
                                    conversation_id=f"sample_{len(conversations)}",
                                    source=f"sample_{dataset_name}",
                                    messages=[Message(role="user", content=text)],
                                    metadata={"tier": self.tier, "sample": True},
                                )
                                conversations.append(conv)
                        except Exception:
                            continue

            elif s3_key.endswith(".json"):
                # JSON is hard to stream-parse without a proper streaming parser
                # For now, if it's a huge JSON list, 10MB might be invalid JSON
                # Check if we can parse it as complete JSON
                try:
                    data = json.loads(content)
                    items_to_process = []

                    if isinstance(data, list):
                        items_to_process = data
                    elif isinstance(data, dict):
                        # Handle wrapped lists (e.g. "filtered_conversations")
                        if "filtered_conversations" in data:
                            items_to_process = data["filtered_conversations"]
                        elif "conversations" in data:
                            items_to_process = data["conversations"]
                        else:
                            # Single object
                            items_to_process = [data]

                    for item in items_to_process[:limit]:
                        conv = self._convert_to_conversation(item)
                        if conv:
                            conversations.append(conv)

                except Exception:
                    logger.warning("Could not parse partial JSON stream")

        except ClientError as e:
            logger.error(f"S3 Client Error: {e}")
            raise

        return conversations
