"""
HuggingFace Mental Health Dataset Loader

Loads mental health datasets from HuggingFace Hub with tier-specific quality validation.
Supports datasets: customized-mental-health-snli2, MentalHealthPreProcessed, DepressionDetection,
and Empathy-Mental-Health.

Part of the Pixelated Empathy AI dataset pipeline.
"""

import hashlib
import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# Handle imports
loader_path = Path(__file__).parent
pipeline_root = loader_path.parent.parent.parent
sys.path.insert(0, str(pipeline_root))

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Please install datasets: uv pip install datasets")

try:
    from schemas.conversation_schema import Conversation, Message
except ImportError:
    try:
        from ai.dataset_pipeline.schemas.conversation_schema import Conversation, Message
    except ImportError:
        from conversation_schema import Conversation, Message

from ai.dataset_pipeline.ingestion.tier_loaders.base_tier_loader import BaseTierLoader

logger = logging.getLogger(__name__)


class HuggingFaceDatasetType(Enum):
    """Supported HuggingFace mental health datasets."""

    MENTAL_HEALTH_SNLI = "mental_health_snli"
    MENTAL_HEALTH_PREPROCESSED = "mental_health_preprocessed"
    DEPRESSION_DETECTION = "depression_detection"
    EMPATHY_MENTAL_HEALTH = "empathy_mental_health"


@dataclass
class HuggingFaceDatasetConfig:
    """Configuration for a HuggingFace dataset."""

    dataset_id: str
    dataset_type: HuggingFaceDatasetType
    split: str = "train"
    subset: Optional[str] = None
    description: str = ""
    license: str = "unknown"
    expected_columns: list[str] = field(default_factory=list)
    conversion_strategy: str = "auto"


# Dataset registry with configurations
HUGGINGFACE_MENTAL_HEALTH_DATASETS: dict[str, HuggingFaceDatasetConfig] = {
    "mental_health_snli": HuggingFaceDatasetConfig(
        dataset_id="iqrakiran/customized-mental-health-snli2",
        dataset_type=HuggingFaceDatasetType.MENTAL_HEALTH_SNLI,
        split="train",
        description="43.9k text pairs for NLI in mental health contexts (entailment/contradiction/neutral)",
        license="unknown",
        expected_columns=["premise", "hypothesis", "label"],
        conversion_strategy="nli_pairs",
    ),
    "mental_health_preprocessed": HuggingFaceDatasetConfig(
        dataset_id="typosonlr/MentalHealthPreProcessed",
        dataset_type=HuggingFaceDatasetType.MENTAL_HEALTH_PREPROCESSED,
        split="train",
        description="3k instruction-response pairs for mental health conversational agents",
        license="unknown",
        expected_columns=["instruction", "output"],
        conversion_strategy="instruction_output",
    ),
    "depression_detection": HuggingFaceDatasetConfig(
        dataset_id="ShreyaR/DepressionDetection",
        dataset_type=HuggingFaceDatasetType.DEPRESSION_DETECTION,
        split="train",
        description="Depression indicator detection dataset",
        license="unknown",
        expected_columns=["text", "label"],
        conversion_strategy="classification",
    ),
}


class HuggingFaceMentalHealthLoader(BaseTierLoader):
    """
    Loader for HuggingFace Mental Health Datasets.

    Loads and converts datasets from HuggingFace Hub to the standard
    Pixelated Empathy conversation format.

    Supported datasets:
    - iqrakiran/customized-mental-health-snli2: NLI pairs for mental health inference
    - typosonlr/MentalHealthPreProcessed: Instruction-output pairs
    - ShreyaR/DepressionDetection: Depression detection classification

    Quality threshold: 85% (Tier 5 Research level with premium for curated data)
    """

    def __init__(
        self,
        quality_threshold: float = 0.85,
        cache_dir: Optional[Path] = None,
        datasets_to_load: Optional[list[str]] = None,
    ):
        """
        Initialize HuggingFace Mental Health loader.

        Args:
            quality_threshold: Quality threshold (default: 0.85 = 85%)
            cache_dir: Optional cache directory for HuggingFace datasets
            datasets_to_load: Optional list of dataset keys to load (from registry).
                            If None, loads all available datasets.
        """
        super().__init__(tier=5, quality_threshold=quality_threshold, base_path=cache_dir)
        self.cache_dir = cache_dir or Path("ai/datasets/huggingface_cache")
        self.datasets_to_load = datasets_to_load or list(HUGGINGFACE_MENTAL_HEALTH_DATASETS.keys())

        # NLI label mapping for SNLI dataset
        self.nli_labels = {0: "entailment", 1: "neutral", 2: "contradiction"}

        logger.info(
            f"Initialized HuggingFaceMentalHealthLoader: "
            f"quality_threshold={quality_threshold}, "
            f"datasets_to_load={self.datasets_to_load}"
        )

    def load_datasets(self) -> dict[str, list[Conversation]]:
        """
        Load all configured HuggingFace mental health datasets.

        Returns:
            Dictionary mapping dataset name to list of conversations
        """
        all_datasets: dict[str, list[Conversation]] = {}

        for dataset_key in self.datasets_to_load:
            if dataset_key not in HUGGINGFACE_MENTAL_HEALTH_DATASETS:
                logger.warning(f"Unknown dataset key: {dataset_key}, skipping")
                continue

            config = HUGGINGFACE_MENTAL_HEALTH_DATASETS[dataset_key]
            logger.info(f"Loading HuggingFace dataset: {config.dataset_id}")

            try:
                conversations = self._load_single_dataset(config)
                if conversations:
                    # Add tier metadata
                    self.add_tier_metadata(
                        conversations,
                        {
                            "dataset_name": dataset_key,
                            "huggingface_id": config.dataset_id,
                            "source": f"huggingface_{dataset_key}",
                            "data_type": "mental_health",
                            "description": config.description,
                            "license": config.license,
                        },
                    )
                    all_datasets[dataset_key] = conversations
                    logger.info(f"Loaded {len(conversations)} conversations from {dataset_key}")
            except Exception as e:
                logger.error(f"Error loading dataset {dataset_key}: {e}", exc_info=True)
                continue

        total_conversations = sum(len(convs) for convs in all_datasets.values())
        logger.info(
            f"HuggingFace Mental Health loading complete: "
            f"{len(all_datasets)} datasets, {total_conversations} total conversations"
        )

        return all_datasets

    def _load_single_dataset(self, config: HuggingFaceDatasetConfig) -> list[Conversation]:
        """
        Load a single dataset from HuggingFace Hub.

        Args:
            config: Dataset configuration

        Returns:
            List of Conversation objects
        """
        try:
            # Load from HuggingFace Hub
            if config.subset:
                dataset = load_dataset(
                    config.dataset_id,
                    config.subset,
                    split=config.split,
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=True,
                )
            else:
                dataset = load_dataset(
                    config.dataset_id,
                    split=config.split,
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=True,
                )

            logger.info(f"Loaded {len(dataset)} records from {config.dataset_id}")

            # Convert based on strategy
            if config.conversion_strategy == "nli_pairs":
                return self._convert_nli_dataset(dataset, config)
            elif config.conversion_strategy == "instruction_output":
                return self._convert_instruction_output_dataset(dataset, config)
            elif config.conversion_strategy == "classification":
                return self._convert_classification_dataset(dataset, config)
            else:
                return self._convert_auto_dataset(dataset, config)

        except Exception as e:
            logger.error(f"Failed to load dataset {config.dataset_id}: {e}", exc_info=True)
            raise

    def _convert_nli_dataset(
        self, dataset: Any, config: HuggingFaceDatasetConfig
    ) -> list[Conversation]:
        """
        Convert NLI dataset (premise/hypothesis/label) to conversations.

        For mental health NLI, we create conversations that test understanding
        of mental health statements and their logical relationships.

        Args:
            dataset: HuggingFace dataset
            config: Dataset configuration

        Returns:
            List of Conversation objects
        """
        conversations = []

        for idx, row in enumerate(dataset):
            try:
                premise = row.get("premise", "")
                hypothesis = row.get("hypothesis", "")
                label = row.get("label", 0)

                if not premise or not hypothesis:
                    continue

                # Map label to string
                label_str = self.nli_labels.get(label, str(label))

                # Generate unique conversation ID
                conv_id = self._generate_conversation_id(
                    config.dataset_id, idx, premise[:50]
                )

                # Create conversation with NLI format
                # User presents the premise, assistant analyzes the hypothesis
                messages = [
                    Message(
                        role="user",
                        content=f"Consider this statement about mental health: \"{premise}\"\n\n"
                        f"How does this relate to: \"{hypothesis}\"?",
                        metadata={"turn_type": "nli_query"},
                    ),
                    Message(
                        role="assistant",
                        content=f"Analyzing the relationship between these mental health statements:\n\n"
                        f"The relationship is: **{label_str}**\n\n"
                        f"This means the hypothesis {'logically follows from' if label_str == 'entailment' else 'contradicts' if label_str == 'contradiction' else 'is unrelated to'} the premise.",
                        metadata={
                            "turn_type": "nli_analysis",
                            "nli_label": label_str,
                        },
                    ),
                ]

                conversation = Conversation(
                    conversation_id=conv_id,
                    source=f"huggingface_{config.dataset_type.value}",
                    messages=messages,
                    metadata={
                        "premise": premise,
                        "hypothesis": hypothesis,
                        "nli_label": label_str,
                        "nli_label_id": label,
                        "dataset_type": "nli",
                        "tier": self.tier,
                        "quality_threshold": self.quality_threshold,
                        "row_index": idx,
                    },
                )

                conversations.append(conversation)

            except Exception as e:
                logger.warning(f"Error converting NLI row {idx}: {e}")
                continue

        return conversations

    def _convert_instruction_output_dataset(
        self, dataset: Any, config: HuggingFaceDatasetConfig
    ) -> list[Conversation]:
        """
        Convert instruction-output dataset to conversations.

        Args:
            dataset: HuggingFace dataset
            config: Dataset configuration

        Returns:
            List of Conversation objects
        """
        conversations = []

        for idx, row in enumerate(dataset):
            try:
                instruction = row.get("instruction", row.get("input", ""))
                output = row.get("output", row.get("response", ""))

                if not instruction or not output:
                    continue

                conv_id = self._generate_conversation_id(
                    config.dataset_id, idx, instruction[:50]
                )

                messages = [
                    Message(
                        role="user",
                        content=instruction,
                        metadata={"turn_type": "instruction"},
                    ),
                    Message(
                        role="assistant",
                        content=output,
                        metadata={"turn_type": "response"},
                    ),
                ]

                conversation = Conversation(
                    conversation_id=conv_id,
                    source=f"huggingface_{config.dataset_type.value}",
                    messages=messages,
                    metadata={
                        "dataset_type": "instruction_output",
                        "tier": self.tier,
                        "quality_threshold": self.quality_threshold,
                        "row_index": idx,
                    },
                )

                conversations.append(conversation)

            except Exception as e:
                logger.warning(f"Error converting instruction row {idx}: {e}")
                continue

        return conversations

    def _convert_classification_dataset(
        self, dataset: Any, config: HuggingFaceDatasetConfig
    ) -> list[Conversation]:
        """
        Convert classification dataset (text/label) to conversations.

        For depression detection, we create conversations that analyze
        text for mental health indicators.

        Args:
            dataset: HuggingFace dataset
            config: Dataset configuration

        Returns:
            List of Conversation objects
        """
        conversations = []

        # Depression detection label mapping
        depression_labels = {0: "not_depressed", 1: "depressed"}

        for idx, row in enumerate(dataset):
            try:
                text = row.get("text", row.get("content", ""))
                label = row.get("label", row.get("is_depression", 0))

                if not text:
                    continue

                # Handle various label formats
                if isinstance(label, str):
                    label_str = label
                else:
                    label_str = depression_labels.get(label, str(label))

                conv_id = self._generate_conversation_id(
                    config.dataset_id, idx, text[:50]
                )

                # Create conversation for classification analysis
                messages = [
                    Message(
                        role="user",
                        content=f"Please analyze this text for mental health indicators:\n\n\"{text}\"",
                        metadata={"turn_type": "classification_query"},
                    ),
                    Message(
                        role="assistant",
                        content=f"Based on the analysis of this text:\n\n"
                        f"**Classification**: {label_str}\n\n"
                        f"This classification is based on linguistic patterns and emotional indicators present in the text.",
                        metadata={
                            "turn_type": "classification_analysis",
                            "classification_label": label_str,
                        },
                    ),
                ]

                conversation = Conversation(
                    conversation_id=conv_id,
                    source=f"huggingface_{config.dataset_type.value}",
                    messages=messages,
                    metadata={
                        "original_text": text,
                        "classification_label": label_str,
                        "dataset_type": "classification",
                        "tier": self.tier,
                        "quality_threshold": self.quality_threshold,
                        "row_index": idx,
                    },
                )

                conversations.append(conversation)

            except Exception as e:
                logger.warning(f"Error converting classification row {idx}: {e}")
                continue

        return conversations

    def _convert_auto_dataset(
        self, dataset: Any, config: HuggingFaceDatasetConfig
    ) -> list[Conversation]:
        """
        Auto-detect and convert dataset format.

        Args:
            dataset: HuggingFace dataset
            config: Dataset configuration

        Returns:
            List of Conversation objects
        """
        conversations = []

        # Get first row to detect format
        if len(dataset) == 0:
            return conversations

        sample_row = dataset[0]
        columns = list(sample_row.keys())

        logger.info(f"Auto-detecting format for columns: {columns}")

        for idx, row in enumerate(dataset):
            try:
                # Try various format patterns
                messages = []

                # Pattern 1: messages array
                if "messages" in row:
                    for msg in row["messages"]:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if content:
                            messages.append(Message(role=role, content=content))

                # Pattern 2: conversation array
                elif "conversation" in row:
                    for turn in row["conversation"]:
                        role = turn.get("role", turn.get("speaker", "user"))
                        content = turn.get("content", turn.get("text", ""))
                        if content:
                            messages.append(Message(role=role, content=content))

                # Pattern 3: user/assistant pairs
                elif "user" in row and "assistant" in row:
                    messages = [
                        Message(role="user", content=str(row["user"])),
                        Message(role="assistant", content=str(row["assistant"])),
                    ]

                # Pattern 4: question/answer pairs
                elif "question" in row and "answer" in row:
                    messages = [
                        Message(role="user", content=str(row["question"])),
                        Message(role="assistant", content=str(row["answer"])),
                    ]

                # Pattern 5: input/output pairs
                elif "input" in row and "output" in row:
                    messages = [
                        Message(role="user", content=str(row["input"])),
                        Message(role="assistant", content=str(row["output"])),
                    ]

                # Pattern 6: text only
                elif "text" in row:
                    messages = [Message(role="user", content=str(row["text"]))]

                if not messages:
                    continue

                conv_id = self._generate_conversation_id(config.dataset_id, idx, "")

                conversation = Conversation(
                    conversation_id=conv_id,
                    source=f"huggingface_{config.dataset_type.value}",
                    messages=messages,
                    metadata={
                        "dataset_type": "auto_detected",
                        "tier": self.tier,
                        "quality_threshold": self.quality_threshold,
                        "row_index": idx,
                        "original_columns": columns,
                    },
                )

                conversations.append(conversation)

            except Exception as e:
                logger.warning(f"Error converting auto row {idx}: {e}")
                continue

        return conversations

    def _generate_conversation_id(
        self, dataset_id: str, row_index: int, content_preview: str
    ) -> str:
        """
        Generate a unique, deterministic conversation ID.

        Args:
            dataset_id: HuggingFace dataset ID
            row_index: Row index in dataset
            content_preview: Preview of content for uniqueness

        Returns:
            Unique conversation ID
        """
        unique_string = f"{dataset_id}_{row_index}_{content_preview}"
        hash_value = hashlib.md5(unique_string.encode()).hexdigest()[:12]
        return f"hf_{hash_value}"

    def get_available_datasets(self) -> dict[str, HuggingFaceDatasetConfig]:
        """
        Get available HuggingFace mental health datasets.

        Returns:
            Dictionary of dataset configurations
        """
        return HUGGINGFACE_MENTAL_HEALTH_DATASETS.copy()

    def load_specific_dataset(self, dataset_key: str) -> list[Conversation]:
        """
        Load a specific dataset by key.

        Args:
            dataset_key: Key from HUGGINGFACE_MENTAL_HEALTH_DATASETS

        Returns:
            List of Conversation objects
        """
        if dataset_key not in HUGGINGFACE_MENTAL_HEALTH_DATASETS:
            raise ValueError(f"Unknown dataset key: {dataset_key}")

        config = HUGGINGFACE_MENTAL_HEALTH_DATASETS[dataset_key]
        conversations = self._load_single_dataset(config)

        self.add_tier_metadata(
            conversations,
            {
                "dataset_name": dataset_key,
                "huggingface_id": config.dataset_id,
                "source": f"huggingface_{dataset_key}",
                "data_type": "mental_health",
            },
        )

        return conversations


def register_huggingface_dataset(
    key: str,
    dataset_id: str,
    dataset_type: HuggingFaceDatasetType,
    split: str = "train",
    subset: Optional[str] = None,
    description: str = "",
    license: str = "unknown",
    expected_columns: Optional[list[str]] = None,
    conversion_strategy: str = "auto",
) -> None:
    """
    Register a new HuggingFace dataset for loading.

    Args:
        key: Unique key for the dataset
        dataset_id: HuggingFace dataset ID (e.g., "username/dataset-name")
        dataset_type: Type of dataset
        split: Dataset split to load
        subset: Optional dataset subset/configuration
        description: Human-readable description
        license: Dataset license
        expected_columns: Expected column names
        conversion_strategy: Strategy for conversion (nli_pairs, instruction_output, classification, auto)
    """
    HUGGINGFACE_MENTAL_HEALTH_DATASETS[key] = HuggingFaceDatasetConfig(
        dataset_id=dataset_id,
        dataset_type=dataset_type,
        split=split,
        subset=subset,
        description=description,
        license=license,
        expected_columns=expected_columns or [],
        conversion_strategy=conversion_strategy,
    )
    logger.info(f"Registered new HuggingFace dataset: {key} -> {dataset_id}")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("HuggingFace Mental Health Dataset Loader")
    print("=" * 50)

    loader = HuggingFaceMentalHealthLoader()

    print("\nAvailable datasets:")
    for key, config in loader.get_available_datasets().items():
        print(f"  - {key}: {config.dataset_id}")
        print(f"    Description: {config.description}")
        print(f"    Strategy: {config.conversion_strategy}")
        print()

    # Test loading a specific dataset
    print("\nTesting dataset loading (mental_health_preprocessed)...")
    try:
        conversations = loader.load_specific_dataset("mental_health_preprocessed")
        print(f"Loaded {len(conversations)} conversations")
        if conversations:
            print(f"\nSample conversation:")
            sample = conversations[0]
            print(f"  ID: {sample.conversation_id}")
            print(f"  Source: {sample.source}")
            print(f"  Messages: {len(sample.messages)}")
            for msg in sample.messages[:2]:
                print(f"    [{msg.role}]: {msg.content[:100]}...")
    except Exception as e:
        print(f"Error loading dataset: {e}")

