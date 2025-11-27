"""
DPO (Direct Preference Optimization) Dataset Loader

Loads DPO-format datasets from HuggingFace Hub for preference learning.
Supports datasets with chosen/rejected response pairs for alignment training.

Supported datasets:
- mlx-community/Human-Like-DPO: Human-like response generation
- flammenai/character-roleplay-DPO: Persona consistency for dual-persona training
- PJMixers/unalignment_toxic-dpo-v0.2-ShareGPT: Toxic input handling/safety alignment

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


class DPODatasetType(Enum):
    """Types of DPO datasets by purpose."""

    HUMAN_LIKE = "human_like"  # Natural conversation style
    ROLEPLAY = "roleplay"  # Character/persona consistency
    SAFETY = "safety"  # Safety alignment / toxic handling
    MEDICAL = "medical"  # Medical dialogue
    GENERAL = "general"  # General preference learning


@dataclass
class DPODatasetConfig:
    """Configuration for a DPO dataset."""

    dataset_id: str
    dataset_type: DPODatasetType
    split: str = "train"
    subset: Optional[str] = None
    description: str = ""
    license: str = "unknown"
    # Column mappings
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    # Format details
    format_type: str = "standard"  # standard, sharegpt, alpaca
    # Use case
    use_case: str = "general"


# DPO Dataset Registry
DPO_DATASETS: dict[str, DPODatasetConfig] = {
    "human_like_dpo": DPODatasetConfig(
        dataset_id="mlx-community/Human-Like-DPO",
        dataset_type=DPODatasetType.HUMAN_LIKE,
        description="DPO dataset for generating more human-like, natural responses",
        license="apache-2.0",
        prompt_column="prompt",
        chosen_column="chosen",
        rejected_column="rejected",
        format_type="standard",
        use_case="natural_conversation_style",
    ),
    "character_roleplay_dpo": DPODatasetConfig(
        dataset_id="flammenai/character-roleplay-DPO",
        dataset_type=DPODatasetType.ROLEPLAY,
        description="DPO dataset for character roleplay and persona consistency",
        license="mit",
        prompt_column="prompt",
        chosen_column="chosen",
        rejected_column="rejected",
        format_type="standard",
        use_case="persona_consistency",
    ),
    "toxic_safety_dpo": DPODatasetConfig(
        dataset_id="PJMixers/unalignment_toxic-dpo-v0.2-ShareGPT",
        dataset_type=DPODatasetType.SAFETY,
        description="DPO dataset for safety alignment and toxic input handling",
        license="apache-2.0",
        prompt_column="conversations",  # ShareGPT format
        chosen_column="chosen",
        rejected_column="rejected",
        format_type="sharegpt",
        use_case="safety_alignment",
    ),
}


@dataclass
class DPOSample:
    """A single DPO training sample with preference pair."""

    conversation_id: str
    prompt: str
    chosen_response: str
    rejected_response: str
    dataset_source: str
    dataset_type: DPODatasetType
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_conversation(self, include_rejected: bool = False) -> Conversation:
        """Convert to standard Conversation format (chosen response only by default)."""
        messages = [
            Message(role="user", content=self.prompt),
            Message(
                role="assistant",
                content=self.chosen_response,
                metadata={"is_chosen": True},
            ),
        ]

        if include_rejected:
            messages.append(
                Message(
                    role="assistant",
                    content=self.rejected_response,
                    metadata={"is_rejected": True},
                )
            )

        return Conversation(
            conversation_id=self.conversation_id,
            source=self.dataset_source,
            messages=messages,
            metadata={
                "dpo_sample": True,
                "dataset_type": self.dataset_type.value,
                "has_preference_pair": True,
                **self.metadata,
            },
        )

    def to_dpo_dict(self) -> dict[str, Any]:
        """Convert to DPO training format dictionary."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen_response,
            "rejected": self.rejected_response,
            "conversation_id": self.conversation_id,
            "source": self.dataset_source,
            "metadata": self.metadata,
        }


class DPODatasetLoader(BaseTierLoader):
    """
    Loader for DPO (Direct Preference Optimization) datasets.

    Loads datasets containing preference pairs (chosen/rejected responses)
    for training models with direct preference optimization.

    Supported formats:
    - Standard DPO: prompt, chosen, rejected columns
    - ShareGPT: conversations array with chosen/rejected
    - Alpaca: instruction/input/output with preference annotations

    Quality threshold: 90% (strict for alignment training)
    """

    def __init__(
        self,
        quality_threshold: float = 0.90,
        cache_dir: Optional[Path] = None,
        datasets_to_load: Optional[list[str]] = None,
        filter_by_type: Optional[DPODatasetType] = None,
    ):
        """
        Initialize DPO dataset loader.

        Args:
            quality_threshold: Quality threshold (default: 0.90 for DPO)
            cache_dir: Optional cache directory for HuggingFace datasets
            datasets_to_load: Optional list of dataset keys to load
            filter_by_type: Optional filter to load only specific dataset types
        """
        super().__init__(tier=5, quality_threshold=quality_threshold, base_path=cache_dir)
        self.cache_dir = cache_dir or Path("ai/datasets/huggingface_cache")
        self.datasets_to_load = datasets_to_load or list(DPO_DATASETS.keys())
        self.filter_by_type = filter_by_type

        logger.info(
            f"Initialized DPODatasetLoader: "
            f"quality_threshold={quality_threshold}, "
            f"datasets_to_load={self.datasets_to_load}"
        )

    def load_datasets(self) -> dict[str, list[Conversation]]:
        """
        Load all configured DPO datasets.

        Returns:
            Dictionary mapping dataset name to list of conversations
        """
        all_datasets: dict[str, list[Conversation]] = {}

        for dataset_key in self.datasets_to_load:
            if dataset_key not in DPO_DATASETS:
                logger.warning(f"Unknown DPO dataset key: {dataset_key}")
                continue

            config = DPO_DATASETS[dataset_key]

            # Filter by type if specified
            if self.filter_by_type and config.dataset_type != self.filter_by_type:
                continue

            logger.info(f"Loading DPO dataset: {config.dataset_id}")

            try:
                samples = self._load_single_dataset(config)
                if samples:
                    # Convert to conversations
                    conversations = [s.to_conversation() for s in samples]

                    # Add tier metadata
                    self.add_tier_metadata(
                        conversations,
                        {
                            "dataset_name": dataset_key,
                            "huggingface_id": config.dataset_id,
                            "source": f"dpo_{dataset_key}",
                            "data_type": "dpo_preference",
                            "dpo_dataset_type": config.dataset_type.value,
                            "use_case": config.use_case,
                            "description": config.description,
                        },
                    )

                    all_datasets[dataset_key] = conversations
                    logger.info(f"Loaded {len(conversations)} DPO samples from {dataset_key}")

            except Exception as e:
                logger.error(f"Error loading DPO dataset {dataset_key}: {e}", exc_info=True)
                continue

        total = sum(len(convs) for convs in all_datasets.values())
        logger.info(f"DPO loading complete: {len(all_datasets)} datasets, {total} total samples")

        return all_datasets

    def load_dpo_samples(self) -> dict[str, list[DPOSample]]:
        """
        Load datasets as DPOSample objects (with both chosen and rejected).

        Returns:
            Dictionary mapping dataset name to list of DPOSample objects
        """
        all_samples: dict[str, list[DPOSample]] = {}

        for dataset_key in self.datasets_to_load:
            if dataset_key not in DPO_DATASETS:
                continue

            config = DPO_DATASETS[dataset_key]

            if self.filter_by_type and config.dataset_type != self.filter_by_type:
                continue

            try:
                samples = self._load_single_dataset(config)
                if samples:
                    all_samples[dataset_key] = samples
                    logger.info(f"Loaded {len(samples)} DPO samples from {dataset_key}")
            except Exception as e:
                logger.error(f"Error loading DPO dataset {dataset_key}: {e}")
                continue

        return all_samples

    def _load_single_dataset(self, config: DPODatasetConfig) -> list[DPOSample]:
        """Load a single DPO dataset from HuggingFace."""
        try:
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

            # Convert based on format
            if config.format_type == "sharegpt":
                return self._convert_sharegpt_format(dataset, config)
            else:
                return self._convert_standard_format(dataset, config)

        except Exception as e:
            logger.error(f"Failed to load dataset {config.dataset_id}: {e}")
            raise

    def _convert_standard_format(
        self, dataset: Any, config: DPODatasetConfig
    ) -> list[DPOSample]:
        """Convert standard DPO format (prompt/chosen/rejected columns)."""
        samples = []

        for idx, row in enumerate(dataset):
            try:
                prompt = row.get(config.prompt_column, "")
                chosen = row.get(config.chosen_column, "")
                rejected = row.get(config.rejected_column, "")

                # Handle nested structures
                if isinstance(prompt, list):
                    prompt = self._extract_text_from_list(prompt)
                if isinstance(chosen, list):
                    chosen = self._extract_text_from_list(chosen)
                if isinstance(rejected, list):
                    rejected = self._extract_text_from_list(rejected)

                if not prompt or not chosen or not rejected:
                    continue

                conv_id = self._generate_conversation_id(config.dataset_id, idx, prompt[:50])

                sample = DPOSample(
                    conversation_id=conv_id,
                    prompt=prompt,
                    chosen_response=chosen,
                    rejected_response=rejected,
                    dataset_source=f"dpo_{config.dataset_type.value}",
                    dataset_type=config.dataset_type,
                    metadata={
                        "row_index": idx,
                        "format": "standard",
                        "use_case": config.use_case,
                    },
                )

                samples.append(sample)

            except Exception as e:
                logger.warning(f"Error converting row {idx}: {e}")
                continue

        return samples

    def _convert_sharegpt_format(
        self, dataset: Any, config: DPODatasetConfig
    ) -> list[DPOSample]:
        """Convert ShareGPT format (conversations array)."""
        samples = []

        for idx, row in enumerate(dataset):
            try:
                conversations = row.get("conversations", row.get(config.prompt_column, []))
                chosen = row.get(config.chosen_column, "")
                rejected = row.get(config.rejected_column, "")

                # Extract prompt from conversations
                prompt = self._extract_prompt_from_sharegpt(conversations)

                if isinstance(chosen, list):
                    chosen = self._extract_text_from_list(chosen)
                if isinstance(rejected, list):
                    rejected = self._extract_text_from_list(rejected)

                if not prompt or not chosen or not rejected:
                    continue

                conv_id = self._generate_conversation_id(config.dataset_id, idx, prompt[:50])

                sample = DPOSample(
                    conversation_id=conv_id,
                    prompt=prompt,
                    chosen_response=chosen,
                    rejected_response=rejected,
                    dataset_source=f"dpo_{config.dataset_type.value}",
                    dataset_type=config.dataset_type,
                    metadata={
                        "row_index": idx,
                        "format": "sharegpt",
                        "original_conversations": len(conversations) if isinstance(conversations, list) else 0,
                        "use_case": config.use_case,
                    },
                )

                samples.append(sample)

            except Exception as e:
                logger.warning(f"Error converting ShareGPT row {idx}: {e}")
                continue

        return samples

    def _extract_text_from_list(self, data: list) -> str:
        """Extract text content from a list structure."""
        texts = []
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                content = item.get("content", item.get("text", item.get("value", "")))
                if content:
                    texts.append(str(content))
        return "\n".join(texts)

    def _extract_prompt_from_sharegpt(self, conversations: Any) -> str:
        """Extract prompt from ShareGPT format conversations."""
        if not conversations:
            return ""

        if isinstance(conversations, str):
            return conversations

        if isinstance(conversations, list):
            # Get the last human/user message as the prompt
            for conv in reversed(conversations):
                if isinstance(conv, dict):
                    role = conv.get("from", conv.get("role", "")).lower()
                    if role in ["human", "user"]:
                        return conv.get("value", conv.get("content", ""))
            # If no user message found, concatenate all
            return self._extract_text_from_list(conversations)

        return str(conversations)

    def _generate_conversation_id(
        self, dataset_id: str, row_index: int, content_preview: str
    ) -> str:
        """Generate unique conversation ID."""
        unique_string = f"{dataset_id}_{row_index}_{content_preview}"
        hash_value = hashlib.md5(unique_string.encode()).hexdigest()[:12]
        return f"dpo_{hash_value}"

    def get_available_datasets(self) -> dict[str, DPODatasetConfig]:
        """Get available DPO datasets."""
        return DPO_DATASETS.copy()

    def get_datasets_by_type(
        self, dataset_type: DPODatasetType
    ) -> dict[str, DPODatasetConfig]:
        """Get DPO datasets filtered by type."""
        return {
            k: v for k, v in DPO_DATASETS.items() if v.dataset_type == dataset_type
        }

    def load_for_training(
        self, include_safety: bool = True, include_roleplay: bool = True
    ) -> list[dict[str, Any]]:
        """
        Load DPO data in format ready for TRL/transformers DPO training.

        Args:
            include_safety: Include safety alignment datasets
            include_roleplay: Include roleplay datasets

        Returns:
            List of dicts with prompt, chosen, rejected keys
        """
        all_training_data = []

        samples_by_dataset = self.load_dpo_samples()

        for dataset_key, samples in samples_by_dataset.items():
            config = DPO_DATASETS[dataset_key]

            # Filter based on flags
            if not include_safety and config.dataset_type == DPODatasetType.SAFETY:
                continue
            if not include_roleplay and config.dataset_type == DPODatasetType.ROLEPLAY:
                continue

            for sample in samples:
                all_training_data.append(sample.to_dpo_dict())

        logger.info(f"Prepared {len(all_training_data)} samples for DPO training")
        return all_training_data


def register_dpo_dataset(
    key: str,
    dataset_id: str,
    dataset_type: DPODatasetType,
    split: str = "train",
    prompt_column: str = "prompt",
    chosen_column: str = "chosen",
    rejected_column: str = "rejected",
    format_type: str = "standard",
    description: str = "",
    use_case: str = "general",
    license: str = "unknown",
) -> None:
    """
    Register a new DPO dataset for loading.

    Args:
        key: Unique key for the dataset
        dataset_id: HuggingFace dataset ID
        dataset_type: Type of DPO dataset
        split: Dataset split to load
        prompt_column: Column name for prompts
        chosen_column: Column name for chosen responses
        rejected_column: Column name for rejected responses
        format_type: Format type (standard, sharegpt, alpaca)
        description: Dataset description
        use_case: Primary use case
        license: Dataset license
    """
    DPO_DATASETS[key] = DPODatasetConfig(
        dataset_id=dataset_id,
        dataset_type=dataset_type,
        split=split,
        prompt_column=prompt_column,
        chosen_column=chosen_column,
        rejected_column=rejected_column,
        format_type=format_type,
        description=description,
        use_case=use_case,
        license=license,
    )
    logger.info(f"Registered new DPO dataset: {key} -> {dataset_id}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("DPO Dataset Loader")
    print("=" * 50)

    loader = DPODatasetLoader()

    print("\nAvailable DPO datasets:")
    for key, config in loader.get_available_datasets().items():
        print(f"  - {key}: {config.dataset_id}")
        print(f"    Type: {config.dataset_type.value}")
        print(f"    Use case: {config.use_case}")
        print(f"    Description: {config.description}")
        print()

    print("\nDatasets by type:")
    for dtype in DPODatasetType:
        datasets = loader.get_datasets_by_type(dtype)
        print(f"  {dtype.value}: {len(datasets)} datasets")

    print("\nTo load datasets, run:")
    print("  samples = loader.load_dpo_samples()")
    print("  training_data = loader.load_for_training()")

