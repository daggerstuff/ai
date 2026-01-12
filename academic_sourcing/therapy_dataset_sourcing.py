"""
Therapy Dataset Sourcing Module

Specialized module for finding and filtering therapy conversation datasets
from various sources (HuggingFace, GitHub, etc.) with specific criteria:
- Multi-turn conversations (20+ turns preferred)
- Therapy/counseling/mental health focus
- Quality filtering and validation
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class DatasetSource(Enum):
    """Dataset source types"""

    HUGGINGFACE = "huggingface"
    GITHUB = "github"
    KAGGLE = "kaggle"
    ZENODO = "zenodo"
    OPENML = "openml"
    PAPERSWITHCODE = "paperswithcode"


class ConversationFormat(Enum):
    """Conversation format types"""

    MULTI_TURN = "multi_turn"  # 20+ turns
    MEDIUM_TURN = "medium_turn"  # 10-19 turns
    SHORT_TURN = "short_turn"  # 5-9 turns
    SINGLE_TURN = "single_turn"  # 1-4 turns


@dataclass
class DatasetMetadata:
    """Metadata for therapy conversation datasets"""

    name: str
    source: str
    url: str
    description: str
    tags: List[str]
    downloads: int = 0
    likes: int = 0
    size_bytes: Optional[int] = None
    num_conversations: Optional[int] = None
    avg_turns: Optional[float] = None
    min_turns: Optional[int] = None
    max_turns: Optional[int] = None
    conversation_format: Optional[str] = None
    languages: List[str] = None
    license: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    quality_score: float = 0.0
    therapeutic_relevance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class TherapyDatasetSourcing:
    """
    Specialized sourcing engine for therapy conversation datasets

    Features:
    - HuggingFace Hub search with therapy-specific filters
    - Multi-turn conversation filtering (20+ turns preferred)
    - Quality scoring and validation
    - Automatic dataset analysis
    - Batch downloading and processing
    """

    def __init__(self, output_path: Optional[str] = None):
        self.output_path = Path(output_path or "ai/training_ready/datasets/sourced")
        self.output_path.mkdir(parents=True, exist_ok=True)

        # API Configuration
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.hf_api_base = "https://huggingface.co/api"
        self.github_token = os.getenv("GITHUB_TOKEN")

        # Therapy-specific keywords
        self.therapy_keywords = [
            "therapy",
            "counseling",
            "psychotherapy",
            "mental health",
            "psychological",
            "clinical",
            "therapeutic",
            "psychiatry",
            "counselor",
            "therapist",
            "patient",
            "client",
            "cbt",
            "dbt",
            "emdr",
            "act",
            "mindfulness",
            "depression",
            "anxiety",
            "trauma",
            "ptsd",
            "conversation",
            "dialogue",
            "session",
            "consultation",
        ]

        # Quality indicators
        self.quality_indicators = [
            "annotated",
            "validated",
            "curated",
            "expert",
            "professional",
            "clinical",
            "real",
            "authentic",
            "multi-turn",
            "long",
            "extended",
            "detailed",
        ]

        logger.info("Initialized TherapyDatasetSourcing")

    # ==================== HuggingFace Hub ====================

    def search_huggingface(
        self, query: str = "therapy conversation", min_turns: int = 20, limit: int = 50
    ) -> List[DatasetMetadata]:
        """
        Search HuggingFace Hub for therapy conversation datasets

        Args:
            query: Search query
            min_turns: Minimum number of conversation turns
            limit: Maximum number of results

        Returns:
            List of DatasetMetadata objects
        """
        logger.info(f"🔍 Searching HuggingFace for: '{query}' (min_turns={min_turns})")

        headers = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        # Search datasets
        search_url = f"{self.hf_api_base}/datasets"
        params = {
            "search": query,
            "limit": limit,
            "sort": "downloads",
            "direction": -1,
        }

        try:
            response = requests.get(
                search_url, params=params, headers=headers, timeout=15
            )

            if response.status_code != 200:
                logger.error(f"HuggingFace Error: {response.status_code}")
                return []

            datasets = response.json()

            results = []
            for dataset in datasets:
                dataset_id = dataset.get("id", "")

                # Filter by therapy relevance
                if not self._is_therapy_relevant(dataset):
                    continue

                # Get detailed info
                metadata = self._get_hf_dataset_details(dataset_id, headers)
                if not metadata:
                    continue

                # Filter by conversation length if we can determine it
                if metadata.avg_turns and metadata.avg_turns < min_turns:
                    logger.debug(
                        f"Skipping {dataset_id}: "
                        f"avg_turns={metadata.avg_turns} < {min_turns}"
                    )
                    continue

                results.append(metadata)

            logger.info(f"✅ Found {len(results)} therapy datasets from HuggingFace")
            return results

        except Exception as e:
            logger.error(f"HuggingFace search exception: {e}")
            return []

    def _get_hf_dataset_details(
        self, dataset_id: str, headers: Dict[str, str]
    ) -> Optional[DatasetMetadata]:
        """Get detailed information about a HuggingFace dataset"""
        try:
            # Get dataset info
            info_url = f"{self.hf_api_base}/datasets/{dataset_id}"
            response = requests.get(info_url, headers=headers, timeout=10)

            if response.status_code != 200:
                return None

            data = response.json()

            # Extract metadata
            tags = data.get("tags", [])
            card_data = data.get("cardData", {})

            # Try to determine conversation statistics
            avg_turns, min_turns, max_turns = self._estimate_conversation_stats(
                dataset_id, headers
            )

            # Determine conversation format
            conv_format = self._classify_conversation_format(avg_turns)

            # Calculate scores
            quality_score = self._calculate_quality_score(data)
            therapeutic_relevance = self._calculate_therapeutic_relevance(data)

            metadata = DatasetMetadata(
                name=dataset_id,
                source=DatasetSource.HUGGINGFACE.value,
                url=f"https://huggingface.co/datasets/{dataset_id}",
                description=data.get("description", ""),
                tags=tags,
                downloads=data.get("downloads", 0),
                likes=data.get("likes", 0),
                size_bytes=card_data.get("dataset_info", {}).get("dataset_size"),
                num_conversations=card_data.get("dataset_info", {})
                .get("splits", {})
                .get("train", {})
                .get("num_examples"),
                avg_turns=avg_turns,
                min_turns=min_turns,
                max_turns=max_turns,
                conversation_format=conv_format,
                languages=card_data.get("language", []),
                license=card_data.get("license"),
                created_at=data.get("createdAt"),
                updated_at=data.get("lastModified"),
                quality_score=quality_score,
                therapeutic_relevance=therapeutic_relevance,
            )

            return metadata

        except Exception as e:
            logger.warning(f"Error getting details for {dataset_id}: {e}")
            return None

    def _estimate_conversation_stats(
        self, dataset_id: str, headers: Dict[str, str]
    ) -> Tuple[Optional[float], Optional[int], Optional[int]]:
        """
        Estimate conversation statistics by sampling the dataset

        Returns:
            (avg_turns, min_turns, max_turns)
        """
        try:
            # Try to get first few rows
            rows_url = "https://datasets-server.huggingface.co/first-rows"
            params = {"dataset": dataset_id, "config": "default", "split": "train"}
            response = requests.get(
                rows_url, params=params, headers=headers, timeout=10
            )

            if response.status_code != 200:
                return None, None, None

            data = response.json()
            rows = data.get("rows", [])

            if not rows:
                return None, None, None

            # Analyze conversation structure
            turn_counts = []
            for row in rows[:10]:  # Sample first 10 rows
                row_data = row.get("row", {})

                # Try different conversation field names
                conversation = None
                for field in [
                    "conversation",
                    "messages",
                    "dialogue",
                    "turns",
                    "history",
                ]:
                    if field in row_data:
                        conversation = row_data[field]
                        break

                if conversation and isinstance(conversation, list):
                    turn_counts.append(len(conversation))

            if turn_counts:
                avg_turns = sum(turn_counts) / len(turn_counts)
                min_turns = min(turn_counts)
                max_turns = max(turn_counts)
                return avg_turns, min_turns, max_turns

            return None, None, None

        except Exception as e:
            logger.debug(f"Could not estimate stats for {dataset_id}: {e}")
            return None, None, None

    def _classify_conversation_format(
        self, avg_turns: Optional[float]
    ) -> Optional[str]:
        """Classify conversation format based on average turns"""
        if avg_turns is None:
            return None

        if avg_turns >= 20:
            return ConversationFormat.MULTI_TURN.value
        elif avg_turns >= 10:
            return ConversationFormat.MEDIUM_TURN.value
        elif avg_turns >= 5:
            return ConversationFormat.SHORT_TURN.value
        else:
            return ConversationFormat.SINGLE_TURN.value

    def _is_therapy_relevant(self, dataset: Dict[str, Any]) -> bool:
        """Check if dataset is therapy-relevant"""
        text_to_check = " ".join(
            [
                dataset.get("id", ""),
                dataset.get("description", ""),
                " ".join(dataset.get("tags", [])),
            ]
        ).lower()

        # Check for therapy keywords
        matches = sum(1 for kw in self.therapy_keywords if kw in text_to_check)
        return matches >= 2  # At least 2 therapy keywords

    def _calculate_quality_score(self, dataset: Dict[str, Any]) -> float:
        """Calculate quality score (0.0-1.0)"""
        score = 0.0

        # Downloads (max 0.3)
        downloads = dataset.get("downloads", 0)
        if downloads > 10000:
            score += 0.3
        elif downloads > 1000:
            score += 0.2
        elif downloads > 100:
            score += 0.1

        # Likes (max 0.2)
        likes = dataset.get("likes", 0)
        if likes > 100:
            score += 0.2
        elif likes > 50:
            score += 0.15
        elif likes > 10:
            score += 0.1

        # Quality indicators in description/tags (max 0.3)
        text_to_check = " ".join(
            [dataset.get("description", ""), " ".join(dataset.get("tags", []))]
        ).lower()

        quality_matches = sum(
            1 for ind in self.quality_indicators if ind in text_to_check
        )
        score += min(0.3, quality_matches * 0.1)

        # Has license (0.1)
        if dataset.get("cardData", {}).get("license"):
            score += 0.1

        # Recent update (0.1)
        last_modified = dataset.get("lastModified", "")
        if (
            last_modified
            and "2024" in last_modified
            or "2025" in last_modified
            or "2026" in last_modified
        ):
            score += 0.1

        return min(score, 1.0)

    def _calculate_therapeutic_relevance(self, dataset: Dict[str, Any]) -> float:
        """Calculate therapeutic relevance score (0.0-1.0)"""
        text_to_check = " ".join(
            [
                dataset.get("id", ""),
                dataset.get("description", ""),
                " ".join(dataset.get("tags", [])),
            ]
        ).lower()

        # Count therapy keyword matches
        matches = sum(1 for kw in self.therapy_keywords if kw in text_to_check)

        # Normalize to 0-1 scale
        score = min(matches / 10, 1.0)

        return score

    # ==================== Advanced Filtering ====================

    def filter_by_conversation_length(
        self,
        datasets: List[DatasetMetadata],
        min_turns: int = 20,
        max_turns: Optional[int] = None,
    ) -> List[DatasetMetadata]:
        """Filter datasets by conversation length"""
        filtered = []

        for dataset in datasets:
            if dataset.avg_turns is None:
                # Include if we can't determine (will need manual check)
                filtered.append(dataset)
                continue

            if dataset.avg_turns >= min_turns:
                if max_turns is None or dataset.avg_turns <= max_turns:
                    filtered.append(dataset)

        logger.info(f"Filtered to {len(filtered)} datasets with {min_turns}+ turns")
        return filtered

    def filter_by_quality(
        self, datasets: List[DatasetMetadata], min_quality: float = 0.5
    ) -> List[DatasetMetadata]:
        """Filter datasets by quality score"""
        filtered = [d for d in datasets if d.quality_score >= min_quality]
        logger.info(
            f"Filtered to {len(filtered)} datasets with quality >= {min_quality}"
        )
        return filtered

    def filter_by_therapeutic_relevance(
        self, datasets: List[DatasetMetadata], min_relevance: float = 0.5
    ) -> List[DatasetMetadata]:
        """Filter datasets by therapeutic relevance"""
        filtered = [d for d in datasets if d.therapeutic_relevance >= min_relevance]
        logger.info(
            f"Filtered to {len(filtered)} datasets with relevance >= {min_relevance}"
        )
        return filtered

    def rank_datasets(
        self,
        datasets: List[DatasetMetadata],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[DatasetMetadata]:
        """
        Rank datasets by composite score

        Args:
            datasets: List of datasets to rank
            weights: Custom weights for scoring factors
                     Default: {"quality": 0.3, "relevance": 0.3,
                               "turns": 0.2, "popularity": 0.2}
        """
        if weights is None:
            weights = {
                "quality": 0.3,
                "relevance": 0.3,
                "turns": 0.2,
                "popularity": 0.2,
            }

        def calculate_composite_score(dataset: DatasetMetadata) -> float:
            score = 0.0

            # Quality score
            score += dataset.quality_score * weights.get("quality", 0.3)

            # Therapeutic relevance
            score += dataset.therapeutic_relevance * weights.get("relevance", 0.3)

            # Conversation length (normalize to 0-1, assuming 50 turns is max)
            if dataset.avg_turns:
                turns_score = min(dataset.avg_turns / 50, 1.0)
                score += turns_score * weights.get("turns", 0.2)

            # Popularity (normalize downloads to 0-1)
            if dataset.downloads:
                pop_score = min(dataset.downloads / 100000, 1.0)
                score += pop_score * weights.get("popularity", 0.2)

            return score

        # Calculate composite scores and sort
        for dataset in datasets:
            dataset.quality_score = calculate_composite_score(dataset)

        ranked = sorted(datasets, key=lambda d: d.quality_score, reverse=True)

        logger.info(f"Ranked {len(ranked)} datasets by composite score")
        return ranked

    # ==================== Export & Reporting ====================

    def export_results(
        self, datasets: List[DatasetMetadata], filename: str = "therapy_datasets.json"
    ) -> Path:
        """Export dataset results to JSON"""
        output_file = self.output_path / filename

        data = [d.to_dict() for d in datasets]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Exported {len(datasets)} datasets to {output_file}")
        return output_file

    def generate_report(self, datasets: List[DatasetMetadata]) -> str:
        """Generate a summary report"""
        if not datasets:
            return "No datasets found."

        report = []
        report.append("=" * 60)
        report.append("THERAPY DATASET SOURCING REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal Datasets Found: {len(datasets)}\n")

        # Statistics
        with_turns = [d for d in datasets if d.avg_turns is not None]
        if with_turns:
            avg_turns = sum(d.avg_turns for d in with_turns) / len(with_turns)
            max_turns_dataset = max(with_turns, key=lambda d: d.avg_turns)
            report.append(f"Average Conversation Turns: {avg_turns:.1f}")
            report.append(
                f"Max Turns Dataset: {max_turns_dataset.name} "
                f"({max_turns_dataset.avg_turns:.0f} turns)\n"
            )

        # Quality distribution
        high_quality = len([d for d in datasets if d.quality_score >= 0.7])
        medium_quality = len([d for d in datasets if 0.4 <= d.quality_score < 0.7])
        low_quality = len([d for d in datasets if d.quality_score < 0.4])

        report.append("Quality Distribution:")
        report.append(f"  High (≥0.7):   {high_quality}")
        report.append(f"  Medium (0.4-0.7): {medium_quality}")
        report.append(f"  Low (<0.4):    {low_quality}\n")

        # Top 10 datasets
        report.append("Top 10 Datasets:")
        report.append("-" * 60)
        for i, dataset in enumerate(datasets[:10], 1):
            turns_info = (
                f"{dataset.avg_turns:.0f} turns"
                if dataset.avg_turns
                else "unknown turns"
            )
            report.append(f"{i}. {dataset.name}")
            report.append(
                f"   Score: {dataset.quality_score:.2f} | {turns_info} | "
                f"{dataset.downloads} downloads"
            )
            report.append(f"   {dataset.url}\n")

        return "\n".join(report)

    # ==================== Main Pipeline ====================

    def find_therapy_datasets(
        self,
        query: Optional[str] = None,
        min_turns: int = 20,
        min_quality: float = 0.5,
        min_relevance: float = 0.5,
        limit: int = 50,
    ) -> List[DatasetMetadata]:
        """
        Main pipeline to find high-quality therapy conversation datasets

        Args:
            query: Search query (optional)
            min_turns: Minimum conversation turns
            min_quality: Minimum quality score
            min_relevance: Minimum therapeutic relevance
            limit: Maximum results to return

        Returns:
            Ranked list of datasets
        """
        logger.info("🚀 Starting therapy dataset sourcing pipeline...")

        # Search HuggingFace
        search_query = (
            query
            if query and query.strip()
            else "therapy conversation mental health counseling"
        )
        datasets = self.search_huggingface(
            query=search_query,
            min_turns=min_turns,
            limit=limit * 2,  # Get more to filter
        )

        # Apply filters
        datasets = self.filter_by_conversation_length(datasets, min_turns=min_turns)
        datasets = self.filter_by_quality(datasets, min_quality=min_quality)
        datasets = self.filter_by_therapeutic_relevance(
            datasets, min_relevance=min_relevance
        )

        # Rank by composite score
        datasets = self.rank_datasets(datasets)

        # Limit results
        datasets = datasets[:limit]

        # Export results
        self.export_results(datasets)

        # Generate report
        report = self.generate_report(datasets)
        print(report)

        logger.info(
            f"✅ Pipeline complete! Found {len(datasets)} high-quality datasets"
        )
        return datasets


# Convenience function
def find_therapy_datasets(
    query: Optional[str] = None,
    min_turns: int = 20,
    min_quality: float = 0.5,
    output_path: Optional[str] = None,
) -> List[DatasetMetadata]:
    """
    Find therapy conversation datasets with specific criteria

    Args:
        query: Search query (optional)
        min_turns: Minimum conversation turns (default: 20)
        min_quality: Minimum quality score (default: 0.5)
        output_path: Output directory path

    Returns:
        List of ranked datasets
    """
    sourcing = TherapyDatasetSourcing(output_path=output_path)
    return sourcing.find_therapy_datasets(
        query=query, min_turns=min_turns, min_quality=min_quality
    )
