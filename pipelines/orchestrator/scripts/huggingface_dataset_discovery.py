#!/usr/bin/env python3
"""
HuggingFace Dataset Discovery Script
Pixelated Empathy - Dataset Expansion

This script searches HuggingFace for mental health and therapeutic conversation datasets
as outlined in the dataset expansion plan (Part 3B).

Categories to explore:
- Mental health conversation datasets
- Chain-of-thought reasoning (non-domain-specific)
- Instruction-following datasets with empathy
- Multi-turn dialogue datasets
- Emotional support conversation
- Crisis intervention training data
- Therapeutic alliance datasets
- Motivational interviewing corpora
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Search queries from the plan
SEARCH_QUERIES = [
    "mental health counseling",
    "therapeutic conversation",
    "empathetic dialogue",
    "chain of thought reasoning",
    "emotional support",
    "crisis intervention",
    "cognitive behavioral therapy",
    "motivational interviewing",
    "psychotherapy dataset",
    "counselor training",
    "mental health chatbot",
    "depression anxiety support",
    "trauma informed care",
]

# Known high-quality datasets to check
KNOWN_DATASETS = {
    "empathetic_dialogues": {
        "hub_id": "facebook/empathetic_dialogues",
        "description": "Facebook's empathy dataset with 25k conversations",
        "relevance": "high",
        "category": "empathetic_dialogue"
    },
    "ESConv": {
        "hub_id": "thu-coai/esconv",
        "description": "Emotional support conversation dataset",
        "relevance": "high",
        "category": "emotional_support"
    },
    "MELD": {
        "hub_id": "declare-lab/MELD",
        "description": "Multimodal emotion lines dataset",
        "relevance": "medium",
        "category": "emotion_recognition"
    },
    "DailyDialog": {
        "hub_id": "daily_dialog",
        "description": "High-quality multi-turn dialogues",
        "relevance": "medium",
        "category": "multi_turn_dialogue"
    },
    "PersonaChat": {
        "hub_id": "bavard/personachat_truecased",
        "description": "Persona-based conversations",
        "relevance": "medium",
        "category": "persona_dialogue"
    },
    "Amod_mental_health": {
        "hub_id": "Amod/mental_health_counseling_conversations",
        "description": "Mental health counseling conversations",
        "relevance": "high",
        "category": "therapeutic"
    },
    "EmoCareAI_Psych8k": {
        "hub_id": "EmoCareAI/Psych8k",
        "description": "Psychology conversations dataset",
        "relevance": "high",
        "category": "therapeutic"
    },
    "heliosbrahma_mental_health": {
        "hub_id": "heliosbrahma/mental_health_chatbot_dataset",
        "description": "Mental health chatbot training data",
        "relevance": "high",
        "category": "therapeutic"
    },
    "counsel_chat": {
        "hub_id": "nbertagnolli/counsel-chat",
        "description": "Counseling chat dataset",
        "relevance": "high",
        "category": "therapeutic"
    },
}


@dataclass
class DatasetInfo:
    """Information about a discovered dataset."""
    hub_id: str
    name: str
    description: str = ""
    downloads: int = 0
    likes: int = 0
    tags: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    category: str = "unknown"
    size_estimate: str = "unknown"
    format: str = "unknown"
    license: str = "unknown"
    last_modified: str = ""
    status: str = "discovered"  # discovered, evaluated, acquired, rejected


def search_huggingface_datasets(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search HuggingFace Hub for datasets matching query."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        datasets = list(api.list_datasets(
            search=query,
            sort="downloads",
            direction=-1,
            limit=limit,
        ))

        results = []
        for ds in datasets:
            results.append({
                "id": ds.id,
                "author": ds.author,
                "downloads": ds.downloads,
                "likes": ds.likes,
                "tags": ds.tags or [],
                "last_modified": str(ds.last_modified) if ds.last_modified else "",
            })

        return results
    except ImportError:
        logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return []
    except Exception as e:
        logger.error(f"Error searching HuggingFace: {e}")
        return []


def get_dataset_info(hub_id: str) -> dict[str, Any] | None:
    """Get detailed info about a specific dataset."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.dataset_info(hub_id)

        return {
            "id": info.id,
            "author": info.author,
            "description": info.description or "",
            "downloads": info.downloads,
            "likes": info.likes,
            "tags": info.tags or [],
            "card_data": info.card_data.__dict__ if info.card_data else {},
            "last_modified": str(info.last_modified) if info.last_modified else "",
        }
    except Exception as e:
        logger.error(f"Error getting dataset info for {hub_id}: {e}")
        return None


def calculate_relevance_score(dataset: dict[str, Any], query: str) -> float:
    """Calculate relevance score for a dataset based on query match and metrics."""
    score = 0.0

    # Downloads weight (log scale)
    downloads = dataset.get("downloads", 0)
    if downloads > 0:
        import math
        score += min(math.log10(downloads + 1) / 6, 1.0) * 30  # Max 30 points

    # Likes weight
    likes = dataset.get("likes", 0)
    score += min(likes / 100, 1.0) * 20  # Max 20 points

    # Tag relevance
    tags = dataset.get("tags", [])
    relevant_tags = ["mental-health", "psychology", "therapy", "counseling",
                     "emotion", "dialogue", "conversation", "empathy",
                     "text-generation", "question-answering"]
    tag_matches = sum(1 for t in tags if any(rt in t.lower() for rt in relevant_tags))
    score += min(tag_matches * 5, 25)  # Max 25 points

    # Query match in description
    description = dataset.get("description", "").lower()
    query_terms = query.lower().split()
    desc_matches = sum(1 for t in query_terms if t in description)
    score += min(desc_matches * 5, 25)  # Max 25 points

    return min(score, 100)


def discover_datasets(output_dir: Path | None = None) -> dict[str, Any]:
    """
    Run full HuggingFace dataset discovery.

    Returns a discovery report with all found datasets.
    """
    logger.info("Starting HuggingFace Dataset Discovery")

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "discoveries"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_datasets: dict[str, DatasetInfo] = {}
    search_results: dict[str, list[str]] = {}

    # Search for each query
    logger.info(f"Searching {len(SEARCH_QUERIES)} queries...")
    for query in SEARCH_QUERIES:
        logger.info(f"  Searching: {query}")
        results = search_huggingface_datasets(query, limit=15)
        search_results[query] = [r["id"] for r in results]

        for result in results:
            hub_id = result["id"]
            if hub_id not in all_datasets:
                relevance = calculate_relevance_score(result, query)
                all_datasets[hub_id] = DatasetInfo(
                    hub_id=hub_id,
                    name=hub_id.split("/")[-1] if "/" in hub_id else hub_id,
                    downloads=result.get("downloads", 0),
                    likes=result.get("likes", 0),
                    tags=result.get("tags", []),
                    relevance_score=relevance,
                    last_modified=result.get("last_modified", ""),
                )

    # Add known datasets
    logger.info("Adding known high-quality datasets...")
    for name, info in KNOWN_DATASETS.items():
        hub_id = info["hub_id"]
        if hub_id not in all_datasets:
            dataset_info = get_dataset_info(hub_id)
            if dataset_info:
                all_datasets[hub_id] = DatasetInfo(
                    hub_id=hub_id,
                    name=name,
                    description=info["description"],
                    downloads=dataset_info.get("downloads", 0),
                    likes=dataset_info.get("likes", 0),
                    tags=dataset_info.get("tags", []),
                    relevance_score=90.0,  # Known high-quality
                    category=info["category"],
                    status="known_quality",
                )

    # Sort by relevance
    sorted_datasets = sorted(
        all_datasets.values(),
        key=lambda x: x.relevance_score,
        reverse=True
    )

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_unique_datasets": len(all_datasets),
        "search_queries": SEARCH_QUERIES,
        "search_results_by_query": search_results,
        "top_datasets": [
            {
                "hub_id": ds.hub_id,
                "name": ds.name,
                "description": ds.description,
                "downloads": ds.downloads,
                "likes": ds.likes,
                "relevance_score": ds.relevance_score,
                "category": ds.category,
                "status": ds.status,
                "tags": ds.tags[:10],  # Limit tags
            }
            for ds in sorted_datasets[:50]  # Top 50
        ],
        "recommendations": {
            "high_priority": [
                ds.hub_id for ds in sorted_datasets
                if ds.relevance_score >= 70 or ds.status == "known_quality"
            ][:15],
            "medium_priority": [
                ds.hub_id for ds in sorted_datasets
                if 40 <= ds.relevance_score < 70
            ][:15],
            "for_review": [
                ds.hub_id for ds in sorted_datasets
                if ds.relevance_score < 40
            ][:10],
        },
        "categories_found": list(set(ds.category for ds in all_datasets.values() if ds.category != "unknown")),
    }

    # Save report
    report_path = output_dir / f"hf_discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Discovery complete! Found {len(all_datasets)} unique datasets")
    logger.info(f"Report saved to: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("HUGGINGFACE DATASET DISCOVERY SUMMARY")
    print("=" * 60)
    print(f"\nTotal unique datasets found: {len(all_datasets)}")
    print(f"\nHigh Priority ({len(report['recommendations']['high_priority'])}):")
    for ds_id in report['recommendations']['high_priority'][:10]:
        ds = all_datasets.get(ds_id)
        if ds:
            print(f"  • {ds_id} (score: {ds.relevance_score:.1f}, downloads: {ds.downloads:,})")

    print(f"\nMedium Priority ({len(report['recommendations']['medium_priority'])}):")
    for ds_id in report['recommendations']['medium_priority'][:5]:
        ds = all_datasets.get(ds_id)
        if ds:
            print(f"  • {ds_id} (score: {ds.relevance_score:.1f})")

    print(f"\nReport saved to: {report_path}")

    return report


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Discover HuggingFace datasets for mental health AI")
    parser.add_argument("--output-dir", type=Path, help="Output directory for discovery report")
    parser.add_argument("--query", help="Single search query (for testing)")

    args = parser.parse_args()

    if args.query:
        # Single query mode
        results = search_huggingface_datasets(args.query)
        print(json.dumps(results, indent=2))
    else:
        # Full discovery
        discover_datasets(args.output_dir)


if __name__ == "__main__":
    main()

