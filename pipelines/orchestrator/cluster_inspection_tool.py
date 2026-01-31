"""
Cluster inspection and manual override tools for Phase 03 standardization.

Provides utilities for examining clusters, viewing cluster details, and allowing 
manual overrides of clustering decisions or deduplication results.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime

from .tfidf_clusterer import TFIDFClusterer, ConversationCluster, ClusteringResult
from .deduplication import ConversationDeduplicator
from .quarantine import QuarantineStore, QuarantineRecord


@dataclass
class ManualOverride:
    """Record of a manual override decision."""
    override_id: str
    original_decision: str
    new_decision: str
    reason: str
    timestamp: datetime
    operator_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClusterInspector:
    """Tools for inspecting clusters and making manual overrides."""
    
    def __init__(self, clusterer: TFIDFClusterer):
        self.clusterer = clusterer
        self.overrides: List[ManualOverride] = []
        
    def inspect_cluster(self, cluster_id: int, clustering_result: ClusteringResult) -> Dict[str, Any]:
        """Inspect detailed information about a specific cluster."""
        cluster = None
        for c in clustering_result.clusters:
            if c.cluster_id == cluster_id:
                cluster = c
                break
        
        if not cluster:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        return {
            "cluster_id": cluster.cluster_id,
            "cluster_label": cluster.cluster_label,
            "size": cluster.cluster_size,
            "centroid": cluster.centroid.tolist() if isinstance(cluster.centroid, np.ndarray) else cluster.centroid,
            "dominant_topics": cluster.dominant_topics,
            "mental_health_patterns": cluster.mental_health_patterns,
            "similarity_score": cluster.similarity_score,
            "quality_metrics": cluster.cluster_quality_metrics,
            "conversation_ids": cluster.conversations[:10],  # First 10 conversation IDs
            "full_conversation_count": len(cluster.conversations)
        }
    
    def list_clusters(self, clustering_result: ClusteringResult, 
                     sort_by: str = "size", 
                     reverse: bool = True) -> List[Dict[str, Any]]:
        """List all clusters with summary information."""
        clusters_info = []
        
        for cluster in clustering_result.clusters:
            clusters_info.append({
                "cluster_id": cluster.cluster_id,
                "cluster_label": cluster.cluster_label,
                "size": cluster.cluster_size,
                "dominant_topics": cluster.dominant_topics,
                "mental_health_patterns": cluster.mental_health_patterns,
                "similarity_score": cluster.similarity_score,
                "quality": cluster.cluster_quality_metrics.get("cohesion", 0.0)
            })
        
        # Sort clusters
        if sort_by == "size":
            clusters_info.sort(key=lambda x: x["size"], reverse=reverse)
        elif sort_by == "similarity":
            clusters_info.sort(key=lambda x: x["similarity_score"], reverse=reverse)
        elif sort_by == "quality":
            clusters_info.sort(key=lambda x: x["quality"], reverse=reverse)
        
        return clusters_info
    
    def find_cluster_by_topic(self, topic: str, clustering_result: ClusteringResult) -> List[Dict[str, Any]]:
        """Find clusters that contain a specific topic."""
        matching_clusters = []
        
        for cluster in clustering_result.clusters:
            if topic.lower() in [t.lower() for t in cluster.dominant_topics]:
                matching_clusters.append({
                    "cluster_id": cluster.cluster_id,
                    "cluster_label": cluster.cluster_label,
                    "size": cluster.cluster_size,
                    "dominant_topics": cluster.dominant_topics,
                    "similarity_score": cluster.similarity_score
                })
        
        return matching_clusters
    
    def apply_manual_override(self, override_type: str, 
                            item_id: str,
                            old_value: Any, 
                            new_value: Any, 
                            reason: str,
                            operator_id: str = "system") -> str:
        """Apply a manual override to a clustering/deduplication decision."""
        override_id = f"override_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{item_id}"
        
        override = ManualOverride(
            override_id=override_id,
            original_decision=str(old_value),
            new_decision=str(new_value),
            reason=reason,
            timestamp=datetime.now(),
            operator_id=operator_id,
            metadata={
                "override_type": override_type,
                "item_id": item_id
            }
        )
        
        self.overrides.append(override)
        
        # Log the override
        print(f"Applied manual override: {override_id} for {item_id} ({override_type})")
        
        return override_id
    
    def get_override_history(self, item_id: Optional[str] = None) -> List[ManualOverride]:
        """Get history of manual overrides, optionally filtered by item ID."""
        if item_id:
            return [o for o in self.overrides if o.metadata.get("item_id") == item_id]
        return self.overrides
    
    def list_cluster_conversations(self, cluster_id: int, clustering_result: ClusteringResult) -> List[str]:
        """List conversation IDs in a specific cluster."""
        for cluster in clustering_result.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster.conversations
        return []


class DeduplicationInspector:
    """Tools for inspecting deduplication results and allowing manual overrides."""
    
    def __init__(self, deduplicator: ConversationDeduplicator):
        self.deduplicator = deduplicator
        self.overrides: List[ManualOverride] = []
        
    def inspect_duplicate_group(self, group_index: int, duplicate_groups: List[List[str]]) -> Dict[str, Any]:
        """Inspect a specific duplicate group."""
        if group_index >= len(duplicate_groups):
            raise ValueError(f"Group index {group_index} out of range")
        
        group = duplicate_groups[group_index]
        
        return {
            "group_index": group_index,
            "group_size": len(group),
            "conversation_ids": group,
            "sample_ids": group[:5]  # Show first 5 IDs
        }
    
    def list_duplicate_groups(self, duplicate_groups: List[List[str]], 
                            min_size: int = 2) -> List[Dict[str, Any]]:
        """List all duplicate groups with summary information."""
        groups_info = []
        
        for i, group in enumerate(duplicate_groups):
            if len(group) >= min_size:
                groups_info.append({
                    "group_index": i,
                    "size": len(group),
                    "conversation_ids": group[:3],  # First 3 IDs
                    "full_list": group
                })
        
        return groups_info
    
    def apply_manual_override(self, override_type: str, 
                            item_id: str,
                            old_value: Any, 
                            new_value: Any, 
                            reason: str,
                            operator_id: str = "system") -> str:
        """Apply a manual override to a deduplication decision."""
        override_id = f"dedup_override_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{item_id}"
        
        override = ManualOverride(
            override_id=override_id,
            original_decision=str(old_value),
            new_decision=str(new_value),
            reason=reason,
            timestamp=datetime.now(),
            operator_id=operator_id,
            metadata={
                "override_type": override_type,
                "item_id": item_id
            }
        )
        
        self.overrides.append(override)
        
        # Log the override
        print(f"Applied deduplication override: {override_id} for {item_id} ({override_type})")
        
        return override_id


def create_cluster_inspection_cli():
    """Create a CLI for cluster inspection and manual overrides."""
    
    def inspect_clusters():
        """Command line interface for cluster inspection."""
        print("🔍 Cluster Inspection Tool")
        print("Commands:")
        print("  inspect <cluster_id> - Inspect details of a specific cluster")
        print("  list - List all clusters with summary")
        print("  search <topic> - Find clusters containing a specific topic")
        print("  override <type> <id> <old_val> <new_val> <reason> - Apply manual override")
        print("  history [item_id] - Show override history")
        print("  conversations <cluster_id> - List conversations in cluster")
        print("  quit - Exit tool")
        
        # This would be expanded into a full CLI, but for now we just provide the framework
        pass
    
    return inspect_clusters


# Example usage functions
def example_cluster_inspection():
    """Example of how to use the cluster inspection tools."""
    # This would typically be used with an existing clusterer instance
    print("Example cluster inspection usage:")
    print("- Initialize clusterer with TFIDFClusterer()")
    print("- Load features with clusterer.load_tfidf_features()")
    print("- Perform clustering with clusterer.cluster_conversations()")
    print("- Create inspector: inspector = ClusterInspector(clusterer)")
    print("- Inspect: details = inspector.inspect_cluster(cluster_id, result)")


def example_dedup_inspection():
    """Example of how to use the deduplication inspection tools."""
    print("Example deduplication inspection usage:")
    print("- Initialize deduplicator with ConversationDeduplicator()")
    print("- Perform deduplication: unique, result = deduplicator.deduplicate_conversations(convs)")
    print("- Create inspector: inspector = DeduplicationInspector(deduplicator)")
    print("- Inspect: details = inspector.inspect_duplicate_group(group_idx, result.duplicate_groups)")


if __name__ == "__main__":
    example_cluster_inspection()
    example_dedup_inspection()