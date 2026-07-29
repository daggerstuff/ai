"""
PIX-3912: Hierarchy-Aware Memory Integration

Integrates the therapeutic concept hierarchy with the existing memory system:
- Memories stored at hierarchical level (not just flat)
- Query expansion: retrieve at current level + parent + sibling
- Memory consolidation: periodic clustering within hierarchy nodes

Enables "similar to what we've seen before" recall for rare presentations.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .therapeutic_concept_hierarchy import TherapeuticConceptHierarchy


@dataclass
class HierarchicalMemory:
    """A memory block anchored to a specific node in the therapeutic hierarchy."""

    memory_id: str
    content: str
    node_id: str  # anchor in the hierarchy
    level: int
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    access_count: int = 0
    last_accessed: str | None = None

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc).isoformat()


class HierarchyAwareMemoryStore:
    """
    Memory store that organizes memories by their position in the therapeutic concept hierarchy.

    Features
    --------
    - Hierarchical indexing: memories attached to nodes at any level
    - Query expansion: when querying a node, also search parent, children, and siblings
    - Consolidation: periodic clustering of memories within hierarchy nodes
    - Semantic retrieval: embedding-based similarity search
    """

    def __init__(
        self,
        hierarchy: TherapeuticConceptHierarchy,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.hierarchy = hierarchy
        self._memories: dict[str, HierarchicalMemory] = {}
        self._node_index: dict[str, list[str]] = {}  # node_id -> list of memory_ids
        self._embedding_model_name = embedding_model
        self._embedding_model: SentenceTransformer | None = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    def add_memory(
        self,
        content: str,
        node_id: str,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> HierarchicalMemory:
        """Add a memory anchored to a hierarchy node."""
        if node_id not in self.hierarchy:
            raise ValueError(f"Node {node_id} not found in hierarchy")

        node = self.hierarchy.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")

        mem_id = memory_id or str(uuid.uuid4())
        emb = self.embedding_model.encode(content, convert_to_numpy=True)

        memory = HierarchicalMemory(
            memory_id=mem_id,
            content=content,
            node_id=node_id,
            level=node.level,
            embedding=emb,
            metadata=metadata or {},
        )

        self._memories[mem_id] = memory
        self._node_index.setdefault(node_id, []).append(mem_id)
        return memory

    def get_memory(self, memory_id: str) -> HierarchicalMemory | None:
        return self._memories.get(memory_id)

    def get_memories_for_node(self, node_id: str) -> list[HierarchicalMemory]:
        """Get all memories directly attached to a node."""
        mem_ids = self._node_index.get(node_id, [])
        return [self._memories[mid] for mid in mem_ids]

    def query(
        self,
        query_text: str,
        node_id: str | None = None,
        expand: bool = True,
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> list[tuple[HierarchicalMemory, float]]:
        """
        Query memories with optional hierarchical expansion.

        Parameters
        ----------
        query_text : text to match
        node_id : optional anchor node; if provided, search within this subtree
        expand : if True, also search parent, siblings, and children
        top_k : number of results
        min_similarity : minimum cosine similarity threshold
        """
        query_emb = self.embedding_model.encode(query_text, convert_to_numpy=True)

        # Determine search scope
        search_nodes: set[str] = set()
        if node_id is not None:
            search_nodes.add(node_id)
            if expand:
                # Parent
                parent = self.hierarchy.get_parent(node_id)
                if parent is not None:
                    search_nodes.add(parent.id)
                # Siblings
                for sibling in self.hierarchy.get_siblings(node_id):
                    search_nodes.add(sibling.id)
                # Children
                for child in self.hierarchy.get_children(node_id):
                    search_nodes.add(child.id)
                # Ancestors (up to root)
                for ancestor in self.hierarchy.get_ancestors(node_id):
                    search_nodes.add(ancestor.id)
        else:
            # Search all nodes
            search_nodes = set(self._node_index.keys())

        # Collect candidate memories
        candidates: list[tuple[HierarchicalMemory, float]] = []
        for nid in search_nodes:
            for mem in self.get_memories_for_node(nid):
                if mem.embedding is None:
                    continue
                sim = float(
                    np.dot(query_emb, mem.embedding)
                    / (np.linalg.norm(query_emb) * np.linalg.norm(mem.embedding) + 1e-8)
                )
                if sim >= min_similarity:
                    mem.touch()
                    candidates.append((mem, sim))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def consolidate_node(self, node_id: str, max_clusters: int = 5) -> list[dict[str, Any]]:
        """
        Consolidate memories within a hierarchy node via simple clustering.
        Returns cluster summaries.
        """
        memories = self.get_memories_for_node(node_id)
        if len(memories) < max_clusters * 2:
            return []

        embeddings = np.stack([m.embedding for m in memories if m.embedding is not None])
        if embeddings.shape[0] < max_clusters:
            return []

        # K-means-ish clustering via random centroids for simplicity
        # In production, use sklearn.cluster.KMeans
        indices = np.random.choice(len(embeddings), size=max_clusters, replace=False)
        centroids = embeddings[indices]

        # Assign to nearest centroid
        assignments: list[list[int]] = [[] for _ in range(max_clusters)]
        for i, emb in enumerate(embeddings):
            dists = np.linalg.norm(centroids - emb, axis=1)
            nearest = int(np.argmin(dists))
            assignments[nearest].append(i)

        clusters: list[dict[str, Any]] = []
        for cluster_idx, mem_indices in enumerate(assignments):
            if not mem_indices:
                continue
            cluster_memories = [memories[i] for i in mem_indices]
            # Simple summary: most common words
            all_text = " ".join(m.content for m in cluster_memories)
            clusters.append({
                "cluster_id": f"{node_id}_cluster_{cluster_idx}",
                "node_id": node_id,
                "memory_count": len(cluster_memories),
                "sample_memories": [m.memory_id for m in cluster_memories[:3]],
                "summary_text": all_text[:200],
            })

        return clusters

    def consolidate_all(self, max_clusters_per_node: int = 5) -> dict[str, list[dict[str, Any]]]:
        """Run consolidation across all nodes with sufficient memories."""
        results: dict[str, list[dict[str, Any]]] = {}
        for node_id in self._node_index:
            clusters = self.consolidate_node(node_id, max_clusters=max_clusters_per_node)
            if clusters:
                results[node_id] = clusters
        return results

    def get_similar_cases(
        self,
        presentation: str,
        condition_id: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[HierarchicalMemory, float]]:
        """
        Retrieve past cases similar to a given presentation.
        If condition_id is provided, restrict search to that condition's subtree.
        """
        return self.query(
            query_text=presentation,
            node_id=condition_id,
            expand=True,
            top_k=top_k,
        )

    def save(self, path: str | Path) -> None:
        data = {
            "memories": {
                mid: {
                    "memory_id": m.memory_id,
                    "content": m.content,
                    "node_id": m.node_id,
                    "level": m.level,
                    "metadata": m.metadata,
                    "created_at": m.created_at,
                    "access_count": m.access_count,
                    "last_accessed": m.last_accessed,
                    "embedding": m.embedding.tolist() if m.embedding is not None else None,
                }
                for mid, m in self._memories.items()
            },
            "node_index": self._node_index,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path, hierarchy: TherapeuticConceptHierarchy) -> HierarchyAwareMemoryStore:
        data = json.loads(Path(path).read_text())
        store = cls(hierarchy=hierarchy)
        for mid, mdata in data["memories"].items():
            memory = HierarchicalMemory(
                memory_id=mdata["memory_id"],
                content=mdata["content"],
                node_id=mdata["node_id"],
                level=mdata["level"],
                embedding=np.array(mdata["embedding"]) if mdata["embedding"] else None,
                metadata=mdata.get("metadata", {}),
                created_at=mdata["created_at"],
                access_count=mdata.get("access_count", 0),
                last_accessed=mdata.get("last_accessed"),
            )
            store._memories[mid] = memory
            store._node_index.setdefault(mdata["node_id"], []).append(mid)
        return store

    def __len__(self) -> int:
        return len(self._memories)
