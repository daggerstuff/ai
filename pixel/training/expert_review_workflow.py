"""
Expert Review Workflow (Tier 1.12)

Self-contained expert review workflow to manage:
- Creating review requests from ExpertValidationDataset
- Assigning experts to requests (round-robin / availability aware)
- Collecting reviews (scores + comments) and computing simple consensus

This module avoids deep dependencies on other validation frameworks to keep
Tier 1 lightweight and model-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
import json
import uuid

from ai.pixel.training.expert_validation_dataset import (
    ExpertValidationDataset,
    ExpertValidationExample,
)


@dataclass
class Expert:
    expert_id: str
    name: str
    email: str = ""
    specialties: List[str] = field(default_factory=list)
    max_concurrent: int = 3
    active: bool = True
    workload: int = 0


@dataclass
class Review:
    expert_id: str
    scores: Dict[str, float]  # keys: clinical_accuracy, emotional_authenticity, safety_compliance
    comments: str = ""


@dataclass
class ReviewRequest:
    request_id: str
    example_id: str
    assigned_experts: Set[str] = field(default_factory=set)
    completed: Dict[str, Review] = field(default_factory=dict)  # expert_id -> Review
    min_reviews: int = 1
    max_reviews: int = 3
    status: str = "pending"  # pending|assigned|in_review|complete

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "example_id": self.example_id,
            "assigned_experts": list(self.assigned_experts),
            "completed": {k: asdict(v) for k, v in self.completed.items()},
            "min_reviews": self.min_reviews,
            "max_reviews": self.max_reviews,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReviewRequest:
        rr = cls(
            request_id=data["request_id"],
            example_id=data["example_id"],
            assigned_experts=set(data.get("assigned_experts", [])),
            completed={},
            min_reviews=int(data.get("min_reviews", 1)),
            max_reviews=int(data.get("max_reviews", 3)),
            status=data.get("status", "pending"),
        )
        for k, v in data.get("completed", {}).items():
            rr.completed[k] = Review(expert_id=v["expert_id"], scores=v["scores"], comments=v.get("comments", ""))
        return rr


class ExpertReviewWorkflow:
    def __init__(self) -> None:
        self.experts: Dict[str, Expert] = {}
        self.requests: Dict[str, ReviewRequest] = {}
        self.example_index: Dict[str, ExpertValidationExample] = {}

    # Expert management
    def register_expert(self, expert: Expert) -> None:
        self.experts[expert.expert_id] = expert

    def list_experts(self, only_active: bool = True) -> List[Expert]:
        values = list(self.experts.values())
        return [e for e in values if e.active] if only_active else values

    # Requests
    def create_requests_from_dataset(self, dataset_path: str, min_reviews: int = 2, max_reviews: int = 3) -> List[str]:
        ds = ExpertValidationDataset.from_jsonl(Path(dataset_path))
        req_ids: List[str] = []
        # Build example index for later lookups
        self.example_index = {ex.example_id: ex for ex in ds.examples}
        for ex in ds.examples:
            rid = f"req_{ex.example_id}"
            self.requests[rid] = ReviewRequest(request_id=rid, example_id=ex.example_id, min_reviews=min_reviews, max_reviews=max_reviews)
            req_ids.append(rid)
        return req_ids

    def assign_experts_round_robin(self, min_reviewers: int = 2, max_reviewers: int = 3) -> Dict[str, List[str]]:
        active = [e for e in self.experts.values() if e.active]
        if not active:
            return {}
        active.sort(key=lambda e: (e.workload, e.expert_id))
        assignments: Dict[str, List[str]] = {}
        cursor = 0
        for req in self.requests.values():
            if len(req.assigned_experts) >= max_reviewers:
                continue
            assigned: List[str] = []
            needed = min_reviewers - len(req.assigned_experts)
            while needed > 0 and active:
                expert = active[cursor % len(active)]
                cursor += 1
                if expert.workload >= expert.max_concurrent:
                    continue
                if expert.expert_id in req.assigned_experts:
                    continue
                req.assigned_experts.add(expert.expert_id)
                expert.workload += 1
                assigned.append(expert.expert_id)
                needed -= 1
            if assigned:
                req.status = "assigned"
                assignments[req.request_id] = assigned
        return assignments

    def submit_review(self, request_id: str, expert_id: str, scores: Dict[str, float], comments: str = "") -> None:
        if request_id not in self.requests:
            raise ValueError("request not found")
        req = self.requests[request_id]
        if expert_id not in req.assigned_experts:
            raise ValueError("expert not assigned")
        if expert_id in req.completed:
            raise ValueError("review already submitted")
        req.completed[expert_id] = Review(expert_id=expert_id, scores=scores, comments=comments)
        req.status = "in_review"
        # decrease workload
        if expert_id in self.experts:
            self.experts[expert_id].workload = max(0, self.experts[expert_id].workload - 1)
        # check completion
        if len(req.completed) >= req.min_reviews:
            req.status = "complete"

    def consensus(self, request_id: str) -> Dict[str, Any]:
        req = self.requests[request_id]
        if not req.completed:
            return {"consensus": None, "reviews": {}}
        keys = ["clinical_accuracy", "emotional_authenticity", "safety_compliance"]
        agg = {k: 0.0 for k in keys}
        for r in req.completed.values():
            for k in keys:
                agg[k] += float(r.scores.get(k, 0.0))
        for k in keys:
            agg[k] /= len(req.completed)
        return {"consensus": agg, "reviews": {eid: asdict(rv) for eid, rv in req.completed.items()}}

    # Persistence helpers
    def save_state(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "experts": {k: asdict(v) for k, v in self.experts.items()},
            "requests": {k: v.to_dict() for k, v in self.requests.items()},
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_state(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        self.experts = {k: Expert(**v) for k, v in data.get("experts", {}).items()}
        self.requests = {k: ReviewRequest.from_dict(v) for k, v in data.get("requests", {}).items()}
