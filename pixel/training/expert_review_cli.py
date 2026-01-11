"""
Expert Review CLI for Tier 1.12

Commands:
- create-requests --dataset <path> --state <state.json> --min-reviews 2 --max-reviews 3
- register-experts --experts <experts.json> --state <state.json>
- assign --state <state.json> --min-reviewers 2 --max-reviewers 3
- submit --state <state.json> --request-id <id> --expert-id <id> --scores '{...}' --comments "..."
- consensus --state <state.json> --request-id <id>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from ai.pixel.training.expert_review_workflow import Expert, ExpertReviewWorkflow


def cmd_create_requests(args) -> None:
    wf = ExpertReviewWorkflow()
    if args.state and Path(args.state).exists():
        wf.load_state(args.state)
    wf.create_requests_from_dataset(args.dataset, min_reviews=args.min_reviews, max_reviews=args.max_reviews)
    wf.save_state(args.state)
    print(json.dumps({"ok": True, "requests": len(wf.requests)}))


def cmd_register_experts(args) -> None:
    wf = ExpertReviewWorkflow()
    if args.state and Path(args.state).exists():
        wf.load_state(args.state)
    experts = json.loads(Path(args.experts).read_text(encoding="utf-8"))
    for e in experts:
        wf.register_expert(Expert(**e))
    wf.save_state(args.state)
    print(json.dumps({"ok": True, "experts": len(wf.experts)}))


def cmd_assign(args) -> None:
    wf = ExpertReviewWorkflow()
    wf.load_state(args.state)
    assigned = wf.assign_experts_round_robin(min_reviewers=args.min_reviewers, max_reviewers=args.max_reviewers)
    wf.save_state(args.state)
    print(json.dumps({"ok": True, "assigned": assigned}))


def cmd_submit(args) -> None:
    wf = ExpertReviewWorkflow()
    wf.load_state(args.state)
    scores: Dict[str, float] = json.loads(args.scores)
    wf.submit_review(args.request_id, args.expert_id, scores, comments=args.comments or "")
    wf.save_state(args.state)
    print(json.dumps({"ok": True}))


def cmd_consensus(args) -> None:
    wf = ExpertReviewWorkflow()
    wf.load_state(args.state)
    result = wf.consensus(args.request_id)
    print(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser(description="Expert Review CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("create-requests")
    p1.add_argument("--dataset", required=True)
    p1.add_argument("--state", required=True)
    p1.add_argument("--min-reviews", type=int, default=2)
    p1.add_argument("--max-reviews", type=int, default=3)
    p1.set_defaults(func=cmd_create_requests)

    p2 = sub.add_parser("register-experts")
    p2.add_argument("--experts", required=True)
    p2.add_argument("--state", required=True)
    p2.set_defaults(func=cmd_register_experts)

    p3 = sub.add_parser("assign")
    p3.add_argument("--state", required=True)
    p3.add_argument("--min-reviewers", type=int, default=2)
    p3.add_argument("--max-reviewers", type=int, default=3)
    p3.set_defaults(func=cmd_assign)

    p4 = sub.add_parser("submit")
    p4.add_argument("--state", required=True)
    p4.add_argument("--request-id", required=True)
    p4.add_argument("--expert-id", required=True)
    p4.add_argument("--scores", required=True, help='JSON string, e.g. {"clinical_accuracy":0.8, ...}')
    p4.add_argument("--comments", default="")
    p4.set_defaults(func=cmd_submit)

    p5 = sub.add_parser("consensus")
    p5.add_argument("--state", required=True)
    p5.add_argument("--request-id", required=True)
    p5.set_defaults(func=cmd_consensus)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
