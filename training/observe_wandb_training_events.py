"""Privately record gradient-step progress from a W&B serverless training job."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from training.launch.wandb_sft_g04 import TrainingAPI

DEFAULT_JOB_ID = "24cc2d06-3ec3-4356-ad3a-179694dd86a7"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "ai/training/output/wandb_local_checkpoints_20261019/v4-training-progress.json"
)
PAGE_SIZE = 100
MAX_INITIAL_PAGES = 2000


def _read_state(path: Path, job_id: str) -> dict[str, Any]:
    if not path.exists():
        return {"job_id": job_id, "last_event_id": None, "latest_gradient_step": None}
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("progress state has an unexpected shape")
    if state.get("job_id") != job_id:
        raise ValueError("progress state belongs to another training job")
    if state.get("last_event_id") is not None and not isinstance(state["last_event_id"], str):
        raise ValueError("progress state has an invalid event cursor")
    return state


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _poll_events(api: TrainingAPI, job_id: str, state: dict[str, Any]) -> dict[str, Any]:
    cursor = state.get("last_event_id")
    pages = 0
    new_events = 0
    latest = state.get("latest_gradient_step")
    while pages < MAX_INITIAL_PAGES:
        query: dict[str, str | int] = {"limit": PAGE_SIZE}
        if cursor:
            query["after"] = cursor
        response = api.request(
            "GET",
            f"/preview/training-jobs/{job_id}/events?{urlencode(query)}",
        )
        if not isinstance(response, dict) or not isinstance(response.get("data"), list):
            raise ValueError("W&B events response has an unexpected shape")
        events = response["data"]
        pages += 1
        for event in events:
            if not isinstance(event, dict):
                continue
            new_events += 1
            cursor = event.get("id", cursor)
            if event.get("type") != "gradient_step":
                continue
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            step = data.get("gradient_step")
            total = data.get("num_gradient_steps")
            if isinstance(step, int) and isinstance(total, int):
                latest = {
                    "event_id": event.get("id"),
                    "gradient_step": step,
                    "total_gradient_steps": total,
                    "training_loss": data.get("loss/train"),
                    "throughput_tokens_per_second": data.get("throughput/train_executed_tok_equiv_per_s"),
                }
        has_more = bool(response.get("has_more"))
        if not events or not has_more:
            break
    else:
        raise RuntimeError("W&B event scan exceeded its safety page limit")

    state.update(
        last_event_id=cursor,
        latest_gradient_step=latest,
        last_poll_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        new_events_seen=new_events,
        event_pages_read=pages,
    )
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--max-polls", type=int, default=432)
    args = parser.parse_args()
    if args.poll_interval < 1 or args.max_polls < 1:
        parser.error("poll interval and max polls must be positive")
    token = os.environ.get("WANDB_API_KEY")
    if not token:
        parser.error("WANDB_API_KEY must be provided through the environment")
    api = TrainingAPI(token)
    state = _read_state(args.output, args.job_id)

    for poll_index in range(args.max_polls):
        state = _poll_events(api, args.job_id, state)
        try:
            status = api.request("GET", f"/preview/training-jobs/{args.job_id}")
        except Exception as exc:
            status = {"status_error": type(exc).__name__}
        if isinstance(status, dict):
            state["job_status"] = status.get("status", status.get("state"))
            state["job_updated_at"] = status.get("updated_at")
            state["job_has_error"] = bool(status.get("error"))
        _write_state(args.output, state)
        if poll_index + 1 < args.max_polls:
            time.sleep(args.poll_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
