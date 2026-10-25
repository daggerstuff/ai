#!/usr/bin/env python3
"""Validate and optionally launch the synthetic agent-tool W&B SFT release.

The default mode is local-only. Pass --execute to upload the verified dataset
and create the W&B model and training job. Durable state prevents accidental
duplicate resource creation after an uncertain API result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "ai"))

from training.launch import wandb_sft_g04 as g04

DEFAULT_MANIFEST = ROOT / "docs/training/agent-tool-sft-2026-10-19/release-manifest.json"
DEFAULT_PACKAGE = ROOT / "ai/training/output/wandb_agent_tool_sft_20261019_v1"
DEFAULT_STATE_DIR = ROOT / "ai/training/output/wandb_agent_tool_sft_state"
# The synthetic tool release pins the campaign's base model: the one approved
# model in the g04 launcher. (g04 itself grew a per-execution override in the
# v7 execute script; this launcher has no override path.)
ALLOWED_MODEL_IDS = frozenset({g04.MODEL_ID})
CONFIG = {"assistant_turns": "all", "batch_size": "auto", "learning_rate": 0.00001}
UNCERTAIN_PHASES = {
    "artifact_upload_submitted_unknown",
    "job_creation_submitted_unknown",
}


class ToolSFTError(g04.QualificationError):
    """Raised when a synthetic tool SFT release or submission is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_tool_trajectory(row: dict[str, Any]) -> int:
    messages = row.get("messages")
    tools = row.get("tools")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ToolSFTError("trajectory must contain a system, user, and assistant message")
    if not isinstance(tools, list) or not tools:
        raise ToolSFTError("trajectory must declare its tools")
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise ToolSFTError("trajectory must begin with system and user messages")
    if messages[-1].get("role") != "assistant" or not isinstance(messages[-1].get("content"), str):
        raise ToolSFTError("trajectory must end with an assistant text response")

    declared = {
        item.get("function", {}).get("name")
        for item in tools
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }
    if None in declared or not declared:
        raise ToolSFTError("tool declarations must use OpenAI function format")

    pending: set[str] = set()
    seen_call_ids: set[str] = set()
    tool_calls = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ToolSFTError("trajectory contains an unsupported message")
        role = message["role"]
        if role == "assistant":
            calls = message.get("tool_calls", [])
            if not isinstance(calls, list):
                raise ToolSFTError("assistant tool_calls must be a list")
            for call in calls:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                call_id = call.get("id") if isinstance(call, dict) else None
                name = function.get("name") if isinstance(function, dict) else None
                if not isinstance(call_id, str) or not call_id or call_id in seen_call_ids:
                    raise ToolSFTError("assistant tool call IDs must be non-empty and unique")
                if name not in declared:
                    raise ToolSFTError("assistant called an undeclared tool")
                try:
                    arguments = json.loads(function.get("arguments", ""))
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ToolSFTError("tool arguments must be valid JSON") from exc
                if not isinstance(arguments, dict):
                    raise ToolSFTError("tool arguments must decode to an object")
                pending.add(call_id)
                seen_call_ids.add(call_id)
                tool_calls += 1
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise ToolSFTError("tool reply must match an earlier tool call")
            if not isinstance(message.get("content"), str):
                raise ToolSFTError("tool reply content must be text")
            pending.remove(call_id)
        elif not isinstance(message.get("content"), str):
            raise ToolSFTError("system and user messages must contain text")
    if pending:
        raise ToolSFTError("trajectory has tool calls without matching replies")
    return tool_calls


def _load_manifest(manifest_path: Path, model_id: str) -> tuple[dict[str, Any], Path, str]:
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolSFTError("cannot read the synthetic tool release manifest") from exc
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    digest_file = manifest_path.with_suffix(".sha256")
    if not digest_file.is_file() or digest_file.read_text(encoding="utf-8").strip() != manifest_sha:
        raise ToolSFTError("release manifest SHA-256 does not match its sidecar")
    if manifest.get("status") != "local_synthetic_research_candidate_not_uploaded":
        raise ToolSFTError("release is not the expected local synthetic research candidate")
    if model_id not in ALLOWED_MODEL_IDS or manifest.get("base_model") != model_id:
        raise ToolSFTError("release base model must match an approved Gemma 4 or Nemotron 3.5 model")
    if manifest.get("real_person_data") is not False or manifest.get("external_content") is not False:
        raise ToolSFTError("release must explicitly exclude real-person and external content")
    root = manifest_path.parent.resolve()
    return manifest, root, manifest_sha


def validate_release(manifest_path: Path, model_id: str) -> dict[str, Any]:
    manifest, root, manifest_sha = _load_manifest(manifest_path, model_id)
    split_groups: dict[str, set[str]] = {}
    split_prompts: dict[str, Counter[str]] = {}
    observed: dict[str, dict[str, Any]] = {}
    all_ids: set[str] = set()
    for split in ("train", "validation", "test"):
        descriptor = manifest.get("splits", {}).get(split)
        if not isinstance(descriptor, dict):
            raise ToolSFTError(f"release has no {split} split")
        payload = (root / descriptor["path"]).resolve()
        rowrefs = (root / descriptor["rowrefs_path"]).resolve()
        if root not in payload.parents or root not in rowrefs.parents:
            raise ToolSFTError("split paths must remain inside the release directory")
        if sha256_file(payload) != descriptor.get("sha256"):
            raise ToolSFTError(f"{split} trajectory SHA-256 mismatch")
        if sha256_file(rowrefs) != descriptor.get("rowrefs_sha256"):
            raise ToolSFTError(f"{split} row-reference SHA-256 mismatch")

        groups: set[str] = set()
        prompts: Counter[str] = Counter()
        tool_messages = 0
        rows = 0
        with payload.open(encoding="utf-8") as trajectory_stream, rowrefs.open(encoding="utf-8") as rowref_stream:
            for index, (line, ref_line) in enumerate(zip(trajectory_stream, rowref_stream, strict=True)):
                row = json.loads(line)
                ref = json.loads(ref_line)
                if hashlib.sha256(line.rstrip("\n").encode()).hexdigest() != ref.get("payload_sha256"):
                    raise ToolSFTError(f"{split} row {index} does not match its row-reference hash")
                if ref.get("split") != split or ref.get("index") != index:
                    raise ToolSFTError(f"{split} row-reference index/split mismatch")
                call_count = validate_tool_trajectory(row)
                row_id = ref.get("id")
                if not isinstance(row_id, str) or row_id in all_ids:
                    raise ToolSFTError("trajectory IDs must be unique across splits")
                all_ids.add(row_id)
                group = ref.get("group_id")
                if not isinstance(group, str) or not group:
                    raise ToolSFTError(f"{split} row reference has no group ID")
                groups.add(group)
                user_prompt = row["messages"][1]["content"]
                prompts[hashlib.sha256(user_prompt.encode()).hexdigest()] += 1
                tool_messages += call_count
                rows += 1
        if rows != descriptor.get("row_count") or tool_messages != descriptor.get("tool_call_messages"):
            raise ToolSFTError(f"{split} observed counts do not match the release manifest")
        split_groups[split] = groups
        split_prompts[split] = prompts
        observed[split] = {"rows": rows, "tool_calls": tool_messages}

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_groups[left] & split_groups[right]:
            raise ToolSFTError(f"scenario groups cross {left}/{right} splits")
    template_overlap = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        shared = split_prompts[left].keys() & split_prompts[right].keys()
        template_overlap[f"{left}_vs_{right}"] = {
            "shared_user_prompt_hashes": len(shared),
            "rows_left": sum(split_prompts[left][key] for key in shared),
            "rows_right": sum(split_prompts[right][key] for key in shared),
        }
    if any(item["shared_user_prompt_hashes"] for item in template_overlap.values()):
        raise ToolSFTError("exact user-prompt overlap across train, validation, or test splits")

    evaluation = manifest.get("heldout_evaluation", {})
    evaluation_path = (root / evaluation.get("path", "")).resolve()
    if root not in evaluation_path.parents or sha256_file(evaluation_path) != evaluation.get("sha256"):
        raise ToolSFTError("held-out evaluation file does not match its manifest digest")
    if evaluation.get("case_count") != observed["test"]["rows"]:
        raise ToolSFTError("held-out case count differs from the release test split")

    train_meta_desc = manifest["splits"]["train"]
    train_metadata = (root / train_meta_desc["metadata_path"]).resolve()
    if sha256_file(train_metadata) != train_meta_desc.get("metadata_sha256"):
        raise ToolSFTError("training metadata SHA-256 mismatch")
    metadata = json.loads(train_metadata.read_text(encoding="utf-8"))
    if metadata.get("total_len") != observed["train"]["rows"] or metadata.get("base_model") != model_id:
        raise ToolSFTError("training metadata row count or base model mismatch")
    if metadata.get("training_config") != CONFIG:
        raise ToolSFTError("training metadata config differs from the launch config")

    return {
        "release_id": manifest["release_id"],
        "base_model": model_id,
        "manifest_sha256": manifest_sha,
        "train_payload_sha256": train_meta_desc["sha256"],
        "train_rows": observed["train"]["rows"],
        "train_tool_calls": observed["train"]["tool_calls"],
        "validation_rows": observed["validation"]["rows"],
        "test_rows": observed["test"]["rows"],
        "heldout_sha256": evaluation["sha256"],
        "prompt_template_overlap": template_overlap,
        "config": CONFIG,
        "train_path": root / train_meta_desc["path"],
        "metadata_path": train_metadata,
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _load_state(path: Path, task_id: str, payload_sha: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolSFTError("cannot parse launcher state; reconcile W&B before resuming") from exc
    if not isinstance(state, dict):
        raise ToolSFTError("launcher state is not an object; reconcile W&B before resuming")
    if state.get("task_id") != task_id or state.get("payload_sha256") != payload_sha:
        raise ToolSFTError("launcher state identity mismatch")
    if state.get("phase") in UNCERTAIN_PHASES:
        raise ToolSFTError("previous remote create/upload result is uncertain; reconcile W&B before resuming")
    return state


def _verify_remote_access(api: g04.TrainingAPI, api_key: str, model_id: str) -> None:
    g04.verify_private_project(g04.ENTITY, g04.PROJECT, api_key)
    response = api.request("GET", "/preview/base-models")
    models = response if isinstance(response, list) else response.get("models", response.get("data", {}).get("models", []))
    ids = {item if isinstance(item, str) else item.get("id") or item.get("model_id") for item in models}
    if model_id not in ids:
        raise ToolSFTError("selected base model is absent from W&B's authenticated training catalog")


def execute(
    package_dir: Path,
    state_dir: Path,
    report: dict[str, Any],
    model_id: str,
    polls: int,
    interval: int,
) -> dict[str, Any]:
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise ToolSFTError("WANDB_API_KEY is unset")
    api = g04.TrainingAPI(api_key)
    _verify_remote_access(api, api_key, model_id)

    task_id = canonical_hash(
        {
            "release_id": report["release_id"],
            "manifest_sha256": report["manifest_sha256"],
            "payload_sha256": report["train_payload_sha256"],
            "config": CONFIG,
        }
    )
    state_path = state_dir / f"wandb-agent-tool-sft-{task_id}.json"
    state = _load_state(state_path, task_id, report["train_payload_sha256"])
    state.update(
        task_id=task_id,
        release_id=report["release_id"],
        manifest_sha256=report["manifest_sha256"],
        payload_sha256=report["train_payload_sha256"],
        train_rows=report["train_rows"],
        model=model_id,
        config=CONFIG,
    )

    if not state.get("training_data_url"):
        state.update(phase="artifact_upload_submitted_unknown", phase_updated_unix=int(time.time()))
        _atomic_json(state_path, state)
        import wandb

        run = None
        try:
            run = wandb.init(entity=g04.ENTITY, project=g04.PROJECT, job_type="agent-tool-sft-data-upload", mode="online")
            artifact = wandb.Artifact(
                f"agent-tool-sft-{task_id[:12]}",
                type="dataset",
                metadata={
                    "release_id": report["release_id"],
                    "payload_sha256": report["train_payload_sha256"],
                    "total_len": report["train_rows"],
                },
            )
            artifact.add_file(str(package_dir / "trajectories.jsonl"), name="trajectories.jsonl")
            artifact.add_file(str(package_dir / "metadata.json"), name="metadata.json")
            logged = run.log_artifact(artifact)
            logged.wait()
            version = getattr(logged, "version", None)
            if not isinstance(version, str):
                raise ToolSFTError("W&B artifact upload returned no version")
            state.update(
                training_data_url=(
                    f"wandb-artifact:///{g04.ENTITY}/{g04.PROJECT}/agent-tool-sft-{task_id[:12]}:{version}"
                ),
                phase="artifact_uploaded",
            )
            _atomic_json(state_path, state)
        except Exception as exc:
            raise ToolSFTError(f"W&B artifact upload outcome requires reconciliation ({type(exc).__name__})") from None
        finally:
            if run is not None:
                run.finish()

    if not state.get("model_id"):
        # Stable name plus return_existing=true makes recovery idempotent.
        state.update(phase="model_creation_submitted_unknown", phase_updated_unix=int(time.time()))
        _atomic_json(state_path, state)
        model = api.request(
            "POST",
            "/preview/models",
            {
                "entity": g04.ENTITY,
                "project": g04.PROJECT,
                "name": f"agent-tool-sft-{task_id[:12]}",
                "base_model": model_id,
                "return_existing": True,
            },
        )
        created_model_id = model.get("id") if isinstance(model, dict) else None
        if not isinstance(created_model_id, str) or not re.fullmatch(r"[0-9a-fA-F-]{36}", created_model_id):
            raise ToolSFTError("W&B model creation outcome requires reconciliation")
        state.update(model_id=created_model_id, phase="model_ready")
        _atomic_json(state_path, state)

    if not state.get("training_job_id"):
        state.update(phase="job_creation_submitted_unknown", phase_updated_unix=int(time.time()))
        _atomic_json(state_path, state)
        job = api.request(
            "POST",
            "/preview/sft-training-jobs",
            {"model_id": state["model_id"], "training_data_url": state["training_data_url"], "config": CONFIG},
        )
        job_id = job.get("id") if isinstance(job, dict) else None
        if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-fA-F-]{36}", job_id):
            raise ToolSFTError("W&B job creation outcome requires reconciliation")
        state.update(training_job_id=job_id, phase="job_created", phase_updated_unix=int(time.time()))
        _atomic_json(state_path, state)

    safe_fields = ("id", "status", "state", "created_at", "updated_at", "completed_at", "step", "total_steps")
    for _ in range(polls):
        status = api.request("GET", f"/preview/training-jobs/{state['training_job_id']}")
        if not isinstance(status, dict):
            raise ToolSFTError("W&B job status response is not an object")
        state["last_status_response"] = {key: status[key] for key in safe_fields if key in status}
        state["last_status"] = status.get("status", status.get("state", "unknown"))
        state["phase_updated_unix"] = int(time.time())
        _atomic_json(state_path, state)
        if str(state["last_status"]).lower() in {"succeeded", "failed", "cancelled", "canceled"}:
            break
        time.sleep(interval)
    return {"task_id": task_id, "phase": state.get("phase"), "training_job_id": state.get("training_job_id"), "state_path": str(state_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(ALLOWED_MODEL_IDS), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--execute", action="store_true", help="upload the verified dataset and create the W&B SFT job")
    parser.add_argument("--polls", type=int, default=0)
    parser.add_argument("--poll-interval", type=int, default=60)
    args = parser.parse_args(argv)
    try:
        report = validate_release(args.manifest, args.model)
        package_trajectories = args.package / "trajectories.jsonl"
        package_metadata = args.package / "metadata.json"
        if not package_trajectories.is_file() or sha256_file(package_trajectories) != report["train_payload_sha256"]:
            raise ToolSFTError("staged local trajectory package does not match the release train payload")
        if not package_metadata.is_file():
            raise ToolSFTError("staged local package has no metadata file")
        package_meta = json.loads(package_metadata.read_text(encoding="utf-8"))
        if (
            package_meta.get("release_id") != report["release_id"]
            or package_meta.get("release_manifest_sha256") != report["manifest_sha256"]
            or package_meta.get("train_payload_sha256") != report["train_payload_sha256"]
            or package_meta.get("total_len") != report["train_rows"]
            or package_meta.get("training_config") != CONFIG
        ):
            raise ToolSFTError("staged local metadata does not match the release and launch configuration")
        result = {
            "mode": "execute" if args.execute else "dry-run",
            "release_id": report["release_id"],
            "manifest_sha256": report["manifest_sha256"],
            "train_payload_sha256": report["train_payload_sha256"],
            "train_rows": report["train_rows"],
            "train_tool_calls": report["train_tool_calls"],
            "validation_rows": report["validation_rows"],
            "test_rows": report["test_rows"],
            "heldout_sha256": report["heldout_sha256"],
            "prompt_template_overlap": report["prompt_template_overlap"],
            "base_model": args.model,
            "config": CONFIG,
            "package_bytes": package_trajectories.stat().st_size + package_metadata.stat().st_size,
            "no_upload_or_job_created": not args.execute,
        }
        if args.execute:
            if args.polls < 0 or args.poll_interval <= 0:
                raise ToolSFTError("poll count must be non-negative and interval must be positive")
            result.update(execute(args.package, args.state_dir, report, args.model, args.polls, args.poll_interval))
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        return 0
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ToolSFTError) as exc:
        sys.stderr.write(f"wandb-agent-tool-sft: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
