#!/usr/bin/env python3
"""Fail-closed W&B Serverless SFT launcher for a G04-approved release.

Dry-run is the default. The launcher never transforms provider-rendered
trajectories; it checks the approved bytes and packages those bytes as the
W&B-required `trajectories.jsonl` artifact member.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

try:
    import wandb
except ImportError:
    wandb = None

API_BASE = "https://api.training.wandb.ai/v1"
HTTP_ERROR_THRESHOLD = 400
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
ENTITY = "wutang"
PROJECT = "pixelated-empathy-kan28"
TERMINAL_STATES = {"succeeded", "failed", "cancelled", "canceled"}
AUDIT_CONFIG_FIELDS = {
    "assistant_turns",
    "batch_size",
    "epochs",
    "learning_rate",
    "max_steps",
    "num_train_epochs",
    "seed",
    "warmup_ratio",
    "weight_decay",
}


class QualificationError(RuntimeError):
    """Raised when a release, route, or durable state fails a safety check."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_segment(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise QualificationError("release id contains unsafe path characters")
    return value


def _manifest_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise QualificationError(f"G04 {label} path must be a repository-relative path")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise QualificationError(f"G04 {label} path escapes the repository") from exc
    return candidate


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise QualificationError(f"G04 {label} has invalid JSONL at line {line_number}") from exc
                if not isinstance(value, dict):
                    raise QualificationError(f"G04 {label} has a non-object row at line {line_number}")
                rows.append(value)
    except OSError as exc:
        raise QualificationError(f"cannot read G04 {label}") from exc
    return rows


def _check_sha_and_rows(path: Path, expected_sha: Any, expected_rows: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise QualificationError(f"G04 {label} requires a SHA-256")
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise QualificationError(f"G04 {label} file/hash does not match the manifest")
    rows = _read_jsonl(path, label)
    if not isinstance(expected_rows, int) or len(rows) != expected_rows:
        raise QualificationError(f"G04 {label} row count does not match the manifest")
    return rows


def _check_file_identity(path: Path, expected_sha: Any, expected_rows: Any, label: str) -> None:
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise QualificationError(f"G04 {label} requires a SHA-256")
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise QualificationError(f"G04 {label} file/hash does not match the manifest")
    if not isinstance(expected_rows, int) or expected_rows < 1:
        raise QualificationError(f"G04 {label} requires a positive row count")
    try:
        with path.open("rb") as stream:
            actual_rows = sum(1 for line in stream if line.strip())
    except OSError as exc:
        raise QualificationError(f"cannot count G04 {label} rows") from exc
    if actual_rows != expected_rows:
        raise QualificationError(f"G04 {label} row count does not match the manifest")


@dataclass(frozen=True)
class Payload:
    path: Path
    sha256: str
    rows: int
    split: str


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError("cannot read G04 release manifest") from exc
    if not isinstance(manifest, dict):
        raise QualificationError("G04 release manifest must be an object")
    return manifest


def _validate_release_gates(manifest: dict[str, Any]) -> None:
    if manifest.get("status") != "approved_for_research_training":
        raise QualificationError("G04 release is not approved; no upload is allowed")
    if manifest.get("upload_gate") is not True:
        raise QualificationError("G04 upload_gate is not explicitly true; no upload is allowed")
    if manifest.get("provider") != "wandb":
        raise QualificationError("G04 manifest is not the W&B provider-specific release")
    scope = manifest.get("scope")
    research_scope = scope == "research-only" or (
        isinstance(scope, dict)
        and "research" in str(scope.get("use", "")).lower()
        and scope.get("clinical_or_production_readiness") is False
    )
    evidence = manifest.get("evidence", {})
    privacy = evidence.get("privacy", {}) if isinstance(evidence, dict) else {}
    limitations = manifest.get("limitations", privacy.get("limitations", []))
    if not research_scope or not isinstance(limitations, list) or not limitations:
        raise QualificationError("G04 manifest must declare research-only scope and limitations")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("source_candidate_sha256", ""))):
        raise QualificationError("G04 manifest has no valid source_candidate_sha256")
    release_id = manifest.get("release_id")
    if not isinstance(release_id, str):
        raise QualificationError("G04 manifest has no immutable release_id")
    _safe_segment(release_id)


def _repo_root(manifest_path: Path) -> Path:
    ancestors = (manifest_path.parent, *manifest_path.parents)
    for parent in ancestors:
        if (parent / "ai").is_dir() and (parent / "pyproject.toml").is_file():
            return parent.resolve()
    for parent in ancestors:
        if (parent / ".git").exists():
            return parent.resolve()
    roots = [parent for parent in ancestors if (parent / "pyproject.toml").is_file()]
    return (roots[-1] if roots else manifest_path.parent).resolve()


def _validate_group_evidence(manifest: dict[str, Any]) -> dict[str, Any]:
    evidence_root = manifest.get("evidence", {})
    if not isinstance(evidence_root, dict):
        raise QualificationError("G04 evidence must be an object")
    evidence = evidence_root.get("group_disjointness", {})
    if not isinstance(evidence, dict):
        raise QualificationError("G04 group-disjointness evidence must be an object")
    expected_pairs = {"train_validation": 0, "train_test": 0, "validation_test": 0}
    group_key = evidence.get("group_key")
    if evidence.get("status") != "verified" or group_key not in {"seed_group_sha256", "content_group_sha256"}:
        raise QualificationError("G04 manifest has no verified supported group-split evidence")
    if evidence.get("pairwise_intersections") != expected_pairs:
        raise QualificationError("G04 split evidence does not show pairwise group-disjoint partitions")
    evidence_path = evidence.get("evidence_path")
    evidence_sha = evidence.get("evidence_sha256")
    if not isinstance(evidence_path, str) or not re.fullmatch(r"[0-9a-f]{64}", str(evidence_sha or "")):
        raise QualificationError("G04 group-disjointness evidence requires a path and SHA-256")
    split_counts = ("train_group_count", "validation_group_count", "test_group_count")
    if any(not isinstance(evidence.get(key), int) or evidence[key] < 1 for key in split_counts):
        raise QualificationError("G04 group-disjointness evidence has invalid group counts")
    return evidence


def _validate_source_candidate(manifest: dict[str, Any], root: Path) -> None:
    source = manifest.get("source_candidate")
    if not isinstance(source, dict):
        raise QualificationError("G04 manifest requires source_candidate path, SHA-256, and row count")
    if source.get("sha256") != manifest.get("source_candidate_sha256"):
        raise QualificationError("G04 source_candidate identity disagrees with its approved hash")
    source_path = _manifest_path(root, source.get("path"), "source candidate")
    _check_file_identity(source_path, source.get("sha256"), source.get("row_count"), "source candidate")


def _load_internal_rows(manifest: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    identity = manifest.get("internal_manifest")
    if not isinstance(identity, dict):
        raise QualificationError("G04 manifest requires exact internal row-membership evidence")
    internal_path = _manifest_path(root, identity.get("path"), "internal manifest")
    return _check_sha_and_rows(internal_path, identity.get("sha256"), identity.get("row_count"), "internal manifest")


def _membership(
    rows: list[dict[str, Any]], split_key: str, label: str, group_key: str = "seed_group_sha256"
) -> tuple[dict[str, set[str]], set[str]]:
    membership: dict[str, set[str]] = {name: set() for name in ("train", "validation", "test")}
    row_ids: set[str] = set()
    for row in rows:
        split = row.get(split_key)
        row_id = row.get("candidate_row_sha256")
        group_id = row.get(group_key)
        valid_identity = (
            isinstance(row_id, str)
            and re.fullmatch(r"[0-9a-f]{64}", row_id)
            and isinstance(group_id, str)
            and re.fullmatch(r"[0-9a-f]{64}", group_id)
        )
        if split not in membership or not valid_identity:
            raise QualificationError(f"G04 {label} has invalid row/group membership fields")
        if row_id in row_ids:
            raise QualificationError(f"G04 {label} repeats a candidate row identity")
        row_ids.add(row_id)
        membership[split].add(group_id)
    return membership, row_ids


def _verify_rowrefs(
    evidence: dict[str, Any], root: Path, internal_membership: dict[str, set[str]], candidate_ids: set[str]
) -> None:
    path = _manifest_path(root, evidence["evidence_path"], "group-disjointness evidence")
    rows = _read_jsonl(path, "group-disjointness evidence")
    rowref_membership, rowref_ids = _membership(
        rows, "provisional_split", "group-disjointness rowref", evidence["group_key"]
    )
    if rowref_ids != candidate_ids or rowref_membership != internal_membership:
        raise QualificationError("G04 rowref evidence does not match exact internal row/group memberships")
    if sha256_file(path) != evidence["evidence_sha256"]:
        raise QualificationError("G04 group-disjointness evidence file/hash does not match the manifest")


def _verify_group_intersections(membership: dict[str, set[str]]) -> None:
    intersections = (
        membership["train"] & membership["validation"],
        membership["train"] & membership["test"],
        membership["validation"] & membership["test"],
    )
    if any(intersections):
        raise QualificationError("G04 internal row memberships are not group-disjoint")


def _validate_partitions(
    manifest: dict[str, Any],
    root: Path,
    evidence: dict[str, Any],
    internal_rows: list[dict[str, Any]],
    membership: dict[str, set[str]],
) -> dict[str, Any]:
    partitions = manifest.get("partitions")
    names = ("train", "validation", "test")
    if not isinstance(partitions, dict) or not all(name in partitions for name in names):
        raise QualificationError("G04 manifest must include train, validation, and test partition identities")
    for name in names:
        part = partitions[name]
        if not isinstance(part, dict) or part.get("split", name) != name:
            raise QualificationError("G04 partition entries must explicitly identify their split")
        group_count = part.get("source_group_count")
        if not isinstance(group_count, int) or group_count < 1:
            raise QualificationError(f"G04 {name} partition has no source_group_count")
        if group_count != evidence[f"{name}_group_count"] or group_count != len(membership[name]):
            raise QualificationError(f"G04 {name} partition group count disagrees with membership evidence")
        path_value, digest, row_count = part.get("path"), part.get("sha256"), part.get("row_count")
        if (
            not isinstance(path_value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(digest or ""))
            or not isinstance(row_count, int)
        ):
            raise QualificationError(f"G04 {name} partition has incomplete path/hash/row-count identity")
        path = _manifest_path(root, path_value, f"{name} partition")
        payload_rows = _check_sha_and_rows(path, digest, row_count, f"{name} partition")
        expected_count = sum(row["split"] == name for row in internal_rows)
        if len(payload_rows) != expected_count:
            raise QualificationError(f"G04 {name} payload row count disagrees with internal row membership")
    return partitions


def _verify_heldout_exclusions(evidence: dict[str, Any], root: Path, membership: dict[str, set[str]]) -> None:
    path_value = evidence.get("heldout_exclusions_path")
    digest = evidence.get("heldout_exclusions_sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise QualificationError("G04 heldout-group exclusion evidence is required")
    path = _manifest_path(root, path_value, "heldout exclusion evidence")
    if not path.is_file() or sha256_file(path) != digest:
        raise QualificationError("G04 heldout exclusion evidence file/hash does not match the manifest")
    heldout_rows = _read_jsonl(path, "heldout exclusion evidence")
    heldout_groups = {row.get(evidence["group_key"]) for row in heldout_rows}
    all_groups = set.union(*membership.values())
    intersections = {name: len(heldout_groups & group_ids) for name, group_ids in membership.items()}
    reserved = evidence.get("reserved_eval_group_count", 0)
    if len(heldout_groups) > reserved or any(
        not isinstance(group, str) or group in all_groups for group in heldout_groups
    ):
        raise QualificationError("G04 heldout groups are inconsistent with row memberships or reservation count")
    if intersections != evidence.get("heldout_group_intersections", intersections):
        raise QualificationError("G04 heldout intersection evidence disagrees with row memberships")


def _selected_train_payload(manifest: dict[str, Any], partitions: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm == "baseline":
        raw = partitions["train"]
    elif arm == "data-ablation":
        ablations = manifest.get("ablations", {})
        raw = ablations.get("data_ablation", {}).get("train") if isinstance(ablations, dict) else None
    else:
        raise QualificationError("unknown experiment arm")
    if not isinstance(raw, dict):
        raise QualificationError("G04 manifest has no explicit approved train payload for this arm")
    if raw.get("split", "train") != "train":
        raise QualificationError("only the explicitly approved train split may be uploaded")
    return raw


def _validate_train_rows(payload_path: Path, expected_rows: Any) -> int:
    rows = 0
    assistant_targets = 0
    with payload_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QualificationError(f"train payload has invalid JSONL at line {line_number}") from exc
            messages = row.get("messages") if isinstance(row, dict) else None
            if not isinstance(messages, list) or not messages:
                raise QualificationError(f"train payload row {line_number} has no messages list")
            roles: list[str] = []
            for message in messages:
                if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
                    raise QualificationError(f"train payload row {line_number} has a non-rendered role")
                if not isinstance(message.get("content"), str) or not message["content"].strip():
                    raise QualificationError(f"train payload row {line_number} has empty/non-text content")
                expected_weight = 0 if message["role"] == "user" else 1
                if message.get("weight") != expected_weight:
                    raise QualificationError(f"train payload row {line_number} has an invalid assistant/user loss mask")
                roles.append(message["role"])
            if roles[-1] != "assistant":
                raise QualificationError(f"train payload row {line_number} does not end with assistant target")
            assistant_targets += sum(role == "assistant" for role in roles)
            rows += 1
    if rows == 0 or assistant_targets == 0:
        raise QualificationError("train payload has no assistant supervision")
    if not isinstance(expected_rows, int) or rows != expected_rows:
        raise QualificationError("train payload row count does not match G04 manifest")
    return rows


def resolve_train_payload(manifest_path: Path, arm: str = "baseline") -> tuple[dict[str, Any], Payload]:
    manifest = _load_manifest(manifest_path)
    _validate_release_gates(manifest)
    root = _repo_root(manifest_path)
    evidence = _validate_group_evidence(manifest)
    _validate_source_candidate(manifest, root)
    internal_rows = _load_internal_rows(manifest, root)
    membership, candidate_ids = _membership(internal_rows, "split", "internal manifest", evidence["group_key"])
    _verify_rowrefs(evidence, root, membership, candidate_ids)
    _verify_group_intersections(membership)
    partitions = _validate_partitions(manifest, root, evidence, internal_rows, membership)
    _verify_heldout_exclusions(evidence, root, membership)

    selected = _selected_train_payload(manifest, partitions, arm)
    path_value = selected.get("path", selected.get("file"))
    digest = selected.get("sha256", selected.get("file_sha256"))
    row_count = selected.get("row_count", selected.get("rows"))
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise QualificationError("train payload requires path and SHA-256 in G04 manifest")
    payload_path = _manifest_path(root, path_value, "train payload")
    if not payload_path.is_file() or sha256_file(payload_path) != digest:
        raise QualificationError("G04 train payload SHA-256 does not match the manifest")
    rows = _validate_train_rows(payload_path, row_count)
    return manifest, Payload(payload_path, digest, rows, "train")


BASELINE_CONFIG: dict[str, Any] = {
    "batch_size": "auto",
    "learning_rate": 0.0001,
    "assistant_turns": "all",
}

# A single composition ablation uses the independently approved provider view
# supplied by G04. It intentionally shares every training hyperparameter.
DATA_ABLATION_CONFIG: dict[str, Any] = {
    "batch_size": "auto",
    "learning_rate": 0.0001,
    "assistant_turns": "all",
}


class TrainingAPI:
    """Authenticated REST client using aiohttp; errors never expose response bodies."""

    def __init__(self, token: str, opener: Callable[..., Any] | None = None):
        self._token = token
        self._opener = opener

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
        if self._opener is None:
            return asyncio.run(self._request_aiohttp(method, path, body))
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{API_BASE}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with self._opener(request, timeout=60) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise QualificationError(f"W&B Training API returned HTTP {exc.code} for {method} {path}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise QualificationError(
                f"W&B Training API request failed for {method} {path}: {type(exc).__name__}"
            ) from None
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError, json.JSONDecodeError:
            raise QualificationError(f"W&B Training API returned invalid JSON for {method} {path}") from None
        if not isinstance(parsed, (dict, list)):
            raise QualificationError(f"W&B Training API returned an unexpected response for {method} {path}")
        return parsed

    async def _request_aiohttp(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any] | list[Any]:
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.request(method, f"{API_BASE}{path}", json=body, headers=headers) as response,
            ):
                if response.status >= HTTP_ERROR_THRESHOLD:
                    raise QualificationError(f"W&B Training API returned HTTP {response.status} for {method} {path}")
                raw = await response.read()
        except QualificationError:
            raise
        except (aiohttp.ClientError, TimeoutError, OSError) as exc:
            raise QualificationError(
                f"W&B Training API request failed for {method} {path}: {type(exc).__name__}"
            ) from None
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError, json.JSONDecodeError:
            raise QualificationError(f"W&B Training API returned invalid JSON for {method} {path}") from None
        if not isinstance(parsed, (dict, list)):
            raise QualificationError(f"W&B Training API returned an unexpected response for {method} {path}")
        return parsed


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temp, 0o600)
    temp.replace(path)
    os.chmod(path, 0o600)


def safe_config_capture(response: dict[str, Any]) -> dict[str, Any]:
    """Keep only known non-secret scalar training settings from API responses."""
    captured: dict[str, Any] = {}
    for container in ("config", "resolved_config", "training_config", "hyperparameters"):
        values = response.get(container)
        if not isinstance(values, dict):
            continue
        for key in AUDIT_CONFIG_FIELDS:
            if key not in values:
                continue
            value = values[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                captured[key] = value
    return captured


def make_training_artifact(
    payload: Payload, manifest: dict[str, Any], config: dict[str, Any], out_dir: Path
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    trajectory_file = out_dir / "trajectories.jsonl"
    shutil.copyfile(payload.path, trajectory_file)
    os.chmod(trajectory_file, 0o600)
    metadata = {
        "schema_version": 1,
        "source": "G04-approved provider-rendered training split",
        "release_id": manifest["release_id"],
        "source_candidate_sha256": manifest["source_candidate_sha256"],
        "release_manifest_sha256": canonical_hash(manifest),
        "train_payload_sha256": payload.sha256,
        "train_row_count": payload.rows,
        # Serverless SFT reads this count from the artifact metadata file.
        "total_len": payload.rows,
        "loss_mask": "assistant turns only",
        "assistant_turns_config": config["assistant_turns"],
        "base_model": MODEL_ID,
        "training_config": config,
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(metadata_path, 0o600)
    return {
        "trajectory_sha256": sha256_file(trajectory_file),
        "metadata_sha256": sha256_file(metadata_path),
        "artifact_bytes": trajectory_file.stat().st_size + metadata_path.stat().st_size,
        "metadata": metadata,
    }


def verify_private_project(entity: str, project: str, api_key: str) -> None:
    query = {
        "query": (
            "query ProjectAccess($entityName: String!, $name: String!) { "
            "project(entityName: $entityName, name: $name) { access } }"
        ),
        "variables": {"entityName": entity, "name": project},
    }
    encoded_auth = base64.b64encode(f"api:{api_key}".encode()).decode("ascii")

    async def fetch_access() -> dict[str, Any]:
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(
                    "https://api.wandb.ai/graphql",
                    json=query,
                    headers={"Authorization": f"Basic {encoded_auth}"},
                ) as response,
            ):
                if response.status >= HTTP_ERROR_THRESHOLD:
                    raise QualificationError("cannot verify W&B project privacy (HTTP error)")
                result = await response.json()
        except QualificationError:
            raise
        except Exception as exc:
            raise QualificationError(f"cannot verify W&B project privacy ({type(exc).__name__})") from None
        if not isinstance(result, dict):
            raise QualificationError("cannot verify W&B project privacy (unexpected response)")
        return result

    result = asyncio.run(fetch_access())
    access = result.get("data", {}).get("project", {}).get("access")
    if access != "PRIVATE":
        raise QualificationError("W&B project is not confirmed PRIVATE; refusing upload/job creation")


@dataclass(frozen=True)
class ExecutionRequest:
    manifest_path: Path
    arm: str
    state_dir: Path
    work_dir: Path
    api: TrainingAPI
    api_key: str
    private_check: Callable[[str, str, str], None] = verify_private_project
    artifact_upload: Callable[[Path, str], str] | None = None
    poll_interval: float = 30
    max_polls: int = 1


@dataclass(frozen=True)
class TrainingPlan:
    manifest: dict[str, Any]
    payload: Payload
    config: dict[str, Any]
    artifact: dict[str, Any]
    task_id: str


def _load_state(path: Path, task_id: str, payload_sha256: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError("cannot read launcher state; refusing to create duplicate resources") from exc
    if state.get("task_id") != task_id or state.get("payload_sha256") != payload_sha256:
        raise QualificationError("launcher state identity mismatch")
    uncertain_phases = {
        "job_creation_submitted_unknown",
        "model_creation_submitted_unknown",
        "artifact_upload_submitted_unknown",
    }
    if state.get("phase") in uncertain_phases:
        raise QualificationError("previous create/upload outcome is unknown; reconcile W&B before resuming")
    return state


def _verify_remote_access(request: ExecutionRequest) -> None:
    request.private_check(ENTITY, PROJECT, request.api_key)
    response = request.api.request("GET", "/preview/base-models")
    if isinstance(response, list):
        catalog = response
    else:
        catalog = response.get("models", response.get("data", {}).get("models", []))
    ids = {
        item if isinstance(item, str) else item.get("id") or item.get("model_id")
        for item in catalog
        if isinstance(item, (str, dict))
    }
    if MODEL_ID not in ids:
        raise QualificationError("selected base model is absent from current authenticated training catalog")


def _initial_state(request: ExecutionRequest, plan: TrainingPlan) -> dict[str, Any]:
    manifest = plan.manifest
    payload = plan.payload
    config = plan.config
    artifact = plan.artifact
    return {
        "task_id": plan.task_id,
        "phase": "preflight_passed",
        "release_id": manifest["release_id"],
        "manifest_sha256": sha256_file(request.manifest_path),
        "payload_sha256": payload.sha256,
        "row_count": payload.rows,
        "arm": request.arm,
        "model": MODEL_ID,
        "config": config,
        "artifact_hashes": {"trajectories": artifact["trajectory_sha256"], "metadata": artifact["metadata_sha256"]},
        "artifact_bytes": artifact["artifact_bytes"],
        "phase_updated_unix": int(time.time()),
    }


def _upload_artifact(request: ExecutionRequest, state: dict[str, Any], plan: TrainingPlan) -> str:
    existing_uri = state.get("training_data_url")
    if existing_uri is not None:
        return existing_uri
    if request.artifact_upload is None:
        raise QualificationError("no W&B artifact uploader configured")
    state["phase"] = "artifact_upload_submitted_unknown"
    state["phase_updated_unix"] = int(time.time())
    atomic_json(request.state_dir / f"wandb-sft-{plan.task_id}.json", state)
    artifact_name = f"g04-{plan.manifest['release_id']}-{request.arm}-{plan.task_id[:12]}"
    artifact_uri = request.artifact_upload(request.work_dir / plan.manifest["release_id"] / request.arm, artifact_name)
    if not artifact_uri.startswith("wandb-artifact:///"):
        raise QualificationError("artifact uploader returned an invalid W&B Training artifact URI")
    state["training_data_url"] = artifact_uri
    state["phase"] = "artifact_uploaded"
    atomic_json(request.state_dir / f"wandb-sft-{plan.task_id}.json", state)
    return artifact_uri


def _create_model(request: ExecutionRequest, state: dict[str, Any], task_id: str) -> str:
    model_id = state.get("model_id")
    if model_id is not None:
        return model_id
    state["phase"] = "model_creation_submitted_unknown"
    state_path = request.state_dir / f"wandb-sft-{task_id}.json"
    atomic_json(state_path, state)
    model = request.api.request(
        "POST",
        "/preview/models",
        {
            "entity": ENTITY,
            "project": PROJECT,
            "name": f"arc-sft-{request.arm}-{task_id[:12]}",
            "base_model": MODEL_ID,
            "return_existing": True,
        },
    )
    model_id = model.get("id")
    if not isinstance(model_id, str) or not re.fullmatch(r"[0-9a-fA-F-]{36}", model_id):
        raise QualificationError("W&B model creation response did not contain a model UUID")
    state.update(model_id=model_id, phase="model_ready")
    atomic_json(state_path, state)
    return model_id


def _create_job(
    request: ExecutionRequest,
    state: dict[str, Any],
    plan: TrainingPlan,
    model_id: str,
    artifact_uri: str,
) -> str:
    job_id = state.get("training_job_id")
    if job_id is not None:
        return job_id
    state["phase"] = "job_creation_submitted_unknown"
    state_path = request.state_dir / f"wandb-sft-{plan.task_id}.json"
    atomic_json(state_path, state)
    job = request.api.request(
        "POST",
        "/preview/sft-training-jobs",
        {"model_id": model_id, "training_data_url": artifact_uri, "config": plan.config},
    )
    job_id = job.get("id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-fA-F-]{36}", job_id):
        raise QualificationError("W&B SFT response did not contain a training job UUID")
    state["training_job_id"] = job_id
    resolved = safe_config_capture(job)
    if resolved:
        state["provider_reported_config"] = resolved
    state["phase"] = "job_created"
    atomic_json(state_path, state)
    return job_id


def _poll_job(request: ExecutionRequest, state: dict[str, Any], job_id: str, task_id: str) -> None:
    state_path = request.state_dir / f"wandb-sft-{task_id}.json"
    safe_fields = ("id", "status", "state", "created_at", "updated_at", "completed_at", "step", "total_steps")
    for poll_index in range(request.max_polls):
        status = request.api.request("GET", f"/preview/training-jobs/{job_id}")
        state["last_status"] = status.get("status", status.get("state", "unknown"))
        state["last_status_response"] = {key: status[key] for key in safe_fields if key in status}
        resolved = safe_config_capture(status)
        if resolved:
            state["provider_reported_config"] = resolved
        state["phase_updated_unix"] = int(time.time())
        if state["last_status"] in TERMINAL_STATES:
            state["phase"] = state["last_status"]
        atomic_json(state_path, state)
        if state["last_status"] in TERMINAL_STATES:
            return
        if request.poll_interval and poll_index + 1 < request.max_polls:
            time.sleep(request.poll_interval)


def execute(request: ExecutionRequest) -> dict[str, Any]:
    config = BASELINE_CONFIG if request.arm == "baseline" else DATA_ABLATION_CONFIG
    manifest, payload = resolve_train_payload(request.manifest_path, request.arm)
    release_id = _safe_segment(manifest["release_id"])
    work_path = request.work_dir / release_id / request.arm
    artifact = make_training_artifact(payload, manifest, config, work_path)
    task_id = canonical_hash(
        {"release": manifest["release_id"], "payload": payload.sha256, "arm": request.arm, "config": config}
    )
    plan = TrainingPlan(manifest, payload, config, artifact, task_id)
    state_path = request.state_dir / f"wandb-sft-{task_id}.json"
    state = _load_state(state_path, task_id, payload.sha256)
    _verify_remote_access(request)
    if not state:
        state = _initial_state(request, plan)
        atomic_json(state_path, state)
    artifact_uri = _upload_artifact(request, state, plan)
    model_id = _create_model(request, state, task_id)
    job_id = _create_job(request, state, plan, model_id, artifact_uri)
    _poll_job(request, state, job_id, task_id)
    return {"task_id": task_id, "phase": state["phase"], "training_job_id": job_id, "state_path": str(state_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "data-ablation"), default="baseline")
    parser.add_argument("--output", type=Path, default=Path("docs/training/execution-2026-10-17/wandb-sft-artifacts"))
    parser.add_argument("--state-dir", type=Path, default=Path("docs/training/execution-2026-10-17/wandb-sft-state"))
    parser.add_argument("--execute", action="store_true", help="upload G04 data and create/monitor the SFT job")
    parser.add_argument("--polls", type=int, default=1)
    parser.add_argument("--poll-interval", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        manifest, payload = resolve_train_payload(args.manifest, args.arm)
        config = BASELINE_CONFIG if args.arm == "baseline" else DATA_ABLATION_CONFIG
        task_id = canonical_hash(
            {"release": manifest["release_id"], "payload": payload.sha256, "arm": args.arm, "config": config}
        )
        if not args.execute:
            prepared = make_training_artifact(
                payload, manifest, config, args.output / manifest["release_id"] / args.arm
            )
            report = {
                "mode": "dry-run",
                "task_id": task_id,
                "release_id": manifest["release_id"],
                "payload_sha256": payload.sha256,
                "rows": payload.rows,
                "assistant_loss_mask": "assistant only; all assistant turns",
                "model": MODEL_ID,
                "config": config,
                "artifact_bytes": prepared["artifact_bytes"],
                "storage_estimate": (
                    "Exact dataset artifact bytes are measured. W&B documents at least 5 GB free storage for all plans "
                    "and says this is typically enough for about 30 LoRAs. Exact account headroom and this model's "
                    "checkpoint bytes are not exposed by the checked read-only APIs."
                ),
                "training_price": (
                    "$0 serverless adapter training during public preview; "
                    "confirm preview eligibility on execution day."
                ),
                "no_upload_or_job_created": True,
            }
            sys.stdout.write(json.dumps(report, indent=2) + "\n")
            return 0
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            raise QualificationError("WANDB_API_KEY is unset")
        if wandb is None:
            raise QualificationError("wandb SDK is required for artifact upload")

        def upload(directory: Path, artifact_name: str) -> str:
            run = None
            try:
                run = wandb.init(entity=ENTITY, project=PROJECT, job_type="sft-data-upload", mode="online")
                artifact = wandb.Artifact(
                    artifact_name,
                    type="dataset",
                    metadata={
                        "release_id": manifest["release_id"],
                        "payload_sha256": payload.sha256,
                        "total_len": payload.rows,
                    },
                )
                artifact.add_file(str(directory / "trajectories.jsonl"), name="trajectories.jsonl")
                artifact.add_file(str(directory / "metadata.json"), name="metadata.json")
                logged = run.log_artifact(artifact)
                logged.wait()
                version = getattr(logged, "version", "v0")
                return f"wandb-artifact:///{ENTITY}/{PROJECT}/{artifact_name}:{version}"
            except Exception as exc:
                raise QualificationError(
                    f"W&B artifact upload failed ({type(exc).__name__}); reconcile artifact before resuming"
                ) from None
            finally:
                if run is not None:
                    run.finish()

        api = TrainingAPI(api_key)
        result = execute(
            ExecutionRequest(
                manifest_path=args.manifest,
                arm=args.arm,
                state_dir=args.state_dir,
                work_dir=args.output,
                api=api,
                api_key=api_key,
                artifact_upload=upload,
                max_polls=args.polls,
                poll_interval=args.poll_interval,
            )
        )
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        return 0
    except QualificationError as exc:
        sys.stderr.write(f"qualification failed: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
