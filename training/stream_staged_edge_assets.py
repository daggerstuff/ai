#!/usr/bin/env python3
"""Stream + reconcile staged edge/nightmare cloud assets against the master gold.

Streams JSONL/CSV/MD assets from gdrive + whitebat via ``rclone cat`` (never
bulk-downloads multi-hundred-MB files), normalizes each record into a canonical
scenario, maps it onto the 10-family edge-case taxonomy, dedups against the local
master gold by SHA-256 (primary) + SHA-1 (secondary), and emits a reconciliation
manifest of scenarios *not yet in master* plus a per-asset/per-family summary.

Authoritative taxonomy: ``scripts/data/designer/configs/edge_cases.py`` (10 families).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pipelines.data_processing.extractors.s3_streamer import S3Streamer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("stream_staged_edge_assets")

_AI_ROOT = Path(__file__).resolve().parents[1]
MASTER_GOLD = _AI_ROOT / "data" / "curated" / "sft_chatml" / "train_master_gold.jsonl"
DEFAULT_MANIFEST = _AI_ROOT / "training" / "output" / "nightmare_fuel" / "staged_edge_reconciliation.jsonl"

# 10 authoritative edge families (mirrors scripts/data/designer/configs/edge_cases.py).
FAMILIES: tuple[str, ...] = (
    "ambiguous crisis language",
    "delusion or paranoia",
    "coercion or abuse",
    "substance use",
    "medical uncertainty",
    "minor or dependent person",
    "therapeutic rupture",
    "cultural or identity conflict",
    "boundary testing",
    "multi-problem complexity",
)

# Keyword rules for first-pass family classification. Structured metadata fields
# (category / diagnostic_tag / scenario_type / tags) are folded into the text before
# classification, so a record's own labels win over body-text co-occurrence.
FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ambiguous crisis language": (
        "suicid", "self-harm", "self harm", "kill myself", "end my life", "give up",
        "death", "crisis", "overdose", "hopeless", "harm myself",
    ),
    "delusion or paranoia": (
        "paranoi", "delusion", "persecut", "monitoring", "watching me", "conspir",
        "following me", "psychotic", "psychosis", "hallucin",
    ),
    "coercion or abuse": (
        "abuse", "coerc", "traffick", "domestic violence", "entrap", "gaslight",
        "sexual assault", "sexual trauma", "ritual abuse", "control",
    ),
    "substance use": (
        "substance", "relapse", "intoxicat", "alcohol", "opioid", "drug", "withdraw",
        "craving", "addiction", "drinking",
    ),
    "medical uncertainty": (
        "medical", "diagnosis", "medication", "refus", "bradycardia", "refeeding",
        "somatic", "symptom", "hospital", "eating disorder",
    ),
    "minor or dependent person": (
        "minor", "child", "dependent", "cps", "mandated report", "adolescent",
        "teen", "guardian", "custody",
    ),
    "therapeutic rupture": (
        "rupture", "alliance", "quit therapy", "idealiz", "devalu", "splitting",
        "transference", "threat to leave", "borderline",
    ),
    "cultural or identity conflict": (
        "cultural", "religious", "identity", "lgbtq", "faith", "belief", "immigrant",
        "ethnic", "extremism",
    ),
    "boundary testing": (
        "boundary", "malinger", "manipulat", "prescription", "extortion", "sign-off",
        "disability doc", "dual relationship", "advance",
    ),
    "multi-problem complexity": (
        "multiple", "comorbid", "co-morbid", "intersecting", "complex", "polysubstance",
        "dual diagnosis", "multi-problem", "everything at once",
    ),
}

# Records whose content is an inert placeholder rather than a real scenario.
PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "this is a simulated llm response for testing purposes",
    "lorem ipsum",
    "placeholder",
)

# Cloud assets to stream. ``kind`` selects the parser; ``remote``/``bucket`` select
# the rclone remote (S3Streamer with an empty prefix builds ``remote:bucket/key``).
ASSETS: list[dict[str, str]] = [
    {"name": "edge_integration_plan", "kind": "md", "remote": "gdrive", "bucket": "pixeldata",
     "key": "archive/chad_drive_imported/datasets_3/consolidated/edge_cases/EDGE_CASE_INTEGRATION_PLAN.md"},
    {"name": "stage3_edge_stress_test", "kind": "jsonl", "remote": "gdrive", "bucket": "pixeldata",
     "key": "staged_datasets/stage3_edge_stress_test.jsonl"},
    {"name": "master_stage3_edge_stress_test", "kind": "jsonl", "remote": "gdrive", "bucket": "pixeldata",
     "key": "training_data_consolidated/final/MASTER_stage3_edge_stress_test.jsonl"},
    {"name": "nightmare_scenarios_batch", "kind": "jsonl", "remote": "gdrive", "bucket": "pixeldata",
     "key": "datasets/training_v2/stage3_edge_crisis/nightmare_scenarios_batch_241.jsonl"},
    {"name": "clinical_redteam", "kind": "jsonl", "remote": "whitebat", "bucket": "whitebat",
     "key": "training/ai-data/raw/clinical_redteam/clinical_redteam.jsonl"},
    {"name": "clinical_redteam_conversations", "kind": "csv", "remote": "whitebat", "bucket": "whitebat",
     "key": "training/ai-data/raw/clinical_redteam/raw/conversations.csv"},
    {"name": "safety_dpo_pairs_10k", "kind": "jsonl", "remote": "gdrive", "bucket": "pixeldata",
     "key": "training/v1/stage3_stress_test/processed/safety_dpo_pairs_10k.jsonl"},
]

ROLE_MAP = {
    "therapist": "assistant", "assistant": "assistant", "clinician": "assistant",
    "counselor": "assistant", "sys": "system", "system": "system",
    "patient": "user", "user": "user", "client": "user", "speaker": "user",
}

MIN_MESSAGES = 2


def _streamer(remote: str, bucket: str) -> S3Streamer:
    return S3Streamer(remote=remote, bucket=bucket, prefix="")


def _iter_jsonl(asset: dict[str, str], limit: int | None) -> Iterator[dict]:
    streamer = _streamer(asset["remote"], asset["bucket"])
    seen = 0
    for record in streamer.stream_jsonl(asset["key"]):
        if isinstance(record, dict):
            yield record
            seen += 1
            if limit is not None and seen >= limit:
                return


def _iter_csv(asset: dict[str, str], limit: int | None) -> Iterator[dict]:
    """Stream a CSV via rclone cat and yield one grouped session dict per row batch."""
    cmd = ["rclone", "cat", f"{asset['remote']}:{asset['bucket']}/{asset['key']}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    assert proc.stdout is not None
    reader = csv.DictReader(proc.stdout)
    sessions: dict[str, list[dict]] = {}
    order: list[str] = []
    for row in reader:
        sid = row.get("session_id") or row.get("pairing_id") or "unknown"
        speaker = (row.get("speaker") or "").strip().lower()
        message = (row.get("message") or "").strip()
        if not message:
            continue
        if sid not in sessions:
            sessions[sid] = []
            order.append(sid)
        sessions[sid].append({"role": ROLE_MAP.get(speaker, "user"), "content": message})
        if limit is not None and len(order) > limit:
            break
    proc.wait()
    for sid in order[:limit] if limit is not None else order:
        yield {"session_id": sid, "messages": sessions[sid]}


def _iter_md(asset: dict[str, str]) -> Iterator[str]:
    cmd = ["rclone", "cat", f"{asset['remote']}:{asset['bucket']}/{asset['key']}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line.rstrip("\n")
    proc.wait()


def _extract_messages(raw: dict) -> list[dict[str, str]]:
    """Return a normalized ``[{role, content}, ...]`` list, or empty if no text found."""
    raw_messages = raw.get("messages")
    if isinstance(raw_messages, list):
        out: list[dict[str, str]] = []
        for m in raw_messages:
            if not isinstance(m, dict):
                continue
            role = ROLE_MAP.get(str(m.get("role", "")).strip().lower(), "user")
            content = str(m.get("content", "")).strip()
            if content:
                out.append({"role": role, "content": content})
        if len(out) >= MIN_MESSAGES:
            return out

    user = str(raw.get("prompt") or raw.get("instruction") or "").strip()
    assistant = str(raw.get("response") or raw.get("text") or raw.get("chosen") or "").strip()
    out = []
    if user:
        out.append({"role": "user", "content": user})
    if assistant:
        out.append({"role": "assistant", "content": assistant})
    return out


def _classification_text(raw: dict, messages: list[dict[str, str]]) -> str:
    labels = " ".join(
        str(raw.get(k) or "")
        for k in ("category", "diagnostic_tag", "scenario_type", "task_type", "intensity")
    )
    tags = raw.get("tags")
    if isinstance(tags, list):
        labels += " " + " ".join(str(t) for t in tags)
    metadata = raw.get("metadata")
    if isinstance(metadata, dict):
        labels += " " + str(metadata.get("category") or "")
    body = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return f"{labels}\n{body}".lower()


def classify_family(text: str) -> str:
    scores = {family: sum(1 for kw in keywords if kw in text) for family, keywords in FAMILY_KEYWORDS.items()}
    best = max(scores, key=lambda f: scores[f])
    return best if scores[best] > 0 else "unmapped"


def _canonical_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages).strip().lower()


def content_hashes(messages: list[dict[str, str]]) -> tuple[str, str]:
    canonical = _canonical_text(messages).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), hashlib.sha1(canonical).hexdigest()


def load_master_hashes(master_path: Path, limit: int | None) -> set[str]:
    hashes: set[str] = set()
    seen = 0
    with master_path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            messages = _extract_messages(raw)
            if not messages:
                continue
            hashes.add(content_hashes(messages)[0])
            seen += 1
            if limit is not None and seen >= limit:
                break
    return hashes


def _looks_placeholder(text: str) -> bool:
    t = text.strip().lower()
    return any(marker in t for marker in PLACEHOLDER_MARKERS)


def normalize_scenario(asset_name: str, raw: dict) -> dict | None:
    """Normalize one raw record into a canonical scenario, or None when it's a
    placeholder / empty. DPO pairs carry both chosen + rejected and route to Stage 5."""
    is_dpo = "chosen" in raw and "rejected" in raw and "prompt" in raw
    messages = _extract_messages(raw)
    if not messages:
        return None

    text = _classification_text(raw, messages)
    if _looks_placeholder(text) or (is_dpo and _looks_placeholder(_canonical_text(messages))):
        return None

    sha256, sha1 = content_hashes(messages)
    family = classify_family(text)
    scenario: dict[str, Any] = {
        "id": str(raw.get("id") or raw.get("session_id") or raw.get("pairing_id") or sha256[:16]),
        "source_asset": asset_name,
        "asset_kind": "dpo" if is_dpo else "scenario",
        "family": family,
        "category": raw.get("category") or raw.get("diagnostic_tag") or "",
        "messages": messages,
        "sha256": sha256,
        "sha1": sha1,
    }
    if is_dpo:
        scenario["rejected"] = str(raw.get("rejected", "")).strip()
        scenario["metadata"] = raw.get("metadata")
    return scenario


def _emit(manifest_path: Path, record: dict) -> None:
    with manifest_path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def reconcile(assets: list[dict[str, str]], master_hashes: set[str], *, limit: int | None, manifest_path: Path) -> dict:
    summary: dict[str, Any] = {"assets": {}, "families": dict.fromkeys(FAMILIES, 0), "unmapped": 0,
                               "new": 0, "duplicates": 0, "placeholders": 0}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest_path.unlink()

    for asset in assets:
        name = asset["name"]
        asset_stat = {"scanned": 0, "new": 0, "duplicates": 0, "placeholders": 0}
        kind = asset["kind"]
        if kind == "md":
            # Plan doc: stream + capture any section headings; not scenario data.
            headings = []
            for line in _iter_md(asset):
                if line.lstrip().startswith("#"):
                    headings.append(line.lstrip())
            summary["assets"][name] = {"kind": "md", "headings": headings[:40]}
            logger.info("%s: streamed plan doc (%d headings)", name, len(headings))
            continue

        iterator: Iterator[dict] = _iter_csv(asset, limit) if kind == "csv" else _iter_jsonl(asset, limit)
        for raw in iterator:
            asset_stat["scanned"] += 1
            scenario = normalize_scenario(name, raw)
            if scenario is None:
                asset_stat["placeholders"] += 1
                summary["placeholders"] += 1
                continue
            if scenario["sha256"] in master_hashes:
                asset_stat["duplicates"] += 1
                summary["duplicates"] += 1
                continue
            asset_stat["new"] += 1
            summary["new"] += 1
            if scenario["family"] in summary["families"]:
                summary["families"][scenario["family"]] += 1
            else:
                summary["unmapped"] += 1
            _emit(manifest_path, scenario)
        summary["assets"][name] = asset_stat
        logger.info("%s: scanned=%d new=%d dup=%d placeholder=%d", name,
                    asset_stat["scanned"], asset_stat["new"], asset_stat["duplicates"], asset_stat["placeholders"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=MASTER_GOLD)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=None, help="max records per asset (smoke test)")
    parser.add_argument("--max-master", type=int, default=None, help="cap master hash build (smoke test)")
    args = parser.parse_args()

    master_hashes = load_master_hashes(args.master, args.max_master)
    logger.info("Loaded %d master content hashes", len(master_hashes))
    summary = reconcile(ASSETS, master_hashes, limit=args.limit, manifest_path=args.manifest)
    logger.info("Reconciliation summary:\n%s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
