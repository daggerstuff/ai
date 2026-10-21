"""Recover structurally incomplete rows from pinned Psy-Insight source dialogues."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import aiohttp

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "ai/data/curated/recovery/gold-20261017/train_master_gold.recovered_candidate.jsonl"
OUTPUT_DIR = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-repairs-20261018"
HF_DATASET = "https://huggingface.co/datasets/xi1019/counseling-psy-insight"
HF_REVISION = "dd6f2e582377ddb71e976ec8f45ad48b5c5fc57a"
SOURCE_HASHES = {
    "en": "f68b64fd8d3936e1d0e6295d7680b2b5a64364385baafa3cb6cf42f5902870c8",
    "zh": "c08fdc4bde30eb1382764d03f88010fda172e36c76079f9dd48b440034c3e35b",
}
MIN_PAIR_SCORE = 0.85
MIN_UTTERANCE_SCORE = 0.70
RELEASE_ID = "wandb-full-corpus-20261018-v2"
EXPECTED_REPAIR_COUNT = 52
W_AND_B_INFERENCE_URL = "https://api.inference.wandb.ai/v1/chat/completions"
BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
GENERATION_MAX_TOKENS = 256
GENERATION_CONCURRENCY = 4
SOURCE_EXCERPT_TURN_COUNT = 2
HTTP_ERROR_THRESHOLD = 400
GENERATION_ATTEMPTS = 5
TARGET_ROW_NUMBERS = {
    6142,
    6146,
    84029,
    84047,
    84057,
    84062,
    84074,
    84093,
    84105,
    84136,
    84137,
    84143,
    84144,
    84145,
    84169,
    84171,
    84172,
    84183,
    84185,
    84215,
    84238,
    84268,
    84269,
    84270,
    84284,
    84285,
    84286,
    84287,
    84288,
    84289,
    84290,
    84291,
    84292,
    84303,
    84319,
    84320,
    84321,
    84332,
    84341,
    84357,
    84359,
    84372,
    84418,
    84448,
    84486,
    84520,
    84526,
    84558,
    84578,
    84589,
    84594,
    84607,
}


@dataclass
class GenerationContext:
    session: aiohttp.ClientSession
    semaphore: asyncio.Semaphore
    token: str
    progress_path: Path
    completed: dict[int, dict[str, Any]]
    progress_lock: asyncio.Lock


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(text: str) -> str:
    text = text.translate(str.maketrans("", "", "\u201c\u201d\u2018\u2019\u200b\ufeff"))
    return re.sub(r"\s+", "", text).casefold()


def download_source(language: str, directory: Path) -> Path:
    path = directory / f"{language}.jsonl"
    url = f"{HF_DATASET}/resolve/{HF_REVISION}/{language}.jsonl"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    if sha256(payload) != SOURCE_HASHES[language]:
        raise ValueError(f"pinned Psy-Insight {language} source hash mismatch")
    path.write_bytes(payload)
    os.chmod(path, 0o600)
    return path


def load_dialogues(paths: dict[str, Path]) -> list[dict[str, Any]]:
    dialogues: list[dict[str, Any]] = []
    for language, path in paths.items():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                dialogue = json.loads(line)
                if dialogue.get("lang") != language or not isinstance(dialogue.get("messages"), list):
                    raise ValueError(f"unexpected Psy-Insight dialogue schema in {path}")
                dialogues.append(dialogue)
    return dialogues


def source_pair_index(
    dialogues: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], int, str, str]], dict[tuple[str, str], list[tuple[dict[str, Any], int]]]]:
    source_pairs: list[tuple[dict[str, Any], int, str, str]] = []
    exact_pairs: dict[tuple[str, str], list[tuple[dict[str, Any], int]]] = {}
    for dialogue in dialogues:
        turns = dialogue["messages"]
        for index in range(len(turns) - 1):
            counselor, client = turns[index : index + 2]
            if counselor.get("role") == "counselor" and client.get("role") == "client":
                normalized_pair = (normalize(counselor.get("content", "")), normalize(client.get("content", "")))
                source_pairs.append((dialogue, index, *normalized_pair))
                exact_pairs.setdefault(normalized_pair, []).append((dialogue, index))
    return source_pairs, exact_pairs


def rank_source_pair(
    assistant_text: str,
    client_text: str,
    source_pairs: list[tuple[dict[str, Any], int, str, str]],
    exact_pairs: dict[tuple[str, str], list[tuple[dict[str, Any], int]]],
) -> tuple[float, float, float, dict[str, Any], int]:
    exact = exact_pairs.get((assistant_text, client_text), [])
    if exact:
        ranked = [(1.0, 1.0, 1.0, dialogue, index) for dialogue, index in exact]
    else:
        ranked = []
        for dialogue, index, source_counselor, source_client in source_pairs:
            counselor_length_ratio = min(len(assistant_text), len(source_counselor)) / max(
                1, len(assistant_text), len(source_counselor)
            )
            client_length_ratio = min(len(client_text), len(source_client)) / max(
                1, len(client_text), len(source_client)
            )
            if counselor_length_ratio < MIN_UTTERANCE_SCORE:
                continue
            if client_length_ratio < MIN_UTTERANCE_SCORE:
                continue
            counselor_score = SequenceMatcher(None, assistant_text, source_counselor).ratio()
            client_score = SequenceMatcher(None, client_text, source_client).ratio()
            ranked.append(((counselor_score + client_score) / 2, counselor_score, client_score, dialogue, index))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        raise ValueError("no matching upstream counselor/client turn pair")
    best = ranked[0]
    if best[0] < MIN_PAIR_SCORE or min(best[1], best[2]) < MIN_UTTERANCE_SCORE:
        raise ValueError("upstream turn pair similarity is below the required threshold")
    if len(ranked) > 1 and ranked[1][0] == best[0] and ranked[1][3]["case_id"] != best[3]["case_id"]:
        raise ValueError("ambiguous tied upstream dialogue matches")
    return best


def make_source_repair(
    row_number: int,
    raw_line: str,
    row: dict[str, Any],
    best: tuple[float, float, float, dict[str, Any], int],
) -> dict[str, Any]:
    dialogue = best[3]
    excerpt = dialogue["messages"]
    if len(excerpt) != SOURCE_EXCERPT_TURN_COUNT:
        raise ValueError(f"expected a two-turn source excerpt at candidate row {row_number}")
    counselor_content = excerpt[0].get("content", "")
    client_content = excerpt[1].get("content", "")
    if not counselor_content.strip() or not client_content.strip():
        raise ValueError(f"empty upstream utterance at candidate row {row_number}")
    system = next(
        (message.get("content", "") for message in row.get("messages", []) if message.get("role") == "system"),
        "Continue the counseling conversation with empathy and care.",
    )
    prompt = (
        "Continue the verified counseling excerpt below as the counselor's next turn. "
        "Keep the response grounded in what the client said, avoid diagnosis or medical advice, "
        "and respond in the same language as the client. Write only the next counselor turn.\n\n"
        f"Counselor: {counselor_content}\nClient: {client_content}"
    )
    return {
        "candidate_row_number_1based": row_number,
        "candidate_row_sha256": sha256(raw_line.rstrip("\n").encode()),
        "repair_kind": "source_dialogue_recovery_with_generated_continuation",
        "source_dialogue_id": dialogue["case_id"],
        "source_language": dialogue["lang"],
        "source_pair_similarity": round(best[0], 6),
        "source_pair_similarity_by_turn": {"counselor": round(best[1], 6), "client": round(best[2], 6)},
        "messages": [
            {
                "role": "system",
                "content": system + " Continue the provided counselor-client transcript. Do not diagnose or prescribe.",
            },
            {"role": "user", "content": prompt},
        ],
    }


def candidate_repairs(dialogues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_pairs, exact_pairs = source_pair_index(dialogues)
    repairs: list[dict[str, Any]] = []
    with CANDIDATE.open(encoding="utf-8") as stream:
        for row_number, raw_line in enumerate(stream, 1):
            if row_number not in TARGET_ROW_NUMBERS or row_number in {6142, 6146}:
                continue
            row = json.loads(raw_line)
            if row.get("source") != "psy_insight":
                raise ValueError(f"unexpected source at Psy-Insight repair row {row_number}")
            messages = [
                message
                for message in row.get("messages", [])
                if isinstance(message, dict)
                and message.get("role") in {"user", "assistant"}
                and isinstance(message.get("content"), str)
                and message["content"].strip()
            ]
            assistant_indices = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
            if not assistant_indices or any(
                message["role"] == "user" for message in messages[: assistant_indices[-1] + 1]
            ):
                raise ValueError(f"candidate row {row_number} no longer matches the recorded exclusion")
            assistant_text = normalize(
                next(message["content"] for message in messages if message["role"] == "assistant")
            )
            client_text = normalize(next(message["content"] for message in messages if message["role"] == "user"))
            best = rank_source_pair(assistant_text, client_text, source_pairs, exact_pairs)
            repairs.append(make_source_repair(row_number, raw_line, row, best))
    if {repair["candidate_row_number_1based"] for repair in repairs} != TARGET_ROW_NUMBERS - {6142, 6146}:
        raise ValueError("source repair targets do not match the recorded exclusion set")
    return repairs


async def generate_continuation(context: GenerationContext, repair: dict[str, Any]) -> None:
    async with context.semaphore:
        body = {
            "model": BASE_MODEL,
            "messages": repair["messages"],
            "temperature": 0,
            "max_tokens": GENERATION_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for attempt in range(GENERATION_ATTEMPTS):
            try:
                async with context.session.post(
                    W_AND_B_INFERENCE_URL,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {context.token}",
                        "Content-Type": "application/json",
                        "OpenAI-Project": "wutang/pixelated-empathy-kan28",
                    },
                ) as response:
                    if response.status >= HTTP_ERROR_THRESHOLD:
                        raise RuntimeError(f"continuation generation failed with HTTP {response.status}")
                    payload = await response.json(content_type=None)
                message = payload["choices"][0]["message"]
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("continuation generation returned empty content")
                repair["messages"].append({"role": "assistant", "content": content.strip()})
                repair["continuation_generation"] = {
                    "provider": "wandb-inference",
                    "model": BASE_MODEL,
                    "temperature": 0,
                    "max_tokens": GENERATION_MAX_TOKENS,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "finish_reason": message.get("finish_reason", payload["choices"][0].get("finish_reason")),
                    "usage": payload.get("usage"),
                    "not_human_reviewed": True,
                }
                async with context.progress_lock:
                    context.completed[repair["candidate_row_number_1based"]] = repair
                    write_progress(context.progress_path, context.completed)
                return
            except (aiohttp.ClientError, TimeoutError, json.JSONDecodeError, KeyError, RuntimeError) as exc:
                if attempt == GENERATION_ATTEMPTS - 1:
                    raise RuntimeError(
                        f"could not generate continuation for candidate row "
                        f"{repair['candidate_row_number_1based']} ({type(exc).__name__}: {exc})"
                    ) from None
                await asyncio.sleep(min(2**attempt, 30))


def write_progress(path: Path, completed: dict[int, dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row_number in sorted(completed):
            stream.write(json.dumps(completed[row_number], ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    temporary.replace(path)


async def generate_source_continuations(repairs: list[dict[str, Any]], progress_path: Path) -> None:
    token = os.environ.get("WANDB_API_KEY")
    if not token:
        raise ValueError("WANDB_API_KEY must be provided through the environment")
    completed: dict[int, dict[str, Any]] = {}
    if progress_path.exists():
        with progress_path.open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                completed[entry["candidate_row_number_1based"]] = entry
        for repair in repairs:
            saved = completed.get(repair["candidate_row_number_1based"])
            if saved is not None and saved["candidate_row_sha256"] == repair["candidate_row_sha256"]:
                repair.update(saved)
    pending = [
        repair
        for repair in repairs
        if repair["repair_kind"] == "source_dialogue_recovery_with_generated_continuation"
        and "continuation_generation" not in repair
    ]
    timeout = aiohttp.ClientTimeout(total=120)
    semaphore = asyncio.Semaphore(GENERATION_CONCURRENCY)
    progress_lock = asyncio.Lock()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        context = GenerationContext(session, semaphore, token, progress_path, completed, progress_lock)
        await asyncio.gather(
            *(
                generate_continuation(context, repair)
                for repair in pending
            )
        )


def book_source_repairs() -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    with CANDIDATE.open(encoding="utf-8") as stream:
        for row_number, raw_line in enumerate(stream, 1):
            if row_number not in {6142, 6146}:
                continue
            row = json.loads(raw_line)
            if row.get("source") != "clinical_book_Get More.....":
                continue
            messages = row.get("messages", [])
            has_target = any(
                isinstance(message, dict)
                and message.get("role") == "assistant"
                and isinstance(message.get("content"), str)
                and message["content"].strip()
                for message in messages
            )
            has_user = any(
                isinstance(message, dict)
                and message.get("role") == "user"
                and isinstance(message.get("content"), str)
                and message["content"].strip()
                for message in messages
            )
            if has_target and has_user:
                continue
            empty_record = not any(
                isinstance(message, dict) and str(message.get("content", "")).strip() for message in messages
            )
            if empty_record:
                user = (
                    'A source record titled "Get More....." has empty user and assistant fields. '
                    "Can we infer a therapeutic example from its title?"
                )
            else:
                user = (
                    'An extracted "Get More....." page is promotional material, not client-therapist dialogue. '
                    "Should it be turned into clinical QA training?"
                )
            repairs.append(
                {
                    "candidate_row_number_1based": row_number,
                    "candidate_row_sha256": sha256(raw_line.rstrip("\n").encode()),
                    "repair_kind": "source_quality_triage_reconstruction",
                    "source_dialogue_id": None,
                    "source_language": "en",
                    "source_pair_similarity": None,
                    "source_pair_similarity_by_turn": None,
                    "messages": [
                        {"role": "user", "content": user},
                        {
                            "role": "assistant",
                            "content": (
                                "No. The title and incomplete record do not provide verified clinical content. "
                                "Do not invent a case or counseling response; mark the source record incomplete "
                                "and exclude it from clinical-response training until its contents are recovered."
                                if empty_record
                                else "No. Promotional material is not client-therapist dialogue. Exclude it from "
                                "clinical QA generation and retain a source-quality disposition."
                            ),
                        },
                    ],
                }
            )
    return repairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    source_dir = args.source_dir or args.output_dir / "source-snapshots"
    source_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    source_paths = {lang: download_source(lang, source_dir) for lang in ("en", "zh")}
    dialogues = load_dialogues(source_paths)
    repairs = candidate_repairs(dialogues) + book_source_repairs()
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress_path = args.output_dir / "row_repairs.progress.jsonl"
    asyncio.run(generate_source_continuations(repairs, progress_path))
    repairs.sort(key=lambda repair: repair["candidate_row_number_1based"])
    if len(repairs) != EXPECTED_REPAIR_COUNT or len(
        {repair["candidate_row_number_1based"] for repair in repairs}
    ) != EXPECTED_REPAIR_COUNT:
        raise ValueError(f"expected exactly {EXPECTED_REPAIR_COUNT} unique repair targets, got {len(repairs)}")
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    repair_path = args.output_dir / "row_repairs.jsonl"
    with repair_path.open("w", encoding="utf-8") as stream:
        for repair in repairs:
            stream.write(json.dumps(repair, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(repair_path, 0o600)
    progress_path.unlink(missing_ok=True)
    manifest = {
        "schema_version": "candidate-row-repairs-v1",
        "release_id": RELEASE_ID,
        "source_candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        "repair_count": len(repairs),
        "repair_count_by_kind": {
            kind: sum(repair["repair_kind"] == kind for repair in repairs)
            for kind in sorted({repair["repair_kind"] for repair in repairs})
        },
        "repair_row_numbers_1based": [repair["candidate_row_number_1based"] for repair in repairs],
        "repairs_path": str(repair_path.relative_to(ROOT)),
        "repairs_sha256": hashlib.sha256(repair_path.read_bytes()).hexdigest(),
        "upstream": {
            "dataset": "Psy-Insight",
            "processor_mirror": HF_DATASET,
            "processor_revision": HF_REVISION,
            "upstream_repository": "https://github.com/ckqqqq/Psy-Insight",
            "files": SOURCE_HASHES,
            "match_method": "adjacent counselor/client utterance similarity, then restore complete source dialogue",
            "minimum_pair_similarity": MIN_PAIR_SCORE,
        },
        "notes": [
            "Original candidate rows remain unchanged; this sidecar is a row-keyed repair overlay.",
            "Fifty Psy-Insight two-turn excerpts were matched to pinned source dialogues and given a "
            "W&B-generated counselor continuation; those continuations are marked unreviewed.",
            "Two empty/promotional source records became source-quality triage examples, not fabricated "
            "clinical examples.",
            "Synthetic source-quality triage is labeled separately from source-dialogue recovery.",
        ],
    }
    manifest_path = args.output_dir / "repair_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    summary = {"repairs": len(repairs), "by_kind": manifest["repair_count_by_kind"], "manifest": str(manifest_path)}
    sys.stdout.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
