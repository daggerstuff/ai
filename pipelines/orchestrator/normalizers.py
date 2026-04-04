#!/usr/bin/env python3
"""
DACT-04: Cross-Source Normalization Registry

Converts source datasets from their native formats into the canonical
ConversationRecord JSONL format used throughout the Pixelated pipeline.

Each normalizer accepts source-specific data and yields dicts conforming to:
{
    "conversation_id": str,
    "source": str,
    "messages": [{"role": str, "content": str, "timestamp": str, "metadata": dict}],
    "metadata": {
        "quality_score": float,
        "topic_tags": list[str],
        "license": str,
        "therapeutic_area": str,
        "gate_status": dict,
        ...
    }
}

Usage:
        python -m ai.pipelines.orchestrator.normalizers --input data.json \
            --format hf_conversation --source my_dataset
        python -m ai.pipelines.orchestrator.normalizers --input data.csv \
            --format reddit_csv --source RedditESS --output normalized.jsonl
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Canonical record structure
CANONICAL_KEYS = {"conversation_id", "source", "messages", "metadata"}


@dataclass
class NormalizationConfig:
    """Configuration for a normalization run."""

    source_name: str
    license: str = "unknown"
    therapeutic_area: str = "general"
    topic_tags: list[str] = field(default_factory=list)
    quality_score: float = 0.5
    gate_status: dict[str, str] = field(
        default_factory=lambda: {
            "gate_1": "PASS",
            "gate_2": "PASS",
            "gate_3": "PASS",
            "gate_4": "PASS",
        }
    )
    stage: str | None = None
    reasoning_type: str | None = None


def _make_id(source: str, index: int, content: str = "") -> str:
    """Generate a stable conversation ID."""
    if content:
        h = hashlib.sha256(f"{source}:{content[:200]}".encode()).hexdigest()[:12]
    else:
        h = hashlib.sha256(f"{source}:{index}".encode()).hexdigest()[:12]
    return f"{source}_{h}"


def _make_record(
    conversation_id: str,
    source: str,
    messages: list[dict[str, Any]],
    config: NormalizationConfig,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical ConversationRecord dict."""
    metadata: dict[str, Any] = {
        "quality_score": config.quality_score,
        "topic_tags": config.topic_tags,
        "license": config.license,
        "therapeutic_area": config.therapeutic_area,
        "gate_status": config.gate_status,
    }
    if config.stage:
        metadata["stage"] = config.stage
    if config.reasoning_type:
        metadata["reasoning_type"] = config.reasoning_type
    if extra_metadata:
        metadata |= extra_metadata

    return {
        "conversation_id": conversation_id,
        "source": source,
        "messages": messages,
        "metadata": metadata,
    }


def _clean_messages_pair(
    input_text: str,
    response_text: str,
    *,
    user_role: str = "user",
    assistant_role: str = "assistant",
) -> list[dict[str, Any]]:
    """Convert input/response pair into messages list."""
    return [
        {"role": user_role, "content": input_text.strip()},
        {"role": assistant_role, "content": response_text.strip()},
    ]


# ─── Normalizer Base ────────────────────────────────────────────────────────


class Normalizer(ABC):
    """Base class for source-specific normalizers."""

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Unique identifier for this normalizer."""
        ...

    @abstractmethod
    def normalize(
        self,
        data: Any,
        config: NormalizationConfig,
    ) -> Iterator[dict[str, Any]]:
        """Yield canonical ConversationRecord dicts from source data."""
        ...


# ─── Hugging Face Normalizers ───────────────────────────────────────────────


class HFConversationNormalizer(Normalizer):
    """Normalize HF datasets with a `conversations` field.

    Supports conversation entries formatted as `{from, value}` or
    `{role, content}`.
    """

    @property
    def format_name(self) -> str:
        return "hf_conversation"

    def normalize(
        self,
        data: Any,
        config: NormalizationConfig,
    ) -> Iterator[dict[str, Any]]:
        """
        data: list of dicts, each with a 'conversations' key containing
              [{"from": "human"|"gpt", "value": "..."}] or
              [{"role": "...", "content": "..."}]
        """
        items = data if isinstance(data, list) else [data]
        for idx, item in enumerate(items):
            conv_data = item.get("conversations", item.get("messages", []))
            if not conv_data:
                continue

            messages = []
            for msg in conv_data:
                role = msg.get("from", msg.get("role", "unknown"))
                content = msg.get("value", msg.get("content", ""))
                if role == "human":
                    role = "user"
                elif role == "gpt":
                    role = "assistant"
                messages.append({"role": role, "content": content})

            if len(messages) < 2:
                continue

            conv_id = item.get("conversation_id", _make_id(config.source_name, idx))
            yield _make_record(conv_id, config.source_name, messages, config)


class HFInstructionNormalizer(Normalizer):
    """Normalizes HF datasets with instruction/input/output fields."""

    @property
    def format_name(self) -> str:
        return "hf_instruction"

    def normalize(
        self,
        data: Any,
        config: NormalizationConfig,
    ) -> Iterator[dict[str, Any]]:
        """
        data: list of dicts with 'instruction', optional 'input', and 'output' fields.
        """
        items = data if isinstance(data, list) else [data]
        for idx, item in enumerate(items):
            instruction = item.get("instruction", "")
            user_input = item.get("input", "")
            output = item.get("output", "")

            if not instruction and not output:
                continue

            user_content = instruction
            if user_input:
                user_content = f"{instruction}\n\nInput: {user_input}"

            messages = _clean_messages_pair(user_content, output)
            conv_id = item.get(
                "id", item.get("conversation_id", _make_id(config.source_name, idx))
            )
            yield _make_record(conv_id, config.source_name, messages, config)


class HFChatMLNormalizer(Normalizer):
    """Normalizes HF datasets already in ChatML format."""

    @property
    def format_name(self) -> str:
        return "hf_chatml"

    def normalize(
        self,
        data: Any,
        config: NormalizationConfig,
    ) -> Iterator[dict[str, Any]]:
        """
        data: list of dicts with 'messages' field in OpenAI chat format.
        """
        items = data if isinstance(data, list) else [data]
        for idx, item in enumerate(items):
            messages = item.get("messages", [])
            if not messages or len(messages) < 2:
                continue

            # Ensure proper role names
            normalized_messages = []
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                normalized_messages.append({"role": role, "content": content})

            conv_id = item.get(
                "id", item.get("conversation_id", _make_id(config.source_name, idx))
            )
            yield _make_record(conv_id, config.source_name, normalized_messages, config)


# ─── Reddit CSV Normalizer ─────────────────────────────────────────────────


class RedditCSVNormalizer(Normalizer):
    """Normalizes Reddit CSV files with post/comment threads."""

    @property
    def format_name(self) -> str:
        return "reddit_csv"

    def normalize(
        self,
        data: Any,
        config: NormalizationConfig,
    ) -> Iterator[dict[str, Any]]:
        """
        data: CSV string content or path.
        Expected columns: title/post, comment/response, or similar.
        """
        if isinstance(data, (str, Path)):
            text = (
                Path(data).read_text(encoding="utf-8")
                if isinstance(data, Path)
                else data
            )
            reader = csv.DictReader(io.StringIO(text))
        else:
            reader = csv.DictReader(data)

        for idx, row in enumerate(reader):
            # Common column patterns
            user_content = (
                row.get("title", "")
                or row.get("post", "")
                or row.get("selftext", "")
                or row.get("input", "")
                or row.get("user", "")
            )
            assistant_content = (
                row.get("comment", "")
                or row.get("response", "")
                or row.get("reply", "")
                or row.get("assistant", "")
            )

            if not user_content and not assistant_content:
                # Try using the entire row as a single conversation
                user_content = json.dumps(dict(row), ensure_ascii=False)
                assistant_content = ""

            if not user_content.strip():
                continue

            messages = _clean_messages_pair(
                user_content.strip(), assistant_content.strip()
            )
            conv_id = row.get(
                "id", _make_id(config.source_name, idx, user_content[:100])
            )
            yield _make_record(conv_id, config.source_name, messages, config)


# ─── CoT JSON Normalizer ────────────────────────────────────────────────────


class CoTJSONNormalizer(Normalizer):
    """Normalizes Chain-of-Thought reasoning JSON datasets."""

    @property
    def format_name(self) -> str:
        return "cot_json"

    def normalize(
        self,
        data: Any,
        config: NormalizationConfig,
    ) -> Iterator[dict[str, Any]]:
        """
        data: list of dicts, each representing a CoT reasoning example.
        Expected fields: scenario/question, reasoning, conclusion/response
        """
        items = data if isinstance(data, list) else [data]
        for idx, item in enumerate(items):
            # Extract question/scenario
            question = (
                item.get("scenario", "")
                or item.get("question", "")
                or item.get("prompt", "")
                or item.get("input", "")
                or item.get("context", "")
            )

            # Extract reasoning chain
            reasoning = (
                item.get("reasoning", "")
                or item.get("chain_of_thought", "")
                or item.get("cot", "")
                or item.get("thought_process", "")
            )

            # Extract conclusion/response
            conclusion = (
                item.get("conclusion", "")
                or item.get("response", "")
                or item.get("output", "")
                or item.get("answer", "")
                or item.get("therapeutic_approach", "")
            )

            if not question:
                continue

            # Build multi-turn conversation: question → reasoning → conclusion
            messages = [{"role": "user", "content": question.strip()}]
            if reasoning.strip():
                messages.append({"role": "assistant", "content": reasoning.strip()})
            if conclusion.strip():
                # If we already have reasoning, add conclusion as a follow-up
                if reasoning.strip():
                    messages.append(
                        {"role": "user", "content": "What is the recommended approach?"}
                    )
                messages.append({"role": "assistant", "content": conclusion.strip()})

            if len(messages) < 2:
                continue

            conv_id = item.get(
                "id",
                item.get(
                    "conversation_id", _make_id(config.source_name, idx, question[:100])
                ),
            )
            extra_meta = {
                "reasoning_type": config.reasoning_type
                or item.get("reasoning_type", ""),
                "therapeutic_focus": item.get("therapeutic_focus", ""),
            }
            yield _make_record(
                conv_id, config.source_name, messages, config, extra_meta
            )


# ─── Clinical/Research Normalizer ───────────────────────────────────────────


class ClinicalNormalizer(Normalizer):
    """Normalizes clinical/research data formats."""

    @property
    def format_name(self) -> str:
        return "clinical"

    def normalize(
        self,
        data: Any,
        config: NormalizationConfig,
    ) -> Iterator[dict[str, Any]]:
        """
        data: list of dicts with clinical case information.
        Expected: case_description, diagnosis, treatment, outcome fields.
        """
        items = data if isinstance(data, list) else [data]
        for idx, item in enumerate(items):
            case_desc = (
                item.get("case_description", "")
                or item.get("case", "")
                or item.get("patient_presentation", "")
                or item.get("scenario", "")
                or item.get("input", "")
            )

            treatment = (
                item.get("treatment", "")
                or item.get("intervention", "")
                or item.get("therapy", "")
                or item.get("response", "")
                or item.get("clinical_approach", "")
            )

            if not case_desc:
                continue

            user_content = f"Case: {case_desc.strip()}"
            assistant_content = (
                treatment.strip() if treatment else "Clinical assessment pending."
            )

            messages = _clean_messages_pair(user_content, assistant_content)
            conv_id = item.get(
                "id",
                item.get("case_id", _make_id(config.source_name, idx, case_desc[:100])),
            )
            extra_meta = {
                "diagnosis": item.get("diagnosis", ""),
                "outcome": item.get("outcome", ""),
            }
            yield _make_record(
                conv_id, config.source_name, messages, config, extra_meta
            )


# ─── Article Normalizer ─────────────────────────────────────────────────────


class ArticleNormalizer(Normalizer):
    """Normalizes single articles (PMC, Bright Data) into instructional pairs."""

    @property
    def format_name(self) -> str:
        return "article"

    def normalize(
        self,
        data: Any,
        config: NormalizationConfig,
    ) -> Iterator[dict[str, Any]]:
        """
        data: list of dicts or single dict with article content.
        Expected: title, sections (list of {heading, text}), or full text.
        """
        items = data if isinstance(data, list) else [data]
        for idx, item in enumerate(items):
            title = item.get("title", config.source_name)
            sections = item.get("sections", [])
            full_text = item.get("text", item.get("content", item.get("body", "")))

            if sections:
                for sec_idx, section in enumerate(sections):
                    heading = section.get("heading", section.get("title", ""))
                    text = section.get("text", section.get("content", ""))
                    if not text.strip():
                        continue

                    user_content = (
                        f"What does current evidence say about {heading or title}?"
                    )
                    messages = _clean_messages_pair(user_content, text.strip())
                    conv_id = _make_id(
                        config.source_name, idx * 1000 + sec_idx, heading[:50]
                    )
                    yield _make_record(conv_id, config.source_name, messages, config)

            elif full_text.strip():
                # Split large articles into chunks
                chunks = _split_text_chunks(full_text.strip(), max_chunk=2000)
                for chunk_idx, chunk in enumerate(chunks):
                    user_content = f"Provide evidence-based guidance on: {title}"
                    messages = _clean_messages_pair(user_content, chunk)
                    conv_id = _make_id(
                        config.source_name, idx * 1000 + chunk_idx, title[:50]
                    )
                    yield _make_record(conv_id, config.source_name, messages, config)


def _split_text_chunks(text: str, max_chunk: int = 2000) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    if len(text) <= max_chunk:
        return [text]

    chunks = []
    sentences = text.replace(". ", ".\n").split("\n")
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > max_chunk and current:
            chunks.append(current.strip())
            current = sent
        else:
            current += f" {sent}" if current else sent
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


# ─── Registry ────────────────────────────────────────────────────────────────

NORMALIZERS: dict[str, Normalizer] = {
    "hf_conversation": HFConversationNormalizer(),
    "hf_instruction": HFInstructionNormalizer(),
    "hf_chatml": HFChatMLNormalizer(),
    "reddit_csv": RedditCSVNormalizer(),
    "cot_json": CoTJSONNormalizer(),
    "clinical": ClinicalNormalizer(),
    "article": ArticleNormalizer(),
}


def get_normalizer(format_name: str) -> Normalizer:
    """Get a normalizer by format name."""
    if format_name not in NORMALIZERS:
        available = ", ".join(NORMALIZERS.keys())
        raise ValueError(f"Unknown format '{format_name}'. Available: {available}")
    return NORMALIZERS[format_name]


def _detect_format_from_object(obj: dict[str, Any]) -> str | None:
    """Detect input format from a single parsed record object."""
    conversations = obj.get("conversations")
    if isinstance(conversations, list) and conversations:
        first_msg = conversations[0]
        return "hf_conversation" if "from" in first_msg else "hf_chatml"

    if "messages" in obj and isinstance(obj["messages"], list):
        return "hf_chatml"
    if "instruction" in obj and "output" in obj:
        return "hf_instruction"
    if "scenario" in obj or "reasoning" in obj or "chain_of_thought" in obj:
        return "cot_json"
    if "case_description" in obj or "diagnosis" in obj:
        return "clinical"
    if "sections" in obj or ("title" in obj and "text" in obj):
        return "article"

    return None


def detect_format(file_path: str | Path) -> str:
    """Auto-detect the format of a data file."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return "reddit_csv"
    if suffix == ".jsonl":
        # Peek at first line to determine
        with open(path) as f:
            first_line = f.readline().strip()
            if not first_line:
                return "hf_conversation"
            with suppress(json.JSONDecodeError):
                obj = json.loads(first_line)
                if detected := _detect_format_from_object(obj):
                    return detected
        return "hf_conversation"
    if suffix == ".json":
        with open(path) as f:
            with suppress(json.JSONDecodeError):
                obj = json.load(f)
                items = obj if isinstance(obj, list) else [obj]
                if items:
                    if detected := _detect_format_from_object(items[0]):
                        return detected
        return "hf_conversation"

    return "hf_conversation"


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _load_data(file_path: str | Path) -> Any:
    """Load data from a file."""
    path = Path(file_path)
    with open(path, encoding="utf-8") as f:
        if path.suffix.lower() != ".jsonl":
            return json.load(f)

        records = []
        for line in f:
            if line := line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line in %s", path)
        return records


def normalize_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    format_name: str | None = None,
    source_name: str | None = None,
    config: NormalizationConfig | None = None,
) -> dict[str, int]:
    """
    Normalize a single file and write output.

    Returns stats dict.
    """
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    detected_format = format_name or detect_format(in_path)
    normalizer = get_normalizer(detected_format)

    src_name = source_name or in_path.stem
    data = _load_data(in_path)

    if config is None:
        config = NormalizationConfig(source_name=src_name)

    stats = {"input": 0, "output": 0, "skipped": 0}
    out_path = (
        Path(output_path)
        if output_path
        else in_path.with_name(f"{in_path.stem}_normalized.jsonl")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for record in normalizer.normalize(data, config):
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["output"] += 1
        stats["input"] = len(data) if isinstance(data, list) else 1
        stats["skipped"] = stats["input"] - stats["output"]

    logger.info(
        "Normalized %s → %s (%d records, %d skipped)",
        in_path,
        out_path,
        stats["output"],
        stats["skipped"],
    )
    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DACT-04: Cross-Source Normalization")
    parser.add_argument("--input", required=True, help="Input data file")
    parser.add_argument(
        "--output", help="Output JSONL file (default: <input>_normalized.jsonl)"
    )
    parser.add_argument(
        "--format",
        choices=list(NORMALIZERS.keys()),
        help="Input format (auto-detected if not specified)",
    )
    parser.add_argument("--source", help="Source name (default: filename stem)")
    parser.add_argument("--license", default="unknown", help="Dataset license")
    parser.add_argument(
        "--therapeutic-area", default="general", help="Therapeutic area"
    )
    parser.add_argument("--stage", help="Training stage")
    parser.add_argument(
        "--quality-score", type=float, default=0.5, help="Default quality score"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    cfg = NormalizationConfig(
        source_name=args.source or Path(args.input).stem,
        license=args.license,
        therapeutic_area=args.therapeutic_area,
        quality_score=args.quality_score,
        stage=args.stage,
    )

    stats = normalize_file(
        args.input,
        args.output,
        format_name=args.format,
        source_name=args.source,
        config=cfg,
    )

    print(f"Normalized: {stats['output']} records written, {stats['skipped']} skipped")


if __name__ == "__main__":
    main()
