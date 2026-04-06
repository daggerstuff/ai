"""
Standard therapeutic source loading and normalization for the integrated pipeline.
"""

from __future__ import annotations

import json
from json import JSONDecodeError, JSONDecoder
from pathlib import Path
from typing import Any, Callable, Protocol

from ai.pipelines.orchestrator.orchestration.storage_resolver import StorageCacheError
from ai.pipelines.orchestrator.utils.logger import get_logger

logger = get_logger("dataset_pipeline.standard_therapeutic_loader")


class StandardTherapeuticConfigProtocol(Protocol):
    source_path: str | None
    source_paths: tuple[str, ...]
    fallback_paths: tuple[str, ...]


class LoaderStatsProtocol(Protocol):
    warnings: list[str]
    errors: list[str]


CacheData = Callable[[str | None], Path | None]


class StandardTherapeuticLoaderService:
    """Own standard therapeutic source loading and format normalization."""

    def __init__(
        self,
        *,
        config: StandardTherapeuticConfigProtocol,
        stats: LoaderStatsProtocol,
        cache_data: CacheData,
    ) -> None:
        self.config = config
        self.stats = stats
        self.cache_data = cache_data

    def load(self) -> list[dict[str, Any]]:
        """Load standard therapeutic data from gdrive-backed JSONL or local JSON files."""
        candidate_paths = self._candidate_source_paths()
        raw_conversations: list[Any] = []
        last_error: Exception | None = None
        for candidate_path in candidate_paths:
            try:
                standard_file = self._resolve_candidate_path(candidate_path)
            except Exception as exc:
                last_error = exc
                continue
            if standard_file is None or not standard_file.exists():
                continue
            logger.info("Attempting to load from: %s", standard_file)
            try:
                if standard_file.suffix == ".jsonl":
                    raw_conversations.extend(self._load_jsonl_file(standard_file))
                else:
                    raw_conversations.extend(self._try_load_json_file(standard_file))
            except Exception as exc:
                last_error = exc
                continue

        if not raw_conversations:
            self._handle_load_error(candidate_paths, last_error)
            return []

        return self._normalize_conversations(raw_conversations)

    def _load_jsonl_file(
        self, file_path: Path, max_samples: int | None = None
    ) -> list[dict[str, Any]]:
        conversations: list[dict[str, Any]] = []
        try:
            with file_path.open(encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if max_samples and index >= max_samples:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        self._warn(f"Skipping malformed line {index + 1}: {exc}")
                        continue
                    if isinstance(record, dict):
                        conversations.append(record)
            logger.info(
                "✅ Loaded %s conversations from JSONL: %s",
                len(conversations),
                file_path,
            )
        except Exception as exc:
            self._error(f"Failed to load JSONL file {file_path}: {exc}")
        return conversations

    @staticmethod
    def _try_load_json_file(file_path: Path) -> list[Any]:
        first_token = StandardTherapeuticLoaderService._first_json_token(file_path)
        if first_token == "[":
            conversations = StandardTherapeuticLoaderService._stream_json_array(file_path)
            logger.info(
                "✅ Loaded %s conversations from %s (list format)",
                len(conversations),
                file_path,
            )
            return conversations
        if first_token == "{":
            conversations = StandardTherapeuticLoaderService._stream_conversations_object(
                file_path
            )
            if conversations:
                logger.info(
                    "✅ Loaded %s conversations from %s (dict format)",
                    len(conversations),
                    file_path,
                )
                return conversations
            logger.warning("File %s loaded but no conversations found", file_path)
            return []

        logger.warning(
            "Unexpected JSON root token in %s: %s",
            file_path,
            first_token or "<empty>",
        )
        return []

    def _candidate_source_paths(self) -> list[str]:
        configured: list[str] = []
        if self.config.source_path:
            configured.append(self.config.source_path)
        configured.extend(path for path in getattr(self.config, "source_paths", ()) if path)
        configured.extend(path for path in getattr(self.config, "fallback_paths", ()) if path)

        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in configured:
            if candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
        return deduped

    def _resolve_candidate_path(self, candidate_path: str) -> Path | None:
        if candidate_path.startswith(("gdrive:", "s3://", "datasets/")):
            try:
                cached_path = self.cache_data(candidate_path)
            except StorageCacheError as exc:
                raise RuntimeError(
                    f"Failed to cache standard therapeutic source {candidate_path}: {exc}"
                ) from exc
            if cached_path and cached_path.exists():
                return cached_path
            self._warn(f"Could not cache remote standard therapeutic path: {candidate_path}")
            return None

        source_root_path = Path(candidate_path)
        if source_root_path.is_dir():
            nested_dataset = source_root_path / "training_dataset.json"
            if nested_dataset.exists():
                return nested_dataset
        return source_root_path

    @staticmethod
    def _first_json_token(file_path: Path) -> str:
        with file_path.open(encoding="utf-8") as handle:
            while True:
                chunk = handle.read(1)
                if not chunk:
                    return ""
                if not chunk.isspace():
                    return chunk

    @staticmethod
    def _stream_json_array(file_path: Path) -> list[Any]:
        with file_path.open(encoding="utf-8") as handle:
            return StandardTherapeuticLoaderService._stream_array_items(handle.read)

    @staticmethod
    def _stream_conversations_object(file_path: Path) -> list[Any]:
        with file_path.open(encoding="utf-8") as handle:
            buffer = ""
            chunk_size = 65536
            found_conversations = False

            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                buffer += chunk
                if not found_conversations:
                    field_index = buffer.find('"conversations"')
                    if field_index == -1:
                        buffer = buffer[-32:]
                        continue
                    found_conversations = True
                    buffer = buffer[field_index + len('"conversations"') :]

                colon_index = buffer.find(":")
                if colon_index == -1:
                    continue
                buffer = buffer[colon_index + 1 :]
                stripped = buffer.lstrip()
                if not stripped:
                    continue
                if stripped[0] != "[":
                    raise ValueError(
                        f"Expected conversations array in {file_path}, found {stripped[0]!r}"
                    )
                prefix_trim = len(buffer) - len(stripped)
                buffer = buffer[prefix_trim:]
                return StandardTherapeuticLoaderService._stream_array_items(
                    handle.read, initial_buffer=buffer
                )

        return []

    @staticmethod
    def _stream_array_items(
        reader: Callable[[int], str],
        *,
        initial_buffer: str = "",
    ) -> list[Any]:
        items: list[Any] = []
        for item in StandardTherapeuticLoaderService._iter_streamed_array_items(
            reader,
            initial_buffer=initial_buffer,
        ):
            items.append(item)
        return items

    @staticmethod
    def _iter_streamed_array_items(
        reader: Callable[[int], str],
        *,
        initial_buffer: str = "",
    ):
        decoder = JSONDecoder()
        buffer = initial_buffer
        chunk_size = 65536
        index = 0
        started = False
        while True:
            index, started, completed = (
                StandardTherapeuticLoaderService._consume_array_prefix(
                    buffer,
                    index,
                    started,
                )
            )
            if completed:
                return
            if index >= len(buffer):
                buffer, index = StandardTherapeuticLoaderService._read_more(
                    reader=reader,
                    chunk_size=chunk_size,
                    buffer=buffer,
                    index=index,
                    started=started,
                )
                continue

            try:
                item, next_index = decoder.raw_decode(buffer, index)
            except JSONDecodeError:
                buffer, index = StandardTherapeuticLoaderService._read_more(
                    reader=reader,
                    chunk_size=chunk_size,
                    buffer=buffer,
                    index=index,
                    started=started,
                    allow_eof=False,
                )
                continue

            index = next_index
            yield item

    @staticmethod
    def _consume_array_prefix(
        buffer: str,
        index: int,
        started: bool,
    ) -> tuple[int, bool, bool]:
        while index < len(buffer):
            current = buffer[index]
            if current.isspace():
                index += 1
                continue
            if not started:
                if current != "[":
                    raise ValueError("Expected JSON array")
                started = True
                index += 1
                continue
            if current == ",":
                index += 1
                continue
            if current == "]":
                return index, started, True
            break
        return index, started, False

    @staticmethod
    def _read_more(
        *,
        reader: Callable[[int], str],
        chunk_size: int,
        buffer: str,
        index: int,
        started: bool,
        allow_eof: bool = True,
    ) -> tuple[str, int]:
        chunk = reader(chunk_size)
        if not chunk:
            if not allow_eof or started:
                raise ValueError("Unexpected EOF while parsing JSON array")
            return buffer, index
        return buffer[index:] + chunk, 0

    def _handle_load_error(
        self, possible_files: list[str], last_error: Exception | None
    ) -> None:
        if last_error:
            self._error(
                f"Failed to load standard therapeutic data. Last error: {last_error}"
            )
            return
        self._error(
            "Standard therapeutic data not found in: "
            + str([str(path) for path in possible_files])
        )

    def _normalize_conversations(
        self, conversations: list[Any]
    ) -> list[dict[str, Any]]:
        training_data: list[dict[str, Any]] = []
        for conv in conversations:
            if not isinstance(conv, dict):
                continue
            text = self._extract_text_from_conv(conv)
            if not text:
                continue
            training_data.append(
                {
                    "text": text,
                    "metadata": {
                        "source": "standard_therapeutic",
                        "is_edge_case": False,
                    },
                }
            )

        logger.info(
            "✅ Converted %s standard therapeutic examples to training format",
            len(training_data),
        )
        return training_data

    def _extract_text_from_conv(self, conv: dict[str, Any]) -> str:
        text = conv.get("text", "")
        if isinstance(text, str) and text:
            return text

        conversation_array = conv.get("conversation", [])
        if isinstance(conversation_array, list) and conversation_array:
            parts = self._parts_from_messages(conversation_array)
            if parts:
                return "\n".join(parts)

        messages = conv.get("messages", [])
        if isinstance(messages, list) and messages:
            parts = self._parts_from_messages(messages)
            if parts:
                return "\n".join(parts)

        content = conv.get("content", "")
        if isinstance(content, str) and content:
            return content

        data = conv.get("data")
        if isinstance(data, dict):
            instruction = data.get("instruction", "")
            user_input = data.get("input", "")
            output = data.get("output", "")
            if isinstance(user_input, str) and isinstance(output, str) and user_input and output:
                parts: list[str] = []
                if isinstance(instruction, str) and instruction:
                    parts.append(f"System: {instruction}")
                parts.append(f"User: {user_input}")
                parts.append(f"Assistant: {output}")
                return "\n".join(parts)

        prompt = conv.get("prompt", "")
        response = conv.get("response", "")
        if isinstance(prompt, str) and isinstance(response, str) and prompt and response:
            return f"User: {prompt}\nAssistant: {response}"

        return ""

    @staticmethod
    def _parts_from_messages(messages: list[Any]) -> list[str]:
        parts: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(role, str) and isinstance(content, str) and role and content:
                parts.append(f"{role.capitalize()}: {content}")
        return parts

    def _warn(self, message: str) -> None:
        logger.warning(message)
        self.stats.warnings.append(message)

    def _error(self, message: str) -> None:
        logger.error(message)
        self.stats.errors.append(message)


__all__ = ["StandardTherapeuticLoaderService"]
