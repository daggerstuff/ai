#!/usr/bin/env python3
"""
Compile Final Training Dataset - Creates manifest + compiled ChatML JSONL export
"""

import argparse
import gc
import hashlib
import json
import logging
import random
import shutil
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from ai.training_ready.scripts.enhanced_deduplication import (
    ConversationEntry,
    EnhancedDeduplicator,
    compute_content_hash,
)
from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console = Console()


@dataclass
class DatasetShard:
    """Represents a sharded dataset file"""

    shard_id: str
    s3_path: str
    size_bytes: int
    sha256: str
    conversation_count: int
    source_families: list[str]


@dataclass
class SplitInfo:
    """Information about a dataset split"""

    conversations: int
    shards: list[DatasetShard]
    total_tokens_approx: int = 0


@dataclass
class ProgressCounters:
    """Counters for progress tracking"""

    raw_count: int
    normalized_count: int
    skipped_count: int


@dataclass
class CheckpointInfo:
    """Enhanced checkpoint information"""

    stage: str
    processed_families: list[str] = field(default_factory=list)
    processed_files: list[str] = field(default_factory=list)
    total_conversations: int = 0
    timestamp: str = ""
    family_stats: dict[str, int] = field(default_factory=dict)
    estimated_progress: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CompilerConfig:
    routing_config_path: Path
    coverage_report_path: Path
    output_dir: Path
    s3_manifest_path: Path
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05
    shard_size_mb: int = 1000
    checkpoint_dir: Path | None = None
    resume: bool = True


# Constants for S3 streaming
PROGRESS_LOG_INTERVAL = 1000  # Log every N records
PROGRESS_LOG_TIME_INTERVAL = 30  # Log every N seconds
LOW_RATE_THRESHOLD = 100  # records/second
LOW_RATE_MEMORY_THRESHOLD = 50000  # records before triggering GC


class CompilationTUI:
    """Beautiful TUI for dataset compilation progress"""

    def __init__(self):
        self.console = Console()
        self.main_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.family_progress = Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        )
        self.stats_table = Table(show_header=True, header_style="bold magenta")
        self.stats_table.add_column("Metric", style="cyan", no_wrap=True)
        self.stats_table.add_column("Value", style="green")

    def display_checkpoint_info(self, checkpoint: CheckpointInfo) -> None:
        """Display checkpoint information in a beautiful panel"""
        checkpoint_time = (
            datetime.fromisoformat(checkpoint.timestamp).strftime("%Y-%m-%d %H:%M:%S UTC")
            if checkpoint.timestamp
            else "Unknown"
        )

        content = f"""\
[bold cyan]Stage:[/bold cyan] {checkpoint.stage}
[bold cyan]Timestamp:[/bold cyan] {checkpoint_time}
[bold cyan]Progress:[/bold cyan] {checkpoint.estimated_progress:.1f}%
[bold cyan]Conversations:[/bold cyan] {checkpoint.total_conversations:,}
[bold cyan]Families Processed:[/bold cyan] {len(checkpoint.processed_families)}
[bold cyan]Files Processed:[/bold cyan] {len(checkpoint.processed_files)}
"""
        if checkpoint.family_stats:
            content += "\n[bold yellow]Family Statistics:[/bold yellow]\n"
            for family, count in sorted(
                checkpoint.family_stats.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                content += f"  • {family}: {count:,} conversations\n"

        panel = Panel(
            content,
            title="[bold green]✓ Checkpoint Loaded[/bold green]",
            border_style="green",
        )
        self.console.print(panel)

    def create_status_table(self, stats: dict[str, Any]) -> Table:
        """Create a live status table"""
        table = Table(show_header=True, header_style="bold blue", box=None)
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="green", width=20)

        for key, value in stats.items():
            display_value = value
            if isinstance(value, (int, float)) and value > 1000:
                display_value = f"{value:,}"
            table.add_row(key, str(display_value))

        return table


class FinalDatasetCompiler:
    """Compiles final training dataset with manifest + compiled export"""

    def __init__(self, config: CompilerConfig):
        self.routing_config_path = config.routing_config_path
        self.coverage_report_path = config.coverage_report_path
        self.output_dir = config.output_dir
        self.s3_manifest_path = config.s3_manifest_path
        self.train_split = config.train_split
        self.val_split = config.val_split
        self.test_split = config.test_split
        self.shard_size_mb = config.shard_size_mb
        self.resume = config.resume

        # Setup checkpoint directory
        self.checkpoint_dir = config.checkpoint_dir or (self.output_dir / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "collection_checkpoint.json"
        self.conversations_cache_dir = self.checkpoint_dir / "conversations_cache"
        self.conversations_cache_dir.mkdir(parents=True, exist_ok=True)

        self.routing_config: dict[str, Any] = {}
        self.coverage_data: dict[str, Any] = {}
        self.s3_manifest: dict[str, Any] = {}
        self.s3_bucket: str = "pixel-data"
        self.all_conversations: list[dict[str, Any]] = []
        self.deduplicator = EnhancedDeduplicator(similarity_threshold=0.95)

        # Track processed families for resume
        self.processed_families: set[str] = set()
        self.processed_files: set[str] = set()
        self.family_stats: dict[str, int] = {}

        # TUI
        self.tui = CompilationTUI()
        self.use_tui = True  # Enable TUI by default

        # Hard holdout families (only in test)
        self.holdout_families = [
            "long_running_therapy",
            "edge_case_crisis",
            "sarcasm",
            "voice_persona",
        ]

    def load_configs(self) -> None:
        """Load routing config and coverage report"""
        with open(self.routing_config_path, encoding="utf-8") as f:
            self.routing_config = json.load(f)

        with open(self.coverage_report_path, encoding="utf-8") as f:
            self.coverage_data = json.load(f)

        with open(self.s3_manifest_path, encoding="utf-8") as f:
            self.s3_manifest = json.load(f)
        bucket = self.s3_manifest.get("bucket")
        if isinstance(bucket, str) and bucket:
            self.s3_bucket = bucket

        endpoint = self.s3_manifest.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("s3_manifest.json missing endpoint")
        self.s3_endpoint = endpoint
        self.s3_loader = S3DatasetLoader(bucket=self.s3_bucket, endpoint_url=self.s3_endpoint)

    def _parse_checkpoint_data(self, checkpoint_data: dict[str, Any]) -> CheckpointInfo:
        """Parse checkpoint data into CheckpointInfo object"""
        self.processed_families = set(checkpoint_data.get("processed_families", []))
        self.processed_files = set(checkpoint_data.get("processed_files", []))
        self.family_stats = checkpoint_data.get("family_stats", {})

        return CheckpointInfo(
            stage=checkpoint_data.get("stage", "unknown"),
            processed_families=list(self.processed_families),
            processed_files=list(self.processed_files),
            total_conversations=checkpoint_data.get("total_conversations", 0),
            timestamp=checkpoint_data.get("timestamp", ""),
            family_stats=self.family_stats,
            estimated_progress=checkpoint_data.get("estimated_progress", 0.0),
            errors=checkpoint_data.get("errors", []),
        )

    def load_checkpoint(self) -> CheckpointInfo | None:
        """Load checkpoint if it exists with enhanced information"""
        if not self.resume or not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file, encoding="utf-8") as f:
                checkpoint_data = json.load(f)

            checkpoint_info = self._parse_checkpoint_data(checkpoint_data)

            if self.use_tui:
                self.tui.display_checkpoint_info(checkpoint_info)
            else:
                logger.info(
                    "Resuming from checkpoint: %s families, %s files already processed",
                    len(self.processed_families),
                    len(self.processed_files),
                )

            return checkpoint_info
        except Exception as e:
            logger.warning("Failed to load checkpoint: %s - starting fresh", e)
            if self.use_tui:
                self.tui.console.print(
                    f"[yellow]⚠ Warning:[/yellow] Failed to load checkpoint: {e} - starting fresh"
                )
            return None

    def save_checkpoint(self, stage: str = "collection", estimated_progress: float = 0.0) -> None:
        """Save checkpoint with enhanced progress tracking"""
        checkpoint = {
            "stage": stage,
            "processed_families": list(self.processed_families),
            "processed_files": list(self.processed_files),
            "total_conversations": len(self.all_conversations),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "family_stats": self.family_stats,
            "estimated_progress": estimated_progress,
            "checkpoint_version": "2.0",  # Version for future compatibility
        }

        # Use atomic write
        temp_file = self.checkpoint_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2)
            temp_file.replace(self.checkpoint_file)

            if self.use_tui:
                self.tui.console.print(
                    f"[dim]💾 Checkpoint saved: {len(self.all_conversations):,} conversations, "
                    f"{len(self.processed_families)} families[/dim]"
                )
            else:
                logger.debug(
                    "Checkpoint saved: %s conversations, %s families processed",
                    len(self.all_conversations),
                    len(self.processed_families),
                )
        except Exception as e:
            logger.error("Failed to save checkpoint: %s", e)
            if self.use_tui:
                self.tui.console.print(f"[red]❌ Failed to save checkpoint: {e}[/red]")

    def save_family_conversations(
        self, family_name: str, conversations: list[dict[str, Any]]
    ) -> None:
        """Save conversations for a family to cache file"""
        cache_file = self.conversations_cache_dir / f"{family_name}.jsonl"
        with open(cache_file, "w", encoding="utf-8") as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    def load_family_conversations(self, family_name: str) -> list[dict[str, Any]] | None:
        """Load cached conversations for a family"""
        cache_file = self.conversations_cache_dir / f"{family_name}.jsonl"
        if not cache_file.exists():
            return None

        conversations = []
        with open(cache_file, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    conversations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        logger.info(
            "Loaded %s cached conversations for family %s",
            len(conversations),
            family_name,
        )
        return conversations

    def _resolve_s3_uri(self, s3_path: str) -> str:
        if s3_path.startswith("s3://"):
            return s3_path
        return f"s3://{self.s3_bucket}/{s3_path.lstrip('/')}"

    @staticmethod
    def _messages_are_chatml(messages: list[Any]) -> bool:
        return all(
            isinstance(m, dict)
            and isinstance(m.get("role"), str)
            and isinstance(m.get("content"), str)
            for m in messages
        )

    @staticmethod
    def _coerce_role(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        v = value.strip().lower()
        if v in {"system"}:
            return "system"
        if v in {"user", "human", "client", "patient"}:
            return "user"
        if v in {"assistant", "bot", "gpt", "therapist", "counselor"}:
            return "assistant"
        return None

    def _conversation_to_messages(self, record: dict[str, Any]) -> list[dict[str, str]] | None:
        """
        Convert common `conversation` formats into ChatML `messages`.

        Many of our S3 JSONLs are shaped as:
        { conversation: [ {from/role/speaker, content/text}, ... ], metadata: {...} }
        """
        conv = record.get("conversation")
        if not isinstance(conv, list) or not conv:
            return None

        messages: list[dict[str, str]] = []
        next_role = "user"

        for turn in conv:
            if isinstance(turn, str):
                content = turn.strip()
                if not content:
                    continue
                role = next_role
            elif isinstance(turn, dict):
                content_val = (
                    turn.get("content")
                    or turn.get("text")
                    or turn.get("message")
                    or turn.get("utterance")
                )
                if not isinstance(content_val, str):
                    continue
                content = content_val.strip()
                if not content:
                    continue

                role = (
                    self._coerce_role(turn.get("role"))
                    or self._coerce_role(turn.get("from"))
                    or self._coerce_role(turn.get("speaker"))
                    or self._coerce_role(turn.get("author"))
                    or next_role
                )
            else:
                continue

            # Skip extra system turns; we can inject a single system prompt if desired.
            if role == "system":
                continue

            messages.append({"role": role, "content": content})
            next_role = "assistant" if role == "user" else "user"

        if not messages:
            return None

        # Optional: ensure there's a system message at the top.
        if all(m.get("role") != "system" for m in messages):
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "You are a therapeutic AI assistant. Respond with empathy and practical support.",
                },
            )

        return messages

    def _check_and_add_hash(self, content_hash: str, seen_hashes: set[str]) -> bool:
        """Check if hash is duplicate and add to set. Returns False if duplicate."""
        if content_hash in seen_hashes:
            return False
        seen_hashes.add(content_hash)
        return True

    def _normalize_chatml_record(
        self, record: Any, *, seen_hashes: set[str]
    ) -> dict[str, Any] | None:
        if not isinstance(record, dict):
            return None

        # Fast path: check hash first if it exists AND messages are already in correct format
        metadata = record.get("metadata")
        messages = record.get("messages")
        if (
            isinstance(metadata, dict)
            and isinstance(messages, list)
            and messages
            and self._messages_are_chatml(messages)
        ):
            content_hash = metadata.get("content_hash")
            if isinstance(content_hash, str) and content_hash:
                # Hash exists and messages are valid - check duplicate immediately
                return record if self._check_and_add_hash(content_hash, seen_hashes) else None

        # Normal path: normalize messages first
        messages = record.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or not self._messages_are_chatml(messages)
        ):
            # Attempt conversion from `conversation` format.
            converted = self._conversation_to_messages(record)
            if converted is None:
                return None
            messages = converted
            record["messages"] = messages

        if not isinstance(metadata, dict):
            metadata = {}
            record["metadata"] = metadata

        # Compute hash if not already present
        content_hash = metadata.get("content_hash")
        if not isinstance(content_hash, str) or not content_hash:
            content_hash = compute_content_hash(messages)
            metadata["content_hash"] = content_hash

        # Check for duplicates
        return record if self._check_and_add_hash(content_hash, seen_hashes) else None

    def _calculate_memory_size(self, out: list[dict[str, Any]]) -> float:
        """Calculate approximate memory size of output list in MB."""
        if not out or not hasattr(sys, "getsizeof"):
            return 0.0
        sample_size = sum(sys.getsizeof(item) for item in out[:1000])
        return (sample_size * len(out) / 1000) / (1024 * 1024)

    def _log_progress(
        self,
        counters: ProgressCounters,
        seen_hashes: set[str],
        out: list[dict[str, Any]],
        elapsed: float,
        last_log_count: int,
    ) -> tuple[int, float]:
        """Log progress and return updated last_log_count and reset stream_start."""
        rate = (counters.raw_count - last_log_count) / elapsed if elapsed > 0 else 0
        out_size_mb = self._calculate_memory_size(out)

        # Trigger GC if rate is very low (possible memory pressure)
        if rate < LOW_RATE_THRESHOLD and counters.raw_count > LOW_RATE_MEMORY_THRESHOLD:
            gc.collect()
            logger.debug("  Triggered GC due to low rate")

        logger.info(
            "  Progress: %s raw, %s normalized, %s skipped (elapsed: %.1fs, rate: %.0f rec/s, seen_hashes: %s, out_size: ~%.1fMB)",
            counters.raw_count,
            counters.normalized_count,
            counters.skipped_count,
            elapsed,
            rate,
            len(seen_hashes),
            out_size_mb,
        )
        return counters.raw_count, time.time()

    def _load_chatml_jsonl_from_s3(self, *, family_name: str, s3_uri: str) -> list[dict[str, Any]]:
        # Retry streaming reads (OVH S3 can occasionally drop large streams).
        max_attempts = 4
        seen_hashes: set[str] = set()

        for attempt in range(1, max_attempts + 1):
            try:
                out: list[dict[str, Any]] = []
                raw_count = 0
                normalized_count = 0
                skipped_count = 0
                last_log_count = 0

                logger.info("Streaming from %s (attempt %s/%s)", s3_uri, attempt, max_attempts)
                stream_start = time.time()

                for rec in self.s3_loader.stream_jsonl(s3_uri):
                    raw_count += 1
                    normalized = self._normalize_chatml_record(rec, seen_hashes=seen_hashes)
                    if normalized is not None:
                        normalized_count += 1
                        out.append(normalized)
                    else:
                        skipped_count += 1

                    # Log progress every N records or every N seconds
                    elapsed = time.time() - stream_start
                    if (raw_count - last_log_count >= PROGRESS_LOG_INTERVAL) or (
                        elapsed >= PROGRESS_LOG_TIME_INTERVAL and raw_count > last_log_count
                    ):
                        counters = ProgressCounters(
                            raw_count=raw_count,
                            normalized_count=normalized_count,
                            skipped_count=skipped_count,
                        )
                        last_log_count, stream_start = self._log_progress(
                            counters, seen_hashes, out, elapsed, last_log_count
                        )

                logger.info(
                    "Loaded %s/%s conversations from %s (skipped %s duplicates/invalid)",
                    normalized_count,
                    raw_count,
                    s3_uri,
                    skipped_count,
                )
                return out
            except FileNotFoundError as e:
                logger.warning("S3 file not found: %s - %s", s3_uri, e)
                return []
            except Exception as e:
                if attempt >= max_attempts:
                    logger.error(
                        "S3 stream error for %s (%s) after %s attempts: %s",
                        family_name,
                        s3_uri,
                        max_attempts,
                        e,
                    )
                    raise
                logger.warning(
                    "S3 stream error for %s (%s) attempt %s/%s: %s",
                    family_name,
                    s3_uri,
                    attempt,
                    max_attempts,
                    e,
                )
                time.sleep(1.5 * attempt)

        return []

    def _load_local_jsonl(self, path: Path) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    items.append(obj)
        return items

    def collect_local_generated_conversations(self) -> None:
        """
        Add locally generated datasets (missing families) into the compilation inputs.
        """
        family_key = "local_generated"
        if family_key in self.processed_families:
            logger.info("Skipping local generated (already processed)")
            return

        generated_dir = self.s3_manifest_path.parent / "generated"
        candidates = [
            (generated_dir / "edge_case_synthetic.jsonl", "edge_case_synthetic"),
            (generated_dir / "long_running_therapy.jsonl", "long_running_therapy"),
            (generated_dir / "cptsd_transcripts.jsonl", "cptsd"),
        ]

        total_convs = 0
        for path, family_name in candidates:
            file_key = f"{family_key}:{path}"
            if file_key in self.processed_files:
                logger.info("Skipping %s (already processed)", path.name)
                continue

            if not path.exists():
                continue

            # Try to load from cache first
            cached = self.load_family_conversations(family_name) if self.resume else None
            if cached is not None:
                convs = cached
            else:
                convs = self._load_local_jsonl(path)
                for conv in convs:
                    conv["metadata"] = conv.get("metadata", {})
                    conv["metadata"]["source_family"] = family_name
                    # Ensure content_hash exists
                    if not conv["metadata"].get("content_hash"):
                        conv["metadata"]["content_hash"] = compute_content_hash(
                            conv.get("messages", [])
                        )
                # Cache it
                if convs:
                    self.save_family_conversations(family_name, convs)

            self.all_conversations.extend(convs)
            self.processed_files.add(file_key)
            total_convs += len(convs)
            self.family_stats[family_name] = len(convs)
            if self.use_tui:
                self.tui.console.print(
                    f"[green]✓[/green] {family_name}: [bold]{len(convs):,}[/bold] conversations from {path.name}"
                )
            else:
                logger.info("Loaded %s conversations from local %s", len(convs), path)
            self.save_checkpoint("collection")

        if total_convs > 0:
            self.processed_families.add(family_key)
            logger.info("Total local generated: %s conversations", total_convs)

    def load_conversations_from_s3(
        self, family_name: str, s3_paths: list[str]
    ) -> list[dict[str, Any]]:
        """
        Load ChatML JSONL datasets from S3.

        This intentionally supports only ChatML-like records:
        - Each JSONL line is a dict with `messages: [{role, content}, ...]`
        - Optional `metadata`
        """
        if not s3_paths:
            logger.warning("No S3 paths provided for family %s", family_name)
            return []

        logger.info("Loading conversations for %s from %s S3 paths", family_name, len(s3_paths))
        out: list[dict[str, Any]] = []

        for s3_path in s3_paths:
            resolved = self._resolve_s3_uri(s3_path)
            logger.info("Processing S3 path: %s -> %s", s3_path, resolved)
            loaded = self._load_chatml_jsonl_from_s3(family_name=family_name, s3_uri=resolved)
            out.extend(loaded)
            logger.info("Accumulated %s total conversations for %s", len(out), family_name)

        logger.info(
            "Total loaded for %s: %s conversations from %s paths",
            family_name,
            len(out),
            len(s3_paths),
        )
        return out

    def load_conversations_for_family(self, family_name: str) -> list[dict[str, Any]]:
        """
        Prefer known-good ChatML JSONL S3 keys for each family.

        This is what lets us compile from the *real* S3 training outputs after
        encoding fixes, rather than relying on placeholder loader logic.
        """
        # Keys are relative to the bucket in ai/training_ready/data/s3_manifest.json
        family_keys: dict[str, list[str]] = {
            "mental_health_datasets": [
                "datasets/training_v3/stage1_foundation/Amod_mental_health_counseling_conversations.jsonl",
                "datasets/training_v3/stage1_foundation/heliosbrahma_mental_health_chatbot_dataset.jsonl",
                "datasets/training_v2/stage1_foundation/Amod_mental_health_counseling_conversations.jsonl",
                "datasets/training_v2/stage1_foundation/heliosbrahma_mental_health_chatbot_dataset.jsonl",
            ],
            "edge_case_generator": [
                "datasets/consolidated/conversations/edge_case_dialogues.jsonl",
                "datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_30_crisis_detection/crisis_detection_conversations.jsonl",
            ],
            "edge_case_resulting_chats": [
                "datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_30_crisis_detection/crisis_detection_conversations.jsonl",
                "datasets/gdrive/tier3_edge_crisis/crisis_detection_conversations.jsonl",
            ],
            "sarcasm": [
                # NOTE: this is JSON (not JSONL). We'll keep it out of ChatML ingestion for now.
            ],
            "professional_therapeutic": [
                "datasets/gdrive/processed/phase_2_professional_datasets/task_5_10_counsel_chat/counsel_chat_conversations.jsonl",
                "datasets/gdrive/processed/phase_2_professional_datasets/task_5_11_llama3_mental_counseling/llama3_mental_counseling_conversations.jsonl",
                "datasets/gdrive/processed/phase_2_professional_datasets/task_5_12_therapist_sft/therapist_sft_conversations.jsonl",
                "datasets/gdrive/processed/phase_2_professional_datasets/task_5_13_neuro_qa_sft/neuro_qa_sft_conversations.jsonl",
                "datasets/gdrive/processed/phase_2_professional_datasets/task_5_9_soulchat/soulchat_2_0_conversations.jsonl",
            ],
            "priority_datasets": [
                "datasets/gdrive/processed/phase_1_priority_conversations/task_5_1_priority_1/priority_1_conversations.jsonl",
                "datasets/gdrive/processed/phase_1_priority_conversations/task_5_2_priority_2/priority_2_conversations.jsonl",
                "datasets/gdrive/processed/phase_1_priority_conversations/task_5_3_priority_3/priority_3_conversations.jsonl",
                "datasets/gdrive/processed/phase_1_priority_conversations/task_5_6_unified_priority/unified_priority_conversations.jsonl",
            ],
            "cot_reasoning": [
                "datasets/gdrive/processed/phase_3_cot_reasoning/task_5_25_tot_reasoning/tot_reasoning_conversations.jsonl",
            ],
            "safety_guardrails_annihilator": [
                # These are large/raw; ChatML outputs are already captured in phase_4_reddit_mental_health
                "datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_27_condition_specific/condition_specific_conversations.jsonl",
                "datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_28_specialized_populations/specialized_populations_conversations.jsonl",
                "datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_29_temporal_analysis/temporal_analysis_conversations.jsonl",
            ],
            "addiction": [
                "datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_28_specialized_populations/specialized_populations_conversations.jsonl",
            ],
            "experimental": [
                "datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_29_temporal_analysis/temporal_analysis_conversations.jsonl",
            ],
        }

        keys = family_keys.get(family_name, [])
        if not keys:
            logger.warning("No S3 keys configured for family: %s", family_name)
            return []
        logger.info("Loading family %s from %s S3 keys", family_name, len(keys))
        return self.load_conversations_from_s3(family_name, keys)

    def _add_metadata_to_conversations(
        self,
        conversations: list[dict[str, Any]],
        family_name: str,
        source_key: str | None = None,
    ) -> None:
        """Add metadata to conversations"""
        for conv in conversations:
            conv["metadata"] = conv.get("metadata", {})
            conv["metadata"]["source_family"] = family_name
            if source_key:
                conv["metadata"]["source_key"] = source_key
            elif "source_key" not in conv["metadata"]:
                conv["metadata"]["source_key"] = (
                    f"s3://{self.s3_bucket}/{conv.get('metadata', {}).get('source_key', '')}"
                )
            if not conv["metadata"].get("content_hash"):
                conv["metadata"]["content_hash"] = compute_content_hash(conv.get("messages", []))

    def _load_or_get_cached_family(
        self, family_name: str, loader_func: Callable[[], list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Load family conversations from cache or loader function"""
        if self.resume and (cached := self.load_family_conversations(family_name)):
            logger.info("  Using cached conversations for %s", family_name)
            return cached

        conversations = loader_func()
        if conversations:
            self.save_family_conversations(family_name, conversations)
        return conversations

    def _process_s3_family(self, family_name: str, idx: int, total: int) -> None:
        """Process a single S3 family"""
        if family_name in self.processed_families:
            if self.use_tui:
                self.tui.console.print(f"[dim]⏭  Skipping {family_name} (already processed)[/dim]")
            else:
                logger.info(
                    "  Skipping family %s/%s: %s (already processed)",
                    idx,
                    total,
                    family_name,
                )
            if cached := self.load_family_conversations(family_name):
                self.all_conversations.extend(cached)
            return

        if self.use_tui:
            self.tui.console.print(f"[cyan]📦 Loading family {idx}/{total}: {family_name}[/cyan]")
        else:
            logger.info("  Loading family %s/%s: %s", idx, total, family_name)
        family_start = time.time()

        convs = self._load_or_get_cached_family(
            family_name, lambda: self.load_conversations_for_family(family_name)
        )

        self._add_metadata_to_conversations(convs, family_name)
        self.all_conversations.extend(convs)
        self.processed_families.add(family_name)
        self.family_stats[family_name] = len(convs)

        family_elapsed = time.time() - family_start
        estimated_progress = (idx / total) * 100 if total > 0 else 0

        if self.use_tui:
            self.tui.console.print(
                f"[green]✓[/green] {family_name}: [bold]{len(convs):,}[/bold] conversations "
                f"({family_elapsed:.1fs}) | Total: {len(self.all_conversations):,}"
            )
        else:
            logger.info(
                "  ✓ %s: %s conversations (%.1fs)",
                family_name,
                len(convs),
                family_elapsed,
            )
            logger.info("  Total so far: %s conversations", len(self.all_conversations))

        self.save_checkpoint("collection", estimated_progress=estimated_progress)

    def _process_routing_family(
        self, family_name: str, family_config: dict[str, Any], idx: int, total: int
    ) -> None:
        """Process a single routing family"""
        if family_name in self.processed_families:
            if self.use_tui:
                self.tui.console.print(f"[dim]⏭  Skipping {family_name} (already processed)[/dim]")
            else:
                logger.info(
                    "  Skipping routing family %s/%s: %s (already processed)",
                    idx,
                    total,
                    family_name,
                )
            if cached := self.load_family_conversations(family_name):
                self.all_conversations.extend(cached)
            return

        if self.use_tui:
            self.tui.console.print(
                f"[cyan]📦 Loading routing family {idx}/{total}: {family_name}[/cyan]"
            )
        else:
            logger.info("  Loading routing family %s/%s: %s", idx, total, family_name)
        family_start = time.time()
        s3_path_raw = family_config.get("s3_path")
        if not isinstance(s3_path_raw, str):
            logger.warning("  Invalid s3_path for family %s, skipping", family_name)
            return

        s3_path: str = s3_path_raw
        conversations = self._load_or_get_cached_family(
            family_name, lambda: self.load_conversations_from_s3(family_name, [s3_path])
        )

        self._add_metadata_to_conversations(conversations, family_name, s3_path)
        self.all_conversations.extend(conversations)
        self.processed_families.add(family_name)
        self.family_stats[family_name] = len(conversations)

        family_elapsed = time.time() - family_start
        estimated_progress = (idx / total) * 100 if total > 0 else 0

        if self.use_tui:
            self.tui.console.print(
                f"[green]✓[/green] {family_name}: [bold]{len(conversations):,}[/bold] conversations "
                f"({family_elapsed:.1fs}) | Total: {len(self.all_conversations):,}"
            )
        else:
            logger.info(
                "  ✓ %s: %s conversations (%.1fs)",
                family_name,
                len(conversations),
                family_elapsed,
            )
            logger.info("  Total so far: %s conversations", len(self.all_conversations))
        self.save_checkpoint("collection", estimated_progress=estimated_progress)

    def collect_all_conversations(self) -> None:
        """Collect all conversations from all dataset families"""
        logger.info("Collecting conversations from all dataset families...")
        total_start = time.time()

        # Always include locally generated missing families (edge_case_synthetic, long_running_therapy, cptsd)
        logger.info("Step 1/3: Loading locally generated conversations...")
        self.collect_local_generated_conversations()
        logger.info("  Total so far: %s conversations", len(self.all_conversations))

        # Also include known-good S3 ChatML exports (post encoding-fix).
        # These will massively increase coverage and fix distribution ratios.
        logger.info("Step 2/3: Loading from S3 families...")
        s3_families = (
            "mental_health_datasets",
            "professional_therapeutic",
            "priority_datasets",
            "cot_reasoning",
            "edge_case_generator",
            "edge_case_resulting_chats",
            "safety_guardrails_annihilator",
        )

        for idx, family_name in enumerate(s3_families, 1):
            self._process_s3_family(family_name, idx, len(s3_families))

        logger.info("Step 3/3: Loading from routing config families...")
        families = self.routing_config.get("families", {})
        routing_families = [
            (name, config)
            for name, config in families.items()
            if config.get("status") == "available" and config.get("s3_path")
        ]

        for idx, (family_name, family_config) in enumerate(routing_families, 1):
            self._process_routing_family(family_name, family_config, idx, len(routing_families))

        total_elapsed = time.time() - total_start
        logger.info(
            "✓ Collected %s total conversations from all families (%.1fs)",
            len(self.all_conversations),
            total_elapsed,
        )

    def assign_splits(self) -> None:
        """Assign conversations to train/val/test splits"""
        logger.info("Assigning conversations to splits...")

        # Separate holdout families
        holdout_conversations = []
        regular_conversations = []

        for conv in self.all_conversations:
            source_family = conv.get("metadata", {}).get("source_family", "")
            if source_family in self.holdout_families:
                holdout_conversations.append(conv)
            else:
                regular_conversations.append(conv)

        # Holdout families go to test only
        for conv in holdout_conversations:
            conv["metadata"]["split"] = "test"

        # Regular conversations: train/val/test split
        random.shuffle(regular_conversations)
        total_regular = len(regular_conversations)

        train_count = int(total_regular * self.train_split)
        val_count = int(total_regular * self.val_split)
        test_count = total_regular - train_count - val_count

        for i, conv in enumerate(regular_conversations):
            if i < train_count:
                conv["metadata"]["split"] = "train"
            elif i < train_count + val_count:
                conv["metadata"]["split"] = "val"
            else:
                conv["metadata"]["split"] = "test"

        logger.info(
            f"Split assignment: train={train_count}, val={val_count}, test={test_count + len(holdout_conversations)}"
        )

    def deduplicate(self) -> None:
        """Deduplicate conversations"""
        logger.info("Deduplicating conversations...")

        # Convert to ConversationEntry format
        for conv in self.all_conversations:
            entry = ConversationEntry(
                messages=conv.get("messages", []),
                source_family=conv.get("metadata", {}).get("source_family", ""),
                source_key=conv.get("metadata", {}).get("source_key", ""),
                content_hash=conv.get("metadata", {}).get("content_hash", ""),
                split=conv.get("metadata", {}).get("split"),
                metadata=conv.get("metadata", {}),
            )
            self.deduplicator.add_conversation(entry)

        # Find and remove duplicates
        exact_dups = self.deduplicator.find_exact_duplicates()
        near_dups = self.deduplicator.find_near_duplicates()

        logger.info(
            f"Found {len(exact_dups)} exact duplicate groups, {len(near_dups)} near-duplicate pairs"
        )

        # Deduplicate
        deduplicated_entries = self.deduplicator.deduplicate(strategy="keep_first")

        # Convert back to conversation format
        self.all_conversations = [
            {"messages": entry.messages, "metadata": entry.metadata}
            for entry in deduplicated_entries
        ]

        logger.info(f"After deduplication: {len(self.all_conversations)} conversations")

    def check_split_leakage(self) -> dict[str, Any]:
        """Check for split leakage violations"""
        logger.info("Checking for split leakage violations...")

        violations = self.deduplicator.check_split_leakage(self.holdout_families)

        if violations["exact_duplicate_leakage"]:
            logger.warning(
                f"Found {len(violations['exact_duplicate_leakage'])} exact duplicate leakage violations"
            )

        if violations["near_duplicate_leakage"]:
            logger.warning(
                f"Found {len(violations['near_duplicate_leakage'])} near-duplicate leakage violations"
            )

        if violations["holdout_family_leakage"]:
            logger.error(
                f"Found {len(violations['holdout_family_leakage'])} holdout family leakage violations"
            )

        return violations

    def create_shards(
        self, conversations: list[dict[str, Any]], split_name: str
    ) -> list[DatasetShard]:
        """Create sharded files for a split"""
        shards = []
        shard_size_bytes = self.shard_size_mb * 1024 * 1024

        current_shard: list[dict[str, Any]] = []
        current_size = 0
        shard_num = 0

        for conv in conversations:
            conv_json = json.dumps(conv, ensure_ascii=False) + "\n"
            conv_size = len(conv_json.encode("utf-8"))

            if current_size + conv_size > shard_size_bytes and current_shard:
                # Save current shard
                shard_id = f"{split_name}_{shard_num:03d}"
                shard_path = self.output_dir / "shards" / f"{shard_id}.jsonl"
                shard_path.parent.mkdir(parents=True, exist_ok=True)

                with open(shard_path, "w", encoding="utf-8") as f:
                    for c in current_shard:
                        f.write(json.dumps(c, ensure_ascii=False) + "\n")

                # Compute hash
                with open(shard_path, "rb") as f:
                    shard_hash = hashlib.sha256(f.read()).hexdigest()

                shard_size = shard_path.stat().st_size
                source_families = list(
                    {c.get("metadata", {}).get("source_family", "") for c in current_shard}
                )

                shard = DatasetShard(
                    shard_id=shard_id,
                    s3_path=f"s3://{self.s3_bucket}/final_dataset/{split_name}/{shard_id}.jsonl",
                    size_bytes=shard_size,
                    sha256=f"sha256:{shard_hash}",
                    conversation_count=len(current_shard),
                    source_families=source_families,
                )
                shards.append(shard)

                # Reset for next shard
                current_shard = []
                current_size = 0
                shard_num += 1

            current_shard.append(conv)
            current_size += conv_size

        # Save final shard
        if current_shard:
            self._extracted_from_create_shards_55(split_name, shard_num, current_shard, shards)
        return shards

    # TODO Rename this here and in `create_shards`
    def _extracted_from_create_shards_55(self, split_name, shard_num, current_shard, shards):
        shard_id = f"{split_name}_{shard_num:03d}"
        shard_path = self.output_dir / "shards" / f"{shard_id}.jsonl"
        shard_path.parent.mkdir(parents=True, exist_ok=True)

        with open(shard_path, "w", encoding="utf-8") as f:
            for c in current_shard:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        with open(shard_path, "rb") as f:
            shard_hash = hashlib.sha256(f.read()).hexdigest()

        shard_size = shard_path.stat().st_size
        source_families = list(
            {c.get("metadata", {}).get("source_family", "") for c in current_shard}
        )

        shard = DatasetShard(
            shard_id=shard_id,
            s3_path=f"s3://{self.s3_bucket}/final_dataset/{split_name}/{shard_id}.jsonl",
            size_bytes=shard_size,
            sha256=f"sha256:{shard_hash}",
            conversation_count=len(current_shard),
            source_families=source_families,
        )
        shards.append(shard)

    def generate_manifest(self) -> dict[str, Any]:
        """Generate dataset manifest"""
        logger.info("Generating dataset manifest...")

        # Group by split
        by_split = defaultdict(list)
        by_family = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

        for conv in self.all_conversations:
            split = conv.get("metadata", {}).get("split", "train")
            family = conv.get("metadata", {}).get("source_family", "unknown")
            by_split[split].append(conv)
            by_family[family][split] += 1

        # Create shards for each split
        splits_info = {}
        for split_name in ["train", "val", "test"]:
            conversations = by_split[split_name]
            shards = self.create_shards(conversations, split_name)

            splits_info[split_name] = {
                "conversations": len(conversations),
                "shards": [
                    {
                        "shard_id": s.shard_id,
                        "s3_path": s.s3_path,
                        "size_bytes": s.size_bytes,
                        "sha256": s.sha256,
                        "conversation_count": s.conversation_count,
                        "source_families": s.source_families,
                    }
                    for s in shards
                ],
            }

        # Build provenance map
        provenance_map = {}
        for conv in self.all_conversations:
            if content_hash := conv.get("metadata", {}).get("content_hash", ""):
                provenance_map[content_hash] = {
                    "source_key": conv.get("metadata", {}).get("source_key", ""),
                    "source_family": conv.get("metadata", {}).get("source_family", ""),
                    "original_format": "jsonl",
                    "processing_steps": ["encoding_fix", "dedup", "chatml_convert"],
                }

        return {
            "manifest_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_conversations": len(self.all_conversations),
            "total_tokens_approx": 0,  # TODO: Compute actual token count
            "splits": splits_info,
            "source_families": {
                family: {"conversations": sum(counts.values()), "splits": counts}
                for family, counts in by_family.items()
            },
            "provenance_map": provenance_map,
            "holdout_families": {
                family: {
                    "test_split_only": True,
                    "rationale": "Hard holdout for evaluation",
                }
                for family in self.holdout_families
            },
        }

    def create_compiled_export(self) -> Path:
        """Create compiled single-file export"""
        logger.info("Creating compiled export...")

        output_path = self.output_dir / "compiled" / "final_training_dataset.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for conv in self.all_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        logger.info(f"Compiled export created: {output_path}")
        return output_path

    def _load_cached_conversations_from_checkpoint(self) -> None:
        """Load cached conversations from processed families in checkpoint"""
        for family in self.processed_families:
            if family != "local_generated" and (cached := self.load_family_conversations(family)):
                self.all_conversations.extend(cached)

    def _update_progress(self, task: int | None, description: str) -> None:
        """Update progress bar if TUI is enabled"""
        if self.use_tui and task is not None:
            self.tui.main_progress.update(task, description=description)  # type: ignore[arg-type]

    def _process_compilation_steps(
        self, task: int | None
    ) -> tuple[dict[str, Any], Path, Path, dict[str, Any]]:
        """Process all compilation steps and return results"""
        self._update_progress(task, "[cyan]Collecting conversations...")
        self.collect_all_conversations()

        self._update_progress(task, "[yellow]Assigning splits...")
        self.assign_splits()

        self._update_progress(task, "[yellow]Deduplicating...")
        self.deduplicate()

        self._update_progress(task, "[yellow]Checking for leakage...")
        violations = self.check_split_leakage()
        if violations["holdout_family_leakage"]:
            if self.use_tui:
                self.tui.console.print(
                    "[red]❌ Holdout family leakage detected! Fix before proceeding.[/red]"
                )
            logger.error("Holdout family leakage detected! Fix before proceeding.")
            raise ValueError("Holdout family leakage detected")

        self._update_progress(task, "[yellow]Generating manifest...")
        manifest = self.generate_manifest()

        self._update_progress(task, "[yellow]Creating compiled export...")
        compiled_path = self.create_compiled_export()

        # Save manifest
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest, manifest_path, compiled_path, violations

    def _display_startup_message(self) -> None:
        """Display startup message"""
        if self.use_tui:
            self.tui.console.print(
                Panel(
                    "[bold cyan]🚀 Starting Final Dataset Compilation[/bold cyan]",
                    border_style="cyan",
                )
            )
        else:
            logger.info("Starting final dataset compilation...")

    def _handle_checkpoint_resume(self, checkpoint: CheckpointInfo) -> None:
        """Handle checkpoint resume logic"""
        if not self.use_tui:
            logger.info("Resuming from checkpoint at stage: %s", checkpoint.stage)
        self._load_cached_conversations_from_checkpoint()
        if not self.use_tui:
            logger.info(
                "Loaded %s conversations from checkpoint",
                len(self.all_conversations),
            )

    def _start_progress_tracking(self) -> int | None:
        """Start progress tracking and return task ID"""
        if not self.use_tui:
            return None
        self.tui.main_progress.start()
        return self.tui.main_progress.add_task("[cyan]Collecting conversations...", total=None)

    def _complete_progress_tracking(self, task: int | None, manifest_path: Path) -> None:
        """Complete progress tracking and display completion message"""
        if self.use_tui:
            if task is not None:
                self.tui.main_progress.update(task, description="[green]✓ Complete!")  # type: ignore[arg-type]
            self.tui.main_progress.stop()
            self.tui.console.print(f"\n[green]✓ Manifest saved:[/green] {manifest_path}")
        else:
            logger.info(f"Manifest saved: {manifest_path}")

    def _handle_leakage_error(self) -> dict[str, Any]:
        """Handle leakage detection error"""
        violations = self.check_split_leakage()
        return {"error": "Holdout family leakage", "violations": violations}

    def _display_final_summary(self, manifest_path: Path, compiled_path: Path) -> None:
        """Display final compilation summary"""
        if not self.use_tui:
            return

        summary_table = self.tui.create_status_table(
            {
                "Total Conversations": f"{len(self.all_conversations):,}",
                "Families Processed": len(self.processed_families),
                "Manifest Path": str(manifest_path),
                "Compiled Export": str(compiled_path),
            }
        )
        self.tui.console.print("\n")
        self.tui.console.print(
            Panel(
                summary_table,
                title="[bold green]✓ Compilation Complete[/bold green]",
                border_style="green",
            )
        )

    def _stop_progress_tracking(self) -> None:
        """Stop progress tracking"""
        if self.use_tui:
            self.tui.main_progress.stop()

    def compile(self) -> dict[str, Any]:
        """Run complete compilation process"""
        self._display_startup_message()
        self.load_configs()

        if checkpoint := self.load_checkpoint():
            self._handle_checkpoint_resume(checkpoint)

        task = self._start_progress_tracking()

        try:
            manifest, manifest_path, compiled_path, violations = self._process_compilation_steps(
                task
            )
            self._complete_progress_tracking(task, manifest_path)
        except ValueError as e:
            if "leakage" not in str(e):
                raise
            return self._handle_leakage_error()
        finally:
            self._stop_progress_tracking()

        self._display_final_summary(manifest_path, compiled_path)

        return {
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "compiled_path": str(compiled_path),
            "total_conversations": len(self.all_conversations),
            "violations": violations,
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Compile final training dataset with checkpoint/resume support"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any existing checkpoints",
    )
    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Clear existing checkpoint and start fresh",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parents[3]

    routing_config_path = (
        project_root / "ai" / "training_ready" / "data" / "dataset_routing_config.json"
    )
    coverage_report_path = (
        project_root / "ai" / "training_ready" / "data" / "dataset_coverage_report.json"
    )
    s3_manifest_path = project_root / "ai" / "training_ready" / "data" / "s3_manifest.json"
    output_dir = project_root / "ai" / "training_ready" / "data" / "final_dataset"

    # Clear checkpoint if requested
    if args.clear_checkpoint:
        checkpoint_dir = output_dir / "checkpoints"
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            logger.info("Cleared checkpoint directory")
        return 0

    compiler = FinalDatasetCompiler(
        CompilerConfig(
            routing_config_path=routing_config_path,
            coverage_report_path=coverage_report_path,
            output_dir=output_dir,
            s3_manifest_path=s3_manifest_path,
            resume=not args.no_resume,  # Enable resume by default
        )
    )

    result = compiler.compile()

    if "error" in result:
        logger.error(f"Compilation failed: {result['error']}")
        return 1

    logger.info("✅ Final dataset compilation complete!")
    logger.info(f"   Total conversations: {result['total_conversations']}")
    logger.info(f"   Manifest: {result['manifest_path']}")
    logger.info(f"   Compiled export: {result['compiled_path']}")
    logger.info("")
    logger.info("💡 Tip: If interrupted, re-run the same command to resume from checkpoint")
    logger.info("   Use --no-resume to start fresh, or --clear-checkpoint to clear checkpoints")

    return 0


if __name__ == "__main__":
    sys.exit(main())
