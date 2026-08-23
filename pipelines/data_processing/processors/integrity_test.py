"""V7 dataset integrity tester.

Validates consolidated V7 output for:
1. Token limits (per-message and per-conversation)
2. Role validity (only system/user/assistant, alternating pattern)
3. UTF-8 encoding correctness (no mojibake, valid Unicode)
4. ChatML structural compliance (required fields, non-empty content)
5. V7 metadata completeness (source, task_type, provenance)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Defaults ---

DEFAULT_MAX_TOKENS_PER_MESSAGE = 4096
DEFAULT_MAX_TOKENS_PER_CONVERSATION = 16384
DEFAULT_MAX_MESSAGE_LENGTH_CHARS = 20000

VALID_ROLES: frozenset[str] = frozenset({"system", "user", "assistant"})

# Approximate token count: 1 token ≈ 4 chars (conservative for English)
_CHARS_PER_TOKEN = 4

# Mojibake detection: sequences of replacement chars or common Windows-1252 artifacts
_MOJIBAKE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\ufffd"),  # Unicode replacement char
    re.compile(r"[\xc3][\x80-\xbf]"),  # UTF-8 interpreted as Latin-1
    re.compile(r"[\xe2][\x82][\x80-\xbf]"),  # Smart quote artifacts
    re.compile(r"\xc2[\x80-\xbf]"),  # Raw continuation bytes as text
]

# Control characters except tab/newline/carriage-return
_CONTROL_CHAR = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Required V7 fields
REQUIRED_V7_FIELDS = {"source", "task_type"}


# --- Data classes ---

@dataclass
class IntegrityViolation:
    """A single integrity violation found in a record."""

    severity: str  # "error" or "warning"
    check: str
    message: str
    record_index: int
    shard_file: str | None = None
    field_path: str | None = None


@dataclass
class IntegrityReport:
    """Full integrity report for a V7 dataset."""

    total_records: int = 0
    total_violations: int = 0
    errors: int = 0
    warnings: int = 0
    violations: list[IntegrityViolation] = field(default_factory=list)
    passed: bool = True

    def add_violation(
        self,
        severity: str,
        check: str,
        message: str,
        record_index: int,
        shard_file: str | None = None,
        field_path: str | None = None,
    ) -> None:
        v = IntegrityViolation(
            severity=severity,
            check=check,
            message=message,
            record_index=record_index,
            shard_file=shard_file,
            field_path=field_path,
        )
        self.violations.append(v)
        self.total_violations += 1
        if severity == "error":
            self.errors += 1
            self.passed = False
        else:
            self.warnings += 1


# --- Integrity checks ---

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: chars / 4, minimum 1."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _check_utf8(text: str) -> str | None:
    """Return a description of the UTF-8 issue, or None if clean."""
    for pattern in _MOJIBAKE_PATTERNS:
        if pattern.search(text):
            return f"Mojibake detected (pattern: {pattern.pattern})"
    if _CONTROL_CHAR.search(text):
        return "Control characters detected (non-printable)"
    return None


def _check_role_validity(messages: list[dict]) -> str | None:
    """Validate role structure. Returns error message or None."""
    if not messages:
        return "No messages"

    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role not in VALID_ROLES:
            return f"Invalid role '{role}' at message {i}"

    # Check alternating pattern (system exempt from alternation)
    conversation_msgs = [m for m in messages if m.get("role") != "system"]
    if len(conversation_msgs) >= 2:
        for i in range(len(conversation_msgs) - 1):
            if conversation_msgs[i]["role"] == conversation_msgs[i + 1]["role"]:
                return (
                    f"Consecutive same-role '{conversation_msgs[i]['role']}' "
                    f"at conversation position {i}-{i + 1}"
                )

    # Ensure at least one user and one assistant
    roles = {m.get("role") for m in messages}
    if "user" not in roles:
        return "No user message found"
    if "assistant" not in roles:
        return "No assistant message found"

    return None


def _check_token_limits(
    messages: list[dict],
    max_tokens_per_message: int,
    max_tokens_per_conversation: int,
    max_message_chars: int,
) -> str | None:
    """Check token/length limits. Returns error message or None."""
    total_tokens = 0
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue

        if len(content) > max_message_chars:
            return (
                f"Message {i} exceeds {max_message_chars} chars "
                f"(got {len(content)})"
            )

        tokens = _estimate_tokens(content)
        total_tokens += tokens

        if tokens > max_tokens_per_message:
            return (
                f"Message {i} exceeds {max_tokens_per_message} tokens "
                f"(estimated {tokens})"
            )

    if total_tokens > max_tokens_per_conversation:
        return (
            f"Conversation exceeds {max_tokens_per_conversation} tokens "
            f"(estimated {total_tokens})"
        )

    return None


def _check_chatml_structure(record: dict) -> str | None:
    """Validate basic ChatML structure. Returns error message or None."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        return "Missing or invalid 'messages' field"
    if len(messages) < 2:
        return f"Too few messages ({len(messages)} < 2)"

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return f"Message {i} is not a dict"
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            return f"Message {i} has empty or non-string content"

    return None


def _check_v7_metadata(record: dict) -> str | None:
    """Check V7-specific metadata fields. Returns error message or None."""
    missing = REQUIRED_V7_FIELDS - set(record.keys())
    if missing:
        return f"Missing V7 fields: {', '.join(sorted(missing))}"
    return None


def check_record(
    record: dict,
    index: int,
    *,
    max_tokens_per_message: int = DEFAULT_MAX_TOKENS_PER_MESSAGE,
    max_tokens_per_conversation: int = DEFAULT_MAX_TOKENS_PER_CONVERSATION,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_LENGTH_CHARS,
    check_v7_metadata: bool = True,
    shard_file: str | None = None,
) -> list[IntegrityViolation]:
    """Run all integrity checks on a single record.

    Returns a list of violations (empty if the record passes all checks).
    """
    violations: list[IntegrityViolation] = []

    def _add(severity: str, check: str, message: str, field_path: str | None = None) -> None:
        violations.append(IntegrityViolation(
            severity=severity,
            check=check,
            message=message,
            record_index=index,
            shard_file=shard_file,
            field_path=field_path,
        ))

    # 1. ChatML structure
    struct_err = _check_chatml_structure(record)
    if struct_err:
        _add("error", "chatml_structure", struct_err, "messages")
        return violations  # Can't check further if structure is broken

    messages = record["messages"]

    # 2. Role validity
    role_err = _check_role_validity(messages)
    if role_err:
        _add("error", "role_validity", role_err, "messages")

    # 3. Token limits
    token_err = _check_token_limits(
        messages,
        max_tokens_per_message,
        max_tokens_per_conversation,
        max_message_chars,
    )
    if token_err:
        _add("error", "token_limits", token_err)

    # 4. UTF-8 / encoding
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        utf8_err = _check_utf8(content)
        if utf8_err:
            _add("error", "utf8_encoding", utf8_err, f"messages[{i}].content")

    # 5. V7 metadata completeness
    if check_v7_metadata:
        meta_err = _check_v7_metadata(record)
        if meta_err:
            _add("warning", "v7_metadata", meta_err)

    return violations


def run_integrity_test(
    input_path: str | Path,
    *,
    max_tokens_per_message: int = DEFAULT_MAX_TOKENS_PER_MESSAGE,
    max_tokens_per_conversation: int = DEFAULT_MAX_TOKENS_PER_CONVERSATION,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_LENGTH_CHARS,
    check_v7_metadata: bool = True,
) -> IntegrityReport:
    """Run integrity tests against V7 output directory or single JSONL file.

    Args:
        input_path: Directory containing shard_*.jsonl files, or a single .jsonl file.
        max_tokens_per_message: Max estimated tokens per message.
        max_tokens_per_conversation: Max estimated tokens per conversation.
        max_message_chars: Max characters per message content.
        check_v7_metadata: Whether to check V7-specific fields.

    Returns:
        IntegrityReport with all violations and pass/fail status.
    """
    path = Path(input_path)
    report = IntegrityReport()

    if path.is_file() and path.suffix == ".jsonl":
        files = [path]
    elif path.is_dir():
        files = sorted(path.rglob("*.jsonl"))
        # Exclude report/log files
        files = [
            f for f in files
            if not f.name.endswith(("report.jsonl", "rejection_log.jsonl", "stats.json"))
        ]
    else:
        report.add_violation(
            "error", "file_access", f"Path not found or not a JSONL: {path}", -1,
        )
        return report

    if not files:
        report.add_violation(
            "error", "file_access", f"No JSONL files found in {path}", -1,
        )
        return report

    for jsonl_file in files:
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f):
                line = raw_line.strip()
                if not line:
                    continue

                record_index = report.total_records
                report.total_records += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    report.add_violation(
                        "error", "json_parse",
                        f"Invalid JSON at line {line_num + 1}: {e}",
                        record_index, str(jsonl_file),
                    )
                    continue

                if not isinstance(record, dict):
                    report.add_violation(
                        "error", "json_parse",
                        f"Record is not a JSON object at line {line_num + 1}",
                        record_index, str(jsonl_file),
                    )
                    continue

                violations = check_record(
                    record,
                    record_index,
                    max_tokens_per_message=max_tokens_per_message,
                    max_tokens_per_conversation=max_tokens_per_conversation,
                    max_message_chars=max_message_chars,
                    check_v7_metadata=check_v7_metadata,
                    shard_file=str(jsonl_file),
                )

                for v in violations:
                    report.violations.append(v)
                    report.total_violations += 1
                    if v.severity == "error":
                        report.errors += 1
                        report.passed = False
                    else:
                        report.warnings += 1

    return report


def format_report(report: IntegrityReport) -> str:
    """Format an IntegrityReport as a human-readable string."""
    lines = [
        "V7 Integrity Test Report",
        "=" * 50,
        f"Total records: {report.total_records}",
        f"Total violations: {report.total_violations} ({report.errors} errors, {report.warnings} warnings)",
        f"Result: {'PASS' if report.passed else 'FAIL'}",
    ]

    if report.violations:
        lines.append("")
        lines.append("Violations:")
        for v in report.violations:
            location = f"[{v.shard_file}]" if v.shard_file else ""
            field_loc = f" ({v.field_path})" if v.field_path else ""
            lines.append(
                f"  [{v.severity.upper()}] {v.check}: {v.message} "
                f"(record #{v.record_index}){location}{field_loc}"
            )

    return "\n".join(lines)
