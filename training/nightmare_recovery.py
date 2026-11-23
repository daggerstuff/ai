"""Pure checkpoint identity and corpus-deduplication rules; no model calls."""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from training.nightmare_records import GeneratedMessage


class CheckpointIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    id: str
    case_index: int | None = Field(default=None, ge=0)
    requested_cases: int | None = Field(default=None, ge=0)


class ResumeIdentityError(RuntimeError):
    """The checkpoint cannot identify completed generation slots unambiguously."""


def pending_indices(
    total: int, records: tuple[CheckpointIdentity, ...], skipped_ids: frozenset[str]
) -> tuple[int, ...]:
    """Decide pending slots before any model request; never guess legacy indices."""
    if total < 0:
        raise ResumeIdentityError("requested case count must be nonnegative")
    selected = [record for record in records if record.id in skipped_ids]
    if {record.id for record in selected} != skipped_ids:
        raise ResumeIdentityError("resume IDs require their matching checkpoint records")
    completed: set[int] = set()
    for record in selected:
        if record.case_index is None or record.requested_cases is None:
            raise ResumeIdentityError(
                "legacy checkpoint lacks slot metadata; preserve it and use an explicit migration or a new checkpoint"
            )
        if record.requested_cases != total or record.case_index >= total or record.case_index in completed:
            raise ResumeIdentityError("checkpoint slot identities conflict with the requested generation recipe")
        completed.add(record.case_index)
    return tuple(index for index in range(total) if index not in completed)


class StoredRecord(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    source: str | None = None
    messages: JsonValue = None


class StoredConversation(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    messages: tuple[GeneratedMessage, ...] = Field(min_length=2)


def session_digest(messages: tuple[GeneratedMessage, ...]) -> str:
    payload = json.dumps(
        [message.model_dump() for message in messages], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def existing_session_digests(path: Path) -> set[str]:
    """Read only the generator's prior conversations for sequential append deduplication."""
    if not path.exists():
        return set()
    digests: set[str] = set()
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            record = StoredRecord.model_validate_json(raw)
            if record.source == "empathy_nightmare_fuel":
                conversation = StoredConversation.model_validate({"messages": record.messages})
                digests.add(session_digest(conversation.messages))
    return digests
