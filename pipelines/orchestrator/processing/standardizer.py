"""
Data format standardization and conversion for the dataset pipeline.
Implements converters to transform raw data into the standard conversation schema.
"""

import datetime
import uuid
from typing import Any

from .conversation_schema import Conversation, Message


def from_simple_message_list(
    messages: list[dict[str, Any]],
    conversation_id: str | None = None,
    source: str | None = None,
) -> Conversation:
    """
    Convert a list of dicts with 'role' and 'content' to a Conversation.
    Each dict: {'role': str, 'content': str, ...}
    """
    msg_objs = []
    for m in messages:
        msg_objs.append(
            Message(
                role=m["role"],
                content=m["content"],
                timestamp=m.get("timestamp", datetime.datetime.utcnow().isoformat()),
                metadata=m.get("metadata", {}),
            )
        )
    return Conversation(
        conversation_id=conversation_id or str(uuid.uuid4()),
        messages=msg_objs,
        source=source,
        created_at=datetime.datetime.utcnow().isoformat(),
        updated_at=datetime.datetime.utcnow().isoformat(),
    )


def from_input_output_pair(
    input_text: str,
    output_text: str,
    input_role: str = "user",
    output_role: str = "assistant",
    conversation_id: str | None = None,
    source: str | None = None,
) -> Conversation:
    """
    Convert a simple input/output pair to a Conversation.
    """
    msgs = [
        Message(
            role=input_role,
            content=input_text,
            timestamp=datetime.datetime.utcnow().isoformat(),
            metadata={}
        ),
        Message(
            role=output_role,
            content=output_text,
            timestamp=datetime.datetime.utcnow().isoformat(),
            metadata={}
        ),
    ]
    return Conversation(
        conversation_id=conversation_id or str(uuid.uuid4()),
        messages=msgs,
        source=source,
        created_at=datetime.datetime.utcnow().isoformat(),
        updated_at=datetime.datetime.utcnow().isoformat(),
    )
