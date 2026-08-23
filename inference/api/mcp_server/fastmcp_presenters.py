from __future__ import annotations

from .fastmcp_store import memory_store_result_id


def memory_store_success_message(
    *,
    user_id: str,
    content: str,
    category: str,
    result,
) -> str:
    lines = [
        f"✅ **Memory Secured** for {user_id}",
        f"- **Content:** {content}",
        f"- **Category:** {category}",
    ]
    record_id = memory_store_result_id(result)
    if record_id:
        lines.append(f"- **ID:** {record_id}")
    return "\n".join(lines)
