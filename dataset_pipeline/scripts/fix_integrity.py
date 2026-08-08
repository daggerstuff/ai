"""Fix V7 shard integrity violations: mojibake + token limits (v2).

Uses regex-based mojibake fix that handles mixed-content strings
where latin-1 round-trip fails (e.g. emoji + mojibake in same string).
Also truncates messages exceeding token limits.
"""

import json
import re
from pathlib import Path

MAX_CHARS = 16380  # under both 20000 char and 4096 token limits


def fix_mojibake_regex(text: str) -> str:
    """Fix double-encoded UTF-8 by converting mojibake byte pairs
    to their correct Unicode codepoints using regex substitution."""

    # 2-byte UTF-8: \xc2/\xc3 followed by \x80-\xbf
    def replace_2byte(m):
        b1 = ord(m.group(1))
        b2 = ord(m.group(2))
        codepoint = ((b1 & 0x1F) << 6) | (b2 & 0x3F)
        return chr(codepoint)

    text = re.sub(r"(\xc2|\xc3)([\x80-\xbf])", replace_2byte, text)

    # 3-byte UTF-8: \xe2 followed by \x82 and \x80-\xbf (smart quotes, etc.)
    def replace_3byte(m):
        b1 = ord(m.group(1))
        b2 = ord(m.group(2))
        b3 = ord(m.group(3))
        codepoint = ((b1 & 0x0F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F)
        return chr(codepoint)

    text = re.sub(r"(\xe2)([\x82])([\x80-\xbf])", replace_3byte, text)

    return text


def truncate_messages(messages: list[dict]) -> bool:
    """Truncate messages exceeding char or token limits. Returns True if changed."""
    changed = False
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        if len(content) > MAX_CHARS:
            msg["content"] = content[:MAX_CHARS] + "…"
            changed = True
    return changed


def process_shard(path: Path) -> tuple[int, int]:
    records = []
    total = 0
    fixed = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            total += 1
            rec = json.loads(line)
            changed = False

            for msg in rec.get("messages", []):
                content = msg.get("content", "")
                if isinstance(content, str):
                    new_content = fix_mojibake_regex(content)
                    if new_content != content:
                        msg["content"] = new_content
                        changed = True
                # Drop messages with empty/non-string content
                if not isinstance(content, str) or content.strip() == "":
                    msg["content"] = "[content unavailable]"
                    changed = True

            if truncate_messages(rec.get("messages", [])):
                changed = True

            if changed:
                fixed += 1
            records.append(rec)

    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return total, fixed


def main():
    shard_dir = Path("/home/vivi/pixelated/ai/data/prepared/v7_master")
    shards = sorted(shard_dir.glob("shard_*.jsonl"))
    print(f"Found {len(shards)} shards")

    grand_total = 0
    grand_fixed = 0

    for shard in shards:
        total, fixed = process_shard(shard)
        grand_total += total
        grand_fixed += fixed
        print(f"  {shard.name}: {total} records, {fixed} fixed")

    print(f"\nTotal: {grand_total} records, {grand_fixed} fixed")


if __name__ == "__main__":
    main()
