"""Featherless API key pool with rotate-on-quota failover.

The host script loads dotenv (override=True) before importing this module.
Keys are read once at pool construction; on quota/rate errors (429/402) the
caller rotates to the next key and retries with the existing backoff.

Env names: FEATHERLESS_API_KEY (primary), FEATHERLESS_API_KEY_2, ...
Duplicates are removed (order preserved). Logs only show the key tail.
"""

import os

_ROTATE_STATUSES = ("http_429", "http_402")


def is_rotate_status(error: str) -> bool:
    """True when a transient error indicates quota/rate exhaustion of the
    current key — rotate to the next key before retrying."""
    return str(error).startswith(_ROTATE_STATUSES)


class KeyPool:
    """Rotating pool of Featherless API keys. Methods contain no awaits, so
    the shared index is safe under a single asyncio event loop."""

    def __init__(self, *env_names: str) -> None:
        keys: list[str] = []
        for name in env_names:
            value = os.environ.get(name, "").strip()
            if value and value not in keys:
                keys.append(value)
        if not keys:
            raise SystemExit(
                f"no Featherless API keys found (expected one of: {', '.join(env_names)})"
            )
        self.keys = keys
        self.idx = 0

    def __len__(self) -> int:
        return len(self.keys)

    def current(self) -> str:
        return self.keys[self.idx]

    def rotate(self) -> str:
        """Advance to the next key and return it."""
        self.idx = (self.idx + 1) % len(self.keys)
        return self.keys[self.idx]

    def describe(self) -> str:
        return f"{len(self.keys)} key(s), active ...{self.current()[-8:]}"
