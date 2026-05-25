"""NVIDIA client helpers for optional upstream integrations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NvidiaClients:
    """Create and expose commonly-used NVIDIA-backed clients lazily."""

    config: dict | None = None

    def __post_init__(self) -> None:
        self.config = self.config or {}
        self._clients: dict[str, object] = {}

    def get_client(self, kind: str) -> object:
        """Return a memoized client for a given integration kind."""

        if kind in self._clients:
            return self._clients[kind]

        if kind == "api":
            client = self._build_basic_client()
        else:
            raise ValueError(f"Unsupported client kind: {kind}")

        self._clients[kind] = client
        return client

    def _build_basic_client(self) -> object:
        """Build a lightweight mock-like client when upstream SDKs are unavailable."""

        class _FallbackClient:
            def __getattr__(self, name: str):
                raise RuntimeError("NVIDIA API client is unavailable in this environment.")

            def __repr__(self) -> str:
                return "NvidiaClients.FallbackClient()"

        # Keep lazy-import behavior for optional dependencies.
        try:
            import nemo

            return nemo
        except Exception:
            return _FallbackClient()


__all__ = ["NvidiaClients"]
