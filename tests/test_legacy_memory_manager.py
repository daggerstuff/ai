from ai.api.memory.memory_manager import get_memory_manager
from ai.memory.local_hindsight_manager import LocalHindsightMemoryManager


def test_legacy_memory_manager_uses_configured_backend(monkeypatch) -> None:
    manager = LocalHindsightMemoryManager(db_path=":memory:")

    monkeypatch.setattr(
        "ai.api.memory.memory_manager._memory_manager_instance",
        None,
        raising=False,
    )
    monkeypatch.setattr(
        "ai.api.memory.memory_manager.get_backend_memory_manager",
        lambda: manager,
    )

    legacy = get_memory_manager()

    assert legacy.client is manager
    assert legacy.get_memory_stats("session-1")["provider"] == "LocalHindsightMemoryManager"
