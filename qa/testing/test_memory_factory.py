import pytest

from ai.research.manager_factory import MemoryManagerFactory


def test_factory_creates_local_foresight_manager_only_when_explicit(monkeypatch):
    class FakeLocalManager:
        def __init__(self, *, db_path: str, bank_id: str) -> None:
            self.db_path = db_path
            self.bank_id = bank_id

    monkeypatch.setenv("MEMORY_PROVIDER", "local_foresight")
    monkeypatch.setenv("HINDSIGHT_LOCAL_DB_PATH", "/tmp/pixelated-memory.db")

    manager = MemoryManagerFactory(local_manager_class=FakeLocalManager).create_manager()

    assert isinstance(manager, FakeLocalManager)
    assert manager.db_path == "/tmp/pixelated-memory.db"
    assert manager.bank_id == "pixelated"


def test_factory_raises_when_memory_is_unconfigured(monkeypatch):
    monkeypatch.delenv("MEMORY_PROVIDER", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="No supported memory provider configured"):
        MemoryManagerFactory().create_manager()


def test_factory_rejects_noncanonical_local_aliases(monkeypatch):
    monkeypatch.setenv("MEMORY_PROVIDER", "foresight")
    monkeypatch.setenv("HINDSIGHT_LOCAL_DB_PATH", "/tmp/pixelated-memory.db")

    with pytest.raises(RuntimeError, match="No supported memory provider configured"):
        MemoryManagerFactory().create_manager()

    monkeypatch.setenv("MEMORY_PROVIDER", "local")

    with pytest.raises(RuntimeError, match="No supported memory provider configured"):
        MemoryManagerFactory().create_manager()


def test_factory_rejects_foresight_api_key_without_explicit_local_mode(monkeypatch):
    monkeypatch.delenv("MEMORY_PROVIDER", raising=False)
    monkeypatch.setenv("HINDSIGHT_API_KEY", "test-key")

    with pytest.raises(RuntimeError, match="No supported memory provider configured"):
        MemoryManagerFactory().create_manager()


def test_factory_allows_provider_override_without_env(monkeypatch):
    class FakeLocalManager:
        def __init__(self, *, db_path: str, bank_id: str) -> None:
            self.db_path = db_path
            self.bank_id = bank_id

    monkeypatch.delenv("MEMORY_PROVIDER", raising=False)
    monkeypatch.setenv("HINDSIGHT_LOCAL_DB_PATH", "/tmp/pixelated-memory.db")

    manager = MemoryManagerFactory(local_manager_class=FakeLocalManager).create_manager(provider="local_foresight")

    assert isinstance(manager, FakeLocalManager)


def test_factory_requires_explicit_local_db_path(monkeypatch):
    monkeypatch.setenv("MEMORY_PROVIDER", "local_foresight")
    monkeypatch.delenv("HINDSIGHT_LOCAL_DB_PATH", raising=False)

    with pytest.raises(RuntimeError, match="FORESIGHT_LOCAL_DB_PATH or db_path is required"):
        MemoryManagerFactory().create_manager()
