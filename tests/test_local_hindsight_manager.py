from ai.memory.local_hindsight_manager import LocalHindsightMemoryManager


def test_local_hindsight_manager_crud_and_recall(tmp_path) -> None:
    manager = LocalHindsightMemoryManager(
        db_path=str(tmp_path / "local-hindsight.db"),
        bank_id="pixelated",
    )

    memory_id = manager.add_memory(
        "Vivi prefers self-hosted memory over cloud billing surprises",
        "vivi",
        metadata={"project_id": "pixelated", "visibility": "private"},
        category="preference",
    )

    stored = manager.get_memory(memory_id)
    assert stored is not None
    assert stored["content"] == "Vivi prefers self-hosted memory over cloud billing surprises"
    assert stored["metadata"]["project_id"] == "pixelated"

    results = manager.search_memories("self-hosted memory", "vivi", limit=5)
    assert results
    assert results[0]["document_id"] == memory_id

    assert manager.update_memory(
        memory_id,
        "Vivi prefers local Hindsight APIs over cloud billing surprises",
        metadata={"visibility": "shared", "project_id": "pixelated", "category": "preference"},
    )
    updated = manager.get_memory(memory_id)
    assert updated is not None
    assert updated["content"] == "Vivi prefers local Hindsight APIs over cloud billing surprises"
    assert updated["metadata"]["visibility"] == "shared"

    all_memories = manager.get_all_memories("vivi", limit=10)
    assert len(all_memories) == 1

    assert manager.delete_memory(memory_id) is True
    assert manager.get_memory(memory_id) is None
