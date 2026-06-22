from infrastructure.database.cms_redis_config import get_invalidation_keys


def test_get_invalidation_keys_document_updated():
    """Test get_invalidation_keys for document_updated event."""
    keys = get_invalidation_keys("document_updated", documentId="123", period="daily")
    assert "doc:123" in keys
    assert "stats:documents:daily" in keys
    expected_len = 2
    assert len(keys) == expected_len


def test_get_invalidation_keys_missing_kwargs():
    """Test get_invalidation_keys when kwargs are missing."""
    # This will leave the placeholder unreplaced since kwargs is empty
    keys = get_invalidation_keys("document_updated")
    assert len(keys) == 2
    assert "doc:{documentId}" in keys
    assert "stats:documents:{period}" in keys


def test_get_invalidation_keys_unknown_event():
    """Test get_invalidation_keys for an unknown event."""
    keys = get_invalidation_keys("unknown_event", foo="bar")
    assert keys == []


def test_get_invalidation_keys_project_updated():
    """Test get_invalidation_keys for project_updated event."""
    keys = get_invalidation_keys("project_updated", period="monthly")
    assert "stats:projects:monthly" in keys
    expected_len = 1
    assert len(keys) == expected_len
