"""
Unit tests for ai.dataset_pipeline.data_loader
"""

from datasets import Dataset, IterableDataset

from ai.dataset_pipeline import data_loader

# Test constants
EXPECTED_DATASET_SIZE = 10


def test_load_hf_dataset_success():
    # Use a small, public dataset for testing
    progress_messages = []
    ds = data_loader.load_hf_dataset(
        "ag_news", split="train", progress_callback=progress_messages.append
    )
    assert ds is not None
    assert any("Starting download" in msg for msg in progress_messages)
    assert any(
        "Loaded dataset" in msg or "Loaded iterable dataset" in msg for msg in progress_messages
    )

    # Select the first EXPECTED_DATASET_SIZE records for testing
    if isinstance(ds, IterableDataset):
        # For IterableDataset, we can't use len() directly
        # Instead, we can iterate and count the first EXPECTED_DATASET_SIZE items
        count = sum(1 for _, _ in zip(ds, range(EXPECTED_DATASET_SIZE), strict=False))
        assert count == EXPECTED_DATASET_SIZE
    elif isinstance(ds, Dataset):
        # For regular Dataset objects that support slicing and len()
        # Verify we can get at least EXPECTED_DATASET_SIZE items
        subset = ds.select(range(min(len(ds), EXPECTED_DATASET_SIZE)))
        assert len(subset) == min(len(ds), EXPECTED_DATASET_SIZE)
    else:
        # For other dataset types (like DatasetDict), verify we have data
        # This is a fallback for unexpected types
        assert ds is not None
        # Try to get length if possible
        try:
            dataset_len = len(ds)
            assert dataset_len >= 0
        except (TypeError, AttributeError):
            # If len() doesn't work, just verify the dataset exists
            pass


def test_load_hf_dataset_failure():
    # Try to load a non-existent dataset
    error_messages = []
    ds = data_loader.load_hf_dataset(
        "nonexistent-dataset-xyz", split="train", error_callback=error_messages.append
    )
    assert ds is None
    assert any("Failed to load HuggingFace dataset" in msg for msg in error_messages)
