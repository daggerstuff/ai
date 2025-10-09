import os
import tempfile

import pandas as pd

from ai.dataset_pipeline.dataset_inventory import get_dataset_metadata, scan_datasets

EXPECTED_DATASET_COUNT = 2


def test_get_dataset_metadata_valid():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df.to_csv(tmp.name, index=False)
        meta = get_dataset_metadata(tmp.name)
        assert meta is not None
        assert meta["name"] == os.path.basename(tmp.name)
        assert meta["size_bytes"] > 0
        assert "a" in meta["columns"]
        assert "b" in meta["columns"]
        assert isinstance(meta["preview_rows"], list)
        os.remove(tmp.name)


def test_get_dataset_metadata_missing():
    meta = get_dataset_metadata("nonexistent_file.csv")
    assert meta is None


def test_scan_datasets(tmp_path):
    # Create two CSVs
    df1 = pd.DataFrame({"x": [1, 2]})
    df2 = pd.DataFrame({"y": [3, 4]})
    csv1 = tmp_path / "a.csv"
    csv2 = tmp_path / "b.csv"
    df1.to_csv(csv1, index=False)
    df2.to_csv(csv2, index=False)
    inventory = scan_datasets(str(tmp_path))
    assert isinstance(inventory, list)
    assert len(inventory) == EXPECTED_DATASET_COUNT
    names = [meta["name"] for meta in inventory]
    assert "a.csv" in names
    assert "b.csv" in names
