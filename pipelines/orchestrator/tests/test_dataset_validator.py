import os

import pandas as pd

from ai.pipelines.orchestrator.dataset_validator import validate_dataset

TEST_CSV = "test_validator_valid.csv"
MISSING_COLS_CSV = "test_validator_missing_cols.csv"
FEW_ROWS_CSV = "test_validator_few_rows.csv"
MALFORMED_CSV = "test_validator_malformed.csv"


def setup_module(_):
    # Valid CSV
    df = pd.DataFrame({"id": [1, 2], "text": ["A", "B"], "label": [0, 1]})
    df.to_csv(TEST_CSV, index=False)
    # Missing columns
    df2 = pd.DataFrame({"id": [1, 2], "text": ["A", "B"]})
    df2.to_csv(MISSING_COLS_CSV, index=False)
    # Few rows
    df3 = pd.DataFrame({"id": [1], "text": ["A"], "label": [0]})
    df3.to_csv(FEW_ROWS_CSV, index=False)
    # Malformed CSV
    with open(MALFORMED_CSV, "w") as f:
        f.write('id,text,label\n1,"Unclosed quote\n2,B,1')


def teardown_module(_):
    for f in [TEST_CSV, MISSING_COLS_CSV, FEW_ROWS_CSV, MALFORMED_CSV]:
        if os.path.exists(f):
            os.remove(f)


def test_valid_dataset():
    result = validate_dataset(TEST_CSV, required_columns=["id", "text", "label"], min_rows=2)
    assert result["exists"]
    assert result["valid_columns"]
    assert result["enough_rows"]
    assert not result["errors"]


def test_missing_file():
    result = validate_dataset("nonexistent.csv", required_columns=["id"], min_rows=1)
    assert not result["exists"]
    assert not result["valid_columns"]
    assert not result["enough_rows"]
    assert result["errors"]


def test_missing_columns():
    result = validate_dataset(
        MISSING_COLS_CSV, required_columns=["id", "text", "label"], min_rows=1
    )
    assert result["exists"]
    assert not result["valid_columns"]
    assert result["enough_rows"]
    assert result["errors"]


def test_not_enough_rows():
    result = validate_dataset(FEW_ROWS_CSV, required_columns=["id", "text", "label"], min_rows=2)
    assert result["exists"]
    assert result["valid_columns"]
    assert not result["enough_rows"]
    assert result["errors"]


def test_malformed_csv():
    result = validate_dataset(MALFORMED_CSV, required_columns=["id", "text", "label"], min_rows=1)
    assert result["exists"]
    assert not result["valid_columns"]
    assert not result["enough_rows"]
    assert result["errors"]
