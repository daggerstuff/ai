import os

import pandas as pd

from ai.dataset_pipeline.psychology_loader import load_psychology_knowledge_csv

TEST_CSV = "test_psychology_knowledge.csv"
MALFORMED_CSV = "malformed_psychology_knowledge.csv"


def setup_module(module):
    # Create a valid test CSV
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "question": ["What is CBT?", "Define mindfulness."],
            "answer": ["Cognitive Behavioral Therapy", "Awareness of the present moment"],
        }
    )
    df.to_csv(TEST_CSV, index=False)
    # Create a malformed CSV
    with open(MALFORMED_CSV, "w") as f:
        f.write('id,question,answer\n1,"Unclosed quote,CBT\n2,Define mindfulness.,Awareness')


def teardown_module(module):
    if os.path.exists(TEST_CSV):
        os.remove(TEST_CSV)
    if os.path.exists(MALFORMED_CSV):
        os.remove(MALFORMED_CSV)


def test_load_valid_psychology_knowledge_csv():
    df = load_psychology_knowledge_csv(TEST_CSV)
    assert df is not None
    assert len(df) == 2
    assert "question" in df.columns
    assert df.iloc[0]["answer"] == "Cognitive Behavioral Therapy"


def test_load_missing_psychology_knowledge_csv():
    df = load_psychology_knowledge_csv("nonexistent_file.csv")
    assert df is None


def test_load_malformed_psychology_knowledge_csv():
    df = load_psychology_knowledge_csv(MALFORMED_CSV)
    assert df is None
