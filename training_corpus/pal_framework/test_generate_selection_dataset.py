from __future__ import annotations

import json
import random
from pathlib import Path

import generate_selection_dataset as g

PERSONAS = [
    {
        "demographics": {"age": 45, "gender": "female", "location": "Hanoi"},
        "healthcare_behavior": {"health_literacy": "low", "preference": "traditional medicine"},
    },
    {
        "demographics": {"age": 30, "gender": "male", "location": "HCMC"},
        "healthcare_behavior": {"health_literacy": "high", "preference": "standard medicine"},
    },
    {
        "demographics": {"age": 60, "gender": "female", "location": "Da Nang"},
        "healthcare_behavior": {"health_literacy": "average", "preference": "standard medicine"},
    },
    {
        "demographics": {"age": 22, "gender": "male", "location": "Hanoi"},
        "healthcare_behavior": {"health_literacy": "low", "preference": "traditional medicine"},
    },
    {
        "demographics": {"age": 50, "gender": "female", "location": "Hue"},
        "healthcare_behavior": {"health_literacy": "high", "preference": "traditional medicine"},
    },
]


def test_sample_distractors():
    rng = random.Random(0)
    distractors = g.sample_distractors(PERSONAS, PERSONAS[0], 3, rng)
    assert len(distractors) == 3
    assert PERSONAS[0] not in distractors


def test_sample_distractors_insufficient_pool():
    rng = random.Random(0)
    try:
        g.sample_distractors(PERSONAS[:2], PERSONAS[0], 3, rng)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_build_messages_chatml():
    messages = g.build_selection_messages("hi", ["A", "B", "C"], 1)
    assert g.is_chatml_compliant(messages)
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "2"


def test_build_messages_bad_index():
    try:
        g.build_selection_messages("hi", ["A", "B"], 5)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_is_chatml_compliant_rejects_garbage():
    assert not g.is_chatml_compliant([])
    assert not g.is_chatml_compliant([{"role": "bot", "content": "x"}])
    assert not g.is_chatml_compliant([{"role": "user", "content": 1}])


def test_build_example_metadata():
    rng = random.Random(1)
    example = g.build_selection_example("d", PERSONAS, 0, n_distractors=3, rng=rng)
    assert g.is_chatml_compliant(example.messages)
    assert example.metadata["correct_option"] == int(example.messages[-1]["content"])
    assert example.metadata["n_options"] == 4  # 1 correct + 3 distractors


def test_no_json_leakage():
    rng = random.Random(2)
    example = g.build_selection_example("d", PERSONAS, 0, n_distractors=3, rng=rng)
    user = example.messages[1]["content"]
    assert "{" not in user and "}" not in user


def test_generate_dataset_roundtrip(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text(
        json.dumps({"dialogue": "x", "personas": PERSONAS, "correct_index": 2}) + "\n",
        encoding="utf-8",
    )
    n = g.generate_dataset(inp, out, seed=3)
    assert n == 1
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert g.is_chatml_compliant(record["messages"])
    assert record["metadata"]["correct_option"] == int(record["messages"][-1]["content"])
