"""
Tests for Defense Mechanism Detection Module

Covers dataset loading, model architecture, loss functions,
prediction output format, and cross-validation integrity.
"""

import json
from unittest.mock import MagicMock

import pytest

from ai.training.defense_mechanisms import (
    DEFENSE_LABELS,
    DEFENSE_MATURITY,
)
from ai.training.defense_mechanisms.dataset import (
    DefenseDataset,
    DialogueSample,
    _normalize_text,
    compute_class_weights,
    create_fold_datasets,
    format_dialogue,
    load_psydefconv,
    normalize_speaker,
)
from ai.training.defense_mechanisms.model import (
    DefensePrediction,
    FocalLoss,
    compute_r_drop_loss,
)
from ai.tools.utilities.torch_proxy import torch

# -- Fixtures --


@pytest.fixture
def sample_turns():
    """Create a realistic multi-turn dialogue."""
    return [
        {"speaker": "Supporter", "text": "Hello, how are you today?"},
        {
            "speaker": "Seeker",
            "text": "Not great. I've been feeling really down lately.",
        },
        {
            "speaker": "Supporter",
            "text": "I'm sorry to hear that. Can you tell me more?",
        },
        {
            "speaker": "Seeker",
            "text": "It's work. Everything is fine, I just need to try harder. It's my own fault really.",
        },
        {
            "speaker": "Supporter",
            "text": "It sounds like you might be being hard on yourself.",
        },
        {
            "speaker": "Seeker",
            "text": "No, it's fine. I don't want to talk about it actually. Let's discuss something else.",
        },
    ]


@pytest.fixture
def sample_dialogue_samples():
    """Create a set of DialogueSample objects for testing."""
    samples = []
    for i in range(20):
        dialogue_id = f"d_{i // 4}"  # 5 unique dialogues
        samples.append(
            DialogueSample(
                sample_id=f"s_{i}",
                dialogue_id=dialogue_id,
                turns=[
                    {"speaker": "Supporter", "text": f"Turn {j} from supporter"}
                    if j % 2 == 0
                    else {"speaker": "Seeker", "text": f"Turn {j} from seeker"}
                    for j in range(6)
                ],
                target_text=f"Turn {i % 6} from seeker",
                label=i % 9,
            )
        )
    return samples


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.randint(0, 1000, (1, 64)),
        "attention_mask": torch.ones(1, 64, dtype=torch.long),
    }
    return tokenizer


@pytest.fixture
def sample_train_json(tmp_path):
    """Create a temporary train.json file."""
    data = []
    for i in range(30):
        data.append(
            {
                "id": f"sample_{i}",
                "dialogue_id": f"dialogue_{i // 6}",
                "dialogue": [
                    {
                        "speaker": "Supporter",
                        "text": "How are you feeling?",
                    },
                    {
                        "speaker": "Seeker",
                        "text": f"I feel {['fine', 'sad', 'angry'][i % 3]}",
                    },
                    {
                        "speaker": "Supporter",
                        "text": "Tell me more about that.",
                    },
                    {
                        "speaker": "Seeker",
                        "text": f"Sample target text {i}",
                    },
                ],
                "current_text": f"Sample target text {i}",
                "label": i % 9,
            }
        )

    json_path = tmp_path / "train.json"
    with open(json_path, "w") as f:
        json.dump(data, f)
    return str(json_path)


# -- Speaker Normalization Tests --


class TestSpeakerNormalization:
    def test_standard_speakers(self):
        assert normalize_speaker("Seeker") == "Seeker"
        assert normalize_speaker("Supporter") == "Supporter"

    def test_case_insensitive(self):
        assert normalize_speaker("seeker") == "Seeker"
        assert normalize_speaker("SUPPORTER") == "Supporter"

    def test_alternative_names(self):
        assert normalize_speaker("help-seeker") == "Seeker"
        assert normalize_speaker("helper") == "Supporter"
        assert normalize_speaker("patient") == "Seeker"
        assert normalize_speaker("therapist") == "Supporter"

    def test_colon_stripping(self):
        assert normalize_speaker("Seeker:") == "Seeker"
        assert normalize_speaker("supporter:") == "Supporter"

    def test_unknown_defaults_to_seeker(self):
        assert normalize_speaker("unknown_role") == "Seeker"


# -- Dialogue Formatting Tests --


class TestDialogueFormatting:
    def test_basic_formatting(self, sample_turns):
        result = format_dialogue(
            turns=sample_turns,
            target_text="No, it's fine. I don't want to talk about it actually. Let's discuss something else.",
        )
        assert "<t>" in result
        assert "</t>" in result
        assert "Seeker:" in result
        assert "Supporter:" in result

    def test_target_wrapping(self, sample_turns):
        target = "Not great. I've been feeling really down lately."
        result = format_dialogue(turns=sample_turns, target_text=target)
        assert f"<t>{target}</t>" in result

    def test_truncation(self, sample_turns):
        result = format_dialogue(
            turns=sample_turns,
            target_text=sample_turns[-1]["text"],
            max_turns=3,
        )
        # Should only keep last 3 turns
        assert "Hello, how are you today?" not in result

    def test_empty_dialogue(self):
        result = format_dialogue(turns=[], target_text="test")
        assert result == ""


# -- Class Weight Tests --


class TestClassWeights:
    def test_uniform_distribution(self):
        labels = list(range(9)) * 10
        weights = compute_class_weights(labels)
        assert weights.shape == (9,)
        # With uniform distribution, weights should be approximately equal
        assert torch.allclose(weights, torch.ones(9), atol=0.01)

    def test_imbalanced_distribution(self):
        # Simulate PSYDEFCONV distribution: class 7 dominant
        labels = [7] * 100 + [0] * 20 + [6] * 10 + [1] * 5
        weights = compute_class_weights(labels)
        assert weights.shape == (9,)
        # Rare classes should have higher weights
        assert weights[1] > weights[7]
        assert weights[0] > weights[7]

    def test_sum_equals_num_labels(self):
        labels = [0, 0, 0, 1, 1, 2, 7, 7, 7, 7, 7]
        weights = compute_class_weights(labels)
        assert abs(weights.sum().item() - 9.0) < 0.01

    def test_no_zero_weights(self):
        labels = [0, 1]  # Most classes have zero count
        weights = compute_class_weights(labels)
        assert (weights > 0).all()


# -- Dataset Loading Tests --


class TestDatasetLoading:
    def test_load_json(self, sample_train_json):
        samples = load_psydefconv(sample_train_json, has_labels=True)
        assert len(samples) == 30
        assert all(s.label is not None for s in samples)
        assert all(0 <= s.label < 9 for s in samples)

    def test_load_without_labels(self, sample_train_json):
        samples = load_psydefconv(sample_train_json, has_labels=False)
        assert len(samples) == 30
        assert all(s.label is None for s in samples)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_psydefconv("/nonexistent/path.json")

    def test_dialogue_ids_present(self, sample_train_json):
        samples = load_psydefconv(sample_train_json)
        assert all(s.dialogue_id for s in samples)


class TestDefenseDataset:
    def test_len(self, sample_dialogue_samples, mock_tokenizer):
        ds = DefenseDataset(sample_dialogue_samples, mock_tokenizer, max_length=64)
        assert len(ds) == 20

    def test_getitem_with_labels(self, sample_dialogue_samples, mock_tokenizer):
        ds = DefenseDataset(sample_dialogue_samples, mock_tokenizer, max_length=64)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["labels"].dtype == torch.long

    def test_get_labels(self, sample_dialogue_samples, mock_tokenizer):
        ds = DefenseDataset(sample_dialogue_samples, mock_tokenizer, max_length=64)
        labels = ds.get_labels()
        assert len(labels) == 20

    def test_get_dialogue_ids(self, sample_dialogue_samples, mock_tokenizer):
        ds = DefenseDataset(sample_dialogue_samples, mock_tokenizer, max_length=64)
        ids = ds.get_dialogue_ids()
        assert len(ids) == 20
        assert len(set(ids)) == 5  # 5 unique dialogues


# -- Cross-Validation Tests --


class TestCrossValidation:
    def test_no_dialogue_leakage(self, sample_dialogue_samples, mock_tokenizer):
        train_ds, val_ds = create_fold_datasets(
            samples=sample_dialogue_samples,
            tokenizer=mock_tokenizer,
            num_folds=5,
            fold_index=0,
            max_length=64,
        )
        train_ids = set(train_ds.get_dialogue_ids())
        val_ids = set(val_ds.get_dialogue_ids())
        assert len(train_ids & val_ids) == 0, "Dialogue leakage detected between train and val"

    def test_all_samples_covered(self, sample_dialogue_samples, mock_tokenizer):
        total_samples = 0
        for fold_idx in range(5):
            _, val_ds = create_fold_datasets(
                samples=sample_dialogue_samples,
                tokenizer=mock_tokenizer,
                num_folds=5,
                fold_index=fold_idx,
                max_length=64,
            )
            total_samples += len(val_ds)
        assert total_samples == 20

    def test_invalid_fold_index(self, sample_dialogue_samples, mock_tokenizer):
        with pytest.raises(ValueError):
            create_fold_datasets(
                samples=sample_dialogue_samples,
                tokenizer=mock_tokenizer,
                num_folds=5,
                fold_index=10,
                max_length=64,
            )


# -- Focal Loss Tests --


class TestFocalLoss:
    def test_basic_computation(self):
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 9)
        targets = torch.randint(0, 9, (4,))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0

    def test_with_class_weights(self):
        weights = torch.ones(9)
        weights[7] = 0.5  # Lower weight for dominant class
        loss_fn = FocalLoss(alpha=weights, gamma=2.0)
        logits = torch.randn(4, 9)
        targets = torch.randint(0, 9, (4,))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_label_smoothing(self):
        loss_no_smooth = FocalLoss(gamma=2.0, label_smoothing=0.0)
        loss_smooth = FocalLoss(gamma=2.0, label_smoothing=0.1)
        logits = torch.randn(8, 9)
        targets = torch.randint(0, 9, (8,))
        l1 = loss_no_smooth(logits, targets)
        l2 = loss_smooth(logits, targets)
        # Both should be valid losses
        assert l1.item() >= 0
        assert l2.item() >= 0

    def test_gamma_zero_is_weighted_ce(self):
        loss_fn = FocalLoss(gamma=0.0)
        logits = torch.randn(4, 9)
        targets = torch.randint(0, 9, (4,))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0


# -- R-Drop Tests --


class TestRDrop:
    def test_identical_logits_zero_loss(self):
        logits = torch.randn(4, 9)
        loss = compute_r_drop_loss(logits, logits.clone())
        assert abs(loss.item()) < 1e-5

    def test_different_logits_positive_loss(self):
        logits_1 = torch.randn(4, 9)
        logits_2 = torch.randn(4, 9)
        loss = compute_r_drop_loss(logits_1, logits_2)
        assert loss.item() > 0

    def test_symmetry(self):
        logits_1 = torch.randn(4, 9)
        logits_2 = torch.randn(4, 9)
        loss_12 = compute_r_drop_loss(logits_1, logits_2)
        loss_21 = compute_r_drop_loss(logits_2, logits_1)
        assert abs(loss_12.item() - loss_21.item()) < 1e-5


# -- Prediction Output Tests --


class TestPredictionOutput:
    def test_prediction_dataclass(self):
        pred = DefensePrediction(
            label=7,
            label_name="High-Adaptive",
            confidence=0.85,
            maturity_score=1.0,
            probabilities=[0.01] * 7 + [0.85] + [0.08],
        )
        assert pred.label == 7
        assert pred.label_name == "High-Adaptive"
        assert 0.0 <= pred.confidence <= 1.0
        assert pred.maturity_score == 1.0

    def test_maturity_score_normalization(self):
        for label, maturity in DEFENSE_MATURITY.items():
            if maturity is not None:
                assert 0.0 <= maturity <= 1.0, f"Label {label} maturity {maturity} out of range"

    def test_all_labels_have_names(self):
        for label_id in range(9):
            assert label_id in DEFENSE_LABELS, f"Label {label_id} missing from DEFENSE_LABELS"

    def test_maturity_monotonic(self):
        """Defense maturity should increase with label number (1-7)."""
        defense_levels = [1, 2, 3, 4, 5, 6, 7]
        maturities = [DEFENSE_MATURITY[lvl] for lvl in defense_levels]
        for i in range(len(maturities) - 1):
            assert maturities[i] < maturities[i + 1], (
                f"Maturity not monotonic: "
                f"level {defense_levels[i]}={maturities[i]} >= "
                f"level {defense_levels[i + 1]}={maturities[i + 1]}"
            )


# -- Text Normalization Tests --


class TestNormalization:
    def test_whitespace_collapse(self):
        assert _normalize_text("hello   world") == "hello world"

    def test_case_folding(self):
        assert _normalize_text("Hello World") == "hello world"

    def test_strip(self):
        assert _normalize_text("  hello  ") == "hello"
