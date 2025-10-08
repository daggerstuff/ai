import pytest
import torch

from ai.pixel.training.multi_objective_loss import MultiObjectiveLoss


@pytest.fixture
def dummy_outputs_targets():
    outputs = {
        "language": torch.tensor([[0.1, 0.9], [0.8, 0.2]], requires_grad=True),
        "eq": torch.tensor([0.5, 0.7], requires_grad=True),
        "persona": torch.tensor([[0.3, 0.7], [0.6, 0.4]], requires_grad=True),
        "clinical": torch.tensor([0.2, 0.8], requires_grad=True),
        "empathy": torch.tensor([0.4, 0.6], requires_grad=True),
    }
    targets = {
        "language": torch.tensor([1, 0]),
        "eq": torch.tensor([0.6, 0.8]),
        "persona": torch.tensor([1, 0]),
        "clinical": torch.tensor([0.0, 1.0]),
        "empathy": torch.tensor([0.5, 0.7]),
    }
    return outputs, targets


def test_multi_objective_loss_forward(dummy_outputs_targets):
    outputs, targets = dummy_outputs_targets
    loss_module = MultiObjectiveLoss()
    loss = loss_module(outputs, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert loss.item() > 0


def test_multi_objective_loss_weights(dummy_outputs_targets):
    outputs, targets = dummy_outputs_targets
    weights = {
        "language": 2.0,
        "eq": 0.5,
        "persona": 1.5,
        "clinical": 1.0,
        "empathy": 0.0,
    }
    loss_module = MultiObjectiveLoss(weights=weights)
    loss = loss_module(outputs, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert loss.item() > 0


def test_multi_objective_loss_gradients(dummy_outputs_targets):
    outputs, targets = dummy_outputs_targets
    loss_module = MultiObjectiveLoss()
    loss = loss_module(outputs, targets)
    loss.backward()
    for key in outputs:
        assert outputs[key].grad is not None


def test_multi_objective_loss_zero_weights(dummy_outputs_targets):
    outputs, targets = dummy_outputs_targets
    weights = {
        "language": 0.0,
        "eq": 0.0,
        "persona": 0.0,
        "clinical": 0.0,
        "empathy": 0.0,
    }
    loss_module = MultiObjectiveLoss(weights=weights)
    loss = loss_module(outputs, targets)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
