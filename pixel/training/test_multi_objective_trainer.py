import pytest
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from ai.pixel.models.pixel_base_model import PixelBaseModel
from ai.pixel.training.multi_objective_loss import MultiObjectiveLoss
from ai.pixel.training.multi_objective_trainer import (
    MultiObjectiveTrainer,
    TrainerConfig,
)


class DummyPixelDataset(Dataset):
    def __init__(self, num_samples=10):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        outputs = {
            "language": torch.randn(2, requires_grad=True),
            "eq": torch.randn(1, requires_grad=True),
            "persona": torch.randn(2, requires_grad=True),
            "clinical": torch.randn(1, requires_grad=True),
            "empathy": torch.randn(1, requires_grad=True),
        }
        targets = {
            "language": torch.randint(0, 2, (1,)),
            "eq": torch.rand(1),
            "persona": torch.randint(0, 2, (1,)),
            "clinical": torch.rand(1),
            "empathy": torch.rand(1),
        }
        return {"outputs": outputs, "targets": targets}


@pytest.fixture
def dummy_train_loader():
    dataset = DummyPixelDataset(num_samples=8)
    return DataLoader(dataset, batch_size=2)


@pytest.fixture
def dummy_val_loader():
    dataset = DummyPixelDataset(num_samples=4)
    return DataLoader(dataset, batch_size=2)


@pytest.fixture
def dummy_model():
    return PixelBaseModel()


@pytest.fixture
def dummy_loss():
    return MultiObjectiveLoss()


@pytest.fixture
def dummy_optimizer(dummy_model):
    return Adam(dummy_model.parameters(), lr=1e-3)


def test_trainer_runs_one_epoch(dummy_model, dummy_loss, dummy_optimizer, dummy_train_loader):
    config = TrainerConfig(val_loader=None, max_epochs=1)
    trainer = MultiObjectiveTrainer(
        model=dummy_model,
        loss_fn=dummy_loss,
        optimizer=dummy_optimizer,
        train_loader=dummy_train_loader,
        config=config,
    )
    trainer.train()


def test_trainer_with_validation(
    dummy_model, dummy_loss, dummy_optimizer, dummy_train_loader, dummy_val_loader
):
    config = TrainerConfig(val_loader=dummy_val_loader, max_epochs=1)
    trainer = MultiObjectiveTrainer(
        model=dummy_model,
        loss_fn=dummy_loss,
        optimizer=dummy_optimizer,
        train_loader=dummy_train_loader,
        config=config,
    )
    trainer.train()


def test_trainer_on_epoch_end_callback(
    dummy_model, dummy_loss, dummy_optimizer, dummy_train_loader
):
    called = []

    def on_epoch_end(epoch, avg_loss):
        called.append((epoch, avg_loss))

    config = TrainerConfig(val_loader=None, max_epochs=EPOCHS_EXPECTED, on_epoch_end=on_epoch_end)
    trainer = MultiObjectiveTrainer(
        model=dummy_model,
        loss_fn=dummy_loss,
        optimizer=dummy_optimizer,
        train_loader=dummy_train_loader,
        config=config,
    )
    trainer.train()
    assert len(called) == EPOCHS_EXPECTED
    for epoch, avg_loss in called:
        assert isinstance(epoch, int)
        assert isinstance(avg_loss, float)


EPOCHS_EXPECTED = 2
