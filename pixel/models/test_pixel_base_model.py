"""
Unit tests for PixelBaseModel architecture and functionality.
"""

import torch

from ai.pixel.models.pixel_base_model import PixelBaseModel


def test_pixel_base_model_instantiation() -> None:
    model = PixelBaseModel()
    assert model is not None


def test_pixel_base_model_forward_shapes() -> None:
    model = PixelBaseModel()
    batch_size = 2
    input_dim = 4096
    x = torch.randn(batch_size, input_dim)
    history = [{"persona_mode": 0}]
    outputs = model(x, history=history)
    # Check output keys
    expected_keys = {
        "eq_outputs",
        "eq_aggregate",
        "persona_logits",
        "persona_mode",
        "persona_valid",
        "persona_transition",
        "persona_response_path",
        "dsm5_logits",
        "pdm2_logits",
        "therapeutic_score",
        "intervention_logits",
        "safety_score",
        "empathy_sim_logits",
        "empathy_progress",
        "empathy_consistency",
        "empathy_calibration",
    }
    assert set(outputs.keys()) == expected_keys
    # Check output shapes
    assert outputs["eq_aggregate"].shape == (batch_size, 1)
    assert outputs["persona_logits"].shape == (batch_size, 2)
    assert outputs["dsm5_logits"].shape == (batch_size, 100)
    assert outputs["pdm2_logits"].shape == (batch_size, 10)
    assert outputs["therapeutic_score"].shape == (batch_size, 1)
    assert outputs["intervention_logits"].shape == (batch_size, 5)
    assert outputs["safety_score"].shape == (batch_size, 1)
    assert outputs["empathy_sim_logits"].shape == (batch_size, 2)
    assert outputs["empathy_progress"].shape == (batch_size, 1)
    assert outputs["empathy_consistency"].shape == (batch_size, 1)
    assert outputs["empathy_calibration"].shape == (batch_size, 1)


def test_pixel_base_model_serialization(tmp_path) -> None:
    model = PixelBaseModel()
    path = tmp_path / "pixel_model.pt"
    model.save(str(path))
    loaded = PixelBaseModel.load(str(path))
    assert isinstance(loaded, PixelBaseModel)
    # Check that parameters match
    for p1, p2 in zip(model.parameters(), loaded.parameters(), strict=False):
        assert torch.allclose(p1, p2)
