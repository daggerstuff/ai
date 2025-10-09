"""
Test suite for supervised fine-tuning orchestration pipeline (Wayfarer).
Covers E2E, config, integration, error, and compliance.
"""
import pytest
from unittest.mock import Mock, patch
from typing import Any

# TDD Anchor: All tests must fail until orchestration is implemented

@pytest.fixture
def dummy_config():
    return {
        "data_path": "dummy/path.jsonl",
        "output_dir": "dummy/out",
        "epochs": 1,
        "model_name": "dummy-model",
        "tokenizer_name": "dummy-tokenizer",
        "use_bias_privacy_hooks": True,
    }

@pytest.fixture
def mock_data_loader():
    return Mock(load=lambda path: [{"text": "sample"}])

@pytest.fixture
def mock_tokenizer():
    return Mock(tokenize=lambda x: [1, 2, 3])

@pytest.fixture
def mock_finetune():
    return Mock(finetune=lambda *a, **kw: "finetuned-model")

@pytest.fixture
def mock_bias_privacy_hooks():
    hooks = Mock()
    hooks.run_bias_privacy_pipeline.return_value = {"bias": {}, "pii": [], "phi": []}
    return hooks

@patch("ai.pipelines.wayfarer_supervised.data_loader", new_callable=lambda: Mock())
@patch("ai.pipelines.wayfarer_supervised.tokenizer", new_callable=lambda: Mock())
@patch("ai.pipelines.wayfarer_supervised.finetune", new_callable=lambda: Mock())
@patch("ai.pipelines.wayfarer_supervised.bias_privacy_hooks", new_callable=lambda: Mock())
def test_orchestration_e2e(mock_hooks, mock_finetune, mock_tokenizer, mock_loader, dummy_config):
    """Given valid config, When pipeline runs, Then model is finetuned and saved."""
    from ai.pipelines import wayfarer_supervised as ws
    mock_loader.load.return_value = [{"text": "sample"}]
    mock_tokenizer.tokenize.return_value = [1, 2, 3]
    mock_finetune.finetune.return_value = "finetuned-model"
    mock_hooks.run_bias_privacy_pipeline.return_value = {"bias": {}, "pii": [], "phi": []}
    result = ws.run_supervised_pipeline(dummy_config)
    assert result == "finetuned-model"

@patch("ai.pipelines.wayfarer_supervised.data_loader", new_callable=lambda: Mock())
def test_missing_config_field(mock_loader, dummy_config):
    """Given missing config, When pipeline runs, Then ValueError is raised."""
    from ai.pipelines import wayfarer_supervised as ws
    bad_config = dict(dummy_config)
    del bad_config["data_path"]
    with pytest.raises(ValueError):
        ws.run_supervised_pipeline(bad_config)

@patch("ai.pipelines.wayfarer_supervised.bias_privacy_hooks", new_callable=lambda: Mock())
def test_integration_bias_privacy(mock_hooks, dummy_config):
    """Given use_bias_privacy_hooks=True, When pipeline runs, Then bias/privacy hooks are called."""
    from ai.pipelines import wayfarer_supervised as ws
    ws.data_loader = Mock(load=lambda path: [{"text": "sample"}])
    ws.tokenizer = Mock(tokenize=lambda x: [1, 2, 3])
    ws.finetune = Mock(finetune=lambda *a, **kw: "finetuned-model")
    ws.bias_privacy_hooks = mock_hooks
    mock_hooks.run_bias_privacy_pipeline.return_value = {"bias": {}, "pii": [], "phi": []}
    ws.run_supervised_pipeline(dummy_config)
    mock_hooks.run_bias_privacy_pipeline.assert_called()

@patch("ai.pipelines.wayfarer_supervised.finetune", new_callable=lambda: Mock())
def test_finetune_error_propagation(mock_finetune, dummy_config):
    """Given finetune error, When pipeline runs, Then exception is propagated."""
    from ai.pipelines import wayfarer_supervised as ws
    ws.data_loader = Mock(load=lambda path: [{"text": "sample"}])
    ws.tokenizer = Mock(tokenize=lambda x: [1, 2, 3])
    ws.finetune = mock_finetune
    ws.bias_privacy_hooks = Mock(run_bias_privacy_pipeline=lambda *a, **kw: {"bias": {}, "pii": [], "phi": []})
    mock_finetune.finetune.side_effect = RuntimeError("finetune failed")
    with pytest.raises(RuntimeError):
        ws.run_supervised_pipeline(dummy_config)

@patch("ai.pipelines.wayfarer_supervised.data_loader", new_callable=lambda: Mock())
def test_data_loader_error(mock_loader, dummy_config):
    """Given data loader error, When pipeline runs, Then exception is propagated."""
    from ai.pipelines import wayfarer_supervised as ws
    mock_loader.load.side_effect = IOError("data load failed")
    ws.tokenizer = Mock(tokenize=lambda x: [1, 2, 3])
    ws.finetune = Mock(finetune=lambda *a, **kw: "finetuned-model")
    ws.bias_privacy_hooks = Mock(run_bias_privacy_pipeline=lambda *a, **kw: {"bias": {}, "pii": [], "phi": []})
    with pytest.raises(IOError):
        ws.run_supervised_pipeline(dummy_config)

@patch("ai.pipelines.wayfarer_supervised.bias_privacy_hooks", new_callable=lambda: Mock())
def test_disable_bias_privacy_hooks(mock_hooks, dummy_config):
    """Given use_bias_privacy_hooks=False, When pipeline runs, Then hooks are not called."""
    from ai.pipelines import wayfarer_supervised as ws
    ws.data_loader = Mock(load=lambda path: [{"text": "sample"}])
    ws.tokenizer = Mock(tokenize=lambda x: [1, 2, 3])
    ws.finetune = Mock(finetune=lambda *a, **kw: "finetuned-model")
    ws.bias_privacy_hooks = mock_hooks
    config = dict(dummy_config)
    config["use_bias_privacy_hooks"] = False
    ws.run_supervised_pipeline(config)
    mock_hooks.run_bias_privacy_pipeline.assert_not_called()

@patch("ai.pipelines.wayfarer_supervised.data_loader", new_callable=lambda: Mock())
def test_edge_case_empty_data(mock_loader, dummy_config):
    """Given empty data, When pipeline runs, Then ValueError is raised."""
    from ai.pipelines import wayfarer_supervised as ws
    mock_loader.load.return_value = []
    ws.tokenizer = Mock(tokenize=lambda x: [1, 2, 3])
    ws.finetune = Mock(finetune=lambda *a, **kw: "finetuned-model")
    ws.bias_privacy_hooks = Mock(run_bias_privacy_pipeline=lambda *a, **kw: {"bias": {}, "pii": [], "phi": []})
    with pytest.raises(ValueError):
        ws.run_supervised_pipeline(dummy_config)