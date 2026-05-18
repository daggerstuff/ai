from unittest.mock import MagicMock, patch

import pytest

# Clean up any potential state
import utils.torch_proxy


def setup_function():
    utils.torch_proxy._torch_module = None
    utils.torch_proxy._torch_import_error = None

@patch("utils.torch_proxy.import_module")
def test_torch_proxy_getattr(mock_import_module):
    # Mock the returned module
    mock_torch = MagicMock()
    mock_torch.Tensor = "mock_tensor"
    mock_import_module.return_value = mock_torch

    # Call proxy
    result = _ = utils.torch_proxy.torch.Tensor

    # Assert lazy import and getattr
    mock_import_module.assert_called_once_with("torch")
    assert result == "mock_tensor"

@patch("utils.torch_proxy.import_module")
def test_torch_proxy_dir_success(mock_import_module):
    mock_torch = MagicMock()
    # Mock dir() behavior on the module
    mock_torch.__dir__ = MagicMock(return_value=["Tensor", "nn"])
    mock_import_module.return_value = mock_torch

    assert "Tensor" in dir(utils.torch_proxy.torch)

@patch("utils.torch_proxy.import_module")
def test_torch_proxy_dir_failure(mock_import_module):
    mock_import_module.side_effect = Exception("Import failed")

    assert dir(utils.torch_proxy.torch) == []

@patch("utils.torch_proxy.import_module")
def test_torch_attr_proxy_getattr(mock_import_module):
    mock_torch = MagicMock()
    mock_nn = MagicMock()
    mock_nn.Linear = "mock_linear"
    mock_torch.nn = mock_nn
    mock_import_module.return_value = mock_torch

    result = utils.torch_proxy.nn.Linear
    assert result == "mock_linear"

@patch("utils.torch_proxy.import_module")
def test_torch_attr_proxy_call(mock_import_module):
    mock_torch = MagicMock()
    mock_func = MagicMock(return_value="called")
    mock_torch.some_func = mock_func
    mock_import_module.return_value = mock_torch

    attr_proxy = utils.torch_proxy._TorchAttrProxy("some_func")
    result = attr_proxy(1, 2, kwarg=3)

    mock_func.assert_called_once_with(1, 2, kwarg=3)
    assert result == "called"

@patch("utils.torch_proxy.import_module")
def test_load_torch_failure(mock_import_module):
    mock_import_module.side_effect = ImportError("No module named torch")

    with pytest.raises(RuntimeError) as excinfo:
        _ = utils.torch_proxy.torch.Tensor

    assert "torch is unavailable in this environment" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ImportError)

    # Should raise immediately on second attempt
    mock_import_module.reset_mock()
    with pytest.raises(RuntimeError) as excinfo2:
        _ = utils.torch_proxy.torch.Tensor

    mock_import_module.assert_not_called()
    assert str(excinfo2.value) == str(excinfo.value)

@patch("utils.torch_proxy.import_module")
def test_load_torch_cached(mock_import_module):
    # First access loads torch
    mock_torch = MagicMock()
    mock_torch.Tensor = "mock_tensor"
    mock_import_module.return_value = mock_torch

    result1 = _ = utils.torch_proxy.torch.Tensor

    # Second access should return cached module, not call import_module again
    mock_import_module.reset_mock()
    result2 = _ = utils.torch_proxy.torch.Tensor

    mock_import_module.assert_not_called()
    assert result1 == "mock_tensor"
    assert result2 == "mock_tensor"
