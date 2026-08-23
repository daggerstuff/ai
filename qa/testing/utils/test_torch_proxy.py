from unittest.mock import MagicMock, patch

import pytest

import utils.torch_proxy
from utils.torch_proxy import _load_torch, _TorchAttrProxy, _TorchModuleProxy


def test_load_torch_success():
    utils.torch_proxy._torch_module = None
    utils.torch_proxy._torch_import_error = None

    with patch("utils.torch_proxy.import_module") as mock_import:
        mock_import.return_value = "mocked_torch"
        module = _load_torch()
        assert module == "mocked_torch"
        assert utils.torch_proxy._torch_module == "mocked_torch"

        # Calling again should return cached module
        mock_import.reset_mock()
        module2 = _load_torch()
        assert module2 == "mocked_torch"
        mock_import.assert_not_called()


def test_load_torch_import_error():
    utils.torch_proxy._torch_module = None
    utils.torch_proxy._torch_import_error = None

    with patch("utils.torch_proxy.import_module") as mock_import:
        mock_import.side_effect = ImportError("No module named 'torch'")

        with pytest.raises(RuntimeError, match="torch is unavailable in this environment"):
            _load_torch()

        assert utils.torch_proxy._torch_import_error is not None

        # Calling again should raise cached error
        mock_import.reset_mock()
        with pytest.raises(RuntimeError, match="torch is unavailable"):
            _load_torch()
        mock_import.assert_not_called()


def test_torch_module_proxy_getattr():
    utils.torch_proxy._torch_module = None
    utils.torch_proxy._torch_import_error = None

    mock_torch = MagicMock()
    mock_torch.tensor = "tensor_func"

    with patch("utils.torch_proxy.import_module", return_value=mock_torch):
        proxy = _TorchModuleProxy()
        assert proxy.tensor == "tensor_func"


def test_torch_module_proxy_dir():
    utils.torch_proxy._torch_module = None
    utils.torch_proxy._torch_import_error = None

    mock_torch = MagicMock()
    mock_torch.__dir__ = MagicMock(return_value=["tensor", "nn"])

    with patch("utils.torch_proxy.import_module", return_value=mock_torch):
        proxy = _TorchModuleProxy()
        assert set(dir(proxy)).issuperset({"tensor", "nn"})


def test_torch_module_proxy_dir_error():
    utils.torch_proxy._torch_module = None
    utils.torch_proxy._torch_import_error = None

    with patch("utils.torch_proxy.import_module", side_effect=ImportError):
        proxy = _TorchModuleProxy()
        assert dir(proxy) == []


def test_torch_attr_proxy():
    utils.torch_proxy._torch_module = None
    utils.torch_proxy._torch_import_error = None

    mock_torch = MagicMock()
    mock_nn = MagicMock()
    mock_nn.Linear = "LinearLayer"
    mock_torch.nn = mock_nn

    with patch("utils.torch_proxy.import_module", return_value=mock_torch):
        proxy = _TorchAttrProxy("nn")

        # Test getattr
        assert proxy.Linear == "LinearLayer"

        # Test call
        mock_nn.return_value = "called_nn"
        assert proxy(1, 2, kw="arg") == "called_nn"
        mock_nn.assert_called_once_with(1, 2, kw="arg")
