from unittest.mock import MagicMock, patch

import pytest

from utils.ngc_cli import NGCCLI, NGCCLINotFoundError


@patch("shutil.which")
@patch("pathlib.Path.exists")
def test_ngc_cli_is_available_false(mock_exists: MagicMock, mock_which: MagicMock):
    mock_which.return_value = None
    mock_exists.return_value = False

    cli = NGCCLI(use_uv=False)
    assert cli.is_available() is False

    with pytest.raises(NGCCLINotFoundError):
        cli.ensure_available()

@patch("shutil.which")
def test_ngc_cli_is_available_true_in_path(mock_which: MagicMock):
    mock_which.return_value = "/usr/local/bin/ngc"

    cli = NGCCLI(use_uv=False)
    assert cli.is_available() is True

    # Should not raise an exception
    cli.ensure_available()
