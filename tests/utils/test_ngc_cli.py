from unittest.mock import patch

import pytest

from utils.ngc_cli import NGCCLI, NGCCLINotFoundError


def test_is_available_mocked():
    with patch("shutil.which", return_value="/usr/bin/ngc"):
        cli = NGCCLI()
        assert cli.is_available() is True

    with (
        patch("shutil.which", return_value=None),
        patch("pathlib.Path.exists", return_value=False),
        patch("utils.ngc_cli.shutil.which", return_value=None),
    ):
        cli = NGCCLI(use_uv=False)
        assert cli.is_available() is False


def test_ensure_available_raises():
    with patch("shutil.which", return_value=None), patch("pathlib.Path.exists", return_value=False):
        cli = NGCCLI(use_uv=False)
        with pytest.raises(NGCCLINotFoundError):
            cli.ensure_available()
