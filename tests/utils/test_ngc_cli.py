from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.ngc_cli import NGCCLI, NGCCLIDownloadError, NGCCLINotFoundError


def test_ngc_cli_is_available_when_ngc_in_path():
    with patch("shutil.which") as mock_which:
        mock_which.side_effect = lambda x: "/usr/bin/ngc" if x == "ngc" else None
        cli = NGCCLI()
        assert cli.is_available() is True
        assert cli.ngc_cmd == "ngc"

def test_ngc_cli_ensure_available_raises_error():
    with patch("shutil.which") as mock_which, patch("pathlib.Path.exists") as mock_exists:
        mock_which.return_value = None
        mock_exists.return_value = False

        # Test without UV
        cli = NGCCLI(use_uv=False)
        assert cli.is_available() is False
        with pytest.raises(NGCCLINotFoundError):
            cli.ensure_available()

def test_ngc_cli_check_config_success():
    with patch("shutil.which") as mock_which:
        mock_which.side_effect = lambda x: "/usr/bin/ngc" if x == "ngc" else None
        cli = NGCCLI()

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "| apikey | ***** | source |"
            mock_run.return_value = mock_result

            config = cli.check_config()
            assert "apikey" in config

def test_ngc_cli_download_resource():
    with patch("shutil.which") as mock_which:
        mock_which.side_effect = lambda x: "/usr/bin/ngc" if x == "ngc" else None
        cli = NGCCLI()

        with patch.object(cli, "check_config"), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            with patch("pathlib.Path.iterdir"), patch("pathlib.Path.mkdir"), patch("os.chdir"):
                mock_file = MagicMock()
                mock_file.stat.return_value.st_mtime = 12345
                mock_iterdir = patch("pathlib.Path.iterdir")
                mock_iterdir.return_value = [mock_file]

                result = cli.download_resource("nvidia/resource", version="1.0", output_dir=Path("test_dir"))
                assert result == mock_file
                mock_run.assert_called_once()
                assert "download-version" in mock_run.call_args[0][0]
                assert "nvidia/resource:1.0" in mock_run.call_args[0][0]

def test_ngc_cli_ensure_available_mock_which():
    with patch("shutil.which") as mock_which:
        mock_which.return_value = "/usr/bin/ngc"
        cli = NGCCLI()
        # Should not raise
        cli.ensure_available()

def test_ngc_cli_download_resource_raises_download_error():
    with patch("shutil.which") as mock_which:
        mock_which.side_effect = lambda x: "/usr/bin/ngc" if x == "ngc" else None
        cli = NGCCLI()

        with patch.object(cli, "check_config"), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="Mock error")

            with patch("pathlib.Path.iterdir"), patch("pathlib.Path.mkdir"), patch("os.chdir"):
                with pytest.raises(NGCCLIDownloadError):
                    cli.download_resource("nvidia/resource", output_dir=Path("test_dir"))