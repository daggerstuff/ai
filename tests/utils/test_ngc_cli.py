<<<<<<< HEAD
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

            with patch("pathlib.Path.iterdir") as mock_iterdir:
                mock_file = MagicMock()
                mock_file.stat.return_value.st_mtime = 12345
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

            with pytest.raises(NGCCLIDownloadError):
                cli.download_resource("nvidia/resource", output_dir=Path("test_dir"))
=======

import pytest

from utils.ngc_cli import NGCCLI, NGCCLINotFoundError


from unittest.mock import patch

@patch('utils.ngc_cli.shutil.which')
def test_ngc_cli_is_available(mock_which):
    mock_which.return_value = '/usr/bin/ngc'
    cli = NGCCLI(use_uv=False)
    assert cli.is_available()

@patch('utils.ngc_cli.shutil.which')
def test_ngc_cli_ensure_available_raises(mock_which):
    mock_which.return_value = None
    cli = NGCCLI(use_uv=False)
    # Clear the ngc_cmd that might have been found by other means
    cli.ngc_cmd = None
    with pytest.raises(NGCCLINotFoundError):
        cli.ensure_available()


def test_parse_json_resources():
    # Test valid list of dicts
    valid_list = '[{"name": "test1", "version": "1.0"}, {"name": "test2", "version": "2.0"}]'
    assert NGCCLI._parse_json_resources(valid_list) == [
        {"name": "test1", "version": "1.0"},
        {"name": "test2", "version": "2.0"}
    ]

    # Test valid dict with resources key
    valid_dict = '{"resources": [{"name": "test1"}, {"name": "test2"}]}'
    assert NGCCLI._parse_json_resources(valid_dict) == [{"name": "test1"}, {"name": "test2"}]

    # Test invalid JSON
    assert NGCCLI._parse_json_resources("not json") == []

    # Test valid JSON but not right structure
    assert NGCCLI._parse_json_resources('{"other": "value"}') == []
    assert NGCCLI._parse_json_resources('["string", "list"]') == []

def test_parse_pipe_delimited_resources():
    valid_pipe = """
+------------------+---------+
| Name             | Version |
+------------------+---------+
| test-resource    | 1.0     |
| another-resource | 2.0     |
+------------------+---------+
    """.strip().splitlines()
    expected = [
        {"Name": "test-resource", "Version": "1.0"},
        {"Name": "another-resource", "Version": "2.0"},
    ]
    assert NGCCLI._parse_pipe_delimited_resources(valid_pipe) == expected

    # Test missing headers
    invalid_pipe = ["| |", "| val1 | val2 |"]
    assert NGCCLI._parse_pipe_delimited_resources(invalid_pipe) == []

    # Test empty or no table
    assert NGCCLI._parse_pipe_delimited_resources(["just some text", "no pipes here"]) == []

def test_parse_whitespace_aligned_resources():
    valid_ws = """
Name                Version   Size
----                -------   ----
test-resource       1.0       1GB
another-resource    2.0       2GB
    """.strip().splitlines()
    expected = [
        {"Name": "test-resource", "Version": "1.0", "Size": "1GB"},
        {"Name": "another-resource", "Version": "2.0", "Size": "2GB"},
    ]
    assert NGCCLI._parse_whitespace_aligned_resources(valid_ws) == expected

    # Test missing headers
    invalid_ws = ["----   ----", "val1   val2"]
    assert NGCCLI._parse_whitespace_aligned_resources(invalid_ws) == []

    # Test empty
    assert NGCCLI._parse_whitespace_aligned_resources([]) == []

def test_parse_resources_output():
    # Test "no resources found"
    assert NGCCLI._parse_resources_output("No resources found for your query.") == []

    # Test JSON output
    json_out = '[{"Name": "res1"}]'
    assert NGCCLI._parse_resources_output(json_out) == [{"Name": "res1"}]

    # Test pipe output
    pipe_out = "| Name |\n| res1 |\n"
    assert NGCCLI._parse_resources_output(pipe_out) == [{"Name": "res1"}]

    # Test whitespace output
    ws_out = "Name   Ver\n----   ---\nres1   1.0\n"
    assert NGCCLI._parse_resources_output(ws_out) == [{"Name": "res1", "Ver": "1.0"}]
>>>>>>> origin/staging
