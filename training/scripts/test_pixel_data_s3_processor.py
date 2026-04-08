import json
import os
from unittest.mock import MagicMock, patch

from training.scripts.pixel_data_s3_processor import run_s3_command


def test_run_s3_command_json_success():
    cmd = ["aws", "s3", "ls", "s3://pixel-data", "--recursive"]
    mock_env = {
        "AWS_ACCESS_KEY_ID": "mock_access",
        "AWS_SECRET_ACCESS_KEY": "mock_secret",
    }
    mock_stdout = json.dumps([{"path": "file1.json", "size": 100}])

    with patch.dict(os.environ, mock_env, clear=True):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=mock_stdout, stderr=""
            )
            result = run_s3_command(cmd)

            mock_run.assert_called_once()
            called_cmd = mock_run.call_args[0][0]
            assert called_cmd == cmd
            assert result == [{"path": "file1.json", "size": 100}]


def test_run_s3_command_raw_string():
    cmd = ["aws", "s3", "ls", "s3://pixel-data", "--recursive"]
    mock_env = {
        "AWS_ACCESS_KEY_ID": "mock_access",
        "AWS_SECRET_ACCESS_KEY": "mock_secret",
    }
    mock_stdout = "2024-01-01 12:00:00 1.2G file1.json"

    with patch.dict(os.environ, mock_env, clear=True):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=mock_stdout, stderr=""
            )
            result = run_s3_command(cmd)

            mock_run.assert_called_once()
            assert result == "2024-01-01 12:00:00 1.2G file1.json"


def test_run_s3_command_error_return_code():
    cmd = ["aws", "s3", "ls", "s3://pixel-data", "--recursive"]
    mock_env = {
        "AWS_ACCESS_KEY_ID": "mock_access",
        "AWS_SECRET_ACCESS_KEY": "mock_secret",
    }

    with patch.dict(os.environ, mock_env, clear=True):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="Access Denied"
            )
            result = run_s3_command(cmd)

            mock_run.assert_called_once()
            assert result == []


def test_run_s3_command_exception():
    cmd = ["aws", "s3", "ls", "s3://pixel-data", "--recursive"]
    mock_env = {
        "AWS_ACCESS_KEY_ID": "mock_access",
        "AWS_SECRET_ACCESS_KEY": "mock_secret",
    }

    with patch.dict(os.environ, mock_env, clear=True):
        with patch("subprocess.run", side_effect=Exception("Mocked subprocess error")):
            result = run_s3_command(cmd)
            assert result == []
