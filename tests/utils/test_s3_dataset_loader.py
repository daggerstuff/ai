import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

import utils.s3_dataset_loader
from utils.s3_dataset_loader import S3DatasetLoader


def test_s3_dataset_loader_stream_jsonl():
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")

    # Mock s3 client
    mock_s3 = MagicMock()

    mock_body = MagicMock()
    mock_body.__iter__.return_value = [b'{"foo": "bar"}\n{"baz": "qux"}\n']

    mock_s3.get_object.return_value = {"Body": mock_body}
    loader._s3_client = mock_s3

    results = list(loader.stream_jsonl("test-key.jsonl"))

    assert len(results) == 2  # noqa: PLR2004
    assert results[0][0] == {"foo": "bar"}
    assert results[1][0] == {"baz": "qux"}


def test_s3_dataset_loader_stream_jsonl_empty_lines():
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")

    mock_s3 = MagicMock()
    mock_body = MagicMock()
    # includes empty lines that should be ignored
    mock_body.__iter__.return_value = [b'\n{"foo": "bar"}\n\n{"baz": "qux"}\n']

    mock_s3.get_object.return_value = {"Body": mock_body}
    loader._s3_client = mock_s3

    results = list(loader.stream_jsonl("test-key.jsonl"))

    assert len(results) == 2  # noqa: PLR2004
    assert results[0][0] == {"foo": "bar"}
    assert results[1][0] == {"baz": "qux"}


def test_s3_dataset_loader_stream_jsonl_invalid_json():
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")

    mock_s3 = MagicMock()
    mock_body = MagicMock()
    # includes invalid json
    mock_body.__iter__.return_value = [b'{"foo": "bar"}\ninvalid json\n{"baz": "qux"}\n']

    mock_s3.get_object.return_value = {"Body": mock_body}
    loader._s3_client = mock_s3

    results = list(loader.stream_jsonl("test-key.jsonl"))

    assert len(results) == 2  # noqa: PLR2004
    assert results[0][0] == {"foo": "bar"}
    assert results[1][0] == {"baz": "qux"}


def test_s3_dataset_loader_stream_jsonl_exception():
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")
    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = Exception("S3 error")
    loader._s3_client = mock_s3

    with pytest.raises(Exception, match="S3 error"):
        list(loader.stream_jsonl("test-key.jsonl"))


def test_s3_dataset_loader_stream_jsonl_without_trailing_newline():
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")
    mock_s3 = MagicMock()
    mock_body = MagicMock()
    # Missing trailing newline at EOF
    mock_body.__iter__.return_value = [b'{"foo": "bar"}\n{"baz": "qux"}']
    mock_s3.get_object.return_value = {"Body": mock_body}
    loader._s3_client = mock_s3

    results = list(loader.stream_jsonl("s3://test-bucket/test-key.jsonl", byte_offset=10))

    assert len(results) == 2  # noqa: PLR2004
    assert results[0][0] == {"foo": "bar"}
    assert results[1][0] == {"baz": "qux"}


def test_s3_dataset_loader_stream_jsonl_large_buffer():
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")
    mock_s3 = MagicMock()
    mock_body = MagicMock()

    # Actually simpler: we can just mock a sequence of chunks
    mock_body.__iter__.return_value = [
        b'{"a": 1}\n',
        b" " * (11 * 1024 * 1024),  # This will increase ptr in the loop
        b'\n{"b": 2}\n',
    ]
    mock_s3.get_object.return_value = {"Body": mock_body}
    loader._s3_client = mock_s3

    results = list(loader.stream_jsonl("test-key.jsonl"))
    # The large space block is treated as invalid json, skipped.
    assert len(results) == 2  # noqa: PLR2004
    assert results[0][0] == {"a": 1}
    assert results[1][0] == {"b": 2}


def test_s3_dataset_loader_s3_client_initialization():
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")
    with patch("utils.s3_dataset_loader.boto3.client") as mock_client:
        mock_client.return_value = "mock_client_instance"

        client1 = loader.s3_client
        assert client1 == "mock_client_instance"

        # Second call should return the cached instance
        client2 = loader.s3_client
        assert client2 == "mock_client_instance"
        assert mock_client.call_count == 1


def test_s3_dataset_loader_import_error():
    # Simulate ImportError
    original_boto3 = sys.modules.get("boto3")
    sys.modules["boto3"] = None

    try:
        # Reload the module to trigger the ImportError block
        importlib.reload(utils.s3_dataset_loader)
        assert utils.s3_dataset_loader.boto3 is None
    finally:
        # Restore original
        if original_boto3 is not None:
            sys.modules["boto3"] = original_boto3
        else:
            del sys.modules["boto3"]
        # Reload again to restore functionality for other tests
        importlib.reload(utils.s3_dataset_loader)
