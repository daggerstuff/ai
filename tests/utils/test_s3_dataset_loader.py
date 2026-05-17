from unittest.mock import MagicMock, patch

import pytest

from utils.s3_dataset_loader import S3DatasetLoader


def test_s3_dataset_loader_stream_jsonl():
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")
    mock_s3 = MagicMock()
    mock_body = MagicMock()
    mock_body.__iter__.return_value = [b'{"foo": "bar"}\n{"baz": "qux"}\n']
    mock_s3.get_object.return_value = {"Body": mock_body}
    loader._s3_client = mock_s3
    results = list(loader.stream_jsonl("test-key.jsonl"))
import pytest
from unittest.mock import MagicMock, patch
from utils.s3_dataset_loader import S3DatasetLoader
import json

def test_s3_dataset_loader_stream_jsonl():
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")

    # Mock s3 client
    mock_s3 = MagicMock()

    mock_body = MagicMock()
    mock_body.__iter__.return_value = [b'{"foo": "bar"}\n{"baz": "qux"}\n']

    mock_s3.get_object.return_value = {"Body": mock_body}
    loader._s3_client = mock_s3

    results = list(loader.stream_jsonl("test-key.jsonl"))

    assert len(results) == 2
    assert results[0][0] == {"foo": "bar"}
    assert results[1][0] == {"baz": "qux"}

def test_s3_dataset_loader_stream_jsonl_empty_lines():
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")
    mock_s3 = MagicMock()
    mock_body = MagicMock()
    mock_body.__iter__.return_value = [b'\n{"foo": "bar"}\n\n{"baz": "qux"}\n']
    mock_s3.get_object.return_value = {"Body": mock_body}
    loader._s3_client = mock_s3
    results = list(loader.stream_jsonl("test-key.jsonl"))

    mock_s3 = MagicMock()
    mock_body = MagicMock()
    # includes empty lines that should be ignored
    mock_body.__iter__.return_value = [b'\n{"foo": "bar"}\n\n{"baz": "qux"}\n']

    mock_s3.get_object.return_value = {"Body": mock_body}
    loader._s3_client = mock_s3

    results = list(loader.stream_jsonl("test-key.jsonl"))

    assert len(results) == 2
    assert results[0][0] == {"foo": "bar"}
    assert results[1][0] == {"baz": "qux"}

def test_s3_dataset_loader_stream_jsonl_invalid_json():
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")
    mock_s3 = MagicMock()
    mock_body = MagicMock()
    mock_body.__iter__.return_value = [b'{"foo": "bar"}\ninvalid json\n{"baz": "qux"}\n']
    mock_s3.get_object.return_value = {"Body": mock_body}
    loader._s3_client = mock_s3
    results = list(loader.stream_jsonl("test-key.jsonl"))
    assert len(results) == 2
    assert results[0][0] == {"foo": "bar"}
    assert results[1][0] == {"baz": "qux"}

@patch("utils.s3_dataset_loader.boto3")
def test_s3_client_lazy_init(mock_boto3):
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")
    mock_boto3.client.return_value = "mock_client"
    assert loader.s3_client == "mock_client"
    assert loader.s3_client == "mock_client"
    mock_boto3.client.assert_called_once()

def test_s3_dataset_loader_stream_jsonl_buffer_trim_exact():
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")
    mock_s3 = MagicMock()
    mock_body = MagicMock()
    chunk1 = b'{"a": 1}\n' * (10 * 1024 * 1024 // 9 + 2)
    mock_body.__iter__.return_value = [chunk1]
    mock_s3.get_object.return_value = {"Body": mock_body}
    loader._s3_client = mock_s3
    iterator = loader.stream_jsonl("test-key.jsonl")
    results = list(iterator)
    assert len(results) > 0

def test_s3_dataset_loader_stream_jsonl_no_newline_at_end():
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")
    mock_s3 = MagicMock()
    mock_body = MagicMock()
    mock_body.__iter__.return_value = [b'{"foo": "bar"}']
    mock_s3.get_object.return_value = {"Body": mock_body}
    loader._s3_client = mock_s3
    results = list(loader.stream_jsonl("test-key.jsonl"))
    assert len(results) == 1
    assert results[0][0] == {"foo": "bar"}

def test_s3_dataset_loader_stream_jsonl_exception_in_s3_call():
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")
    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = Exception("S3 Error")
    loader._s3_client = mock_s3
    with pytest.raises(Exception, match="S3 Error"):
        list(loader.stream_jsonl("test-key.jsonl"))

def test_s3_dataset_loader_stream_jsonl_with_byte_offset():
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")
    mock_s3 = MagicMock()
    mock_body = MagicMock()
    mock_body.__iter__.return_value = [b'{"foo": "bar"}\n']
    mock_s3.get_object.return_value = {"Body": mock_body}
    loader._s3_client = mock_s3
    list(loader.stream_jsonl("test-key.jsonl", byte_offset=100))
    mock_s3.get_object.assert_called_with(Bucket="test-bucket", Key="test-key.jsonl", Range="bytes=100-")

def test_s3_dataset_loader_stream_jsonl_no_newline_at_end_invalid_json():
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")
    mock_s3 = MagicMock()
    mock_body = MagicMock()
    mock_body.__iter__.return_value = [b'{"foo": ']
    mock_s3.get_object.return_value = {"Body": mock_body}
    loader._s3_client = mock_s3
    results = list(loader.stream_jsonl("test-key.jsonl"))
    assert len(results) == 0

def test_s3_dataset_loader_s3_prefix():
    loader = S3DatasetLoader(bucket="test-bucket", endpoint_url="http://localhost")
    mock_s3 = MagicMock()
    mock_body = MagicMock()
    mock_body.__iter__.return_value = [b'{"foo": "bar"}\n']
    mock_s3.get_object.return_value = {"Body": mock_body}
    loader._s3_client = mock_s3
    list(loader.stream_jsonl("s3://test-bucket/test-key.jsonl"))
    mock_s3.get_object.assert_called_with(Bucket="test-bucket", Key="test-key.jsonl")

def test_import_error_boto3(monkeypatch):
    import importlib
    import sys
    monkeypatch.setitem(sys.modules, "boto3", None)
    import utils.s3_dataset_loader
    importlib.reload(utils.s3_dataset_loader)
    assert utils.s3_dataset_loader.boto3 is None
    monkeypatch.undo()
    importlib.reload(utils.s3_dataset_loader)

    mock_s3 = MagicMock()
    mock_body = MagicMock()
    # includes invalid json
    mock_body.__iter__.return_value = [b'{"foo": "bar"}\ninvalid json\n{"baz": "qux"}\n']

    mock_s3.get_object.return_value = {"Body": mock_body}
    loader._s3_client = mock_s3

    results = list(loader.stream_jsonl("test-key.jsonl"))

    assert len(results) == 2
    assert results[0][0] == {"foo": "bar"}
    assert results[1][0] == {"baz": "qux"}
