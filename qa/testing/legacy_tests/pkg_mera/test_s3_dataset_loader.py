import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai.tools.utilities.utils.s3_dataset_loader import (
    S3DatasetLoader,
    get_s3_dataset_path,
    load_dataset_from_s3,
)


@pytest.fixture
def loader():
    return S3DatasetLoader(
        bucket="test-bucket",
        aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
        aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        endpoint_url="http://localhost:4566",
        region_name="us-east-1",
    )


@pytest.fixture
def mock_s3_client():
    """Return a mock S3 client that can be injected into a loader."""
    return MagicMock()


@pytest.fixture
def mock_body():
    """Return a mock S3 object body with iter_lines support."""
    body = MagicMock()
    body.__enter__ = MagicMock(return_value=body)
    body.__exit__ = MagicMock(return_value=False)
    return body


def test_s3_dataset_loader_stream_jsonl(loader, mock_s3_client, mock_body):
    mock_body.iter_lines.return_value = [b'{"foo": "bar"}', b'{"baz": "qux"}']
    mock_s3_client.get_object.return_value = {"Body": mock_body}
    loader._client = mock_s3_client

    results = list(loader.stream_jsonl("test-bucket", "test-key.jsonl"))

    assert len(results) == 2
    assert results[0] == {"foo": "bar"}
    assert results[1] == {"baz": "qux"}
    mock_s3_client.get_object.assert_called_once_with(
        Bucket="test-bucket", Key="test-key.jsonl"
    )


def test_s3_dataset_loader_stream_jsonl_empty_lines(loader, mock_s3_client, mock_body):
    mock_body.iter_lines.return_value = [
        b"",
        b'{"foo": "bar"}',
        b"",
        b'{"baz": "qux"}',
        b"",
    ]
    mock_s3_client.get_object.return_value = {"Body": mock_body}
    loader._client = mock_s3_client

    results = list(loader.stream_jsonl("test-bucket", "test-key.jsonl"))

    assert len(results) == 2
    assert results[0] == {"foo": "bar"}
    assert results[1] == {"baz": "qux"}


def test_s3_dataset_loader_stream_jsonl_invalid_json(loader, mock_s3_client, mock_body):
    mock_body.iter_lines.return_value = [b'{"foo": "bar"}', b"invalid json", b'{"baz": "qux"}']
    mock_s3_client.get_object.return_value = {"Body": mock_body}
    loader._client = mock_s3_client

    results = list(loader.stream_jsonl("test-bucket", "test-key.jsonl"))

    assert len(results) == 3
    assert results[0] == {"foo": "bar"}
    assert results[1] == {"text": "invalid json"}
    assert results[2] == {"baz": "qux"}


def test_s3_dataset_loader_stream_jsonl_no_newline_at_end(loader, mock_s3_client, mock_body):
    mock_body.iter_lines.return_value = [b'{"foo": "bar"}']
    mock_s3_client.get_object.return_value = {"Body": mock_body}
    loader._client = mock_s3_client

    results = list(loader.stream_jsonl("test-bucket", "test-key.jsonl"))

    assert len(results) == 1
    assert results[0] == {"foo": "bar"}


def test_s3_dataset_loader_stream_jsonl_exception(loader, mock_s3_client):
    mock_s3_client.get_object.side_effect = Exception("S3 error")
    loader._client = mock_s3_client

    with pytest.raises(Exception, match="S3 error"):
        list(loader.stream_jsonl("test-bucket", "test-key.jsonl"))


def test_s3_dataset_loader_s3_prefix(loader, mock_s3_client, mock_body):
    mock_body.iter_lines.return_value = [b'{"foo": "bar"}']
    mock_s3_client.get_object.return_value = {"Body": mock_body}
    loader._client = mock_s3_client

    list(loader.stream_jsonl("unused", "s3://test-bucket/test-key.jsonl"))

    mock_s3_client.get_object.assert_called_once_with(
        Bucket="test-bucket", Key="test-key.jsonl"
    )


def test_s3_dataset_loader_ensure_client_lazy_init(loader):
    with patch("ai.tools.utilities.utils.s3_dataset_loader.boto3") as mock_boto3:
        mock_boto3.client.return_value = "mock_client"
        assert loader._ensure_client() == "mock_client"
        assert loader._ensure_client() == "mock_client"
        mock_boto3.client.assert_called_once()


def test_s3_dataset_loader_load_json(loader, mock_s3_client, mock_body):
    mock_s3_client.get_object.return_value = {"Body": mock_body}
    mock_body.read.return_value = b'{"hello": "world"}'
    loader._client = mock_s3_client

    result = loader.load_json("test-bucket", "test-key.json")

    assert result == {"hello": "world"}
    mock_s3_client.get_object.assert_called_once_with(
        Bucket="test-bucket", Key="test-key.json"
    )


def test_s3_dataset_loader_stream_json_array(loader, mock_s3_client, mock_body):
    mock_s3_client.get_object.return_value = {"Body": mock_body}
    mock_body.read.return_value = b'[{"foo": "bar"}, {"baz": "qux"}]'
    loader._client = mock_s3_client

    results = list(loader.stream_json_array("test-bucket", "test-key.json"))

    assert len(results) == 2
    assert results[0] == {"foo": "bar"}
    assert results[1] == {"baz": "qux"}


def test_s3_dataset_loader_stream_json_dispatches_to_jsonl(loader, mock_s3_client, mock_body):
    mock_body.iter_lines.return_value = [b'{"foo": "bar"}']
    mock_s3_client.get_object.return_value = {"Body": mock_body}
    loader._client = mock_s3_client

    results = list(loader.stream_json("test-bucket", "test-key.jsonl"))

    assert len(results) == 1
    assert results[0] == {"foo": "bar"}
    mock_s3_client.get_object.assert_called_once_with(
        Bucket="test-bucket", Key="test-key.jsonl"
    )


def test_s3_dataset_loader_stream_json_dispatches_to_json_array(loader, mock_s3_client, mock_body):
    mock_s3_client.get_object.return_value = {"Body": mock_body}
    mock_body.read.return_value = b'[{"foo": "bar"}]'
    loader._client = mock_s3_client

    results = list(loader.stream_json("test-bucket", "test-key.json"))

    assert len(results) == 1
    assert results[0] == {"foo": "bar"}


def test_s3_dataset_loader_upload_file(loader, mock_s3_client):
    loader._client = mock_s3_client

    result = loader.upload_file("test-bucket", "test-key.json", {"hello": "world"})

    assert result is True
    assert mock_s3_client.put_object.call_count == 1
    call_kwargs = mock_s3_client.put_object.call_args.kwargs
    assert call_kwargs["Bucket"] == "test-bucket"
    assert call_kwargs["Key"] == "test-key.json"
    assert call_kwargs["Body"] == b'{"hello": "world"}'


def test_s3_dataset_loader_download_file(loader, mock_s3_client, mock_body):
    mock_s3_client.get_object.return_value = {"Body": mock_body}
    mock_body.read.return_value = b'{"downloaded": true}'
    loader._client = mock_s3_client

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        result = loader.download_file("test-bucket", "test-key.json", tmp.name)
        assert result is True
        assert Path(tmp.name).read_text() == '{"downloaded": true}'

    mock_s3_client.get_object.assert_called_once_with(
        Bucket="test-bucket", Key="test-key.json"
    )


def test_s3_dataset_loader_list_datasets(loader, mock_s3_client):
    mock_s3_client.list_objects_v2.return_value = {
        "Contents": [{"Key": "a.jsonl"}, {"Key": "b.jsonl"}]
    }
    loader._client = mock_s3_client

    results = loader.list_datasets("test-bucket", prefix="datasets/")

    assert results == ["a.jsonl", "b.jsonl"]
    mock_s3_client.list_objects_v2.assert_called_once_with(
        Bucket="test-bucket", Prefix="datasets/"
    )


def test_s3_dataset_loader_object_exists_true(loader, mock_s3_client):
    loader._client = mock_s3_client
    assert loader.object_exists("test-bucket", "test-key.json") is True
    mock_s3_client.head_object.assert_called_once_with(
        Bucket="test-bucket", Key="test-key.json"
    )


def test_s3_dataset_loader_object_exists_false(loader, mock_s3_client):
    from botocore.exceptions import ClientError

    loader._client = mock_s3_client
    error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
    mock_s3_client.head_object.side_effect = ClientError(error_response, "HeadObject")

    assert loader.object_exists("test-bucket", "test-key.json") is False


def test_s3_dataset_loader_local_file_fallback(tmp_path, monkeypatch):
    local_file = tmp_path / "test-key.jsonl"
    local_file.write_text('{"foo": "bar"}\n{"baz": "qux"}\n')

    loader = S3DatasetLoader(
        bucket="test-bucket",
        aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
        aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    # Force the local-fallback path without relying on cwd/absolute-path quirks.
    monkeypatch.setattr(loader, "_maybe_local_path", lambda _bucket, _key: local_file)

    results = list(loader.stream_jsonl("test-bucket", "test-key.jsonl"))

    assert len(results) == 2
    assert results[0] == {"foo": "bar"}
    assert results[1] == {"baz": "qux"}


def test_s3_dataset_loader_init_requires_credentials(monkeypatch):
    for key in (
        "HETZNER_S3_ACCESS_KEY",
        "HETZNER_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "HETZNER_S3_SECRET_KEY",
        "HETZNER_SECRET_KEY",
        "AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError, match="S3 credentials not found"):
        S3DatasetLoader(bucket="test-bucket")


def test_get_s3_dataset_path(monkeypatch):
    monkeypatch.setenv("HETZNER_S3_BUCKET", "pixel-data")
    path = get_s3_dataset_path("my-dataset.jsonl")
    assert path == "s3://pixel-data/my-dataset.jsonl"


def test_load_dataset_from_s3_local_jsonl(tmp_path):
    local_file = tmp_path / "dataset.jsonl"
    local_file.write_text('{"foo": "bar"}\n{"baz": "qux"}\n')

    result = load_dataset_from_s3(str(local_file))

    assert result == [{"foo": "bar"}, {"baz": "qux"}]


def test_load_dataset_from_s3_local_json(tmp_path):
    local_file = tmp_path / "dataset.json"
    local_file.write_text('{"hello": "world"}')

    result = load_dataset_from_s3(str(local_file))

    assert result == {"hello": "world"}


def test_s3_dataset_loader_stream_jsonl_import_error(loader):
    with patch("ai.tools.utilities.utils.s3_dataset_loader.boto3", None):
        loader._client = None
        with pytest.raises(ImportError, match="boto3 is required"):
            list(loader.stream_jsonl("test-bucket", "test-key.jsonl"))
