from unittest.mock import MagicMock

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
    expected_results_len = 2
    assert len(results) == expected_results_len
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
    expected_results_len = 2
    assert len(results) == expected_results_len
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
    expected_results_len = 2
    assert len(results) == expected_results_len
    assert results[0][0] == {"foo": "bar"}
    assert results[1][0] == {"baz": "qux"}
