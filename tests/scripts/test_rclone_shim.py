"""Tests for rclone boto3 shim paginator extensions."""

from ai.scripts.rclone_boto3_shim import RcloneS3Client


class TestRclonePaginator:
    def test_list_objects_v2_supported(self) -> None:
        client = RcloneS3Client()
        paginator = client.get_paginator("list_objects_v2")
        assert paginator is not None

    def test_list_object_versions_supported(self) -> None:
        client = RcloneS3Client()
        paginator = client.get_paginator("list_object_versions")
        assert paginator is not None

    def test_list_multipart_uploads_supported(self) -> None:
        client = RcloneS3Client()
        paginator = client.get_paginator("list_multipart_uploads")
        assert paginator is not None

    def test_unsupported_action_raises(self) -> None:
        import pytest

        client = RcloneS3Client()
        with pytest.raises(NotImplementedError, match="not implemented"):
            client.get_paginator("delete_objects")
