import logging
import subprocess

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class RcloneS3Paginator:
    def __init__(self, remote):
        self.remote = remote

    def paginate(self, Bucket, Prefix=""):
        cmd = ["rclone", "lsf", f"{self.remote}:{Bucket}/{Prefix}", "-R", "--format", "ps", "--files-only"]
        logger.info(f"Rclone list: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        contents = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split(";")
            if len(parts) >= 2:
                path = parts[0]
                size = int(parts[1])
                contents.append({"Key": Prefix + path, "Size": size, "LastModified": ""})
        yield {"Contents": contents}


class RcloneBody:
    def __init__(self, cmd):
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    def iter_lines(self):
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            yield line

    def read(self):
        assert self.proc.stdout is not None
        return self.proc.stdout.read()


class RcloneS3Client:
    """A drop-in replacement for boto3.client('s3') that uses rclone subprocesses."""

    def __init__(self, remote="HetznerS3"):
        self.remote = remote

    def get_paginator(self, action):
        if action in ("list_objects_v2", "list_object_versions", "list_multipart_uploads"):
            return RcloneS3Paginator(self.remote)
        raise NotImplementedError(f"Paginator for {action} not implemented in rclone shim")

    def head_bucket(self, Bucket):
        # Just check if bucket exists by listing root
        cmd = ["rclone", "lsd", f"{self.remote}:"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if Bucket not in result.stdout:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket")

    def create_bucket(self, Bucket):
        cmd = ["rclone", "mkdir", f"{self.remote}:{Bucket}"]
        subprocess.run(cmd, check=True)

    def get_object(self, Bucket, Key, Range=None):
        cmd = ["rclone", "cat", f"{self.remote}:{Bucket}/{Key}"]
        # Range is ignored: rclone cat doesn't support byte offsets out-of-the-box
        return {"Body": RcloneBody(cmd)}

    def put_object(self, Bucket, Key, Body):
        cmd = ["rclone", "rcat", f"{self.remote}:{Bucket}/{Key}"]
        
        if isinstance(Body, str):
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
            assert proc.stdin is not None
            proc.stdin.write(Body)
            proc.stdin.close()
        elif hasattr(Body, '__iter__') and not isinstance(Body, (bytes, bytearray)):
            # Handle string iterators (generators)
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
            assert proc.stdin is not None
            for chunk in Body:
                proc.stdin.write(chunk)
            proc.stdin.close()
        else:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            assert proc.stdin is not None
            proc.stdin.write(Body)
            proc.stdin.close()
            
        proc.wait()

    def upload_file(self, Filename, Bucket, Key):
        cmd = ["rclone", "copyto", Filename, f"{self.remote}:{Bucket}/{Key}"]
        subprocess.run(cmd, check=True)

    def download_file(self, Bucket, Key, Filename):
        cmd = ["rclone", "copyto", f"{self.remote}:{Bucket}/{Key}", Filename]
        subprocess.run(cmd, check=True)


def get_client(*args, **kwargs):
    return RcloneS3Client(remote="HetznerS3")
