import contextlib
import json
import logging
import os
import subprocess

logger = logging.getLogger(__name__)

DEFAULT_DATASTORE = "US-EAST-VA"
DEFAULT_BUCKET = "pixeldata"


class S3Streamer:
    """Streams datasets to and from OVHcloud AI Object Storage using ovhai CLI."""

    def __init__(self, datastore: str = DEFAULT_DATASTORE, bucket: str = DEFAULT_BUCKET):
        self.datastore = datastore
        self.bucket = bucket
        self.container_spec = f"{self.bucket}@{self.datastore}"

    def list_files(self, prefix: str = ""):
        """Yields object names in the OVH AI container."""
        cmd = ["ovhai", "bucket", "object", "list", self.container_spec]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("DATE") and "BYTES" not in line:
                parts = line.split()
                if len(parts) >= 4:
                    obj_name = parts[3]
                    if not prefix or obj_name.startswith(prefix):
                        yield obj_name

    def download_to_file(self, key: str, local_path: str):
        """Downloads an object from OVH AI container to disk."""
        cmd = ["ovhai", "bucket", "object", "download", self.container_spec, key, local_path]
        subprocess.run(cmd, check=True)

    def write_jsonl(self, key: str, iterator):
        """
        Writes an iterable of dicts as JSONL to disk and syncs directly to OVH AI Object Storage.
        """
        local_canonical = "dataset/final_dataset.jsonl"
        os.makedirs(os.path.dirname(local_canonical), exist_ok=True)

        count = 0
        with open(local_canonical, "w", encoding="utf-8") as f:
            for item in iterator:
                f.write(json.dumps(item) + "\n")
                count += 1

        logger.info("Saved %d clean JSONL records locally to %s", count, local_canonical)

        # Upload to OVH AI container via ovhai CLI
        cmd = ["ovhai", "bucket", "object", "upload", self.container_spec, local_canonical]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            uploaded_name = f"dataset/{os.path.basename(local_canonical)}"
            if uploaded_name != key:
                move_cmd = ["ovhai", "bucket", "object", "move", self.container_spec, uploaded_name, key]
                subprocess.run(move_cmd, capture_output=True, text=True)
            logger.info("Successfully uploaded dataset to OVH AI object store %s/%s", self.container_spec, key)
        else:
            err_msg = f"ovhai bucket upload to {self.container_spec} failed: {res.stderr.strip()}"
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        return count
