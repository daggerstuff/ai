import os
import json
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class S3Streamer:
    """Streams datasets to and from Hetzner S3 without loading everything into memory."""
    
    def __init__(self):
        # Configure boto3 for Hetzner S3
        self.endpoint_url = os.environ.get('HETZNER_S3_ENDPOINT', 'https://hel1.your-objectstorage.com')
        self.access_key = os.environ.get('HETZNER_S3_ACCESS_KEY')
        self.secret_key = os.environ.get('HETZNER_S3_SECRET_KEY')
        self.region = os.environ.get('HETZNER_S3_REGION', 'hel1')
        self.bucket = os.environ.get('HETZNER_S3_BUCKET', 'pixeldata')

        if not self.access_key or not self.secret_key:
            raise ValueError("HETZNER_S3_ACCESS_KEY and HETZNER_S3_SECRET_KEY must be set in .env")

        self.client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region
        )

    def list_files(self, prefix):
        """Yields all object keys under a specific prefix."""
        paginator = self.client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                yield obj['Key']

    def stream_jsonl(self, key):
        """Yields parsed JSON objects line-by-line from a JSONL file in S3."""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            # Use iter_lines to stream the response body
            for line in response['Body'].iter_lines():
                if line:
                    yield json.loads(line.decode('utf-8'))
        except self.client.exceptions.NoSuchKey:
            print(f"Warning: Key {key} not found in bucket {self.bucket}")
            return

    def stream_text(self, key):
        """Yields decoded text line-by-line from a text file in S3."""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            for line in response['Body'].iter_lines():
                if line is not None:
                    yield line.decode('utf-8')
        except self.client.exceptions.NoSuchKey:
            print(f"Warning: Key {key} not found in bucket {self.bucket}")
            return

    def download_to_file(self, key, local_path):
        """Downloads a file directly to disk for processing (e.g. for EPUB/PDF)."""
        self.client.download_file(self.bucket, key, local_path)

    def write_jsonl(self, key, iterator):
        """
        Streams a large iterable of dicts as JSONL directly to S3.
        Note: Boto3 upload_fileobj accepts a file-like object. We can use a generator
        with smart_open, but for simplicity we will write to a local temp file and upload.
        """
        import tempfile
        
        # We use a named temporary file to avoid keeping it all in RAM
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as temp_file:
            temp_path = temp_file.name
            for item in iterator:
                temp_file.write(json.dumps(item) + '\n')
                
        # Upload the temp file
        self.client.upload_file(temp_path, self.bucket, key)
        
        # Cleanup
        os.unlink(temp_path)
        print(f"Successfully uploaded dataset to {key}")
