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

    def stream_jsonl(self, key, chunk_bytes=512 * 1024 * 1024):
        """
        Yields parsed JSON objects line-by-line from a JSONL file in S3.
        Reads in byte-range chunks to avoid connection drops on very large files.
        """
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=key)
            total_size = head['ContentLength']
        except self.client.exceptions.NoSuchKey:
            print(f"Warning: Key {key} not found in bucket {self.bucket}")
            return

        leftover = b""
        offset = 0

        while offset < total_size:
            end = min(offset + chunk_bytes - 1, total_size - 1)
            try:
                response = self.client.get_object(
                    Bucket=self.bucket, Key=key,
                    Range=f"bytes={offset}-{end}"
                )
                data = response['Body'].read()
            except Exception as e:
                print(f"Warning: range read failed at offset {offset}: {e}. Retrying...")
                import time; time.sleep(3)
                response = self.client.get_object(
                    Bucket=self.bucket, Key=key,
                    Range=f"bytes={offset}-{end}"
                )
                data = response['Body'].read()

            chunk = leftover + data
            lines = chunk.split(b'\n')
            # Last element may be incomplete — carry it over
            leftover = lines[-1]

            for line in lines[:-1]:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line.decode('utf-8'))
                    except Exception:
                        pass

            offset = end + 1

        # Handle any remaining bytes
        if leftover.strip():
            try:
                yield json.loads(leftover.decode('utf-8'))
            except Exception:
                pass


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

    def write_jsonl(self, key, iterator, chunk_size_mb=64):
        """
        Streams an iterable of dicts as JSONL directly to S3 using multipart upload.
        This avoids writing anything to local disk, preventing disk quota issues.
        """
        import io
        
        part_number = 1
        parts = []
        buffer = io.BytesIO()
        chunk_bytes = chunk_size_mb * 1024 * 1024  # e.g. 64MB chunks
        
        # Initiate multipart upload
        mpu = self.client.create_multipart_upload(Bucket=self.bucket, Key=key)
        upload_id = mpu['UploadId']
        
        try:
            for item in iterator:
                line = (json.dumps(item) + '\n').encode('utf-8')
                buffer.write(line)
                
                # When buffer hits the chunk size, upload that part
                if buffer.tell() >= chunk_bytes:
                    buffer.seek(0)
                    response = self.client.upload_part(
                        Bucket=self.bucket,
                        Key=key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=buffer.read()
                    )
                    parts.append({'PartNumber': part_number, 'ETag': response['ETag']})
                    print(f"  Uploaded part {part_number} ({chunk_size_mb}MB) to S3...")
                    part_number += 1
                    buffer = io.BytesIO()
            
            # Upload whatever remains in the buffer as the final part
            if buffer.tell() > 0:
                buffer.seek(0)
                response = self.client.upload_part(
                    Bucket=self.bucket,
                    Key=key,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=buffer.read()
                )
                parts.append({'PartNumber': part_number, 'ETag': response['ETag']})
                print(f"  Uploaded final part {part_number} to S3.")
            
            # Complete the multipart upload
            self.client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            print(f"Successfully uploaded dataset to {key} ({part_number} parts)")
            
        except Exception as e:
            # Abort multipart upload on failure to avoid leaving partial data
            self.client.abort_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id
            )
            raise e
