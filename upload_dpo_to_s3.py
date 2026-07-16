import os
from pathlib import Path
import boto3
from dotenv import load_dotenv

load_dotenv("ai/.env")

endpoint = os.environ.get("HETZNER_S3_ENDPOINT")
access_key = os.environ.get("HETZNER_S3_ACCESS_KEY")
secret_key = os.environ.get("HETZNER_S3_SECRET_KEY")
region = os.environ.get("HETZNER_S3_REGION", 'hel1')

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=region,
)

bucket = os.environ.get("HETZNER_S3_BUCKET", "pixel-data")
if bucket == "pixeldata":
    pass

local_file = "ai/data/generated/dpo_preference_pairs_10k.jsonl"
s3_key = "training/v1/stage3_stress_test/processed/safety_dpo_pairs_10k.jsonl"

print(f"Uploading {local_file} to s3://{bucket}/{s3_key}...")
s3.upload_file(local_file, bucket, s3_key)
print("Done!")
