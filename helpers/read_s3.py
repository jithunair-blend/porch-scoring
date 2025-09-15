import boto3
import os
from config.config import S3_ASSETS, LOCAL_DIR

s3 = boto3.client("s3")
os.makedirs(LOCAL_DIR, exist_ok=True)

def get_asset(name: str) -> str:
    """
    Download the requested asset from S3 if not already cached locally.
    Returns the local file path.
    """
    if name not in S3_ASSETS:
        raise ValueError(f"Unknown asset requested: {name}")

    bucket = S3_ASSETS[name]["bucket"]
    key = S3_ASSETS[name]["key"]
    filename = os.path.join(LOCAL_DIR, os.path.basename(key))

    # Download only if not already present
    if not os.path.exists(filename):
        print(f"Downloading {key} from {bucket} ...")
        s3.download_file(bucket, key, filename)
    else:
        print(f"Using cached file: {filename}")

    return filename

