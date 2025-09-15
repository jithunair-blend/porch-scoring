import boto3
import os
from config.config import S3_OUTPUTS

s3 = boto3.client("s3")

def put_output(name: str, local_path: str) -> str:
    """
    Upload a local file to S3 based on the mapping in S3_OUTPUTS.
    Returns the full S3 URI where the file was written.
    """
    if name not in S3_OUTPUTS:
        raise ValueError(f"Unknown output requested: {name}")

    bucket = S3_OUTPUTS[name]["bucket"]
    key = S3_OUTPUTS[name]["key"]

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file does not exist: {local_path}")

    print(f"Uploading {local_path} → s3://{bucket}/{key} ...")
    s3.upload_file(local_path, bucket, key)

    return f"s3://{bucket}/{key}"
