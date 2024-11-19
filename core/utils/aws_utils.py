import base64
import logging
import os
from functools import lru_cache
from pathlib import Path
import tempfile
from typing import Tuple

import boto3
from botocore.exceptions import ClientError
from bson import ObjectId
from dotenv import load_dotenv
from mypy_boto3_s3 import S3Client
from pymongo.database import Database

from .file_extensions import detect_file_type

load_dotenv()
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = "databridge-storage"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "test")


@lru_cache(maxsize=1)
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="us-east-2",
    )



def upload_from_encoded_string(s3_client: S3Client, content: str, doc_id: str, bucket_name=BUCKET_NAME):
    extension = detect_file_type(content)
    try:
        # Create temporary file for content
        decoded_content = base64.b64decode(content)
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(decoded_content)
            temp_file_path = temp_file.name
            bucket, key = upload_to_s3_from_path(s3_client, temp_file_path, doc_id, bucket_name)
        os.remove(temp_file_path)
        return bucket, key
    except Exception as e:
        logging.error(f"Error decoding base64 content: {e}")
        return None, None


def upload_from_file(s3_client: S3Client, file, bucket_name=BUCKET_NAME):
    bytestream = file.file
    key = f"{file.filename}"
    try:
        s3_client.upload_fileobj(bytestream, bucket_name, key)
    except ClientError as e:
        logging.error(e)
        return None, None
    return bucket_name, key


def upload_video_to_s3_from_file(file):
    s3_client = get_s3_client()
    return upload_from_file(s3_client, file)


def upload_to_s3_from_path(s3_client, file_name, object_name, bucket_name=BUCKET_NAME):
    try:
        s3_client.upload_file(file_name, bucket_name, object_name)
    except ClientError as e:
        logging.error(e)
        return None, None
    return bucket_name, object_name


def upload_video_to_s3(video_path: str, video_name: str):
    s3_client = get_s3_client()
    return upload_to_s3_from_path(
        s3_client=s3_client, file_name=str(video_path), bucket_name=BUCKET_NAME, object_name=video_name
    )


def upload_image_to_s3(thumbnail_path: Path, thumbnail_name: str):
    s3_client = get_s3_client()
    return upload_to_s3_from_path(
        s3_client=s3_client, file_name=str(thumbnail_path), bucket_name=BUCKET_NAME, object_name=thumbnail_name
    )


def create_presigned_url(s3_client, bucket_name, object_name, expiration=3600) -> str:
    if not object_name or not bucket_name:
        return ""
    try:
        response = s3_client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket_name, "Key": object_name}, ExpiresIn=expiration
        )
    except ClientError as e:
        logging.error(e)
        return ""
    return response


def delete_file_from_s3(s3_client, bucket_name, key):
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=key)
        logging.info(f"File {key} deleted successfully from bucket {bucket_name}")
        return True
    except ClientError as e:
        logging.error(f"Error deleting file from S3: {e}")
        return False


def get_file_from_s3(s3_client, bucket_name, key):
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        file_content = response["Body"].read()
        logging.info(f"File {key} retrieved successfully from bucket {bucket_name}")
        return file_content
    except ClientError as e:
        logging.error(f"Error retrieving file from S3: {e}")
        return None


def get_specific_asset(db: Database, s3_client: S3Client, asset_id: str) -> Tuple[str, bytes]:
    asset = db[COLLECTION_NAME].find_one({"_id": ObjectId(asset_id)})
    bucket_name, key = asset["bucket_name"], asset["key"]
    s3_object = s3_client.get_object(Bucket=bucket_name, Key=key)
    return key, s3_object["Body"].read()
