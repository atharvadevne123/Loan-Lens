"""AWS S3 integration for model artifact storage (boto3 stub)."""

import logging

logger = logging.getLogger(__name__)


def upload_model_to_s3(
    bucket: str,
    prefix: str = "loan-lens/models",
    local_path: str = "model.joblib",
) -> str | None:
    """Upload the trained model artifact to S3 and return the S3 URI."""
    try:
        import boto3

        s3 = boto3.client("s3")
        key = f"{prefix}/model.joblib"
        s3.upload_file(local_path, bucket, key)
        uri = f"s3://{bucket}/{key}"
        logger.info("model_uploaded_to_s3 uri=%s", uri)
        return uri
    except ImportError:
        logger.debug("boto3 not installed — S3 upload skipped")
    except Exception as exc:
        logger.warning("S3 upload failed: %s", exc)
    return None


def download_model_from_s3(
    bucket: str,
    prefix: str = "loan-lens/models",
    local_path: str = "model.joblib",
) -> bool:
    """Download the latest model artifact from S3."""
    try:
        import boto3

        s3 = boto3.client("s3")
        key = f"{prefix}/model.joblib"
        s3.download_file(bucket, key, local_path)
        logger.info("model_downloaded_from_s3 bucket=%s key=%s", bucket, key)
        return True
    except ImportError:
        logger.debug("boto3 not installed — S3 download skipped")
    except Exception as exc:
        logger.warning("S3 download failed: %s", exc)
    return False


def list_model_versions(bucket: str, prefix: str = "loan-lens/models") -> list[str]:
    """List all model artifact versions stored in S3."""
    try:
        import boto3

        s3 = boto3.client("s3")
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]
    except ImportError:
        logger.debug("boto3 not installed")
    except Exception as exc:
        logger.warning("S3 list failed: %s", exc)
    return []
