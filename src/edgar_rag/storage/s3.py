"""S3-backed object store.

Same contract as the local store, so switching is a `STORAGE_BACKEND=s3`
env change rather than a code change. boto3 is imported lazily so local
development never pays for it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from edgar_rag.storage.base import ObjectStore

if TYPE_CHECKING:  # pragma: no cover
    pass


class S3ObjectStore(ObjectStore):
    """Stores objects in an S3 bucket under an optional key prefix."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str = "us-east-1",
        client: Any | None = None,
    ) -> None:
        if client is None:
            import boto3  # imported here to keep it optional for local runs

            client = boto3.client("s3", region_name=region)
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> str:
        full_key = self._key(key)
        extra = {"ContentType": content_type} if content_type else {}
        self.client.put_object(Bucket=self.bucket, Key=full_key, Body=data, **extra)
        return f"s3://{self.bucket}/{full_key}"

    def get_bytes(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                raise KeyError(key) from exc
            raise
        return response["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "403"}:
                return False
            raise
        return True

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        strip = len(self.prefix) + 1 if self.prefix else 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
            for obj in page.get("Contents", []):
                yield obj["Key"][strip:]

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))
