from src.storage.s3 import S3Storage


class MinioStorage(S3Storage):
    """
    MinioStorage is fully S3 compatible. 
    It inherits from S3Storage and uses the same async aioboto3 implementation.
    """
    pass

