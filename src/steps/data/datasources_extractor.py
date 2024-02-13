import os

from zenml import step

from src.models.model_bucket_client import BucketClient
from src.models.model_dataset import Dataset


@step()
def datasources_extractor(
    dataset: Dataset, bucket_client: BucketClient, destination_path: str
) -> str:
    """
    Extracts the dataset from the data source to the destination path.

    Args:
        bucket_client (BucketClient): The Minio client.
        destination_path (str): The destination path where the dataset will be extracted.

    Returns:
        str: The path where the dataset has been extracted.
    """
    folder_name = dataset.uuid
    if os.path.isdir(os.path.join(destination_path, folder_name)):
        return os.path.join(destination_path, folder_name)

    dataset.download(bucket_client, destination_path)
    return os.path.join(destination_path, folder_name)
