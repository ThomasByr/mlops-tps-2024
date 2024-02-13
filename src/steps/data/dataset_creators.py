from zenml import step

from src.materializers.materializer_dataset import DatasetMaterializer
from src.models.model_dataset import Dataset


@step(output_materializers=DatasetMaterializer)
def dataset_creator() -> Dataset:
    """
    Creates a dataset for training.

    Returns:
        Dataset: The dataset for training.
    """
    dataset = Dataset(
        bucket_name="data-sources",
        seed=42,
        uuid="hardhat",
        annotations_path="annotations",
        images_path="images",
        distribution_weights=[0.6, 0.2, 0.2],
        label_map={
            0: "helmet",
            1: "head",
        },
    )

    return dataset
