import os
import shutil

from zenml import step

from src.config.settings import DATASET_YOLO_CONFIG_NAME
from src.models.model_dataset import Dataset


@step
def dataset_preparator(dataset: Dataset, dataset_path: str) -> None:
    """
    Prepares the dataset for training.

    Args:
        dataset (Dataset): The dataset to prepare.
        dataset_path (str): The path where dataset has been downloaded.
    """
    generate_train_test_val_splits(dataset, dataset_path)
    dataset_to_yolo_converter(dataset, dataset_path)


@step
def generate_train_test_val_splits(dataset: Dataset, dataset_path: str) -> None:
    """
    Generates train, test and validation splits for the dataset.

    Args:
        dataset (Dataset): The dataset to split.
        dataset_path (str): The path where dataset has been downloaded.
    """
    split_names = dataset.split_names

    # if the dataset preparation has already been done, do nothing
    if os.path.isdir(os.path.join(dataset_path, split_names[0])) or os.path.isfile(
        os.path.join(dataset_path, DATASET_YOLO_CONFIG_NAME)
    ):
        return
    # create the directories

    for split in split_names:
        os.makedirs(os.path.join(dataset_path, split), exist_ok=True)
        os.makedirs(
            os.path.join(dataset_path, split, dataset.annotations_path), exist_ok=True
        )
        os.makedirs(
            os.path.join(dataset_path, split, dataset.images_path), exist_ok=True
        )

    # split the files into the directories according to the distribution weights
    for filename_json in os.listdir(
        os.path.join(dataset_path, dataset.annotations_path)
    ):
        split = dataset.get_next_split()
        filename_img = filename_json.replace(
            dataset.annotations_path, dataset.images_path
        ).replace(".json", ".png")
        old_file = os.path.join(dataset_path, dataset.annotations_path, filename_json)
        new_file = os.path.join(
            dataset_path, split, dataset.annotations_path, filename_json
        )
        shutil.move(old_file, new_file)
        old_file = os.path.join(dataset_path, dataset.images_path, filename_img)
        new_file = os.path.join(dataset_path, split, dataset.images_path, filename_img)
        shutil.move(old_file, new_file)

    # remove the empty directories
    os.rmdir(os.path.join(dataset_path, dataset.annotations_path))
    os.rmdir(os.path.join(dataset_path, dataset.images_path))


@step
def dataset_to_yolo_converter(dataset: Dataset, dataset_path: str) -> None:
    """
    Converts a custom dataset to YOLO format.

    Args:
        dataset (Dataset): The dataset to convert.
        dataset_path (str): The path where dataset has been downloaded.
    """
    dataset.to_yolo_format(dataset_path)
