import os

from ultralytics import YOLO
from zenml import step

from src.config.settings import (
    YOLO_PRE_TRAINED_WEIGHTS_NAME,
    YOLO_PRE_TRAINED_WEIGHTS_URL,
)


@step
def get_pre_trained_weights_path(model_dir_path: str) -> str:
    """
    Get the pre-trained weights path.

    Returns:
        The pre-trained weights path.
    """
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    pre_trained_weights_path = os.path.join(
        model_dir_path, YOLO_PRE_TRAINED_WEIGHTS_NAME
    )
    if not os.path.exists(pre_trained_weights_path):
        os.system(f"wget -O {pre_trained_weights_path} {YOLO_PRE_TRAINED_WEIGHTS_URL}")
    return pre_trained_weights_path


@step
def model_trainer(
    model_dir_path: str,
    dataset_path: str,
    pre_trained_weights_path: str,
    pipeline_config: dict,
) -> str:
    """
    Train the model.

    Args:
        dataset_path: The path of the dataset.
        pre_trained_weights_path: The path of the pre-trained weights.
        pipeline_config: The pipeline configuration.

    Returns:
        The path of the trained model.
    """
    # Load the pre-trained weights
    model = YOLO()  # pre_trained_weights_path)

    # Train the model
    params = pipeline_config["model"]
    data_path = os.path.join(dataset_path, "dataset.yaml")

    # Model name
    number_of_trained_models = len(os.listdir(model_dir_path))

    model.train(
        data=data_path,
        project=model_dir_path,
        name=f"yolo_model_v{number_of_trained_models}",  # specify the name of the experiment
        epochs=params["epochs"],
        batch=params["batch_size"],
        imgsz=params["imgsz"],
    )

    # Save the trained model
    trained_model_path = os.path.join(
        model_dir_path, f"yolo_model_v{number_of_trained_models}", "weights", "best.pt"
    )

    return trained_model_path
