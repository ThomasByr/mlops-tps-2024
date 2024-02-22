# imports
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from zenml import step
import os


@step
def model_evaluator(
    trained_model_path: str,
    dataset_path: str,
    pipeline_config: dict,
):
    """
    Evaluates the model

    Args:
        dataset_path: The path of the dataset.
        pipeline_config: The pipeline configuration.

    Returns:
        dictionnary of model metrics
    """
    params = pipeline_config.get("model", {})

    data_path = os.path.join(dataset_path, "dataset.yaml")

    args = dict(
        model=trained_model_path,
        data=data_path,
        split="test",
        mode="val",
        batch=params["batch_size"],
        imgsz=params["imgsz"],
        iou=0.6,
    )
    validator = DetectionValidator(args=args)
    validator.training = False
    validator()

    tmp = validator.get_stats()

    return dict(
        precision=tmp.get("metrics/precision(B)", -1),
        recall=tmp.get("metrics/recall(B)", -1),
        map50=tmp.get("metrics/mAP50(B)", -1),
        map50_95=tmp.get("metrics/mAP50-95(B)", -1),
    )
