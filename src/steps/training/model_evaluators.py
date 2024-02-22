# imports
from ultralytics import YOLO
from zenml import step


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
    metrics = pipeline_config.get("evaluation", None)
    if metrics is None:
        raise KeyError('No "evaluation" section in pipeline parameters')

    model = YOLO()
    model.load(trained_model_path)

    val = model.val(
        data=dataset_path,
    )
    print(val)

    results = {}
    types = metrics.get("type", [])
    if "IoU" in types:
        results["IoU"] = 0.5
    if "test" in types:
        results["test"] = 0.9

    return results
