# imports
from zenml import step


@step
def model_evaluator(
    trained_model_path: str,
    pipeline_config: dict,
):
    """
    Evaluates the model

    Args:
        dataset_path: The path of the dataset.
        threshold
        pipeline_config: The pipeline configuration.

    Returns:
        dictionnary of model metrics
    """
    metrics = pipeline_config.get("evaluation", None)
    if metrics is None:
        raise KeyError('No "evaluation" section in pipeline parameters')
    print(metrics)
    return {"IoU": 0.5, "test": 0.1}
