# imports
from zenml import step


@step
def model_validation(
    metrics: dict,
    pipeline_config: dict,
) -> bool:
    """
    Validates the model

    Args:
        metrics: performance metrics of the model.
        pipeline_config: The pipeline configuration.

    Returns:
        True if model is valid
        else False
    """
    thresholds = pipeline_config.get("validation", {}).get("threshold", {})
    found_bad_metric = False
    for metric_type, value in metrics.items():
        thresh = thresholds.get(metric_type, 0)
        # print(f"metric : {metric_type}")
        # print(f"  value : {value}")
        # print(f"  threshold : {thresh}")
        # print(f"  =>", "good" if value >= thresh else "bad")
        if value < thresh:
            found_bad_metric = True
    return not found_bad_metric
