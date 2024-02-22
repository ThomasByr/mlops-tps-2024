import os

from omegaconf import OmegaConf
from zenml import pipeline

from src.config.settings import EXTRACTED_DATASETS_PATH, MLFLOW_EXPERIMENT_PIPELINE_NAME
from src.models.model_dataset import Dataset
from src.steps.data.datalake_initializers import (
    data_source_list_initializer,
    minio_client_initializer,
)
from src.steps.data.dataset_creators import dataset_creator
from src.steps.data.dataset_preparators import (
    dataset_preparator,
)
from src.steps.data.datasources_extractor import datasources_extractor
from src.steps.training.model_evaluators import model_evaluator
from src.steps.training.model_trainers import (
    get_pre_trained_weights_path,
    model_trainer,
)
from src.steps.training.model_validation import model_validation


@pipeline(name=MLFLOW_EXPERIMENT_PIPELINE_NAME, enable_cache=False)
def gitflow_experiment_pipeline(cfg: str) -> None:
    """
    Experiment a local training and evaluate if the model can be deployed.

    Args:
        cfg: The Hydra configuration.
    """
    pipeline_config = OmegaConf.to_container(OmegaConf.create(cfg))
    # pipeline_config = {'project': {'name': 'mlops-hardshat'}, 'model': {'epochs': 10}, 'evaluation': {'key': 'value'}, 'pipeline': {'name': 'experiment'}}

    bucket_client = minio_client_initializer()
    data_source_list = data_source_list_initializer()

    dataset = dataset_creator()

    destination_path = "./datasets/"
    dataset_path = datasources_extractor(dataset, bucket_client, destination_path)

    dataset_preparator(dataset, dataset_path)

    # Train the model

    model_dir_path = "./models/"
    pre_trained_weights_path = get_pre_trained_weights_path(model_dir_path)

    # comet_ml.init()

    # TODO temporaire pour test eval
    # trained_model_path = model_trainer(
    #     model_dir_path, dataset_path, pre_trained_weights_path, pipeline_config
    # )
    trained_model_path = os.path.join(
        model_dir_path, f"yolo_model_v6", "weights", "best.pt"
    )

    # Evaluate the model
    test_metrics_result = model_evaluator(
        trained_model_path=trained_model_path,
        dataset_path=dataset_path,
        pipeline_config=pipeline_config,
    )

    # Validation
    decision = model_validation(
        metrics=test_metrics_result,
        pipeline_config=pipeline_config,
    )
