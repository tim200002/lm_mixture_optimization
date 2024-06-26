from attrs import define, frozen
from typing import Tuple, List, Optional
from enum import Enum, unique

from mixture_optimization.datamodels.data_mixing_config import DataMixingConfig
from mixture_optimization.datamodels.open_lm_config import OpenLMConfig
from mixture_optimization.datamodels.trial_tracking_config import Experiment
from mixture_optimization.datamodels.weight_selector_config import ExperimentManagerConfig, WeightSelectorConfig

@unique
class DatasetTag(Enum):
    DOLMA_V1_6_SMALL = "dolma_v1_6_small"

@frozen
class ExperimentTrackingConfig:
    log_path: str
    runs_folder: str
    config_path: str

@define
class Config:
    name: str
    description: str
    train_data: List[Tuple[str, str]]
    val_data: List[Tuple[str, str]]
    val_weights: List[float]
    workspace: str
    data_workspace: str
    delete_dataset_after_run: bool
    dataset_tag: DatasetTag
    experiment_manager_config: ExperimentManagerConfig
    open_lm_config: OpenLMConfig
    data_mixing_config: DataMixingConfig
    experiment_history: List[Experiment] = []
    experiment_tracking_config: Optional[ExperimentTrackingConfig] = None
