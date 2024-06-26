from attrs import define, frozen
from typing import Dict, Optional, Tuple, List
from enum import Enum, unique

from mixture_optimization.datamodels.data_mixing_config import DataMixingConfig
from mixture_optimization.datamodels.open_lm_config import OpenLMConfig
from mixture_optimization.datamodels.trial_tracking_config import Experiment
from mixture_optimization.datamodels.weight_selector_config import ExperimentManagerConfig, WeightSelectorConfig

@unique
class DatasetTag(Enum):
    DOLMA_V1_6_SMALL = "dolma_v1_6_small"

@frozen
class LogConfig:
    log_path: str
    runs_folder: str
    config_path: str
    experiment_history_path: str

@define
class Config:
    name: str
    description: str
    train_data: Dict[str, str]
    val_data: Dict[str, str]
    val_weights: List[float]
    data_workspace: str
    delete_dataset_after_run: bool
    dataset_tag: DatasetTag
    experiment_manager_config: ExperimentManagerConfig
    open_lm_config: OpenLMConfig
    data_mixing_config: DataMixingConfig
    id: Optional[str] = None
