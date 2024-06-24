from attrs import define, frozen
from enum import Enum, auto, unique
from typing import Dict, Optional, List

@unique
class TrialStatus(Enum):
    INITIALIZED = auto()
    MIXED = auto()
    MANIFEST_CREATED = auto()
    RUNNING = auto()
    RAN = auto()
    PARSED = auto()
    DELETED = auto()

@unique
class TrialType(Enum):
    INITIALIZATION = auto()
    OPTIMIZATION = auto()


@frozen
class ValResult:
    domain: str
    loss: float
    perplexity: float
    eval_tokens: int
    train_tokens_seen: int
    loss_tokens_lower_95: float
    loss_tokens_upper_95: float


@define
class Trial:
    idx: int
    experiment_idx: int
    name: str
    status: TrialStatus
    type: TrialType
    workspace: str
    weights: list[float]
    kwargs: dict = {}

    # initializes later
    mixing_log_path: Optional[str] = None
    dataset: Optional[str] = None
    true_mixing_weights: Optional[List[float]] = None
    open_lm_log_dir : Optional[str] = None
    val_file_path: Optional[str] = None
    val_results: Optional[List[Dict[str, ValResult]]] = None
    weighted_val_perplexity: Optional[float] = None

@frozen
class ExperimentConfig:
    initialization_weights: List[List[float]] 
    experiment_idx: int

@define
class Experiment:
    config: ExperimentConfig
    trials: List[Trial] = []