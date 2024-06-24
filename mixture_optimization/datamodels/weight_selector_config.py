from typing import Optional
from attrs import frozen
from enum import Enum

class WeightSelectorType(Enum):
    BAYESIAN = "bayesian"
    TURBO = "turbo"
    SIMPLEX = "simplex"
    DETERMINISTIC = "deterministic"




@frozen
class WeightSelectorConfig:
    type: WeightSelectorType
    no_weights: int
    no_initializations: int
    maximize: bool
    no_optimizations: Optional[int] = None


@frozen
class ExperimentManagerConfig:
    no_experiments: int
    weight_selector_config: WeightSelectorConfig