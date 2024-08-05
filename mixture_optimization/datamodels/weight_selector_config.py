from typing import List, Optional, Tuple
from attrs import frozen
from enum import Enum

class WeightSelectorType(Enum):
    BAYESIAN = "bayesian"
    TURBO = "turbo"
    SIMPLEX = "simplex"
    DETERMINISTIC = "deterministic"
    LIN_INTERP = "lin_interp"




@frozen
class WeightSelectorConfig:
    type: WeightSelectorType
    no_weights: int
    no_initializations: int
    maximize: bool
    bounds: Optional[List[Tuple[float, float] | None]] = None
    no_optimizations: Optional[int] = None
    normalize_bounds: bool = False
    kwargs: dict = {}


@frozen
class ExperimentManagerConfig:
    no_experiments: int
    weight_selector_config: WeightSelectorConfig