from typing import List, Optional, Tuple

from mixture_optimization.datamodels.trial_tracking_config import Experiment, ExperimentConfig, TrialType
from mixture_optimization.datamodels.weight_selector_config import WeightSelectorConfig
from attrs import define

from mixture_optimization.weight_selector.utils.botorch_constraints import get_bounds_from_config
import botorch.utils.transforms as bo_transforms

@define
class TrialMemoryUnit:
    trial_index: int
    trial_type: TrialType
    weights: List[float]
    value: Optional[float] = None



class WeightSelectorInterface:
    def __init__(self, config: WeightSelectorConfig, experiment_config: ExperimentConfig):
        self.config = config
        self.experiment_config = experiment_config
        assert self.config.no_weights > 1, "Optimization requires at least 2 weights"
        self.trial_memory: List[TrialMemoryUnit] = []

    def setup(self, experiment_history: Optional[Experiment]) -> Tuple[bool, ExperimentConfig]:
        """"
        Initialize from previous config. REturn (didUpdateCOnfig, UpdatedConfig)
        """
        raise NotImplementedError

    def add_evaluation(self, value: float, trial_index: int) -> None:
        assert self.trial_memory[-1].trial_index == trial_index, "Trial index mismatch"
        value_optim = value if self.config.maximize else -value
        self.trial_memory[-1].value = value_optim

    def propose_next_weights(self) -> Tuple[List[float], TrialType]:
        assert self.no_initialization_started() == self.no_initialization_completed(), "Inconsistent trial count. We can only have one trial running at one time"
        assert self.no_optimization_started() == self.no_optimization_completed(), "Inconsistent trial count. We can only have one trial running at one time"

        if self.no_initialization_started() < self.config.no_initializations:
            assert self.no_optimization_started() == 0, "Optimization started before all initializations are done"
            return self._propose_next_weights_initialization()
        else:
            assert self.no_initialization_completed() == self.config.no_initializations, "Not all initializations are done"
            return self._propose_next_weights_optimization()
    
    def _propose_next_weights_optimization(self):
        raise NotImplementedError

    def _propose_next_weights_initialization(self):  
        """
        Helper function to sample next weigthts from the initialization array
        """
        assert self.no_optimization_started() == 0, "Optimization started before all initializations are done"
        trial_idx = self.get_next_trial_idx()      
        assert trial_idx == self.no_initialization_completed(), "Trial idx mismatch. Trial index must increase prev trial by 1"
        weights = self.experiment_config.initialization_weights[trial_idx]

        unit = TrialMemoryUnit(
            trial_index=trial_idx,
            trial_type=TrialType.INITIALIZATION,
            weights=weights
        )
        self.trial_memory.append(unit)
        return weights, TrialType.INITIALIZATION
    
    def get_best_weights(self) -> Tuple[List[float], float]:
        best_value = -float('inf')
        best_weights = None
        for trial in self.trial_memory:
            if trial.value > best_value:
                best_value = trial.value
                best_weights = trial.weights
        
        best_value_optim = best_value if self.config.maximize else -best_value
        return (best_weights, best_value_optim)

    def experiment_done(self) -> bool:
        if self.config.no_optimizations == None:
            raise ValueError("No optimization number specified, please override")
        
        assert self.no_initialization_completed() <= self.config.no_initializations, "Too many initializations started"
        assert self.no_optimization_completed() <= self.config.no_optimizations, "Too many optimizations started"

        return (self.no_initialization_completed() == self.config.no_initializations) and (self.no_optimization_completed() == self.config.no_optimizations)
    
    def attach_trial(self, weights: List[float], trial_type: TrialType) -> None:
        if trial_type == TrialType.INITIALIZATION:
            assert self.no_initialization_started() < self.config.no_initializations, "Too many initializations started"
        elif trial_type == TrialType.OPTIMIZATION:
            assert self.no_optimization_started() < self.config.no_optimizations, "Too many optimizations started"
            assert self.no_initialization_completed() == self.config.no_initializations, "Optimization started before all initializations are done"
        
        self.trial_memory.append(TrialMemoryUnit(trial_index=self.get_next_trial_idx(), trial_type=trial_type, weights=weights))

    def no_initialization_started(self) -> int:
        no_initialization_started = 0
        for trial in self.trial_memory:
            if trial.trial_type == TrialType.INITIALIZATION:
                no_initialization_started += 1
        return no_initialization_started
    
    def no_optimization_started(self) -> int :
        no_optimization_started = 0
        for trial in self.trial_memory:
            if trial.trial_type == TrialType.OPTIMIZATION:
                no_optimization_started += 1
        return no_optimization_started
    
    def no_initialization_completed(self) -> int:
        no_initialization_completed = 0
        for trial in self.trial_memory:
            if trial.trial_type == TrialType.INITIALIZATION and trial.value is not None:
                no_initialization_completed += 1
        return no_initialization_completed
    
    def no_optimization_completed(self) -> int:
        no_optimization_completed = 0
        for trial in self.trial_memory:
            if trial.trial_type == TrialType.OPTIMIZATION and trial.value is not None:
                no_optimization_completed += 1
        return no_optimization_completed

    def get_next_trial_idx(self) -> int:
        assert self.no_initialization_started() == self.no_initialization_completed(), "Inconsistent trial count. We can only have one trial running at one time"
        assert self.no_optimization_started() == self.no_optimization_completed(), "Inconsistent trial count. We can only have one trial running at one time"
        return len(self.trial_memory)
    
    def _convert_free_weights_to_pdf(self, free_weights):
        fixed_weight = 1 - sum(free_weights)
        return [*free_weights, fixed_weight]
    

    def _normalize(self, X):
        # without specific bounds, constraints are from 0,1 does not require scaling
        if self.config.bounds is None:
            return X
        
        bounds = get_bounds_from_config(self.config.bounds)
        return bo_transforms.normalize(X, bounds)
    
    def _unnormalize(self, X):
        if self.config.bounds is None:
            return X
        
        bounds = get_bounds_from_config(self.config.bounds)
        return bo_transforms.unnormalize(X, bounds)

        
    
    