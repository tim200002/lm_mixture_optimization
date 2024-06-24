from typing import List, Optional, Tuple

from mixture_optimization.datamodels.trial_tracking_config import Experiment, ExperimentConfig, TrialType
from mixture_optimization.datamodels.weight_selector_config import WeightSelectorConfig
from attrs import define

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
        self.trial_offset = experiment_config.trial_offset
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
        raise NotImplementedError
    
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
            self.no_initialization_started += 1
        elif trial_type == TrialType.OPTIMIZATION:
            assert self.no_optimization_started() < self.config.no_optimizations, "Too many optimizations started"
            assert self.no_initialization_completed() == self.config.no_initializations, "Optimization started before all initializations are done"
            self.no_optimization_started += 1
        
        trial_idx = self.experiment_config.trial_offset + len(self.trial_memory) - 1 # first run is index 0
        self.trial_memory.append(TrialMemoryUnit(trial_index=trial_idx, trial_type=trial_type, weights=weights))

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
    
    