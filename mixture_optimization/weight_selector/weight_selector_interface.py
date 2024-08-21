import logging
import random
from typing import List, Optional, Tuple, Dict

import torch

from mixture_optimization.datamodels.trial_tracking_config import Experiment, ExperimentConfig, Trial, TrialType
from mixture_optimization.datamodels.weight_selector_config import WeightSelectorConfig
from attrs import define
from botorch.utils.sampling import sample_simplex, HitAndRunPolytopeSampler

from mixture_optimization.weight_selector.utils.botorch_constraints import get_bounds_from_config
import botorch.utils.transforms as bo_transforms

logger = logging.getLogger("experiment_runner")

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
        self.no_weights = self.config.no_weights
        assert self.no_weights > 1, "Optimization requires at least 2 weights"
        self.trial_memory: List[TrialMemoryUnit] = []


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
        if self.config.no_optimizations is None:
            raise ValueError("No optimization number specified, please override")
        
        assert self.no_initialization_completed() <= self.config.no_initializations, "Too many initializations started"
        assert self.no_optimization_completed() <= self.config.no_optimizations, "Too many optimizations started"

        return (self.no_initialization_completed() == self.config.no_initializations) and (self.no_optimization_completed() == self.config.no_optimizations)
    
    def attach_trial(self, weights: Dict[str, float], trial_type: TrialType) -> None:
        if trial_type == TrialType.INITIALIZATION:
            assert self.no_initialization_started() < self.config.no_initializations, "Too many initializations started"
        elif trial_type == TrialType.OPTIMIZATION:
            if self.config.no_optimizations is not None:
                assert self.no_optimization_started() < self.config.no_optimizations, "Too many optimizations started"
            assert self.no_initialization_completed() == self.config.no_initializations, "Optimization started before all initializations are done"
        
        weights_values = list(weights.values())
        
        self.trial_memory.append(TrialMemoryUnit(trial_index=self.get_next_trial_idx(), trial_type=trial_type, weights=weights_values))

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
    
    @classmethod
    def _convert_free_weights_to_pdf(cls, free_weights):
        fixed_weight = 1 - sum(free_weights)
        eps = 1e-6
        assert fixed_weight >= -eps, f"Fixed weight must be positive, got {fixed_weight}"
        if fixed_weight < 0:
            logger.warning(f"Fixed weight is negative of value {fixed_weight}. Setting to 0.")
            fixed_weight = 0
        return [*free_weights, fixed_weight]
    
    @classmethod
    def _convert_free_weights_to_pdf_tensor(cls, free_weights:torch.Tensor):
        sums = free_weights.sum(dim=-1) # sum along rows
        fixed_weight = 1 - sums
        return torch.cat([free_weights, fixed_weight.unsqueeze(-1)], dim=-1)
    
    def _get_bounds_botorch(self):
        if self.config.bounds is None:
            return None
        return get_bounds_from_config(self.config.bounds)
    
    @classmethod
    def _sample_uniform(cls, no_samples: int, no_weights: int, bounds: Optional[List[Tuple[float, float]]] = None):
        # ToDo, test uniform sampler
        # when no bounds we can simply sample a simplex
        if bounds is None:
            logger.info("No bounds provided, sampling from simplex")
            return sample_simplex(no_weights, no_samples, qmc=True)
        
        logger.info("Bounds provided, sampling from polytope")
        assert len(bounds) == no_weights, "Bounds must be provided for all weights"
        # inequality constraints, i.e. bounds
        # I1 = torch.eye(no_weights)
        # A = torch.cat([I1,-I1], dim=-1).view(-1, I1.shape[-1]) # fill all rows with patten [[1,0,0,...][-1,0,0,...][0,1,0,...][0,-1,0,...]...
        # bounds_concat = [[bounds[1], - bounds[0]] for bounds in bounds]
        # b = torch.tensor(bounds_concat).reshape(-1, 1)
        lb = torch.tensor([0.0 if b is None else b[0] for b in bounds])
        ub = torch.tensor([1.0 if b is None else b[1] for b in bounds])
        bounds = torch.stack([lb, ub], dim=0)

        # equality constraints, i.e. valid pdf
        C = torch.ones(1, no_weights)
        d = torch.ones(1, 1) 

        sampler = HitAndRunPolytopeSampler(None, (C, d), bounds)
        return sampler.draw(no_samples)

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
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

        
    
    