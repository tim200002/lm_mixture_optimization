import logging
from mixture_optimization.datamodels.trial_tracking_config import Experiment, ExperimentConfig, TrialType
from mixture_optimization.datamodels.weight_selector_config import WeightSelectorConfig
from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface
from typing import List, Optional, Tuple
import numpy as np
import torch

from botorch.utils.sampling import sample_simplex
logger = logging.getLogger("experiment_runner")

class RandomSimplexSelector(WeightSelectorInterface):

    @staticmethod
    def from_scratch(config: WeightSelectorConfig, exp_idx: int) -> Tuple[WeightSelectorInterface, ExperimentConfig]:
        samples = sample_simplex(config.no_weights, config.no_initializations, qmc=True).tolist()
        exp_config = ExperimentConfig(initialization_weights=samples, experiment_idx=exp_idx)
        return RandomSimplexSelector(config, exp_config), exp_config
    
    @staticmethod
    def from_history(config: WeightSelectorConfig, experiment: Experiment) -> WeightSelectorInterface:
        return RandomSimplexSelector(config, experiment.experiment_config)


    def __init__(self, config: WeightSelectorConfig, experiment_config: ExperimentConfig):
        super().__init__(config, experiment_config)
        assert config.no_optimizations == None, "Simplex selector does not support optimizations"

    
    def parse_history(self, run_history: List[dict]):
        for i, run in enumerate(run_history):
            trial_index = run["idx"]
            assert trial_index == i, f"Trial index {trial_index} does not match the expected index {i}"
            reference_weights = self.experiment_config.initialization_weights[i]
            mixing_weights = run["true_mixing_weights"]
            assert torch.allclose(torch.tensor(reference_weights), torch.tensor(mixing_weights), atol=1e-4), f"True mixing weights {mixing_weights} do not match the reference weights {reference_weights}"
            
            if "weighted_values" in run:
                self.add_evaluation()  
    
    def propose_next_weights(self):
       next_trial_idx = self.no_initialization_started()
       weights = self.experiment_config.initialization_weights[next_trial_idx]
       assert sum(weights) - 1 < 1e-4, f"Sum of weights should be 1, but is {sum(weights)}"
       return weights, TrialType.INITIALIZATION

    def add_evaluation(self, value:float, trial_index: int):
        super().add_evaluation(value, trial_index)
    
    def experiment_done(self):
        # Override as no optimizations are supported
        return self.no_initialization_completed() == self.config.no_initializations