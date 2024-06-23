import logging
from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface
from typing import List
import numpy as np
import torch

from botorch.utils.sampling import sample_simplex
logger = logging.getLogger("experiment_runner")

class RandomSimplexSelector(WeightSelectorInterface):
    def __init__(self, config: dict):
        super().__init__(config)
        self.no_weights = config['no_weights']
        self.no_trials = config['no_trials']
        self.no_completed_trials = 0

        if "initialization_weights" in config:
            logger.info("Using initialization weights")
            self.initialization_weights = config["initialization_weights"]
        else:
            samples = sample_simplex(self.no_weights, self.no_trials, qmc=True)
            self.initialization_weights = samples.tolist()
            config["initialization_weights"] = self.initialization_weights

    
    def parse_history(self, run_history: List[dict]):
        for i, run in enumerate(run_history):
            trial_index = run["idx"]
            assert trial_index == i, f"Trial index {trial_index} does not match the expected index {i}"
            reference_weights = self.initialization_weights[i]
            mixing_weights = run["true_mixing_weights"]
            assert torch.allclose(torch.tensor(reference_weights), torch.tensor(mixing_weights), atol=1e-4), f"True mixing weights {mixing_weights} do not match the reference weights {reference_weights}"
            
            if "weighted_values" in run:
                self.add_evaluation()  
    
    def propose_next_weights(self):
       run_idx = self.no_completed_trials
       weights = self.initialization_weights[run_idx]
       assert sum(weights) - 1 < 1e-4, f"Sum of weights should be 1, but is {sum(weights)}"
       return weights, {}

    def get_best_weights(self):
        return None

    def add_evaluation(self, perplexity, run_index):
        self.no_completed_trials += 1
    
    def experiment_done(self):
        assert self.no_completed_trials <= self.no_trials, "Too many runs completed"
        return self.no_completed_trials == self.no_trials