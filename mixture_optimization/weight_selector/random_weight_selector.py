from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface
from typing import List
import numpy as np

class RandomWeightSelector(WeightSelectorInterface):
    def __init__(self, config: dict):
        super().__init__(config)
        self.no_weights = config['no_weights']
        self.no_runs = config['no_runs']
        self.no_completed_runs = 0
    
    def parse_history(self, run_history: List[dict]):
        for run in run_history:
            if "weighted_values" in run:
                self.no_completed_runs += 1       
    
    def propose_next_weights(self):
       # Randomly select weights summing up to 1
        weights = np.random.rand(self.no_weights)
        weights /= np.sum(weights)
        
        return weights.tolist()

    def get_best_weights(self):
        return None

    def add_evaluation(self, perplexity, run_index):
        self.no_completed_runs += 1
    
    def experiment_done(self):
        assert self.no_completed_runs <= self.no_runs, "Too many runs completed"
        return self.no_completed_runs == self.no_runs