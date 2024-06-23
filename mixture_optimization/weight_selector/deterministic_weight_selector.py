from typing import List
from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface

class DeterministicWeightSelector(WeightSelectorInterface):
    def __init__(self, config: dict, run_history=None):
        super().__init__(config)
        self.weights = config["weights"]
        self.did_complete = False

    def propose_next_weights(self):
        return self.weights, {}
    
    def get_best_weights(self):
        return self.weights
    
    def add_evaluation(self, perplexity, run_index):
        self.did_complete = True

    def experiment_done(self):
        return self.did_complete
    
    def parse_history(self, run_history: List[dict]):
        pass
    
