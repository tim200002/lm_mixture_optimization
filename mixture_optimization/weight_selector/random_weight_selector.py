from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface
from typing import List
import numpy as np

class RandomWeightSelector(WeightSelectorInterface):
    def __init__(self, config: dict):
        super().__init__(config)
        self.no_weights = config['no_weights']
    
    def propose_next_weights(self):
       # Randomly select weights summing up to 1
        weights = np.random.rand(self.no_weights)
        weights /= np.sum(weights)
        
        return weights.tolist()