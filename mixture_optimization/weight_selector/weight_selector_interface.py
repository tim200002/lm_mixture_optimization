from typing import List

class WeightSelectorInterface:
    def __init__(self, config: dict):
        self.config = config

        self.no_weights = config['no_weights']
    
    def propose_next_weights(self, run_history: List[dict]):
        raise NotImplementedError