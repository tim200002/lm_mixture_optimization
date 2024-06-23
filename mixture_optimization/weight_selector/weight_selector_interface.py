from typing import List

class WeightSelectorInterface:
    def __init__(self, config: dict, run_history = None):
        self.config = config
    
    def parse_history(self, run_history: List[dict]):
        raise NotImplementedError

    def add_evaluation(self, perplexity, run_index):
        raise NotImplementedError

    def propose_next_weights(self):
        raise NotImplementedError
    
    def get_best_weights(self):
        raise NotImplementedError

    def experiment_done(self):
        raise NotImplementedError