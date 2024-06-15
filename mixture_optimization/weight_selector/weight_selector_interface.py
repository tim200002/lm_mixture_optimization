from typing import List

class WeightSelectorInterface:
    def __init__(self, config: dict):
        self.config = config
    
    def parse_history(self, run_history: List[dict]):
        pass

    def add_evaluation(self, perplexity, run_index):
        pass

    def propose_next_weights(self):
        raise NotImplementedError
    
    def get_best_weights(self):
        raise NotImplementedError

    def experiment_done(self):
        raise NotImplementedError