from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface

class DeterministicWeightSelector(WeightSelectorInterface):
    def __init__(self, config: dict):
        super().__init__(config)
        self.weights = config["weights"]

    def propose_next_weights(self):
        return self.weights