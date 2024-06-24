from typing import List
from mixture_optimization.datamodels.trial_tracking_config import ExperimentConfig, TrialType
from mixture_optimization.datamodels.weight_selector_config import WeightSelectorConfig
from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface

class DeterministicWeightSelector(WeightSelectorInterface):
    def __init__(self, config: WeightSelectorConfig, experiment_config: ExperimentConfig):
        super().__init__(config)
        assert experiment_config.experiment_idx == 0, "Deterministic selector does not support multiple experiments"
        assert config.no_optimizations == None, "Deterministic selector does not support optimizations"
        assert config.no_initializations == 1, "Deterministic selector requires exactly 1 initialization"
        self.weights = config.kwargs["weights"]

    def propose_next_weights(self):
        return self.weights, TrialType.INITIALIZATION

    def experiment_done(self):
        return self.no_initialization_completed() == 1
    
