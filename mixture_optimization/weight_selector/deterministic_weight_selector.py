from typing import Tuple
from mixture_optimization.datamodels.trial_tracking_config import Experiment, ExperimentConfig, TrialType
from mixture_optimization.datamodels.weight_selector_config import WeightSelectorConfig
from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface

class DeterministicWeightSelector(WeightSelectorInterface):
    @staticmethod
    def from_scratch(config: WeightSelectorConfig, exp_idx: int) -> Tuple[WeightSelectorInterface, ExperimentConfig]:
        initialization_weights = config.kwargs["weights"]
        print(initialization_weights)
        exp_config = ExperimentConfig(initialization_weights=[initialization_weights], experiment_idx=exp_idx)
        return DeterministicWeightSelector(config, exp_config), exp_config

    @staticmethod
    def from_history(config: WeightSelectorConfig, experiment: Experiment) -> WeightSelectorInterface:
        return DeterministicWeightSelector(config, experiment.config)

    def __init__(self, config: WeightSelectorConfig, experiment_config: ExperimentConfig):
        super().__init__(config, experiment_config)
        assert experiment_config.experiment_idx == 0, "Deterministic selector does not support multiple experiments"
        assert config.no_optimizations is None, "Deterministic selector does not support optimizations"
        assert config.no_initializations == 1, "Deterministic selector requires exactly 1 initialization"
        self.weights = config.kwargs["weights"]

    def propose_next_weights(self):
        return self.weights, TrialType.INITIALIZATION

    def experiment_done(self):
        return self.no_initialization_completed() == 1
    
