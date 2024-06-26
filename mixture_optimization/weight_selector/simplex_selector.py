import logging
from mixture_optimization.datamodels.trial_tracking_config import Experiment, ExperimentConfig, TrialType
from mixture_optimization.datamodels.weight_selector_config import WeightSelectorConfig
from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface
from typing import Tuple

from botorch.utils.sampling import sample_simplex
logger = logging.getLogger("experiment_runner")

class RandomSimplexSelector(WeightSelectorInterface):

    @staticmethod
    def from_scratch(config: WeightSelectorConfig, exp_idx: int) -> Tuple[WeightSelectorInterface, ExperimentConfig]:
        samples = sample_simplex(config.no_weights, config.no_initializations, qmc=True).tolist()
        exp_config = ExperimentConfig(initialization_weights=samples, experiment_idx=exp_idx)
        return RandomSimplexSelector(config, exp_config), exp_config
    
    @staticmethod
    def from_history(config: WeightSelectorConfig, experiment: Experiment) -> WeightSelectorInterface:
        return RandomSimplexSelector(config, experiment.config)


    def __init__(self, config: WeightSelectorConfig, experiment_config: ExperimentConfig):
        super().__init__(config, experiment_config)
        assert config.no_optimizations == None, "Simplex selector does not support optimizations"

    
    def propose_next_weights(self):
       next_trial_idx = self.no_initialization_started()
       weights = self.experiment_config.initialization_weights[next_trial_idx]
       assert sum(weights) - 1 < 1e-4, f"Sum of weights should be 1, but is {sum(weights)}"
       return weights, TrialType.INITIALIZATION

    
    def experiment_done(self):
        # Override as no optimizations are supported
        return self.no_initialization_completed() == self.config.no_initializations