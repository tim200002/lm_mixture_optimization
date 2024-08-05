import logging
from mixture_optimization.datamodels.trial_tracking_config import Experiment, ExperimentConfig, TrialType
from mixture_optimization.datamodels.weight_selector_config import WeightSelectorConfig
from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface
from typing import Tuple

logger = logging.getLogger("experiment_runner")

class RandomSimplexSelector(WeightSelectorInterface):

    @staticmethod
    def from_scratch(config: WeightSelectorConfig, exp_idx: int) -> Tuple[WeightSelectorInterface, ExperimentConfig]:
        samples = WeightSelectorInterface._sample_uniform(config.no_initializations, config.no_weights, config.bounds).tolist()
        exp_config = ExperimentConfig(initialization_weights=samples, experiment_idx=exp_idx)
        return RandomSimplexSelector(config, exp_config), exp_config
    
    @staticmethod
    def from_history(config: WeightSelectorConfig, experiment: Experiment) -> WeightSelectorInterface:
        return RandomSimplexSelector(config, experiment.config)


    def __init__(self, config: WeightSelectorConfig, experiment_config: ExperimentConfig):
        super().__init__(config, experiment_config)
        assert config.no_optimizations is None, "Simplex selector does not support optimizations"
    
    def experiment_done(self):
        # Override as no optimizations are supported
        return self.no_initialization_completed() == self.config.no_initializations