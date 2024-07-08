import logging
from mixture_optimization.datamodels.trial_tracking_config import Experiment, ExperimentConfig, TrialType
from mixture_optimization.datamodels.weight_selector_config import WeightSelectorConfig
from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface
from typing import Tuple
import numpy as np

logger = logging.getLogger("experiment_runner")

class LinInterpSelector(WeightSelectorInterface):

    @staticmethod
    def from_scratch(config: WeightSelectorConfig, exp_idx: int) -> Tuple[WeightSelectorInterface, ExperimentConfig]:
        assert config.no_optimizations is None, "Lin interp selector does not support optimizations"
        assert config.no_weights == 2, "Lin interp selector only supports 2 weights"

        steps = np.linspace(0, 1, config.no_initializations)
        samples = [[float(step), float(1-step)] for step in steps]

        exp_config = ExperimentConfig(initialization_weights=samples, experiment_idx=exp_idx)
        return LinInterpSelector(config, exp_config), exp_config
    
    @staticmethod
    def from_history(config: WeightSelectorConfig, experiment: Experiment) -> WeightSelectorInterface:
        return LinInterpSelector(config, experiment.config)


    def __init__(self, config: WeightSelectorConfig, experiment_config: ExperimentConfig):
        super().__init__(config, experiment_config)
        assert config.no_optimizations is None, "Lin interp selector does not support optimizations"
        assert config.no_weights == 2, "Lin interp selector only supports 2 weights"

    
    def experiment_done(self):
        # Override as no optimizations are supported
        return self.no_initialization_completed() == self.config.no_initializations