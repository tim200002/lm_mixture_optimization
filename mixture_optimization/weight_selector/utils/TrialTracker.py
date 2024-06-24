from typing import List, Optional
from mixture_optimization.datamodels.trial_tracking_config import Experiment
from mixture_optimization.datamodels.weight_selector_config import WeightSelectorConfig


class Runner:
    def __init__(self, config: WeightSelectorConfig, experiment_history: Optional[List[Experiment]] = None):
        self.config = config
        assert self.no_weights > 1, "Optimization requires at least 2 weights"
        self.no_experiments = 0
        self.no_trials = 0

    
    def create_new_experiment(self):
        self.weight_selector = BayesianWeightSelector(self.config)