from typing import List, Optional, Tuple
from mixture_optimization.datamodels.trial_tracking_config import Experiment, TrialType
from mixture_optimization.datamodels.weight_selector_config import ExperimentManagerConfig
from mixture_optimization.weight_selector.weight_selector_factory import weight_selector_from_scratch, weight_selector_from_history
from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface


class ExperimentManager:
    def __init__(self, config: ExperimentManagerConfig):
        self.config = config
        self.no_experiments = 0
        self.no_trials = 0
        self.weight_selector:Optional[WeightSelectorInterface] = None

    def check_experiment_done(self) -> bool:
        return self.weight_selector.experiment_done()
    
    def check_all_completed(self) -> bool:
        assert self.no_experiments <= self.config.no_experiments, "Too many experiments started"
        if self.no_experiments != self.config.no_experiments:
            return False
        
        # in case we are at last experiment, check if all trials are done
        return self.check_experiment_done()
    
    def create_next_experiment(self) -> ExperimentManagerConfig:
        self.no_experiments += 1
        weight_selector, expconfig = weight_selector_from_scratch(self.config.weight_selector_config, self.no_experiments - 1) # 0-indexed
        self.weight_selector = weight_selector
        return expconfig
    
    def parse_history(self, experiment_history: List[Experiment]):
        assert len(experiment_history) > 0, "No experiments to parse"
        self.no_experiments = len(experiment_history)
        last_experiment = experiment_history[-1]
        self.weight_selector = weight_selector_from_history(self.config.weight_selector_config, last_experiment)

        for trial in last_experiment.trials:
            self.weight_selector.attach_trial(trial.weights, trial.type)
            if trial.weighted_val_perplexity:
                self.weight_selector.add_evaluation(trial.weighted_val_perplexity, trial.idx)


    def get_best_weights(self) -> Tuple[List[float], float]:
        return self.weight_selector.get_best_weights()
    
    def add_evaluation(self, perplexity: float, trial_index: int):
        return self.weight_selector.add_evaluation(perplexity, trial_index)
    
    def propose_next_weights(self) -> Tuple[List[float], TrialType]:
        return self.weight_selector.propose_next_weights()
    
    def get_experiment_idx(self) -> int:
        assert self.no_experiments > 0, "No experiments started"
        return self.no_experiments - 1

    def get_next_trial_idx(self) -> int:
        assert self.no_experiments > 0, "No experiments started"
        assert self.weight_selector, "No weight selector initialized"
        return self.weight_selector.get_next_trial_idx()
    
    

        