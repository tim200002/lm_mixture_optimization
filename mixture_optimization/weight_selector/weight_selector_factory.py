from typing import List, Optional, Tuple
from mixture_optimization.datamodels.config import Config
from mixture_optimization.datamodels.trial_tracking_config import Experiment, ExperimentConfig
from mixture_optimization.datamodels.weight_selector_config import WeightSelectorConfig, WeightSelectorType
from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface


def _weight_selector_factory(config: WeightSelectorConfig) -> WeightSelectorInterface:
    type = config.type
    # if type == 'random':
    #     from mixture_optimization.weight_selector.random_weight_selector import RandomWeightSelector
    #     cls =  RandomWeightSelector
       
    if type == WeightSelectorType.DETERMINISTIC:
        from mixture_optimization.weight_selector.deterministic_weight_selector import DeterministicWeightSelector
        cls = DeterministicWeightSelector
    elif type == WeightSelectorType.BAYESIAN:
        from mixture_optimization.weight_selector.bayesian_selector import BayesianWeightSelector
        cls = BayesianWeightSelector
    elif type == WeightSelectorType.TURBO:
        from mixture_optimization.weight_selector.turbo_selector import TurboWeightSelector
        cls = TurboWeightSelector
    elif type == WeightSelectorType.SIMPLEX:
        from mixture_optimization.weight_selector.simplex_selector import RandomSimplexSelector
        cls = RandomSimplexSelector
    else:
        raise ValueError(f"Weight selector type {type} not recognized.")

    return cls

def weight_selector_from_scratch(weight_selector_config: WeightSelectorConfig, experiment_history: List[Experiment]) -> Tuple[WeightSelectorInterface, ExperimentConfig]:
    # get latest offset
    trial_offset = 0
    if len(experiment_history) > 0 and len(experiment_history[-1].trials) > 0:
        trial_offset = experiment_history[-1].trials[-1].idx + 1
    elif len(experiment_history) > 1:
        trial_offset = experiment_history[-2].trials[-1].idx + 1
    
    cls = _weight_selector_factory(weight_selector_config)
    return cls.from_scratch(weight_selector_config, trial_offset)

def weight_selector_from_history(weight_selecor_config: WeightSelectorConfig, experiment:Experiment) -> WeightSelectorInterface:
    cls = _weight_selector_factory(weight_selecor_config)
    return cls.from_history(weight_selecor_config.experiment_manager_config.weight_selector_config, experiment)
    