from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface


def weight_selector_factory(config) -> WeightSelectorInterface:
    type = config["weight_selector"]['type']
    if type == 'random':
        from mixture_optimization.weight_selector.random_weight_selector import RandomWeightSelector
        return RandomWeightSelector(config["weight_selector"], config["run_history"])
    elif type == 'deterministic':
        from mixture_optimization.weight_selector.deterministic_weight_selector import DeterministicWeightSelector
        return DeterministicWeightSelector(config["weight_selector"], config["run_history"])
    elif type == 'bayesian':
        from mixture_optimization.weight_selector.bayesian_selector import BayesianWeightSelector
        return BayesianWeightSelector(config["weight_selector"], config["run_history"])
    elif type == 'turbo':
        from mixture_optimization.weight_selector.turbo_selector import TurboWeightSelector
        return TurboWeightSelector(config["weight_selector"], config["run_history"])
    else:
        raise ValueError(f"Weight selector type {type} not recognized.")