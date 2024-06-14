from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface


def weight_selector_factory(config) -> WeightSelectorInterface:
    type = config['type']
    if type == 'random':
        from mixture_optimization.weight_selector.random_weight_selector import RandomWeightSelector
        return RandomWeightSelector(config)
    elif type == 'deterministic':
        from mixture_optimization.weight_selector.deterministic_weight_selector import DeterministicWeightSelector
        return DeterministicWeightSelector(config)
    else:
        raise ValueError(f"Weight selector type {type} not recognized.")