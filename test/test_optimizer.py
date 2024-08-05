import sys
from typing import List
sys.path.append(".")

import cattrs
import numpy as np
import yaml
import matplotlib.pyplot as plt

from mixture_optimization.datamodels.weight_selector_config import WeightSelectorConfig
from mixture_optimization.weight_selector.weight_selector_factory import weight_selector_from_scratch
import logging

logger = logging.getLogger("experiment_runner")


def func(r, B, b):
    eps = 1e-6
    return min(10, B / (r**b + eps))

best_parameters = {'Books': [4.06201743, 0.06136096], 'CC': [4.51788667, 0.06288045], 'stack-v4': [2.3069031 , 0.27325697]}

def test_function(weights: List):
    sum = 0
    for domain_weight, domain_parameters in zip(weights, best_parameters.values()):
        sum += func(domain_weight, *domain_parameters)
    sum /= len(weights)
    return sum


def load_config(path:str) -> WeightSelectorConfig:
    with open(path, 'r') as f:
        obj = yaml.safe_load(f)
        return cattrs.structure(obj["weight_selector_config"], WeightSelectorConfig)
    

def optimization_loop(config: WeightSelectorConfig):
    weight_selector, _ = weight_selector_from_scratch(config, 0)
    i = 0
    while not weight_selector.experiment_done():
        weights, type = weight_selector.propose_next_weights()
        value = test_function(weights)
        weight_selector.add_evaluation(value, i)
        logger.info(f"Completed optimization {i} with weights {weights} and value {value}")
        i += 1

    return weight_selector.trial_memory

if __name__ == "__main__":

    # create logger
    
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))

    config = load_config("test/test_config.yaml")
    trial_memory = optimization_loop(config)

    logger.info("Optimization completed. Visualizing results")
    values = [ - trial.value for trial in trial_memory if trial.value is not None]
    
    best_value = min(values)
    print(f"Best value: {best_value}")

    
