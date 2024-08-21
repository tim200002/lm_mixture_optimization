import sys, os
sys.path.append("/root/code/mixture_optimization")
sys.path.append("/root/code/mixture_optimization/mixture_optimization")


import cattrs
from typing import List
import yaml

from  mixture_optimization.datamodels.trial_tracking_config import Experiment, TrialType, ExperimentConfig


def read_experiments(experimnt_history):
    with open(experimnt_history, 'r') as f:
        obj = yaml.safe_load(f)
    
    experiments = cattrs.structure(obj, List[Experiment])
    return experiments

def save_experiments(experiments, path):
    with open(path, 'w') as f:
        yaml.dump(cattrs.unstructure(experiments), f)

if __name__ == "__main__":
    experiment_history_path = "/root/code/mixture_optimization/logs/bayesian_books_cc_stack_pes20_small_0/experiment_history_original.yaml"

    experiments = read_experiments(experiment_history_path)


    init_weight_sets = set()
    output_experiments = []
    for i, experiment in enumerate(experiments):
        init_weights = experiment.config.initialization_weights
        init_weights = tuple([tuple(x) for x in init_weights])


        if init_weights in init_weight_sets:
            print(f"Duplicate initialization weights found in experiment {i}. Skipping")
            continue
        
        init_weight_sets.add(init_weights)
        experiment.trials = [trial for trial in experiment.trials if trial.experiment_idx == i]
        output_experiments.append(experiment)

    save_path = "/root/code/mixture_optimization/logs/bayesian_books_cc_stack_pes20_small_0/experiment_history_filtered.yaml"
    save_experiments(output_experiments, save_path)
    print(f"Filtered experiments saved to {save_path}")