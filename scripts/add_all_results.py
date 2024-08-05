import sys
sys.path.append("/root/code/mixture_optimization")
sys.path.append("/root/code/mixture_optimization/mixture_optimization")

from typing import List
import cattrs
import numpy as np
import yaml
from mixture_optimization.datamodels.trial_tracking_config import Experiment


def read_experiments(experimnt_history):
    with open(experimnt_history, 'r') as f:
        obj = yaml.safe_load(f)
    
    experiments = cattrs.structure(obj, List[Experiment])
    return experiments

if __name__ == "__main__":
    path = "/root/code/mixture_optimization/logs/lin_interp_cc_stack_0/experiment_history.yaml"
    experiments = read_experiments(path)

    for experiment in experiments:
        for trial in experiment.trials:
            assert trial.all_results is None
            assert trial.val_results is not None

            trial.all_results = [trial.val_results]
    
    with open(path, 'w') as f:
        yaml.dump(cattrs.unstructure(experiments), f)
    
