import sys, os
sys.path.append("/root/code/mixture_optimization")
sys.path.append("/root/code/mixture_optimization/mixture_optimization")

from typing import List
import cattrs
import numpy as np
import yaml
from mixture_optimization.datamodels.trial_tracking_config import Experiment, TrialType, ExperimentConfig
from mixture_optimization.datamodels.config import Config
from mixture_optimization.utils.workspace_setup import create_unique_folder

def read_experiments(experimnt_history):
    with open(experimnt_history, 'r') as f:
        obj = yaml.safe_load(f)
    
    experiments = cattrs.structure(obj, List[Experiment])
    return experiments

def read_config(config_path):
    with open(config_path, 'r') as f:
        obj = yaml.safe_load(f)
    
    config = cattrs.structure(obj, Config)
    return config

if __name__ == "__main__":
    config_path = "config/config_bayesian_cc_books_stack_pes2o.yaml"
    init_run_path = "/root/code/mixture_optimization/logs/uniform_books_cc_stack_pes2o/experiment_history.yaml"
    experiments = read_experiments(init_run_path)
    experiment = experiments[0]
    config = read_config(config_path)
    no_initializations_to_sample = config.experiment_manager_config.weight_selector_config.no_initializations

    init_trials = [] 
    for trial in experiment.trials:
        if trial.val_results is None:
            print(f"Trial {trial.idx} has did not finish, aborting")
            break

        if trial.type == TrialType.INITIALIZATION:
            init_trials.append(trial)
    
    no_init_to_sample_from = len(init_trials)
    assert no_init_to_sample_from >= no_initializations_to_sample
    
    print(f"Sampling {no_initializations_to_sample} initializations from in total {no_init_to_sample_from}")
    sampled_init_trials = np.random.choice(init_trials, no_initializations_to_sample, replace=False).tolist()
    for i, trial in enumerate(sampled_init_trials):
        trial.idx = i
    
    new_experiment_config = ExperimentConfig(
        initialization_weights=[trial.weights.values() for trial in sampled_init_trials],
        experiment_idx=0
    )
    new_experiment = Experiment(config=new_experiment_config, trials=sampled_init_trials)


    config = read_config(config_path)
    
    # create folder
    logs_dir = "/root/code/mixture_optimization/logs"
    experiment_dir = create_unique_folder(logs_dir, config.name)

    print(f"Writing new setup to {experiment_dir}")

    # save config
    config_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(cattrs.unstructure(config), f)
    
    # save experiment
    experiment_history_path = os.path.join(experiment_dir, "experiment_history.yaml")
    with open(experiment_history_path, 'w') as f:
        yaml.dump(cattrs.unstructure([new_experiment]), f)