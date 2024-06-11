import os
from typing import Optional
from mixture_optimization.experiment_runner import ExperimentRunner

def main(config_path: Optional[str] = None, experiment_dir: Optional[str] = None):
    assert config_path or experiment_dir, "Either config_path or experiment_dir must be provided."
    assert not (config_path and experiment_dir), "Only one of config_path or experiment_dir must be provided."

    if config_path:
        assert os.path.exists(config_path), f"Config file {config_path} does not exist."
        experiment_runner = ExperimentRunner.from_scratch(config_path) 
    else:
        assert os.path.exists(experiment_dir), f"Experiment directory {experiment_dir} does not exist."
        config_path = os.path.join(experiment_dir, "config.yaml")
        assert os.path.exists(config_path), f"Config file {config_path} does not exist."
        experiment_runner = ExperimentRunner.from_checkpoint(experiment_dir)
    

    experiment_runner.execute_next_run() 


if __name__ == "__main__":
    config_path = "/root/code/mixture_optimization/config/config.yaml"
    main(config_path=config_path)
    # experiment_dir = "/root/code/mixture_optimization/logs/Test_5"
    # main(experiment_dir=experiment_dir)