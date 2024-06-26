import os
from typing import Optional
from mixture_optimization.experiment_runner import ExperimentRunner
from argparse import ArgumentParser
import wandb

def main(config_path: Optional[str] = None, experiment_dir: Optional[str] = None):
    assert config_path or experiment_dir, "Either config_path or experiment_dir must be provided."
    assert not (config_path and experiment_dir), "Only one of config_path or experiment_dir must be provided."

    if config_path:
        assert os.path.exists(config_path), f"Config file {config_path} does not exist."
        logs_dir = "./logs" # hardcoded for now
        experiment_runner = ExperimentRunner.from_scratch(config_path, logs_dir=logs_dir) 
    else:
        assert os.path.exists(experiment_dir), f"Experiment directory {experiment_dir} does not exist."
        config_path = os.path.join(experiment_dir, "config.yaml")
        assert os.path.exists(config_path), f"Config file {config_path} does not exist."
        experiment_runner = ExperimentRunner.from_checkpoint(experiment_dir)
    
    while not experiment_runner.is_done():
        experiment_runner.logger.info("Running next Iteration")  
        experiment_runner.check_and_create_new_experiment()
        experiment_runner.execute_next_trial() 
    
    experiment_runner.logger.info("All experiments are done.")
    best_weights, best_perplexity = experiment_runner.get_best_weights()
    experiment_runner.logger.info(f"Best perplexity of {best_perplexity} achieved with weights: {best_weights}")
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment-dir", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    args = parser.parse_args()

    main(experiment_dir=args.experiment_dir, config_path=args.config_path)