import math
from typing import Dict, List, Optional
import uuid
from mixture_optimization.datamodels.config import Config, LogConfig
from mixture_optimization.datamodels.trial_tracking_config import Experiment, Trial, TrialStatus, ValResult
from mixture_optimization.experiment_manager import ExperimentManager
from mixture_optimization.utils.list_to_argv import list_to_argv
from mixture_optimization.utils.memory_allocation import allocate_memory_on_gpu, deallocate_memory_on_gpu
from mixture_optimization.utils.wandb import setup_wandb
from mixture_optimization.utils.workspace_setup import get_experiment_dir, setup_logs
from mixture_optimization.data_mixing.mix_tokenized_single_thread_2 import mix_tokenized_data
from mixture_optimization.data_mixing.create_manifest import create_manifest
import torch
import datetime
import coolname

from open_lm.main import main as open_lm_main

import yaml
import os
import logging
import shutil
import sys
import json
import cattrs
import wandb

class ExperimentRunner:
    
    def __init__(self, config: Config, experiment_history: List[Experiment], log_config: LogConfig):
        self.config = config
        self.config.assert_validity()

        self.experiment_history = experiment_history
        self.log_config = log_config

        self.logger = logging.getLogger("experiment_runner")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(stream=sys.stdout))
        self.logger.addHandler(logging.FileHandler(self.log_config.log_path))
        
        # Initialize experiment manager
        self.experimemt_manager = ExperimentManager(config.experiment_manager_config)
        if len(self.experiment_history) == 0:
            exp_config = self.experimemt_manager.create_next_experiment()
            self.experiment_history.append(Experiment(exp_config))
        elif len(self.experiment_history) > 0:
            self.experimemt_manager.parse_history(self.experiment_history, config.domain_names)
        self._save_experiment_history()
        
    @classmethod
    def from_wandb():
        # Todo imprlement downloading previous state from wandb create folder to continue, ...
        raise NotImplementedError("Not implemented yet.")

    @classmethod
    def from_checkpoint(cls, experiment_dir: str):
        assert os.path.exists(experiment_dir), f"Experiment directory {experiment_dir} does not exist."
        config_path = os.path.join(experiment_dir, "config.yaml")
        assert os.path.exists(config_path), f"Config file {config_path} does not exist."
        experiment_history_path = os.path.join(experiment_dir, "experiment_history.yaml")
        assert os.path.exists(experiment_history_path), f"Experiment history file {experiment_history_path} does not exist."
        with open(config_path, "r") as config_path:
            obj = yaml.safe_load(config_path)
            config = cattrs.structure(obj, Config)

        # When no id create new wand experiment
        if config.id is None:
            config.id = str(uuid.uuid4())
            name = coolname.generate_slug(2)
            setup_wandb(config, experiment_dir, name)
        else:
            setup_wandb(config, experiment_dir)

        with open(experiment_history_path, "r") as experiment_history_file:
            obj = yaml.safe_load(experiment_history_file)
            experiment_history = cattrs.structure(obj, List[Experiment])
        
        log_config = setup_logs(experiment_dir, exist_ok=True)
        instance = cls(config, experiment_history, log_config)
        instance._save_config()

        return instance

    @classmethod
    def from_scratch(cls, config_path: str, logs_dir: str):
        assert os.path.exists(config_path), f"Config file {config_path} does not exist."
        with open(config_path, "r") as config_path:
            obj = yaml.safe_load(config_path)
            config = cattrs.structure(obj, Config)

        config.id = str(uuid.uuid4())

        # create workspace
        experiment_dir = get_experiment_dir(logs_dir, config.name)
        # Setup wandb
        name = coolname.generate_slug(2)
        setup_wandb(config, experiment_dir, name)
        
        log_config = setup_logs(experiment_dir)
        instance = cls(config, [], log_config)
        instance._save_config()
        return instance
    
    def check_and_create_new_experiment(self):
        if self.experimemt_manager.check_experiment_done():
            self.logger.info("Experiment done. Creating new experiment.")
            exp_config = self.experimemt_manager.create_next_experiment()
            self.experiment_history.append(Experiment(exp_config))
            self._save_experiment_history()

    
    def initialize_new_trial(self, trial: Optional[Trial]) -> Trial:
        """
        Initialize new run by getting the next weights
        """
        assert not trial, "Run must be None before initializing new run. Parameter necessary for pipeline."
        
        # trial_idx = sum([len(experiment.trials) for experiment in self.config.experiment_history])
        experiment_idx = self.experimemt_manager.get_experiment_idx()
        trial_idx = self.experimemt_manager.get_next_trial_idx()
        other_run_currenlty_running = self.get_latest_run_and_status() is not None
        assert not other_run_currenlty_running, "Another run is currently running. Exiting."
        
        weights, trial_type = self.experimemt_manager.propose_next_weights()
        self.logger.info(f"Next run proposed weights: {weights}")
        weights_dict = {}
        for i, domain_name in enumerate(self.config.domain_names):
            weights_dict[domain_name] = weights[i]
        trial_name =  f"exp_{experiment_idx}_trial_{trial_idx}"
       
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        data_dir = os.path.join(self.config.data_workspace, timestamp)

        new_trial = Trial(
            idx=trial_idx,
            experiment_idx=experiment_idx,
            name=trial_name,
            status=TrialStatus.INITIALIZED,
            weights=weights_dict,
            type=trial_type,
            data_dir=data_dir
        )

        workspace = new_trial.get_workspace(self.log_config)
        if os.path.exists(workspace):
            self.logger.error(f"Initializing new trial, however, workspace {workspace} already exists. Exiting.")
            exit(1)
        os.makedirs(workspace)

        self.experiment_history[-1].trials.append(new_trial)
        self._save_experiment_history()

        return new_trial

    def mix_dataset(self, trial: Trial):
        assert trial.status == TrialStatus.INITIALIZED, "Run must be initialized before mixing dataset."
        if os.path.exists(trial.data_dir):
            self.logger.warning(f"Data directory {trial.data_dir} already exists. Deleting previous data to create new mixture.")
            shutil.rmtree(trial.data_dir)
        os.makedirs(trial.data_dir)

        # setup special logger for mixing
        mixing_log_path = os.path.join(trial.get_workspace(self.log_config), "mixing.log")
        trial.mixing_log_path = mixing_log_path

        domains = self.config.domain_names 
        manifests = [self.config.train_data[dom] for dom in self.config.domain_names]
        mixing_weights = [trial.weights[dom] for dom in self.config.domain_names]
        output_dir = trial.data_dir
        token_count = self.config.open_lm_config.complete_train_token_count
        chunk_size = self.config.data_mixing_config.chunk_size
        shard_size = self.config.data_mixing_config.shard_size
        oversample_factor = self.config.data_mixing_config.oversample_factor
        shard_selection_multiplier = self.config.data_mixing_config.shard_selection_multiplier
        mixing_seed = self.config.data_mixing_config.seed

        self.logger.info(f"Mixing dataset. Domains: {domains}, mixing weights: {mixing_weights}, output_dir: {output_dir}")
        true_mixing_weights = mix_tokenized_data(domains, manifests, mixing_weights, output_dir=output_dir, output_token_count=token_count, chunk_size=chunk_size, shard_size=shard_size, oversample_factor=oversample_factor, log_path=mixing_log_path, shard_selection_multiplier=shard_selection_multiplier, seed=mixing_seed)
        self.logger.info(f"Dataset mixed. True mixing weights: {true_mixing_weights}")

        # Format to dict
        true_mixing_weights_dict = {}
        for i, domain_name in enumerate(self.config.domain_names):
            true_mixing_weights_dict[domain_name] = true_mixing_weights[i]

        trial.true_mixing_weights = true_mixing_weights_dict
        trial.status = TrialStatus.MIXED
        self._save_experiment_history()

        # Add logs to wandb
        log_obj = trial.get_mixing_weights_wandb()
        experiment = self.experiment_history[trial.experiment_idx]
        mixing_table = experiment.generate_mixing_table()
        log_obj.update({
            "mixing_table": mixing_table
        })
        wandb.log({
            f"exp_{trial.experiment_idx}": log_obj
        })
        
        return trial
    
    def create_manifest_for_dataset(self, trial: Trial):
        assert trial.status == TrialStatus.MIXED, "Run must be mixed before creating manifest."
        data_dir = trial.data_dir
        assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist. Exiting."

        num_workers = self.config.data_mixing_config.no_workers
        self.logger.info(f"Creating manifest for dataset in {data_dir}")
        create_manifest(data_dir, num_workers=num_workers)

        trial.status = TrialStatus.MANIFEST_CREATED
        trial.dataset = os.path.join(data_dir, "manifest.jsonl")
        self._save_experiment_history()

        return trial

    def execute_run(self, trial: Trial):
        assert trial.status in [TrialStatus.MANIFEST_CREATED, TrialStatus.RUNNING], "Run must be mixed before executing."
        
        dataset_exists = os.path.exists(trial.dataset)
        open_lm_dir_exist = os.path.exists(trial.open_lm_log_dir) if trial.open_lm_log_dir else False
        if open_lm_dir_exist:
            checkpoints_dir = os.path.join(trial.open_lm_log_dir, "checkpoints")
            checkpoints_exist = os.path.exists(checkpoints_dir)
            checkpoints_exist = len(os.listdir(checkpoints_dir)) > 0 if checkpoints_exist else False
        else:
            checkpoints_exist = False

        def create_directories():
            self.logger.info("Creating open-lm log directory.")
            open_lm_log_dir = os.path.join(trial.get_workspace(self.log_config), "open_lm")
            if os.path.exists(open_lm_log_dir):
                self.logger.error(f"Open-lm log directory {open_lm_log_dir} already exists. Exiting run.")
                exit(1)
            os.makedirs(open_lm_log_dir)
            trial.open_lm_log_dir = open_lm_log_dir

        if trial.status == TrialStatus.MANIFEST_CREATED and dataset_exists and not checkpoints_exist:
            restore_run = False
            self.logger.info("Startung run from scarch after manifest creation")
            create_directories()
        elif trial.status == TrialStatus.RUNNING and dataset_exists and open_lm_dir_exist and not checkpoints_exist:
            restore_run = False
            self.logger.info("Starting run from scarch after run was interrupted before checkpointing. Must delete old log directory.")
            shutil.rmtree(trial.open_lm_log_dir)
            create_directories()
        elif trial.status == TrialStatus.RUNNING and dataset_exists and open_lm_dir_exist and checkpoints_exist:
            restore_run = True
            self.logger.info("Resuming run from checkpoints.")
        else:
            self.logger.error(f"Run {trial['name']} is in invalid state. Exiting.")
            self.logger.error(f"Dataset exists: {dataset_exists}, open_lm_dir exists: {open_lm_dir_exist}, checkpoints exist: {checkpoints_exist}")
            exit(1)
        
        trial.status = TrialStatus.RUNNING
        self._save_experiment_history()

        # Assemble config for open-lm
        self.logger.info("Assembling open-lm config")
        open_lm_config = self._create_open_lm_config(trial, restore_run)

        # Run open-lm
        self.logger.info("Starting open-lm")
        open_lm_main(open_lm_config)

        trial.status = TrialStatus.RAN
        self._save_experiment_history()

        return trial

    def parse_results(self, trial: Trial):
        assert trial.status == TrialStatus.RAN, "Run must be ran before parsing results."
        val_file_path = os.path.join(trial.open_lm_log_dir, "checkpoints", "results.jsonl")
        assert os.path.exists(val_file_path), f"Validation file {val_file_path} does not exist. Exiting."
        trial.val_file_path = val_file_path

        all_results:List[Dict[str, ValResult]]  = []
        with open(val_file_path, "r") as val_file:
            for line in val_file:
                epoch_val_results = json.loads(line)
                per_domain_results = {}
                for i, domain_name in enumerate(self.config.val_data.keys()):
                    domain_result = epoch_val_results[i]
                    loss = domain_result["loss"]
                    perplexity = loss
                    eval_tokens = domain_result["tokens"]
                    train_tokens_seen = domain_result["train_tokens"]
                    loss_tokens_lower_95 = domain_result["loss_tokens_lower_95"]
                    loss_tokens_upper_95 = domain_result["loss_tokens_upper_95"]

                    val_results = ValResult(
                        domain=domain_name,
                        loss=loss,
                        perplexity=perplexity,
                        eval_tokens=eval_tokens,
                        train_tokens_seen=train_tokens_seen,
                        loss_tokens_lower_95=loss_tokens_lower_95,
                        loss_tokens_upper_95=loss_tokens_upper_95
                    )
                    
                    per_domain_results[domain_name] = val_results
                all_results.append(per_domain_results)
        
        trial.all_results = all_results
        trial.val_results = all_results[-1]
        
        # calc weighted perplexity on last run
        weighted_perplexity = 0
        val_weights_normalized = [weight / sum(self.config.val_weights) for weight in self.config.val_weights]
        assert sum(val_weights_normalized) - 1 < 1e-6, "Weights must sum to 1."
        for i, domain_name in enumerate(self.config.val_data.keys()):
            weight = val_weights_normalized[i]
            perplexity = trial.val_results[domain_name].perplexity
            weighted_perplexity += weight * perplexity
        
        trial.weighted_val_perplexity = weighted_perplexity
        self.experimemt_manager.add_evaluation(weighted_perplexity, trial.idx)
        trial.status = TrialStatus.PARSED
        

        # Add logs to wandb
        experiment = self.experiment_history[trial.experiment_idx]
        perplexity_table = experiment.generate_perplexity_table()
        log_obj = trial.get_perplexity_wandb()
        log_obj.update({
            "perplexity_table": perplexity_table
        })
        wandb.log({
            f"exp_{trial.experiment_idx}" :log_obj
        })

        self._save_experiment_history()

        if self.config.delete_run_after_run:
            trial_workspace = trial.get_workspace(self.log_config)
            self.logger.info(f"Deleting run workspace {trial_workspace}")
            shutil.rmtree(trial_workspace)

        return trial
    
    def delete_dataset(self, trial: Trial):
        assert trial.status == TrialStatus.PARSED, "Run must be parsed before deleting dataset."
        dataset_manifest = trial.dataset
        dataset_dir = os.path.dirname(dataset_manifest)
        self.logger.info(f"Deleting dataset {dataset_dir} to save space")
        shutil.rmtree(dataset_dir)
        trial.status = TrialStatus.DELETED
        trial.dataset = None
        self._save_experiment_history()
        return trial

    
    def _create_open_lm_config(self, trial: Trial, restore_run):
        params = []
        open_lm_config = self.config.open_lm_config
        per_episode_token_count = open_lm_config.complete_train_token_count // open_lm_config.epochs

        # that logging structure is proper we must do a trick
        open_lm_log_dir = trial.open_lm_log_dir # i.e. /root/code/mixture_optimization/logs/Test_3/runs/run_0/open_lm
        log_dir_base_path = os.path.dirname(open_lm_log_dir)  # i.e. /root/code/mixture_optimization/logs/Test_3/runs/run_0
        name = os.path.basename(open_lm_log_dir) # i.e. open_lm
    
        # Add general experiment parameters
        params.append(("model", open_lm_config.model))
        params.append(("train-num-samples", per_episode_token_count))
        params.append(("dataset-manifest", trial.dataset))
        val_data_paths = [path for  path in self.config.val_data.values()]
        val_data_keys = [open_lm_config.data_key for _ in val_data_paths]
        params.append(("val-data", val_data_paths))
        params.append(("val-data-key", val_data_keys))
        params.append(("logs-dir", log_dir_base_path))
        params.append(("name", name))
        if open_lm_config.seed:
            params.append(("seed", open_lm_config.seed))
        params.append(("data-key", open_lm_config.data_key))
        params.append(("report-to", open_lm_config.report_to))
        params.append(("workers", open_lm_config.workers))
        params.append(("epochs", open_lm_config.epochs))
        params.append(("log-every-n-steps", open_lm_config.log_every_n_steps))

        # Add training specific parameters
        params.append(("global-batch-size", open_lm_config.global_batch_size))
        params.append(("grad-clip-norm", open_lm_config.grad_clip_norm))
        params.append(("lr", open_lm_config.lr))
        params.append(("warmup", open_lm_config.warmup))
        params.append(("wd", open_lm_config.wd))
        params.append(("beta2", open_lm_config.beta2))
        params.append(("lr-cooldown-end", open_lm_config.lr_cooldown_end))
        params.append(("z-loss-coefficient", open_lm_config.z_loss_coefficient))
        params.append(("accum-freq", open_lm_config.accum_freq))
        params.append(("model-norm", open_lm_config.model_norm))
        
        # Add validation specific parameters
        params.append(("val-frequency", open_lm_config.val_frequency))
        params.append(("global-val-batch-size", open_lm_config.global_val_batch_size))
        params.append(("val-num-samples", open_lm_config.val_num_samples))

        # Add set true parameters
        if open_lm_config.qk_norm:
            params.append("qk-norm")
        if open_lm_config.fsdp:
            params.append("fsdp")
        if open_lm_config.fsdp_amp:
            params.append("fsdp-amp")
        if open_lm_config.grad_checkpointing:
            params.append("grad-checkpointing")

        if restore_run:
            params.append(("resume", "latest"))

        return list_to_argv(params)


    def _save_config(self):
        obj = cattrs.unstructure(self.config)
        with open(self.log_config.config_path, "w") as config_file:
            yaml.safe_dump(obj, config_file)
    
    def _save_experiment_history(self):
        obj = cattrs.unstructure(self.experiment_history)
        with open(self.log_config.experiment_history_path, "w") as experiment_history_file:
            yaml.safe_dump(obj, experiment_history_file)
        wandb.save(self.log_config.experiment_history_path)

    
    def get_latest_run_and_status(self):
        latest_trials = self.experiment_history[-1].trials
        if len(latest_trials) == 0:
            return None
        last_trial = latest_trials[-1]
        end_stage = TrialStatus.PARSED if not self.config.delete_dataset_after_run else TrialStatus.DELETED
        if last_trial.status.value >= end_stage.value:
            return None
        
        return (last_trial, last_trial.status)

    def execute_next_trial(self):
        pipeline = [
            ("Initialize new trial", self.initialize_new_trial),
            ("Mix dataset", self.mix_dataset),
            ("Create manifest", self.create_manifest_for_dataset),
            ("Execute trial", self.execute_run),
            ("Parse results", self.parse_results)
        ]

        if self.config.delete_dataset_after_run:
            pipeline.append(("Delete dataset", self.delete_dataset))

        res = self.get_latest_run_and_status()
        if not res:
            self.logger.info("No run to exectute. Initializing new run.")
            pipeline = pipeline
            last_trial = None
        else:
            last_trial, last_trial_status = res
            if last_trial_status == TrialStatus.INITIALIZED:
                self.logger.info("Last run was initialized. Initializing pipeline form dataset mixing.")
                pipeline = pipeline[1:]
            elif last_trial_status == TrialStatus.MIXED:
                self.logger.info("Last run was mixed. Initializing pipeline form execution.")
                pipeline = pipeline[2:]
            elif last_trial_status == TrialStatus.MANIFEST_CREATED:
                self.logger.info("Last run was mixed. Initializing pipeline form execution.")
                pipeline = pipeline[3:]
            elif last_trial_status == TrialStatus.RUNNING:
                self.logger.info("Last run was running. Resuming last run")
                pipeline = pipeline[3:]
            elif last_trial_status == TrialStatus.RAN:
                self.logger.info("Last run was ran. Parsing results.")
                pipeline = pipeline[4:]
            elif last_trial_status == TrialStatus.PARSED:
                self.logger.info("Last run was parsed. Deciding if dataset should be deleted.")
                if self.config.delete_dataset_after_run:
                    self.logger.info("Decided to delete dataset. Adding stage to pipeline.")
                    pipeline = pipeline[5:]
                else:
                    self.logger.info("Decided not to delete dataset. No further steps and returning.")
                    return
            elif last_trial_status == TrialStatus.DELETED:
                self.logger.error("Last run was deleted. We should not be here. Exiting.")
                exit(1)
            else:
                self.logger.error(f"Last run status {last_trial_status} not recognized. Exiting.")
                exit(1)

        pipeline_stages = [stage for (stage, _) in pipeline]
        self.logger.info(f"Executing pipeline: {pipeline_stages}")

        trial = last_trial
        for stage, func in pipeline:
            self.logger.info(f"Executing stage: {stage}")
            trial = func(trial)
        
        return trial

    def is_done(self):
        return self.experimemt_manager.check_all_completed()

    def get_best_weights_and_perplexity(self):
        best_perplexity = float("inf")
        for experiment in self.experiment_history:
            exp_best_weight, exp_best_perplexity = experiment.get_best_weights_and_perplexity()
            if exp_best_perplexity < best_perplexity:
                best_perplexity = exp_best_perplexity
                best_weights = exp_best_weight
        
        return best_weights, best_perplexity
    


if __name__ == "__main__":
    experiment_dir = "logs/Test_33"
    runner = ExperimentRunner.from_checkpoint(experiment_dir)

    run_0 = runner.config['run_history'][0]
    print(run_0)

    open_lm_config = runner._create_open_lm_config(run_0, restore_run=False)
    print(" ".join(open_lm_config))
