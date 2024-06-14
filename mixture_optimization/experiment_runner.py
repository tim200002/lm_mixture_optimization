import math
from mixture_optimization.utils.list_to_argv import list_to_argv
from mixture_optimization.utils.workspace_setup import setup_workspace
from mixture_optimization.weight_selector.weight_selector_factory import weight_selector_factory
from mixture_optimization.data_mixing.mix_tokenized_single_thread_2 import mix_tokenized_data
from mixture_optimization.data_mixing.create_manifest import create_manifest

from open_lm.main import main as open_lm_main

import yaml
import os
import logging
import shutil
import sys
import json



class ExperimentRunner:
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("experiment_runner")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(stream=sys.stdout))
        self.logger.addHandler(logging.FileHandler(config['experiment_tracking']['log_path']))

        self.weight_selector = weight_selector_factory(config['weight_selector'])
        

    @classmethod
    def from_checkpoint(cls, experiment_dir: str):
        assert os.path.exists(experiment_dir), f"Experiment directory {experiment_dir} does not exist."
        config_path = os.path.join(experiment_dir, "config.yaml")
        assert os.path.exists(config_path), f"Config file {config_path} does not exist."
        with open(config_path, "r") as config_path:
            config = yaml.safe_load(config_path)
        
        instance = cls(config)
        return instance

    @classmethod
    def from_scratch(cls, config_path: str):
        assert os.path.exists(config_path), f"Config file {config_path} does not exist."
        with open(config_path, "r") as config_path:
            config = yaml.safe_load(config_path)

        # create workspace
        config = setup_workspace(config)
        instance = cls(config)
        instance._save_config()

        return instance

    
    def initialize_new_run(self, run):
        """
        Initialize new run by getting the next weights
        """
        assert not run, "Run must be None before initializing new run. Parameter necessary for pipeline."
        
        run_history = self.config['run_history']
        if run_history:
            last_run = run_history[-1]
            assert last_run["status"] == "finished", "Last run must be finished before initializing new run."
        
        weights = self.weight_selector.propose_next_weights(run_history)
        experiment_name =  f"run_{len(run_history)}"
        workspace = os.path.join(self.config['experiment_tracking']['runs_folder'], experiment_name)
        new_run = {
            "id": experiment_name,
            "status": "initialized",
            "weights": weights,
            "workspace": workspace
        }

        if os.path.exists(workspace):
            self.logger.error(f"Initializing new run, however, workspace {workspace} already exists. Exiting.")
            exit(1)
        os.makedirs(workspace)

        run_history.append(new_run)
        self._save_config()

        return new_run

    def mix_dataset(self, run):
        assert run['status'] == "initialized", "Run must be initialized before mixing dataset."
        data_dir = os.path.join(self.config['data_workspace'], run['id'])
        
        if os.path.exists(data_dir):
            self.logger.warning(f"Data directory {data_dir} already exists. Deleting previous data to create new mixture.")
            shutil.rmtree(data_dir)
        os.makedirs(data_dir)

        # setup special logger for mixing
        mixing_log_path = os.path.join(run['workspace'], "mixing.log")
        run['mixing_log_path'] = mixing_log_path

        domains = [domain for (domain, _) in self.config["train_data"]]
        manifests = [manifest for (_, manifest) in self.config["train_data"]]
        mixing_weights = run['weights']
        output_dir = data_dir
        token_count = self.config['open_lm_config']['complete_train_token_count']
        chunk_size = self.config['data_mixing']['chunk_size']
        shard_size = self.config['data_mixing']['shard_size']
        oversample_factor = self.config['data_mixing']['oversample_factor']

        self.logger.info(f"Mixing dataset. Domains: {domains}, mixing weights: {mixing_weights}, output_dir: {output_dir}")
        true_mixing_weights = mix_tokenized_data(domains, manifests, mixing_weights, output_dir=output_dir, output_token_count=token_count, chunk_size=chunk_size, shard_size=shard_size, oversample_factor=oversample_factor, log_path=mixing_log_path)
        self.logger.info(f"Dataset mixed. True mixing weights: {true_mixing_weights}")

        run["true_mixing_weights"] = true_mixing_weights
        run['status'] = "mixed"
        self._save_config()

        return run
    
    def create_manifest_for_dataset(self, run):
        assert run['status'] == "mixed", "Run must be mixed before creating manifest."
        data_dir = os.path.join(self.config['data_workspace'], run['id'])
        assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist. Exiting."

        num_workers = self.config['data_mixing']['no_workers']
        self.logger.info(f"Creating manifest for dataset in {data_dir}")
        create_manifest(data_dir, num_workers=num_workers)

        run['status'] = "manifest_created"
        run['dataset'] = os.path.join(data_dir, "manifest.jsonl")
        self._save_config()

        return run



    def execute_run(self, run):
        assert run['status'] == "manifest_created" or run['status'] == "running", "Run must be mixed before executing."
        
        dataset_exists = os.path.exists(run['dataset'])
        open_lm_dir = run.get('open_lm_log_dir', None)
        open_lm_dir_exist = os.path.exists(open_lm_dir) if open_lm_dir else False
        if open_lm_dir_exist:
            checkpoints_dir = os.path.join(open_lm_dir, "checkpoints")
            checkpoints_exist = os.path.exists(checkpoints_dir)
            checkpoints_exist = len(os.listdir(checkpoints_dir)) > 0 if checkpoints_exist else False
        else:
            checkpoints_exist = False

        def create_directories():
            self.logger.info("Creating open-lm log directory.")
            open_lm_log_dir = os.path.join(run["workspace"], "open_lm")
            if os.path.exists(open_lm_log_dir):
                self.logger.error(f"Open-lm log directory {open_lm_log_dir} already exists. Exiting run.")
                exit(1)
            os.makedirs(open_lm_log_dir)
            run['open_lm_log_dir'] = open_lm_log_dir

        if run["status"] == "manifest_created" and dataset_exists and not checkpoints_exist:
            restore_run = False
            self.logger.info("Startung run from scarch after manifest creation")
            create_directories()
        elif run["status"] == "running" and dataset_exists and open_lm_dir_exist and not checkpoints_exist:
            restore_run = False
            self.logger.info("Starting run from scarch after run was interrupted before checkpointing. Must delete old log directory.")
            shutil.rmtree(open_lm_dir)
            create_directories()
        elif run["status"] == "running" and dataset_exists and open_lm_dir_exist and checkpoints_exist:
            restore_run = True
            self.logger.info("Resuming run from checkpoints.")
        else:
            self.logger.error(f"Run {run['id']} is in invalid state. Exiting.")
            self.logger.error(f"Dataset exists: {dataset_exists}, open_lm_dir exists: {open_lm_dir_exist}, checkpoints exist: {checkpoints_exist}")
            exit(1)
        
        run['status'] = "running"
        self._save_config()

        # Assemble config for open-lm
        self.logger.info("Assembling open-lm config")
        open_lm_config = self._create_open_lm_config(run, restore_run)

        # Run open-lm
        self.logger.info("Starting open-lm")
        open_lm_main(open_lm_config)

        run['status'] = "ran"
        self._save_config()

        return run

    def parse_results(self, run):
        assert run['status'] == "ran", "Run must be ran before parsing results."
        open_lm_log_dir = run['open_lm_log_dir']
        val_file_path = os.path.join(open_lm_log_dir, "checkpoints", "results.jsonl")
        val_file_path = "logs/Test_3/runs/run_0/open_lm/run_0/checkpoints/results.jsonl"
        assert os.path.exists(val_file_path), f"Validation file {val_file_path} does not exist. Exiting."
        run["val_file_path"] = val_file_path

        all_results = []
        with open(val_file_path, "r") as val_file:
            for line in val_file:
                epoch_val_results = json.loads(line)
                per_domain_results = {}
                for i, (domain_name, _) in enumerate(self.config['val_data']):
                    domain_result = epoch_val_results[i]
                    loss = domain_result["loss"]
                    perplexity = math.exp(loss)
                    eval_tokens = domain_result["tokens"]
                    train_tokens_seen = domain_result["train_tokens"]
                    loss_tokens_lower_95 = domain_result["loss_tokens_lower_95"]
                    loss_tokens_upper_95 = domain_result["loss_tokens_upper_95"]

                    obj = {
                        "domain": domain_name,
                        "loss": loss,
                        "perplexity": perplexity,
                        "eval_tokens": eval_tokens,
                        "train_tokens_seen": train_tokens_seen,
                        "loss_tokens_lower_95": loss_tokens_lower_95,
                        "loss_tokens_upper_95": loss_tokens_upper_95
                    }
                    per_domain_results[domain_name] = obj
                all_results.append(per_domain_results)
        
        run["val_results"] = all_results
        run['status'] = "parsed"
        self._save_config()

        return run
    
    def delete_dataset(self, run):
        assert run['status'] == "parsed", "Run must be parsed before deleting dataset."
        dataset_manifest = run['dataset']
        dataset_dir = os.path.dirname(dataset_manifest)
        self.logger.info(f"Deleting dataset {dataset_dir} to save space")
        shutil.rmtree(dataset_dir)
        run["status"] = "deleted"
        run["dataset"] = None
        self._save_config()
        return run

    
    def _create_open_lm_config(self, run, restore_run):
        params = []
        open_lm_config = self.config['open_lm_config']
        per_episode_token_count = open_lm_config['complete_train_token_count'] // open_lm_config['epochs']

        # that logging structure is proper we must do a trick
        open_lm_log_dir = run['open_lm_log_dir'] # i.e. /root/code/mixture_optimization/logs/Test_3/runs/run_0/open_lm
        log_dir_base_path = os.path.dirname(open_lm_log_dir)  # i.e. /root/code/mixture_optimization/logs/Test_3/runs/run_0
        name = os.path.basename(open_lm_log_dir) # i.e. open_lm
    
        # Add general experiment parameters
        params.append(("model", open_lm_config['model']))
        params.append(("train-num-samples", per_episode_token_count))
        params.append(("dataset-manifest", run['dataset']))
        val_data_paths = [path for (_, path) in self.config['val_data']]
        val_data_keys = [open_lm_config['data_key'] for _ in self.config['val_data']]
        params.append(("val-data", val_data_paths))
        params.append(("val-data-key", val_data_keys))
        params.append(("logs-dir", log_dir_base_path))
        params.append(("name", name))
        if open_lm_config.get("seed", None):
            params.append(("seed", open_lm_config['seed']))
        params.append(("data-key", open_lm_config['data_key']))
        params.append(("report-to", open_lm_config['report_to']))
        params.append(("workers", open_lm_config['workers']))
        params.append(("epochs", open_lm_config['epochs']))
        params.append(("log-every-n-steps", open_lm_config['log_every_n_steps']))

        # Add training specific parameters
        params.append(("global-batch-size", open_lm_config['global_batch_size']))
        params.append(("grad-clip-norm", open_lm_config['grad_clip_norm']))
        params.append(("lr", open_lm_config['lr']))
        params.append(("warmup", open_lm_config['warmup']))
        params.append(("wd", open_lm_config['wd']))
        params.append(("beta2", open_lm_config['beta2']))
        params.append(("lr-cooldown-end", open_lm_config['lr_cooldown_end']))
        params.append(("z-loss-coefficient", open_lm_config['z_loss_coefficient']))
        params.append(("accum-freq", open_lm_config['accum_freq']))
        params.append(("model-norm", open_lm_config['model_norm']))
        
        # Add validation specific parameters
        params.append(("val-frequency", open_lm_config['val_frequency']))
        params.append(("global-val-batch-size", open_lm_config['global_val_batch_size']))
        params.append(("val-num-samples", open_lm_config['val_num_samples']))

        # Add set true parameters
        if open_lm_config["qk_norm"]:
            params.append("qk-norm")
        if open_lm_config["fsdp"]:
            params.append("fsdp")
        if open_lm_config["fsdp_amp"]:
            params.append("fsdp-amp")
        if open_lm_config["grad_checkpointing"]:
            params.append("grad-checkpointing")

        if restore_run:
            params.append(("resume", "latest"))

        return list_to_argv(params)


    def _save_config(self):
        with open(self.config["experiment_tracking"]["config_path"], "w") as config_file:
            yaml.safe_dump(self.config, config_file)

    
    def get_latest_run_and_status(self):
        run_history = self.config['run_history']
        if not run_history:
            return None
        last_run = run_history[-1]
        last_run_status = last_run['status']
        run_should_be_deleted = self.config['delete_dataset_after_run']
        if last_run_status == "deleted":
            return None

        if not run_should_be_deleted and last_run_status == "parsed":
            return None

        return (last_run, last_run_status)
    
    def execute_next_run(self):
        pipeline = [
            ("Initialize new run", self.initialize_new_run),
            ("Mix dataset", self.mix_dataset),
            ("Create manifest", self.create_manifest_for_dataset),
            ("Execute run", self.execute_run),
            ("Parse results", self.parse_results)
        ]

        if self.config['delete_dataset_after_run']:
            pipeline.append(("Delete dataset", self.delete_dataset))

        res = self.get_latest_run_and_status()
        if not res:
            self.logger.info("No run to exectute. Initializing new run.")
            pipeline = pipeline
            last_run = None
        else:
            last_run, last_run_status = res
            if last_run_status == "initialized":
                self.logger.info("Last run was initialized. Initializing pipeline form dataset mixing.")
                pipeline = pipeline[1:]
            elif last_run_status == "mixed":
                self.logger.info("Last run was mixed. Initializing pipeline form execution.")
                pipeline = pipeline[2:]
            elif last_run_status == "manifest_created":
                self.logger.info("Last run was mixed. Initializing pipeline form execution.")
                pipeline = pipeline[3:]
            elif last_run_status == "running":
                self.logger.info("Last run was running. Resuming last run")
                pipeline = pipeline[3:]
            elif last_run_status == "ran":
                self.logger.info("Last run was ran. Parsing results.")
                pipeline = pipeline[4:]
            elif last_run_status == "parsed":
                self.logger.info("Last run was parsed. Deciding if dataset should be deleted.")
                if self.config['delete_dataset_after_run']:
                    self.logger.info("Decided to delete dataset. Deleting dataset.")
                    pipeline = pipeline[5:]
                else:
                    self.logger.info("Decided not to delete dataset. No further steps and returning.")
                    return
            elif last_run_status == "deleted":
                self.logger.error("Last run was deleted. We should not be here. Exiting.")
                exit(1)
            else:
                self.logger.error(f"Last run status {last_run_status} not recognized. Exiting.")
                exit(1)

        pipeline_stages = [stage for (stage, _) in pipeline]
        self.logger.info(f"Executing pipeline: {pipeline_stages}")

        run = last_run
        for stage, func in pipeline:
            self.logger.info(f"Executing stage: {stage}")
            run = func(run)
        
        return run

    def is_done(self):
        max_runs = self.config['max_no_runs']
        no_runs = len(self.config['run_history'])
        assert no_runs <= max_runs, f"Number of runs {no_runs} exceeds maximum number of runs {max_runs}. Exiting."
        if no_runs < max_runs:
            return False
        
        last_run_state = self.get_latest_run_and_status()
        if last_run_state != None: # this indicates that the last run did not yet finish but was aborted
            return False
        
        return True
    





if __name__ == "__main__":
    experiment_dir = "logs/Test_33"
    runner = ExperimentRunner.from_checkpoint(experiment_dir)

    run_0 = runner.config['run_history'][0]
    print(run_0)

    open_lm_config = runner._create_open_lm_config(run_0, restore_run=False)
    print(" ".join(open_lm_config))
