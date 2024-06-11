from mixture_optimization.utils.list_to_argv import list_to_argv
from mixture_optimization.utils.workspace_setup import setup_workspace
from mixture_optimization.weight_selector.weight_selector_factory import weight_selector_factory
from mixture_optimization.data_mixing.mix_tokenized_single_thread import mix_tokenized_data
from mixture_optimization.data_mixing.create_manifest import create_manifest

from open_lm.main import main as open_lm_main

import yaml
import os
import logging
import shutil
import sys



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
        self._save_config()

        domains = [domain for (domain, _) in self.config["train_data"]]
        manifests = [manifest for (_, manifest) in self.config["train_data"]]
        mixing_weights = run['weights']
        output_dir = data_dir
        token_count = self.config['open_lm_config']['complete_train_token_count']
        chunk_size = self.config['data_mixing']['chunk_size']
        no_workers = self.config['data_mixing']['no_workers']

        self.logger.info(f"Mixing dataset. Domains: {domains}, mixing weights: {mixing_weights}, output_dir: {output_dir}")
        mix_tokenized_data(domains, manifests, mixing_weights, output_dir, token_count, chunk_size, no_workers, mixing_log_path)
        self.logger.info(f"Dataset mixed. Now creating manifest")
        create_manifest(data_dir, num_workers=no_workers)
        self.logger.info(f"Manifest created.")

        run['status'] = "mixed"
        run['dataset'] = os.path.join(data_dir, "manifest.jsonl")
        self._save_config()

        return run



    def execute_run(self, run):
        assert run['status'] == "mixed" or run['status'] == "running", "Run must be mixed before executing."
        if run['status'] == "mixed":
            restore_run = False
            self.logger.info(f"Starting run {run['id']}. Validating that dataset exist.")
            self._validate_dataset_exist(run)
        else:
            restore_run = True
            self.logger.info(f"Restoring run {run['id']}. Validating that dataset and checkpoints exist.")
            self._validate_dataset_exist(run)
            self._validate_checkpoints_exist(run)
        
        # create relevant folders
        self.logger.info("Creating open-lm log directory.")
        open_lm_log_dir = os.path.join(run["workspace"], "open_lm")
        if os.path.exists(open_lm_log_dir) and not restore_run:
            self.logger.error(f"Open-lm log directory {open_lm_log_dir} already exists. However we are not restoring run. This should not happen. Exiting run.")
            exit(1)
        os.makedirs(open_lm_log_dir, exist_ok=True)
        run['open_lm_log_dir'] = open_lm_log_dir
        self._save_config()

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

        # ToDo parse results

        run['status'] = "parsed"
        self._save_config()

        return run
    
    def delete_dataset(self, run):
        assert run['status'] == "parsed", "Run must be parsed before deleting dataset."
        dataset_manifest = run['dataset']
        dataset_dir = os.path.dirname(dataset_manifest)
        self.logger.info(f"Would delete folder {dataset_dir}")
        # ToDo delete dataset

        return run



    def _validate_dataset_exist(self, run):
        dataset = run['dataset']
        if not os.path.exists(dataset):
            self.logger.error(f"Dataset {dataset} does not exist. Exiting run.")
            exit(1)
        self.logger.info(f"Dataset {dataset} exists.")
    
    def _validate_checkpoints_exist(self, run):
        open_lm_dir = run['open_lm_log_dir']
        checkpoints_dir = os.path.join(open_lm_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            self.logger.error(f"Checkpoints directory {checkpoints_dir} does not exist. Exiting run.")
            exit(1)
        checkpoints = os.listdir(checkpoints_dir)
        if not checkpoints:
            self.logger.error(f"No checkpoints found in {checkpoints_dir}. Exiting run.")
            exit(1)
        
        self.logger.info(f"Checkpoints to resume from found in {checkpoints_dir}.")
    
    def _create_open_lm_config(self, run, restore_run):
        params = []

        open_lm_config = self.config['open_lm_config']

        # configs to be played around
        params.append(("train-num-samples", open_lm_config['complete_train_token_count'] // open_lm_config["epochs"]))
        params.append(("workers", open_lm_config['workers']))
        params.append(("dataset-manifest", run['dataset']))
        params.append(("global-batch-size", open_lm_config['global_batch_size']))
        params.append(("log-every-n-steps", open_lm_config['log_every_n_steps']))
        params.append(("grad-clip-norm", open_lm_config['grad_clip_norm']))
        params.append(("lr", open_lm_config['lr']))
        params.append(("warmup", open_lm_config['warmup']))
        params.append(("wd", open_lm_config['wd']))
        params.append(("beta2", open_lm_config['beta2']))
        params.append(("epochs", open_lm_config['epochs']))
        params.append(("report-to", open_lm_config['report_to']))
        params.append(("data-key", open_lm_config['data_key']))
        params.append(("lr-cooldown-end", open_lm_config['lr_cooldown_end']))

        params.append(("logs", run["open_lm_log_dir"]))
        params.append(("name", run["id"]))

        # set configs
        # params.append("fsdp")
        # params.append("fsdp-limit-all-gathers")
        # params.append("fsdp-amp")

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
        if last_run_status == "finished":
            return None

        return (last_run, last_run_status)
    
    def execute_next_run(self):
        pipeline = [
            ("Initialize new run", self.initialize_new_run),
            ("Mix dataset", self.mix_dataset),
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
            elif last_run_status == "running":
                self.logger.info("Last run was running. Resuming last run")
                pipeline = pipeline[2:]
            elif last_run_status == "ran":
                self.logger.info("Last run was ran. Parsing results.")
                pipeline = pipeline[3:]
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




        
