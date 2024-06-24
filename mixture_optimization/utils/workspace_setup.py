import os
import re

from mixture_optimization.datamodels.config import Config, ExperimentTrackingConfig

def create_unique_folder(base_path, experiment_name):
    os.makedirs(base_path, exist_ok=True)
    highest_idx = -1
    for folder_name in os.listdir(base_path):
        match = re.match(rf"^({experiment_name})_(\d+)$", folder_name)
        if match:
            idx = int(match.group(2))
            if idx > highest_idx:
                highest_idx = idx

    new_idx = highest_idx + 1
    new_folder_name = f"{experiment_name}_{new_idx}"
    workspace_folder = os.path.join(base_path, new_folder_name)
    os.makedirs(workspace_folder)
    return workspace_folder

def setup_workspace(config: Config):
    experiment_name = config.name
    workspace = config.workspace
    data_workspace = config.data_workspace

    # create new folder for this experiment based on name and workspace
    workspace_folder = create_unique_folder(workspace, experiment_name)
    config.workspace = workspace_folder
    data_workspace_folder = create_unique_folder(data_workspace, experiment_name)
    config.data_workspace = data_workspace_folder

    # set important paths and folders for experiment tracking
    log_path = os.path.join(workspace_folder, "logs.log")
    config_path = os.path.join(workspace_folder, "config.yaml")
    runs_folder = os.path.join(workspace_folder, "runs")
    os.makedirs(runs_folder)
    config.experiment_tracking_config = ExperimentTrackingConfig(log_path=log_path, config_path=config_path, runs_folder=runs_folder)

    return config