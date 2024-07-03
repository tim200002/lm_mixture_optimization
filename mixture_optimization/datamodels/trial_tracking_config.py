import os
from attrs import define, frozen
from enum import Enum, auto, unique
from typing import Dict, Optional, List
import wandb



@unique
class TrialStatus(Enum):
    INITIALIZED = auto()
    MIXED = auto()
    MANIFEST_CREATED = auto()
    RUNNING = auto()
    RAN = auto()
    PARSED = auto()
    DELETED = auto()

@unique
class TrialType(Enum):
    INITIALIZATION = auto()
    OPTIMIZATION = auto()


@frozen
class ValResult:
    domain: str
    loss: float
    perplexity: float
    eval_tokens: int
    train_tokens_seen: int
    loss_tokens_lower_95: float
    loss_tokens_upper_95: float


@define
class Trial:
    idx: int
    experiment_idx: int
    name: str
    status: TrialStatus
    type: TrialType
    weights: Dict[str, float]
    data_dir: str
    kwargs: dict = {}

    # initializes later
    mixing_log_path: Optional[str] = None
    dataset: Optional[str] = None
    true_mixing_weights: Optional[Dict[str,float]] = None
    open_lm_log_dir : Optional[str] = None
    val_file_path: Optional[str] = None
    val_results: Optional[Dict[str, ValResult]] = None
    weighted_val_perplexity: Optional[float] = None

    def get_workspace(self, log_config): # No type hints due to circular import
        workspace = os.path.join(log_config.runs_folder, f"exp_{self.experiment_idx}_trial_{self.idx}")
        return workspace

    def get_perplexity_wandb(self):
        assert self.status.value >= TrialStatus.PARSED.value, "Trial has not been parsed yet"
        domain_performance = {domain: res.perplexity for domain, res in self.val_results.items()}
        return {
            "domain_perplexities": domain_performance,
            "weighted_val_perplexity": self.weighted_val_perplexity,
        }

    def get_mixing_weights_wandb(self):
        assert self.status.value >= TrialStatus.MIXED.value, "Trial has not been mixed yet"
        return {
            "mixing_weights": self.true_mixing_weights
        }

@frozen
class ExperimentConfig:
    initialization_weights: List[List[float]] 
    experiment_idx: int

@define
class Experiment:
    config: ExperimentConfig
    trials: List[Trial] = []

    def generate_mixing_table(self):
        if len(self.trials) == 0:
            return None
        domain_names = []
        domain_weights = []
        for trial in self.trials:
            if trial.status.value < TrialStatus.MIXED.value:
                continue
            domain_names.append(list(trial.weights.keys()))
            domain_weights.append(list(trial.weights.values()))

        # assert everyone is using same domains, i.e. equailty along columns
        for row, next_row in zip(domain_names[:-1], domain_names[1:]):
            assert row == next_row, "Domains are not same across trials"
        
        table = wandb.Table(data=domain_weights, columns=domain_names[0])
        return table

    def generate_perplexity_table(self):
        if len(self.trials) == 0:
            return None
        domain_names = []
        perplexities = []
        for trial in self.trials:
            if trial.status.value < TrialStatus.PARSED.value:
                continue
            domain_names.append(list(trial.val_results.keys()) + ["weighted"])
            perplexities.append([domain_result.perplexity for domain_result in trial.val_results.values()] + [trial.weighted_val_perplexity])

        # assert everyone is using same domains, i.e. equailty along columns
        for row, next_row in zip(domain_names[:-1], domain_names[1:]):
            assert row == next_row, "Domains are not same across trials"

        table = wandb.Table(data=perplexities, columns=domain_names[0])
        return table
    
    def get_best_weights_and_perplexity(self):
        best_trial = None
        best_perplexity = float("inf")
        for trial in self.trials:
            if trial.status.value < TrialStatus.PARSED.value:
                continue
            if trial.weighted_val_perplexity < best_perplexity:
                best_trial = trial
                best_perplexity = trial.weighted_val_perplexity
        return best_trial.weights, best_perplexity