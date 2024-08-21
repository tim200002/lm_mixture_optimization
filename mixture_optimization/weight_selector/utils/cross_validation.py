from copy import deepcopy
from typing import List

from mixture_optimization.datamodels.trial_tracking_config import Trial, Experiment, TrialType
from mixture_optimization.datamodels.weight_selector_config import WeightSelectorConfig
from mixture_optimization.weight_selector.bayesian_selector import BayesianWeightSelector
from mixture_optimization.weight_selector.weight_selector_factory import weight_selector_from_history


def cross_validate_leave_one_out(experiment: Experiment):
    experiment = deepcopy(experiment)
    experiment.trials = [trial for trial in experiment.trials if trial.weighted_val_perplexity is not None]

    true_perplexities = []
    pred_perplexities = []
    lower_bounds = []
    upper_bounds = []
    for i in range(len(experiment.trials)):
        print(f"Iteration {i}/{len(experiment.trials)}")
        train_idxs = list(range(len(experiment.trials)))
        train_idxs.remove(i)
        val_idx = i
        true_perplexity, pred_perplexity, (lower, upper) = cross_validate_run(experiment, train_idxs, val_idx)
        true_perplexities.append(true_perplexity)
        pred_perplexities.append(pred_perplexity)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        
    return true_perplexities, pred_perplexities, (lower_bounds, upper_bounds)

def cross_validate_over_time(experiment: Experiment):
    experiment = deepcopy(experiment)
    experiment.trials = [trial for trial in experiment.trials if trial.weighted_val_perplexity is not None]
    no_initialization_trials = len([trial for trial in experiment.trials if trial.type == TrialType.INITIALIZATION])

    true_perplexities = []
    pred_perplexities = []
    lower_bounds = []
    upper_bounds = []
    for i in range(no_initialization_trials, len(experiment.trials)):
        print(f"Iteration {i}/{len(experiment.trials)}")
        train_idxs = list(range(i))
        val_idx = i
        true_perplexity, pred_perplexity, (lower, upper) = cross_validate_run(experiment, train_idxs, val_idx)
        true_perplexities.append(true_perplexity)
        pred_perplexities.append(pred_perplexity)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        
    return true_perplexities, pred_perplexities, (lower_bounds, upper_bounds)

def cross_validate_run(experiment, train_idxs: List[int], val_idx: List[int]):
    train_trials = [trial for trial in experiment.trials if trial.idx in train_idxs]
    val_trial: Trial = experiment.trials[val_idx]
    
    val_pred, (lower, upper) = BayesianWeightSelector.predict(train_trials, val_value=list(val_trial.true_mixing_weights.values()))
    
    return val_trial.weighted_val_perplexity, val_pred, (lower, upper)