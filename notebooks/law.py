import sys
sys.path.append("/root/code/mixture_optimization")
sys.path.append("/root/code/mixture_optimization/mixture_optimization")

from collections import defaultdict
from typing import List
import cattrs
from scipy.optimize import curve_fit, minimize
import numpy as np
import yaml
from mixture_optimization.datamodels.trial_tracking_config import Experiment


def read_experiments(experimnt_history):
    with open(experimnt_history, 'r') as f:
        obj = yaml.safe_load(f)
    
    experiments = cattrs.structure(obj, List[Experiment])
    return experiments

def parse_results(experiment: Experiment, eval_idx):
    results = defaultdict(list)

    tokens_seen = None
    for trial in experiment.trials:
        if trial.val_results is None:
            continue
        
        for key, res in trial.all_results[eval_idx].items():
            w = trial.true_mixing_weights[key]
            results[key].append({
                "weight": w,
                "perplexity": res.perplexity
            })
            if not tokens_seen:
                tokens_seen = res.train_tokens_seen

    
    # sort by weight
    # for domain, result in results.items():
    #     results[domain] = sorted(result, key=lambda x: x["weight"])
    
    
    return {
        "results": results,
        "tokens_seen": tokens_seen
    }

def fit(experiment: Experiment, function, eval_idx = -1, region=(0.1, 0.9)):
    results = parse_results(experiment, eval_idx)

    best_parameters = {}
    domain_results_filtered = {}
    for domain, domain_result in results["results"].items():
        best_parameters[domain], domain_results_filtered[domain] = fit_res(domain_result, function, region)
    
    return best_parameters, results
    

def fit_res(domain_result, function, region):
    domain_result_filtered = [x for x in domain_result if region[0] <= x["weight"] <= region[1]]

    weights = np.array([x["weight"] for x in domain_result_filtered])
    perplexities = np.array([x["perplexity"] for x in domain_result_filtered])
    
    popt, pcov = curve_fit(function, weights, perplexities, maxfev=5000)
    return popt, domain_result_filtered

def optimize(function, parameters):

    def weighted_sum_func(domain_weights: List[float]):
        sum = 0
        for domain_weight, domain_parameters in zip(domain_weights, parameters.values()):
            sum += function(domain_weight, *domain_parameters)
        sum /= len(domain_weights)
        return sum

    # optimize weights s.t. sum weights == 1
    res = minimize(weighted_sum_func, [1/len(parameters)] * len(parameters), bounds=[(0, 1)] * len(parameters), constraints={"type": "eq", "fun": lambda x: np.sum(x) - 1})

    # return best weights as well as predicted value
    return res.x, res.fun



        


if __name__ == "__main__":
    path = "/root/code/mixture_optimization/logs/uniform_books_cc_stack_0/experiment_history.yaml"
    experiments = read_experiments(path)
    experiment = experiments[0]

    def func(r,  B, b):
        return  B / (r**b)
    
    best_parameters = fit(experiment, func)
    print(best_parameters)