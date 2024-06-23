import json
import math
import os
import yaml

def parse_results(run):
    open_lm_log_dir = run['open_lm_log_dir']
    val_file_path = os.path.join(open_lm_log_dir, "checkpoints", "results.jsonl")
    assert os.path.exists(val_file_path), f"Validation file {val_file_path} does not exist. Exiting."
    run["val_file_path"] = val_file_path

    all_results = []
    with open(val_file_path, "r") as val_file:
        for line in val_file:
            epoch_val_results = json.loads(line)
            per_domain_results = {}
            for i, (domain_name, _) in enumerate(config['val_data']):
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
    
    # calc weighted perplexity on last run
    val_weights = config['val_weights']
    weighted_perplexity = 0
    for i, (domain_name, _) in enumerate(config['val_data']):
        weight = val_weights[i]
        perplexity = all_results[-1][domain_name]["perplexity"]
        weighted_perplexity += weight * perplexity
    run["weighted_val_perplexity"] = weighted_perplexity

    run['status'] = "parsed"


if __name__ == "__main__":
    config_path = "/root/code/mixture_optimization/logs/bayesian_first_try_0/config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    runs = config['run_history']
    runs = runs[:-1]
    for run in runs:
        parse_results(run)

    save_path = "/root/code/mixture_optimization/logs/bayesian_first_try_0/config.yaml"
    with open(save_path, "w") as f:
        yaml.dump(config, f)