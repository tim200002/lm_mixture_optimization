from argparse import ArgumentParser
import yaml
import json
import os

def run_completed(run):
    return run["status"] in ["parsed", "deleted"]

def create_index_keys(config, run):
    open_lm_config_keys = ["model", "true_mixing_weights", "accum_freq", "beta2", "complete_train_token_count", "global_batch_size", "grad_clip_norm", "lr", "lr_cooldown_end", "model_norm", "precision", "qk_norm", "warmup", "wd", "z_loss_coefficient"]
    run_keys = ["true_mixing_weights"]

    key_dict = {}
    for key, value in config["open_lm_config"].items():
        if key in open_lm_config_keys:
            key_dict[key] = value
    
    for key, value in run.items():
        if key in run_keys:
            key_dict[key] = value

    return key_dict

def check_for_existing_entry(key, db):
    for entry in db:
        if entry["index_keys"] == key:
            return True
    return False

def add_entry_to_db(entry, db):
    db.append(entry)
    return db

def get_performance_metrics(run):
    val_results = run["val_results"][-1]
    out = {}
    for domain, res in val_results.items():
        obj = {
            "domain": domain,
            "perplexity": res["perplexity"],
            "loss": res["loss"],
        }
        out[domain] = obj
    return out

def get_entry_by_key(key, db):
    for entry in db:
        if entry["index_keys"] == key:
            return entry
    return None

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--db-base", type=str, default="/media/ssd1/tim/database", help="Database name")
    parser.add_argument("--config-path", type=str, help="Path to config file")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_tag = config["dataset_tag"]
    db_path = os.path.join(args.db_base, f"{dataset_tag}.json")
    if not os.path.exists(db_path):
        print(f"Database {db_path} does not exist. Creating new one.")
        with open(db_path, "w") as f:
            json.dump([], f)
    else:
        print(f"Database {db_path} exists. Loading.")
        with open(db_path, "r") as f:
            db = json.load(f)
        assert isinstance(db, list), "Database is not a list"
    
    for i, run in enumerate(config["run_history"]):
        run_name = run["name"]
        print(f"Processing run {run_name}")
        if not run_completed(run):
            print("Run is not completed. Skipping.")
            continue

        index_keys = create_index_keys(config, run)
        entry_already_exists = check_for_existing_entry(index_keys, db)
        if entry_already_exists:
            print("Entry already exists in database.")
            print(f"The keys are {index_keys}")
            entry = get_entry_by_key(index_keys, db)
            print(f"The entry is {entry}")
            print("Please manually remove entry and try again.")
        
        performance_metrics = get_performance_metrics(run)
        entry = {
            "index_keys": index_keys,
            "val_results": performance_metrics,
        }
        db = add_entry_to_db(entry, db)

    with open(db_path, "w") as f:
        json.dump(db, f, indent=4)
