import wandb
import cattrs

def setup_wandb(config, experiment_dir, name=None):
    config_dict = cattrs.unstructure(config)
    val_weight_tag = "_".join([str(weight) for weight in config.val_weights])
    wandb.init(
        project="mixture_optimization",
        id=config.id,
        config=config_dict,
        dir=experiment_dir,
        save_code=False,
        tags=[config.dataset_tag.value, val_weight_tag, config.name],
        notes=config.description,
        resume="allow",
        name=name,
    )