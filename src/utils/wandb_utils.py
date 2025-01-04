# src/utils/wandb_utils.py

import wandb

def init_wandb(config):
    project = config.get("wandb_project", "default_project")
    run_name = config.get("wandb_run_name", "unnamed_run")
    entity = config.get("wandb_entity", None)
    run = wandb.init(
        project=project,
        name=run_name,
        entity=entity,
        config=config
    )
    return run

def finish_wandb():
    wandb.finish()
