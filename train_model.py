import json
import logging
import os
import sys
from collections import OrderedDict

import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf
import wandb

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


@hydra.main(config_path="config", config_name="train_model_config")
def main(cfg: DictConfig):
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("train_model.py main() running.")
    log.info(OmegaConf.to_yaml(cfg))
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    
    cfg.problem.model.test_iterations = list(range(cfg.problem.model.test_iterations["low"],
                                                   cfg.problem.model.test_iterations["high"] + 1))
    assert 0 <= cfg.problem.hyp.alpha <= 1, "Weighting for loss (alpha) not in [0, 1], exiting."
    
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    
    net, start_epoch, optimizer_state_dict = dt.utils.load_model_from_checkpoint(cfg.problem.name,
                                                                                 cfg.problem.model,
                                                                                 device)
    
    param_count = np.sum(x.size for x in jax.tree_leaves(net))
    log.info(f"This {cfg.problem.model.model} has {param_count/1e6:0.3f} million parameters.")
    log.info(f"Training will start at epoch {start_epoch}.")