# --------------------------------------------------------
# ?
# https://arxiv.org/abs/todo
# Copyright (c) 2024 Osher & Co
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------

import os
# noinspection PyUnresolvedReferences
import isaacgym
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

import re
from isaacgyminsertion.utils.utils import set_np_formatting, set_seed
from algo.deploy.deploy_pos_pred import HardwarePlayer

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(config_name="config", config_path="./cfg")
def main(config: DictConfig):
    test = False
    set_np_formatting()
    config.seed = set_seed(config.seed)

    agent = HardwarePlayer(config)

    if test:
        config.offline_train.model.transformer.tact_path = to_absolute_path(f'outputs/gt_test/tact/checkpoints'
                                                                            f'/model_last.pt')
        config.offline_train.train.normalize_file = to_absolute_path(f'outputs/gt_test/tact/normalization.pkl')
        agent.restore(config.checkpoint)  # restore policy

    agent.deploy()


if __name__ == '__main__':
    main()
