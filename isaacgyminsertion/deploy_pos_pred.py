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
    set_np_formatting()
    config.seed = set_seed(config.seed)

    agent = HardwarePlayer(config)
    test = False

    if test:
        # model_path = 'outputs/gt_test/tact/'

        model_path = '/home/roblab20/osher3_workspace/src/isaacgym/python/IsaacGymInsertion/isaacgyminsertion/outputs/depth'
        config.offline_train.model.transformer.load_tact = True
        config.offline_train.model.transformer.tact_path = to_absolute_path(model_path + '/checkpoints/model_last.pt')
        config.offline_train.train.load_stats = True
        config.offline_train.train.normalize_file = to_absolute_path(model_path + '/normalization.pkl')
        agent.restore(config.checkpoint)  # restore policy

    agent.deploy(test=test)


if __name__ == '__main__':
    main()
