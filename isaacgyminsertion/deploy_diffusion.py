# --------------------------------------------------------
# ?
# https://arxiv.org/abs/todo
# Copyright (c) 2022 Osher & Co
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------

import os
from datetime import datetime
# noinspection PyUnresolvedReferences
import isaacgym
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

import re
import argparse
from typing import Optional
from termcolor import cprint
from isaacgyminsertion.utils.utils import set_np_formatting, set_seed
from algo.models.diffusion.deploy import HardwarePlayer
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

cfg_path = '/home/roblab20/tactile_diffusion/datastore_real/ckpts/diff/'


@hydra.main(config_name="task_config", config_path=f"{cfg_path}")
def main(config: DictConfig):

    set_np_formatting()
    config.seed = set_seed(config.seed)
    config.diffusion_train.load_path = cfg_path
    # TODO change output dir to  teacher folder

    agent = HardwarePlayer(config)
    agent.deploy()


if __name__ == '__main__':
    main()