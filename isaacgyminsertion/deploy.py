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


import argparse
from typing import Optional
from termcolor import cprint
from isaacgyminsertion.utils.utils import set_np_formatting, set_seed
from algo.deploy.deploy import HardwarePlayer
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

@hydra.main(config_name="config", config_path="./cfg")
def main(config: DictConfig):

    config.checkpoint = 'outputs/215_stage2_nn/stage2_nn/last.pth'
    if config.checkpoint:
        config.checkpoint = to_absolute_path(config.checkpoint)

    set_np_formatting()
    config.seed = set_seed(config.seed)
    output_dif = os.path.join('outputs', str(datetime.now().strftime("%m-%d-%y")))
    output_dif = os.path.join(output_dif, str(datetime.now().strftime("%H-%M-%S")))
    os.makedirs(output_dif, exist_ok=True)
    agent = HardwarePlayer(output_dif, config)
    agent.restore(config.checkpoint)
    agent.deploy()

if __name__ == '__main__':
    main()