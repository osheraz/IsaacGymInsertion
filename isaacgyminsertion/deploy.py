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
from algo.deploy.deploy import HardwarePlayer
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Enter here the model you want to test (base folder)
model_to_test = 'gt2'


def find_config_folder(base_folder):
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if re.match(r'config_\w+\.yaml', file):
                folder_name = os.path.basename(root)
                match = re.match(r'config_(\w+)\.yaml', file)
                if match:
                    config_id = match.group(1)
                    return folder_name, config_id

    return None, None


cfg_name = find_config_folder(f"./outputs/{model_to_test}/")


@hydra.main(config_name=f"config_{cfg_name[1]}", config_path=f"./outputs/{model_to_test}/{cfg_name[0]}")
def main(config: DictConfig):

    config.checkpoint = f'outputs/{model_to_test}/stage1_nn/last.pth'
    if config.checkpoint:
        config.checkpoint = to_absolute_path(config.checkpoint)

    set_np_formatting()
    config.seed = set_seed(config.seed)

    # TODO change output dir to  teacher folder
    output_dif = os.path.join(model_to_test + str(datetime.now().strftime("%m-%d-%y")))
    # output_dif = os.path.join(output_dif, str(datetime.now().strftime("%H-%M-%S")))

    os.makedirs(output_dif, exist_ok=True)
    agent = HardwarePlayer(output_dif, config)
    agent.restore(config.checkpoint)
    agent.deploy()


if __name__ == '__main__':
    main()
