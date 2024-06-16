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
from algo.models.transformer.deploy import HardwarePlayer

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


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


# Enter here the model you want to test (base folder)
teacher = 'gt_test'
student = 'student'
cfg_name = find_config_folder(f"./outputs/{teacher}/")

@hydra.main(config_name=f"config_{cfg_name[1]}", config_path=f"./outputs/{teacher}/{cfg_name[0]}")
def main(config: DictConfig):

    config.checkpoint = f'outputs/{teacher}/stage1_nn/last.pth'
    config.offline_train.train.student_ckpt_path = f'outputs/{teacher}/{student}/checkpoints/model_last.pt'
    config.offline_train.train.normalize_file = f'outputs/{teacher}/{student}/normalization.pkl'

    if config.checkpoint:
        config.checkpoint = to_absolute_path(config.checkpoint)
    if config.offline_train.train.student_ckpt_path:
        config.offline_train.train.student_ckpt_path = to_absolute_path(config.offline_train.train.student_ckpt_path)
    if config.offline_train.train.normalize_file:
        config.offline_train.train.normalize_file = to_absolute_path(config.offline_train.train.normalize_file)

    set_np_formatting()
    config.seed = set_seed(config.seed)

    agent = HardwarePlayer(config)
    agent.restore(config.checkpoint)  # restore policy

    agent.deploy()


if __name__ == '__main__':
    main()
