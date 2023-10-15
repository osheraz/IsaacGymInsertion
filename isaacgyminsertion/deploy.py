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

import hydra
import os
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from isaacgyminsertion.utils.misc import set_np_formatting, set_seed
# noinspection PyUnresolvedReferences
from algo.deploy.deploy import HardwarePlayer


# ---- OmegaConf & Hydra Config
# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config.
# used primarily for num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)


@hydra.main(config_name="config", config_path="./cfg")
def main(config: DictConfig):

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