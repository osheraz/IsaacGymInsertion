# --------------------------------------------------------
# Now its my turn.
# https://arxiv.org/abs/todo.
# Copyright (c) 2023
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# --------------------------------------------------------
# Based on: In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
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

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from termcolor import cprint

from algo.ppo.frozen_ppo import PPO
from algo.ext_adapt.ext_adapt import ExtrinsicAdapt

from isaacgyminsertion.tasks import isaacgym_task_map
from isaacgyminsertion.utils.reformat import omegaconf_to_dict
from isaacgyminsertion.utils.utils import set_np_formatting, set_seed


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(config_name="config", config_path="./cfg")
def run(cfg: DictConfig):

    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    if cfg.train.ppo.multi_gpu:
        rank = int(os.getenv("LOCAL_RANK", "0"))
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        cfg.sim_device = f"cuda:{rank}"
        cfg.rl_device = f"cuda:{rank}"
        cfg.graphics_device_id = int(rank)
        # sets seed. if seed is -1 will pick a random one
        cfg.seed = set_seed(cfg.seed + rank)
    else:
        rank = -1
        cfg.seed = set_seed(cfg.seed)

    # sets seed. if seed is -1 will pick a random one
    # cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)
    if cfg.train_diffusion:
        # TODO edit
        from algo.models.diffusion.train_diffusion import Runner

        # perform train
        runner = Runner(cfg)

        exit()

    if cfg.offline_real_training:
        # TODO edit
        from algo.models.transformer.runner import Runner as TransformerRunner

        agent = None

        # perform train
        runner = TransformerRunner(cfg.offline_train, agent=agent)
        runner.run()

        exit()

    # for training the transformer with offline data only
    if cfg.offline_training:
        from algo.models.transformer.runner import Runner as TransformerRunner

        agent = None
        
        # perform train
        runner = TransformerRunner(cfg, agent=agent)
        runner.run()
        
        exit()

    cprint("Start Building the Environment", "green", attrs=["bold"])

    envs = isaacgym_task_map[cfg.task_name](
        cfg=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device,
        rl_device=cfg.rl_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        virtual_screen_capture=False,  # cfg.capture_video,
        force_render=cfg.force_render  # if not cfg.headless else False,
    )

    output_dif = os.path.join('outputs', str(datetime.now().strftime("%m-%d-%y")))
    output_dif = os.path.join(output_dif, str(datetime.now().strftime("%H-%M-%S")))
    os.makedirs(output_dif, exist_ok=True)
    agent = eval(cfg.train.algo)(envs, output_dif, full_config=cfg)

    if cfg.test:
        assert cfg.train.load_path
        agent.restore_test(cfg.train.load_path)
        # Test insertion accuracy in the simulation with online Student\Teacher
        if not cfg.offline_training_w_env:
            num_success, total_trials = agent.test()
            print(f"Success rate: {num_success / total_trials}")
        else:
            # Test transformer (trained offline)
            print("Loading Teacher model from", cfg.train.load_path)
            print("Loading Student model from", cfg.offline_train.train.student_ckpt_path)

            from algo.models.transformer.runner import Runner as TransformerRunner 
            agent = PPO(envs, output_dif, full_config=cfg)
            agent.restore_test(cfg.train.load_path)
            agent.set_eval()

            runner = TransformerRunner(cfg, agent,
                                       action_regularization=cfg.offline_train.train.action_regularization)
            runner.run()

    else:
        if rank <= 0:
            date = str(datetime.now().strftime("%m%d%H"))
            with open(os.path.join(output_dif, f"config_{date}.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg))

        agent.restore_train(cfg.train.load_path)
        agent.train()

    cprint("Finished", "green", attrs=["bold"])


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "config_path", type=argparse.FileType("r"), help="Path to hydra config."
    # )
    # args = parser.parse_args()

    run()