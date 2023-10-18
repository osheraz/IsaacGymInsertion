# --------------------------------------------------------
# now its our turn.
# https://arxiv.org/abs/todo
# Copyright (c) 2023 Osher & Co.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# --------------------------------------------------------
# Based on: In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import time
import torch
import numpy as np
from termcolor import cprint

from algo.ppo.experience import ExperienceBuffer
from algo.models.models import ActorCritic
from algo.models.running_mean_std import RunningMeanStd
from isaacgyminsertion.utils.misc import AverageScalarMeter
from tensorboardX import SummaryWriter
import torch.distributed as dist


class ExtrinsicAdapt(object):
    def __init__(self, env, output_dir, full_config):
        # ---- MultiGPU ----
        self.multi_gpu = full_config.train.ppo.multi_gpu
        if self.multi_gpu:
            self.rank = int(os.getenv("LOCAL_RANK", "0"))
            self.rank_size = int(os.getenv("WORLD_SIZE", "1"))
            dist.init_process_group("nccl", rank=self.rank, world_size=self.rank_size)
            self.device = "cuda:" + str(self.rank)
            print(f"current rank: {self.rank} and use device {self.device}")
        else:
            self.rank = -1
            self.device = full_config["rl_device"]
        # ------
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        self.action_space = self.env.action_space
        self.actions_num = self.action_space.shape[0]
        # ---- Tactile Info ---
        self.tactile_info = self.ppo_config["tactile_info"]
        tactile_seq_length = self.ppo_config["tactile_seq_length"]
        self.tactile_info_dim = self.network_config.tactile_mlp.units[0]
        # ---- ft Info ---
        self.ft_info = self.ppo_config["ft_info"]
        self.ft_seq_length = self.ppo_config["ft_seq_length"]
        self.ft_input_dim = self.ppo_config["ft_input_dim"]
        self.ft_info_dim = self.ft_input_dim * self.ft_seq_length
        # ---- Priv Info ----
        self.priv_info = self.ppo_config['priv_info']
        self.priv_info_dim = self.ppo_config['priv_info_dim']
        self.extrin_adapt = self.ppo_config['extrin_adapt']
        # ---- Model ----
        net_config = {
            'actor_units': self.network_config.mlp.units,
            'actions_num': self.actions_num,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'input_shape': self.obs_shape,
            'extrin_adapt': self.extrin_adapt,
            'priv_info_dim': self.priv_info_dim,
            'priv_info': self.priv_info,
            "tactile_info": self.tactile_info,
            "tactile_input_shape": self.tactile_info_dim,
            "ft_input_shape": self.ft_info_dim,
            "ft_info": self.ft_info,
            "tactile_units": self.network_config.tactile_mlp.units,
            "tactile_decoder_embed_dim": self.network_config.tactile_mlp.units[0],
            "ft_units": self.network_config.ft_mlp.units,
        }

        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.priv_mean_std = RunningMeanStd(self.priv_info_dim).to(self.device)
        self.priv_mean_std.eval()

        # Currently ft is not supported
        self.ft_mean_std = RunningMeanStd((self.ft_seq_length, 6)).to(self.device)
        self.ft_mean_std.train()

        # tactile is already normalized in task.

        # ---- Output Dir ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, 'stage2_nn')
        self.tb_dir = os.path.join(self.output_dir, 'stage2_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        writer = SummaryWriter(self.tb_dir)
        self.writer = writer
        self.direct_info = {}
        # ---- Misc ----
        self.batch_size = self.num_actors
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.mean_eps_length = AverageScalarMeter(window_size=20000)
        self.mean_eps_success = AverageScalarMeter(window_size=20000)

        self.best_rewards = -10000
        self.agent_steps = 0

        # ---- Optim ----
        adapt_params = []
        for name, p in self.model.named_parameters():
            if 'tactile_decoder' in name or 'tactile_mlp' in name:# or 'ft_adapt_tconv' in name:
                adapt_params.append(p)
            else:
                p.requires_grad = False

        self.optim = torch.optim.Adam(adapt_params, lr=3e-4)
        # ---- Training Misc
        self.internal_counter = 0
        self.latent_loss_stat = 0
        self.loss_stat_cnt = 0
        batch_size = self.num_actors
        self.step_reward = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.step_length = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

    def set_eval(self):
        self.model.eval()
        self.running_mean_std.eval()
        self.priv_mean_std.eval()
        self.ft_mean_std.eval()

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        while True:
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
                'ft_hist': self.ft_mean_std(obs_dict['ft_hist'].detach()),
                'tactile_hist': obs_dict['tactile_hist'].detach(),
            }
            mu = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)

    def train(self):
        _t = time.time()
        _last_t = time.time()

        obs_dict = self.env.reset()
        self.agent_steps += self.batch_size
        while self.agent_steps <= 1e9:
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']).detach(),
                'priv_info': self.priv_mean_std(obs_dict['priv_info']).detach(),
                'ft_hist': self.ft_mean_std(obs_dict['ft_hist'].detach()),
                'tactile_hist': obs_dict['tactile_hist'].detach(),
            }
            mu, _, _, e, e_gt = self.model._actor_critic(input_dict)
            loss = ((e - e_gt.detach()) ** 2).mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            mu = mu.detach()
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu) # online
            self.agent_steps += self.batch_size

            # ---- statistics
            self.step_reward += r
            self.step_length += 1
            done_indices = done.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])

            not_dones = 1.0 - done.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            self.log_tensorboard()

            if self.agent_steps % 500 == 0:
                self.save(os.path.join(self.nn_dir, f'{self.agent_steps // 1e8}00m'))
                self.save(os.path.join(self.nn_dir, f'last'))

            mean_rewards = self.mean_eps_reward.get_mean()
            self.best_rewards = mean_rewards
            # if mean_rewards > self.best_rewards:
            #     self.save(os.path.join(self.nn_dir, f'best'))
            #     self.best_rewards = mean_rewards

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Mean Best: {self.best_rewards:.2f}'
            cprint(info_string)

    def log_tensorboard(self):
        self.writer.add_scalar('episode_rewards/step', self.mean_eps_reward.get_mean(), self.agent_steps)
        self.writer.add_scalar('episode_lengths/step', self.mean_eps_length.get_mean(), self.agent_steps)
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f'{k}/frame', v, self.agent_steps)

    def log_wandb(self):
        # TODO
        pass

    def restore_train(self, fn):
        checkpoint = torch.load(fn)
        cprint('careful, using non-strict matching', 'red', attrs=['bold'])
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        # self.ft_mean_std.load_state_dict(checkpoint['ft_mean_std'])

    def restore_test(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.model.load_state_dict(checkpoint['model'])
        self.ft_mean_std.load_state_dict(checkpoint['ft_mean_std'])
        self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.priv_mean_std:
            weights['priv_mean_std'] = self.priv_mean_std.state_dict()
        if self.ft_mean_std:
            weights['ft_mean_std'] = self.ft_mean_std.state_dict()
        torch.save(weights, f'{name}.pth')
