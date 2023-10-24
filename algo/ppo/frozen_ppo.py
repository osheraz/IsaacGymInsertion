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
# Based on: RLGames
# Copyright (c) 2019 Denys88
# Licence under MIT License
# https://github.com/Denys88/rl_games/
# --------------------------------------------------------

import os
import time
import cv2
import imageio
import torch
import torch.distributed as dist
import numpy as np

from algo.ppo.experience import ExperienceBuffer, DataLogger
from algo.ppo.experience import VectorizedExperienceBuffer
from algo.models.models import ActorCritic
from algo.models.running_mean_std import RunningMeanStd

from isaacgyminsertion.utils.misc import AverageScalarMeter
from isaacgyminsertion.utils.misc import add_to_fifo, multi_gpu_aggregate_stats
from tensorboardX import SummaryWriter


class PPO(object):
    def __init__(self, env, output_dif, full_config):

        self.rank = -1
        self.device = full_config["rl_device"]
        
        # ------
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        
        action_space = full_config.task.env.numActions
        self.actions_num = action_space.shape[0]
       
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        # ---- Tactile Info ---
        self.tactile_info = self.ppo_config["tactile_info"]
        self.tactile_seq_length = self.network_config.tactile_decoder.tactile_seq_length
        self.tactile_input_dim = (self.network_config.tactile_decoder.img_width,
                                  self.network_config.tactile_decoder.img_height,
                                  self.network_config.tactile_decoder.num_channels)
        self.mlp_tactile_info_dim = self.network_config.tactile_mlp.units[0]
        # ---- ft Info ---
        self.ft_info = self.ppo_config["ft_info"]
        self.ft_seq_length = self.ppo_config["ft_seq_length"]
        self.ft_input_dim = self.ppo_config["ft_input_dim"]
        self.ft_info_dim = self.ft_input_dim * self.ft_seq_length
        # ---- Priv Info ----
        self.priv_info = self.ppo_config['priv_info']
        self.priv_info_dim = self.ppo_config['priv_info_dim']
        self.extrin_adapt = self.ppo_config['extrin_adapt']
        self.priv_info_embed_dim = self.network_config.priv_mlp.units[-1]
        # ---- Obs Info (student)----
        self.obs_info = self.ppo_config["obs_info"]
        # ---- Model ----
        net_config = {
            'actor_units': [512, 256, 128],
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'extrin_adapt': False,
            'priv_info_dim': 31,
            'priv_info': True,
            "ft_input_shape": self.ft_info_dim,
            "ft_info": self.ft_info,
            "ft_units": self.network_config.ft_mlp.units,
            "obs_units": self.network_config.obs_mlp.units,
            "obs_info": self.obs_info,

            "tactile_info": self.tactile_info,
            "mlp_tactile_input_shape": self.mlp_tactile_info_dim,
            "mlp_tactile_units": self.network_config.tactile_mlp.units,
            'tactile_input_dim': self.tactile_input_dim,
            'tactile_seq_length': self.tactile_seq_length,
            "tactile_decoder_embed_dim": self.network_config.tactile_mlp.units[0],
        }

        self.model = ActorCritic(net_config)
        self.model.to(self.device)

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.priv_mean_std = RunningMeanStd((self.priv_info_dim,)).to(self.device)

        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, 'stage1_nn')
        self.tb_dif = os.path.join(self.output_dir, 'stage1_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)

        # ---- Optim ----
        self.last_lr = float(self.ppo_config['learning_rate'])
        self.weight_decay = self.ppo_config.get('weight_decay', 0.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.last_lr, weight_decay=self.weight_decay)

        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']
        self.clip_value = self.ppo_config['clip_value']
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.gamma = self.ppo_config['gamma']
        self.tau = self.ppo_config['tau']
        self.truncate_grads = self.ppo_config['truncate_grads']
        self.grad_norm = self.ppo_config['grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        self.normalize_input = self.ppo_config['normalize_input']
        self.normalize_value = self.ppo_config['normalize_value']

        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config['horizon_length']
        self.batch_size = self.horizon_length * self.num_actors
        self.mini_epochs_num = self.ppo_config['mini_epochs']
        self.minibatch_size = self.batch_size // self.mini_epochs_num  # self.ppo_config['minibatch_size']
        assert self.batch_size % self.minibatch_size == 0 or full_config.test

        # ---- scheduler ----
        self.kl_threshold = self.ppo_config['kl_threshold']
        self.scheduler = AdaptiveScheduler(self.kl_threshold)

        # ---- Snapshot
        self.save_freq = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']

        # ---- Tensorboard Logger ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer

        # ---- Rollout Videos ----
        self.it = 0
        self.log_video_every = self.env.cfg_task.env.record_video_every
        self.log_ft_every = self.env.cfg_task.env.record_ft_every

        self.last_recording_it = 0
        self.last_recording_it_ft = 0

        self.episode_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.episode_success = AverageScalarMeter(100)

        self.obs = None
        self.epoch_num = 0
        self.storage = ExperienceBuffer(self.num_actors,
                                        self.horizon_length,
                                        self.batch_size,
                                        self.minibatch_size,
                                        self.obs_shape[0],
                                        self.actions_num,
                                        self.priv_info_dim,
                                        self.device, )

        # ---- Data Logger ----
        # getting the shapes for the data logger initialization
        log_items = {
            'arm_joints_shape': self.env.arm_dof_pos.shape[-1],
            'eef_pos_shape': self.env.fingertip_centered_pos.size()[-1] + self.env.fingertip_centered_quat.size()[-1],
            'socket_pos_shape': self.env.socket_pos.size()[-1] + self.env.socket_quat.size()[-1],
            'noisy_socket_pos_shape': self.env.socket_pos.size()[-1] + self.env.socket_quat.size()[-1],
            'plug_pos_shape': self.env.plug_pos.size()[-1] + self.env.plug_quat.size()[-1],
            'action_shape': self.actions_num,
            'target_shape': self.env.targets.shape[-1],
            'tactile_shape': self.env.tactile_imgs.shape[1:],
            'latent_shape': net_config['priv_mlp_units'][-1],
            'rigid_physics_params_shape': self.env.rigid_physics_params.shape[-1],
            'plug_socket_pos_error_shape': self.env.plug_socket_pos_error.shape[-1],
            'plug_socket_quat_error_shape': self.env.plug_socket_quat_error.shape[-1],

        }

        # initializing data logger, the device should be changed
        self.data_logger_init = lambda x: DataLogger(self.env.num_envs, self.env.max_episode_length, self.env.device, os.path.join(self.env.cfg_task.data_logger.base_folder, self.env.cfg_task.data_logger.sub_folder), self.env.cfg_task.data_logger.total_trajectories, **log_items)
        self.data_logger = None
        
        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.best_rewards = -10000

        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0


    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
            self.priv_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
            self.priv_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    def model_act(self, obs_dict):

        processed_obs = self.running_mean_std(obs_dict['obs'])
        processed_priv = self.priv_mean_std(obs_dict['priv_info'])
        input_dict = {
            'obs': processed_obs,
            'priv_info': processed_priv,
        }
        res_dict = self.model.act(input_dict)
        res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict

    def restore_test(self, fn):
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        action, latent, done = None, None, None
        while True:
            
            self.log_video()
            
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
                'priv_info': self.priv_mean_std(obs_dict['priv_info']),
            }
            mu, latent = self.model.act_inference(input_dict)
            action = mu.clone()
            mu = torch.clamp(mu, -1.0, 1.0)

            # collect data
            if self.env.cfg_task.data_logger.collect_data:
                if self.data_logger is None:
                    self.data_logger = self.data_logger_init(None)
                self.log_trajectory_data(action, latent, done)

            obs_dict, r, done, info = self.env.step(mu)