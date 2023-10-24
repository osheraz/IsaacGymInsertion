# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: RLGames
# Copyright (c) 2019 Denys88
# Licence under MIT License
# https://github.com/Denys88/rl_games/
# --------------------------------------------------------

import gym
import torch
from torch.utils.data import Dataset
import os
from datetime import datetime
from pathlib import Path
import json
import cv2
import deepdish as dd
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

def transform_op(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


class ExperienceBuffer(Dataset):
    def __init__(self, num_envs, horizon_length, batch_size, minibatch_size, obs_dim, act_dim, priv_dim, device):
        self.device = device
        self.num_envs = num_envs
        self.transitions_per_env = horizon_length
        self.priv_info_dim = priv_dim

        self.data_dict = None
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.priv_dim = priv_dim
        self.storage_dict = {
            'obses': torch.zeros((self.transitions_per_env, self.num_envs, self.obs_dim), dtype=torch.float32,
                                 device=self.device),
            'priv_info': torch.zeros((self.transitions_per_env, self.num_envs, self.priv_dim), dtype=torch.float32,
                                     device=self.device),
            'rewards': torch.zeros((self.transitions_per_env, self.num_envs, 1), dtype=torch.float32,
                                   device=self.device),
            'values': torch.zeros((self.transitions_per_env, self.num_envs, 1), dtype=torch.float32,
                                  device=self.device),
            'neglogpacs': torch.zeros((self.transitions_per_env, self.num_envs), dtype=torch.float32,
                                      device=self.device),
            'dones': torch.zeros((self.transitions_per_env, self.num_envs), dtype=torch.uint8, device=self.device),
            'actions': torch.zeros((self.transitions_per_env, self.num_envs, self.act_dim), dtype=torch.float32,
                                   device=self.device),
            'mus': torch.zeros((self.transitions_per_env, self.num_envs, self.act_dim), dtype=torch.float32,
                               device=self.device),
            'sigmas': torch.zeros((self.transitions_per_env, self.num_envs, self.act_dim), dtype=torch.float32,
                                  device=self.device),
            'returns': torch.zeros((self.transitions_per_env, self.num_envs, 1), dtype=torch.float32,
                                   device=self.device),
        }

        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.length = self.batch_size // self.minibatch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        input_dict = {}
        for k, v in self.data_dict.items():
            if type(v) is dict:
                v_dict = {kd: vd[start:end] for kd, vd in v.items()}
                input_dict[k] = v_dict
            else:
                input_dict[k] = v[start:end]
        return input_dict['values'], input_dict['neglogpacs'], input_dict['advantages'], input_dict['mus'], \
               input_dict['sigmas'], input_dict['returns'], input_dict['actions'], \
               input_dict['obses'], input_dict['priv_info']

    def update_mu_sigma(self, mu, sigma):
        start = self.last_range[0]
        end = self.last_range[1]
        self.data_dict['mus'][start:end] = mu
        self.data_dict['sigmas'][start:end] = sigma

    def update_data(self, name, index, val):
        if type(val) is dict:
            for k, v in val.items():
                self.storage_dict[name][k][index, :] = v
        else:
            self.storage_dict[name][index, :] = val

    def computer_return(self, last_values, gamma, tau):
        last_gae_lam = 0
        mb_advs = torch.zeros_like(self.storage_dict['rewards'])
        for t in reversed(range(self.transitions_per_env)):
            if t == self.transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.storage_dict['values'][t + 1]
            next_nonterminal = 1.0 - self.storage_dict['dones'].float()[t]
            next_nonterminal = next_nonterminal.unsqueeze(1)
            delta = self.storage_dict['rewards'][t] + gamma * next_values * next_nonterminal - \
                    self.storage_dict['values'][t]
            mb_advs[t] = last_gae_lam = delta + gamma * tau * next_nonterminal * last_gae_lam
            self.storage_dict['returns'][t, :] = mb_advs[t] + self.storage_dict['values'][t]

    def prepare_training(self):
        self.data_dict = {}
        for k, v in self.storage_dict.items():
            self.data_dict[k] = transform_op(v)
        advantages = self.data_dict['returns'] - self.data_dict['values']
        self.data_dict['advantages'] = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).squeeze(1)
        return self.data_dict


class VectorizedExperienceBuffer:
    def __init__(self, obs_shape, priv_shape, tactile_shape, action_shape, capacity, device):
        """Create Vectorized Experience buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        See Also
        --------
        ReplayBuffer.__init__
        """

        self.device = device

        self.obses = torch.empty((capacity, *obs_shape), dtype=torch.float32, device=self.device)
        self.priv_obses = torch.empty((capacity, *priv_shape), dtype=torch.float32, device=self.device)
        # actually, tactile_imgs will be huge..
        self.tactile_imgs = torch.empty((capacity, *tactile_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((capacity, *action_shape), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((capacity, 1), dtype=torch.bool, device=self.device)

        self.capacity = capacity
        self.idx = 0
        self.full = False

    def add(self, obs, priv_obs, tactile_img, action, reward, done):
        num_observations = obs.shape[0]
        remaining_capacity = min(self.capacity - self.idx, num_observations)
        overflow = num_observations - remaining_capacity
        if remaining_capacity < num_observations:
            self.obses[0: overflow] = obs[-overflow:]
            self.priv_obses[0: overflow] = priv_obs[-overflow:]
            self.tactile_imgs[0: overflow] = tactile_img[-overflow:]
            self.actions[0: overflow] = action[-overflow:]
            self.rewards[0: overflow] = reward[-overflow:]
            self.dones[0: overflow] = done[-overflow:]
            self.full = True

        self.obses[self.idx: self.idx + remaining_capacity] = obs[:remaining_capacity]
        self.priv_obses[self.idx: self.idx + remaining_capacity] = priv_obs[:remaining_capacity]
        self.tactile_imgs[self.idx: self.idx + remaining_capacity] = tactile_img[:remaining_capacity]
        self.actions[self.idx: self.idx + remaining_capacity] = action[:remaining_capacity]
        self.rewards[self.idx: self.idx + remaining_capacity] = reward[:remaining_capacity]
        self.dones[self.idx: self.idx + remaining_capacity] = done[:remaining_capacity]

        self.idx = (self.idx + num_observations) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obses, priv_obses, tactile_imgs: torch tensor
            batch of observations
        actions: torch tensor
            batch of actions executed given obs
        rewards: torch tensor
            rewards received as results of executing act_batch
        next_obses: torch tensor
            next set of observations seen after executing act_batch
        not_dones: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not
        not_dones_no_max: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not, specifically exlcuding maximum episode steps
        """

        idxs = torch.randint(0,
                             self.capacity if self.full else self.idx,
                             (batch_size,), device=self.device)
        obses = self.obses[idxs]
        priv_obses = self.priv_obses[idxs]
        tactile_imgs = self.obses[idxs]

        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        dones = self.dones[idxs]

        return obses, priv_obses, tactile_imgs, actions, rewards, dones


class DataLogger():

    def __init__(self, num_envs, episode_length, device, dir_path, total_trajectories, **kwargs):

        self.buffer = []
        self.id = 0
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        self.dir = dir_path
        self.count = 0

        self.num_envs = num_envs
        self.device = device
        self.transitions_per_env = episode_length
        
        self.data_shapes = {}
        for key, value in kwargs.items():
            if key.endswith("_shape"):
                self.data_shapes[key.replace("_shape", "")] = value

        self.trajectory_ctr = 0
        self.total_trajectories = total_trajectories
        self.pbar = tqdm(total=self.total_trajectories)

        self._init_buffers()
        save_data = False
        if save_data:
            self.num_workers = 24
            try:
                self.q_s = [mp.JoinableQueue(maxsize=500) for _ in range(self.num_workers)]
                self.workers = [mp.Process(target=self.worker, args=(q, idx)) for idx, q in enumerate(self.q_s)]
                for worker in self.workers:
                    worker.daemon = True
                    worker.start()
            except KeyboardInterrupt:
                for q in self.q_s:
                    q.put(None)
                for q in self.q_s:
                    q.join()
                for worker in self.workers:
                    worker.terminate()
                exit()

    def _init_buffers(self):

        self.log_data = {}
        for key, shape in self.data_shapes.items():
            if shape is not None:
                data_shape = (self.num_envs, self.transitions_per_env, shape)
                if isinstance(shape, torch.Size):
                    data_shape = (self.num_envs, self.transitions_per_env, *shape)
                self.log_data[key] = torch.zeros(data_shape, dtype=torch.float32, device=self.device)

        self.done = torch.zeros((self.num_envs), self.transitions_per_env, dtype=torch.bool, device=self.device)

        self.env_step_counter = torch.zeros((self.num_envs, 1), dtype=torch.long, device=self.device).view(-1, 1)

        self.env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device).unsqueeze(-1)
    
    def _reset_buffers(self, env_ids):
        for key, buffer in self.log_data.items():
            buffer[env_ids, ...] = 0.
        self.done[env_ids, ...] = 0.
        self.env_step_counter[env_ids, ...] = 0.

    def _save_batch_trajectories(self, data):
        q_id = np.random.randint(0, self.num_workers)
        self.q_s[q_id].put(data)
        
    def update(self, save_trajectory=True, **kwargs):

        for key, value in kwargs.items():
            if key == "done":
                continue
            if value is None:
                value = torch.zeros((self.num_envs, self.data_shapes[key]), dtype=torch.float32, device=self.device)
            self.log_data[key][self.env_ids, self.env_step_counter, ...] = value.clone().unsqueeze(1)

        
        done = kwargs.get('done', None)
        if done is None:
            done = torch.zeros_like(self.done[:, 0, ...]).to(torch.bool)
        if done.dtype != torch.bool:
            done = done.clone().to(torch.bool)

        self.done[self.env_ids, self.env_step_counter, ...] = done.clone().unsqueeze(1)
        self.env_step_counter += 1
        dones = done.to(torch.long).nonzero()
        if len(dones) > 0:
            self.pbar.update(len(dones))
            save_env_ids = dones.squeeze(1)
            if save_trajectory:
                self.trajectory_ctr += len(dones)
                for save_env_id in save_env_ids:
                    batch_data = {key: self.log_data[key][save_env_id, ...].clone().cpu() for key in self.log_data}
                    batch_data['done'] = self.done[save_env_id, ...].clone().cpu()
                    self._save_batch_trajectories(batch_data)
            self._reset_buffers(save_env_ids)
            if save_trajectory and (self.trajectory_ctr >= self.total_trajectories):
                self.pbar.close()
                for q in self.q_s:
                    q.put(None)
                for q in self.q_s:
                    q.join()
                for worker in self.workers:
                    worker.terminate()
                print('Data collection finished!')
                exit()

    def worker(self, q, q_idx):
        data_path = os.path.join(self.dir, f'{q_idx}')
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        try:
            while True:
                item = q.get()
                if item is None:
                    break
                # save the data
                for k, v in item.items():
                    if isinstance(item[k], torch.Tensor):
                        item[k] = item[k].numpy()
                np.savez_compressed(os.path.join(data_path, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.npz'), **item)
                q.task_done()
        except KeyboardInterrupt:
            print("Ctrl+C detected. Exiting gracefully.")

    def get_data(self):
        return self.log_data
    
    def reset(self):
        self._reset_buffers(torch.arange(self.num_envs, dtype=torch.long, device=self.device).unsqueeze(-1))