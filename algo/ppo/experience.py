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
import os
import numpy as np
import torch
import multiprocessing as mp
from tqdm import tqdm
from datetime import datetime

NUM_JOINTS = 7
ROT_MAT_SIZE = 9
POS_SIZE = 3
QUAT_SIZE = 4
ACT_SIZE = 6


def transform_op(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


class StudentBuffer(Dataset):
    def __init__(self, num_envs, horizon_length, batch_size, minibatch_size,
                 obs_dim, act_dim, priv_dim, student_dims, device):

        self.device = device
        self.num_envs = num_envs
        self.transitions_per_env = horizon_length
        self.priv_info_dim = priv_dim

        self.data_dict = None
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.priv_dim = priv_dim

        self.tactile_info = student_dims.get('tactile') is not None
        self.img_info = student_dims.get('img') is not None
        self.seg_info = student_dims.get('seg') is not None
        self.student_obs_info = student_dims.get('student_obs') is not None
        self.pcl_info = student_dims.get('pcl') is not None

        self.stud_obs_dim = student_dims['student_obs'] if self.student_obs_info else None
        self.tactile_dim = student_dims['tactile'] if self.tactile_info else None
        self.img_dim = student_dims['img'] if self.img_info else None
        self.seg_dim = student_dims['seg'] if self.seg_info else None
        self.pcl_dim = student_dims['pcl'] if self.pcl_info else None

        self.storage_dict = {
            'n_obs': torch.zeros((self.transitions_per_env, self.num_envs, self.obs_dim), dtype=torch.float32,
                                 device=self.device),
            'n_priv_info': torch.zeros((self.transitions_per_env, self.num_envs, self.priv_dim), dtype=torch.float32,
                                       device=self.device),
            'rewards': torch.zeros((self.transitions_per_env, self.num_envs, 1), dtype=torch.float32,
                                   device=self.device),
            'teacher_actions': torch.zeros((self.transitions_per_env, self.num_envs, self.act_dim), dtype=torch.float32,
                                           device=self.device),
            'student_actions': torch.zeros((self.transitions_per_env, self.num_envs, self.act_dim), dtype=torch.float32,
                                           device=self.device),
            'latent_gt': torch.zeros((self.transitions_per_env, self.num_envs, 8), dtype=torch.float32,
                                     device=self.device),
        }

        if self.tactile_info:
            self.storage_dict['n_tactile'] = torch.zeros((self.transitions_per_env, self.num_envs, *self.tactile_dim),
                                                         dtype=torch.float32,
                                                         device=self.device)
        if self.img_info:
            self.storage_dict['n_img'] = torch.zeros((self.transitions_per_env, self.num_envs, *self.img_dim),
                                                     dtype=torch.float32,
                                                     device=self.device)
        if self.seg_info:
            self.storage_dict['n_seg'] = torch.zeros((self.transitions_per_env, self.num_envs, *self.seg_dim),
                                                     dtype=torch.float32,
                                                     device=self.device)
        if self.pcl_info:
            self.storage_dict['n_pcl'] = torch.zeros((self.transitions_per_env, self.num_envs, *self.pcl_dim),
                                                     dtype=torch.float32,
                                                     device=self.device)
        if self.student_obs_info:
            self.storage_dict['n_student_obs'] = torch.zeros(
                (self.transitions_per_env, self.num_envs, self.stud_obs_dim),
                dtype=torch.float32,
                device=self.device)

        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.length = self.batch_size // self.minibatch_size
        self.indices = torch.randperm(self.batch_size, requires_grad=False, device=self.device)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        batch_idx = self.indices[start:end]
        input_dict = {}
        for k, v in self.data_dict.items():
            if type(v) is dict:
                v_dict = {kd: vd[batch_idx] for kd, vd in v.items()}
                input_dict[k] = v_dict
            else:
                input_dict[k] = v[batch_idx]
        return input_dict

    def update_data(self, name, index, val):
        if type(val) is dict:
            for k, v in val.items():
                self.storage_dict[name][k][index, :] = v
        else:
            self.storage_dict[name][index, :] = val

    def prepare_training(self):
        self.data_dict = {}
        for k, v in self.storage_dict.items():
            self.data_dict[k] = transform_op(v)
        return self.data_dict


class ExperienceBuffer(Dataset):
    def __init__(self, num_envs, horizon_length, batch_size, minibatch_size, obs_dim, act_dim, priv_dim, pts_dim, vt_poilcy,
                 device):
        self.device = device
        self.num_envs = num_envs
        self.transitions_per_env = horizon_length
        self.priv_info_dim = priv_dim

        self.data_dict = None
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.priv_dim = priv_dim
        self.pts_dim = pts_dim
        self.vt_policy = vt_poilcy

        self.storage_dict = {
            'obses': torch.zeros((self.transitions_per_env, self.num_envs, self.obs_dim), dtype=torch.float32,
                                 device=self.device),
            'priv_info': torch.zeros((self.transitions_per_env, self.num_envs, self.priv_dim), dtype=torch.float32,
                                     device=self.device),
            'contacts': torch.zeros((self.transitions_per_env, self.num_envs, self.pts_dim), dtype=torch.float32,
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

        if self.vt_policy:
            self.storage_dict['img'] = torch.zeros((self.transitions_per_env, self.num_envs, 54*96),
                                                     dtype=torch.float32,
                                                     device=self.device)
            self.storage_dict['seg'] = torch.zeros((self.transitions_per_env, self.num_envs, 54*96),
                                                     dtype=torch.float32,
                                                     device=self.device)
            self.storage_dict['student_obs'] = torch.zeros(
                (self.transitions_per_env, self.num_envs, 18),
                dtype=torch.float32,
                device=self.device)

        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.length = self.batch_size // self.minibatch_size
        self.indices = torch.randperm(self.batch_size, requires_grad=False, device=self.device)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        batch_idx = self.indices[start:end]
        input_dict = {}
        for k, v in self.data_dict.items():
            if type(v) is dict:
                v_dict = {kd: vd[batch_idx] for kd, vd in v.items()}
                input_dict[k] = v_dict
            else:
                input_dict[k] = v[batch_idx]
        if self.vt_policy:
            return input_dict['values'], input_dict['neglogpacs'], input_dict['advantages'], input_dict['mus'], \
                input_dict['sigmas'], input_dict['returns'], input_dict['actions'], \
                input_dict['obses'], input_dict['priv_info'], input_dict['contacts'], input_dict['img'], input_dict['seg'], input_dict['student_obs']
        else:
            return input_dict['values'], input_dict['neglogpacs'], input_dict['advantages'], input_dict['mus'], \
                input_dict['sigmas'], input_dict['returns'], input_dict['actions'], \
                input_dict['obses'], input_dict['priv_info'], input_dict['contacts']

    def update_mu_sigma(self, mu, sigma):
        start = self.last_range[0]
        end = self.last_range[1]
        batch_idx = self.indices[start:end]
        self.data_dict['mus'][batch_idx] = mu
        self.data_dict['sigmas'][batch_idx] = sigma

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


class DataLoggerSim():

    def __init__(self, num_envs, episode_length, device, dir_path, total_trajectories, save_trajectory, **kwargs):

        self.buffer = []
        self.id = 0
        self.count = 0
        self.num_envs = num_envs
        self.device = device
        self.transitions_per_env = episode_length
        self.data_shapes = {}

        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        self.dir = dir_path

        for key, value in kwargs.items():
            if key.endswith("_shape"):
                self.data_shapes[key.replace("_shape", "")] = value

        self.pbar = None
        self.trajectory_ctr = 0
        self.total_trajectories = None

        self._init_buffers()

        if save_trajectory:
            self.pbar = tqdm(total=self.total_trajectories)
            self.total_trajectories = total_trajectories
            self.num_workers = 8
            try:
                self.q_s = [mp.JoinableQueue(maxsize=episode_length) for _ in range(self.num_workers)]
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

            self.log_data[key][self.env_ids, self.env_step_counter, ...] = value.clone().unsqueeze(1).to(
                self.log_data[key].dtype)

        done = kwargs.get('done', None)
        if done is None:
            done = torch.zeros_like(self.done[:, 0, ...]).to(torch.bool)
        if done.dtype != torch.bool:
            done = done.clone().to(torch.bool)

        self.done[self.env_ids, self.env_step_counter, ...] = done.clone().unsqueeze(1)
        self.env_step_counter += 1
        dones = done.to(torch.long).nonzero()
        if len(dones) > 0:
            save_env_ids = dones.squeeze(1)
            if save_trajectory:
                self.pbar.update(len(dones))
                self.trajectory_ctr += len(dones)
                for save_env_id in save_env_ids:
                    batch_data = {key: self.log_data[key][save_env_id, ...].clone().cpu() for key in self.log_data}
                    batch_data['done'] = self.done[save_env_id, ...].clone().cpu()
                    self._save_batch_trajectories(batch_data)
            self._reset_buffers(save_env_ids)
            if save_trajectory:
                if self.trajectory_ctr >= self.total_trajectories:
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
                np.savez_compressed(os.path.join(data_path, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.npz'),
                                    **item)
                q.task_done()
        except KeyboardInterrupt:
            print("Ctrl+C detected. Exiting gracefully.")

    def get_data(self):
        return self.log_data

    def reset(self):
        self._reset_buffers(torch.arange(self.num_envs, dtype=torch.long, device=self.device).unsqueeze(-1))


class DataLoggerReal:

    def __init__(self, num_envs, episode_length, device, dir_path, total_trajectories, save_trajectory, **kwargs):
        self.buffer = []
        self.id = 0
        self.count = 0
        self.num_envs = num_envs
        self.device = device
        self.transitions_per_env = episode_length
        self.data_shapes = {}

        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        self.dir = dir_path

        for key, value in kwargs.items():
            if key.endswith("_shape"):
                self.data_shapes[key.replace("_shape", "")] = value

        self.pbar = None
        self.trajectory_ctr = 0
        self.total_trajectories = total_trajectories

        self._init_buffers()

        if save_trajectory:
            self.pbar = tqdm(total=self.total_trajectories)
            self.num_workers = 3
            self.manager = mp.Manager()
            self.q_s = [self.manager.Queue(maxsize=episode_length) for _ in range(self.num_workers)]
            self.workers = [mp.Process(target=self.worker, args=(q, idx)) for idx, q in enumerate(self.q_s)]
            for worker in self.workers:
                worker.daemon = True
                worker.start()

    def _init_buffers(self):
        self.log_data = {}
        for key, shape in self.data_shapes.items():
            if shape is not None:
                data_shape = (self.num_envs, self.transitions_per_env, shape)
                if isinstance(shape, torch.Size):
                    data_shape = (self.num_envs, self.transitions_per_env, *shape)
                self.log_data[key] = torch.zeros(data_shape, dtype=torch.float32, device=self.device)

        self.done = torch.zeros((self.num_envs, self.transitions_per_env), dtype=torch.bool, device=self.device)
        self.env_step_counter = torch.zeros((self.num_envs, 1), dtype=torch.long, device=self.device).view(-1, 1)
        self.env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device).unsqueeze(-1)

    def _reset_buffers(self, env_ids):
        for key, buffer in self.log_data.items():
            buffer[env_ids, ...] = 0.
        self.done[env_ids, ...] = 0.
        self.env_step_counter[env_ids, ...] = 0.

    def _save_batch_trajectories(self, data):
        q_id = np.random.randint(0, self.num_workers)
        while True:
            try:
                self.q_s[q_id].put(data)
                break
            except mp.queues.Full:
                q_id = (q_id + 1) % self.num_workers

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
            save_env_ids = dones.squeeze(1)
            if save_trajectory:
                self.pbar.update(len(dones))
                self.trajectory_ctr += len(dones)
                for save_env_id in save_env_ids:
                    batch_data = {key: self.log_data[key][save_env_id, ...].clone().cpu() for key in self.log_data}
                    batch_data['done'] = self.done[save_env_id, ...].clone().cpu()
                    self._save_batch_trajectories(batch_data)
            self._reset_buffers(save_env_ids)
            if save_trajectory:
                if self.trajectory_ctr >= self.total_trajectories:
                    self.pbar.close()
                    self._shutdown_workers()
                    print('Data collection finished!')
                    # exit()

    def worker(self, q, q_idx):
        data_path = os.path.join(self.dir, f'{q_idx}')
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        try:
            while True:
                try:
                    item = q.get()
                except mp.queues.Empty:
                    continue
                if item is None:
                    print(f"Worker {q_idx} received shutdown signal.")
                    break
                # save the data
                for k, v in item.items():
                    if isinstance(item[k], torch.Tensor):
                        item[k] = item[k].numpy()
                np.savez_compressed(os.path.join(data_path, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.npz'),
                                    **item)
                q.task_done()
        except (KeyboardInterrupt, EOFError) as e:
            print(f"Exception {e} detected in worker {q_idx}. Exiting gracefully.")
        except Exception as e:
            print(f"Unexpected error in worker {q_idx}: {e}")
        finally:
            q.task_done()

    def _shutdown_workers(self):
        print("Shutting down workers...")
        for q in self.q_s:
            q.put(None)
        for worker in self.workers:
            worker.join()
            worker.terminate()
        print("All workers shut down successfully.")

    def get_data(self):
        return self.log_data

    def reset(self):
        self._reset_buffers(torch.arange(self.num_envs, dtype=torch.long, device=self.device).unsqueeze(-1))


class SimLogger():

    def __init__(self, env):
        self.env = env
        self.gt_contact = env.cfg_task.env.compute_contact_gt

        log_items = {
            # 'contacts_shape': env.gt_extrinsic_contact.shape[-1],
            'arm_joints_shape': env.arm_dof_pos.shape[-1],
            'eef_pos_shape': env.fingertip_centered_pos.size()[-1] + ROT_MAT_SIZE,
            'socket_pos_shape': env.socket_pos.size()[-1] + ROT_MAT_SIZE,
            'noisy_socket_pos_shape': env.socket_pos.size()[-1] + ROT_MAT_SIZE,
            'plug_pos_shape': env.plug_pos.size()[-1] + ROT_MAT_SIZE,
            'action_shape': env.cfg_task.env.numActions,
            'target_shape': env.cfg_task.env.numTargets,
            'rigid_physics_params_shape': env.rigid_physics_params.shape[-1],
            'plug_hand_pos_shape': env.plug_hand_pos.shape[-1],
            'plug_hand_quat_shape': env.plug_hand_quat.shape[-1],
            'plug_pos_error_shape': env.plug_pos_error.shape[-1],
            'plug_quat_error_shape': env.plug_quat_error.shape[-1],
            'finger_normalized_forces_shape': env.finger_normalized_forces.shape[-1],
            'plug_heights_shape': env.plug_heights.shape[-1],
            'obs_hist_shape': env.obs_queue.shape[-1],
            'obs_hist_stud_shape': env.obs_student_buf.shape[-1],
            'priv_obs_shape': env.states_buf.shape[-1],
            'hand_joints_shape': env.hand_joints.shape[-1],
            'latent_shape': env.cfg_ppo.network.priv_mlp.units[-1],
        }

        if self.env.external_cam:
            log_items.update({
                'img_shape': env.image_buf.shape[1:],
                'seg_shape': env.seg_buf.shape[1:],
            })
        if self.env.cfg_task.env.tactile:
            log_items.update({
                'tactile_shape': env.tactile_imgs.shape[1:],
            })
        if self.env.cfg_task.env.compute_contact_gt:
            log_items.update({
                'contact_latent_shape': 4,
            })

        # log_folder = env.cfg_task.data_logger.base_folder
        log_folder = os.path.expanduser("~")
        log_folder = os.path.join(log_folder, 'tactile_insertion')

        self.data_logger_init = lambda x: DataLoggerSim(env.num_envs,
                                                        env.max_episode_length,
                                                        env.device,
                                                        os.path.join(log_folder,
                                                                     env.cfg_task.data_logger.sub_folder),
                                                        env.cfg_task.data_logger.total_trajectories,
                                                        save_trajectory=env.cfg_task.data_logger.collect_data,
                                                        **log_items)

        self.data_logger = None

    def log_trajectory_data(self, action, latent, done, save_trajectory=True):

        eef_pos = torch.cat(self.env.pose_world_to_robot_base(self.env.fingertip_centered_pos.clone(),
                                                              self.env.fingertip_centered_quat.clone()), dim=-1)
        plug_pos = torch.cat(self.env.pose_world_to_robot_base(self.env.plug_pos.clone(),
                                                               self.env.plug_quat.clone()), dim=-1)
        socket_pos = torch.cat(self.env.pose_world_to_robot_base(self.env.socket_pos.clone(),
                                                                 self.env.socket_quat.clone()), dim=-1)
        noisy_socket_pos = torch.cat(self.env.pose_world_to_robot_base(self.env.noisy_gripper_goal_pos.clone(),
                                                                       self.env.noisy_gripper_goal_quat.clone()),
                                     dim=-1)

        new_action = None
        if action is not None:
            new_action = action.clone()
        new_done = None
        if done is not None:
            new_done = done.clone()

        log_data = {
            # 'contacts': self.env.gt_extrinsic_contact.clone(),
            'arm_joints': self.env.arm_dof_pos.clone(),
            'eef_pos': eef_pos,
            'socket_pos': socket_pos,
            'noisy_socket_pos': noisy_socket_pos,
            'plug_pos': plug_pos,
            'action': new_action,
            'target': self.env.targets.clone(),
            'rigid_physics_params': self.env.rigid_physics_params.clone(),
            'plug_hand_pos': self.env.plug_hand_pos.clone(),
            'plug_hand_quat': self.env.plug_hand_quat.clone(),
            'plug_pos_error': self.env.plug_pos_error.clone(),
            'plug_quat_error': self.env.plug_quat_error.clone(),
            'finger_normalized_forces': self.env.finger_normalized_forces.clone(),
            'plug_heights': self.env.plug_heights.clone(),
            'obs_hist': self.env.obs_buf.clone(),
            'obs_hist_stud': self.env.obs_student_buf.clone(),
            'priv_obs': self.env.states_buf.clone(),
            'done': new_done,
            'hand_joints': self.env.hand_joints.clone(),
        }

        if self.env.external_cam:
            log_data.update({
                'img': self.env.image_buf.clone(),
                'seg': self.env.seg_buf.clone(),
            })
        if self.env.cfg_task.env.tactile:
            log_data.update({
                'tactile': self.env.tactile_imgs.clone(),
            })

        if latent is not None:
            new_latent = latent.clone()
            log_data.update({
                'latent': new_latent
            })

            if self.env.cfg_task.env.compute_contact_gt:
                log_data.update({
                    'contact_latent': new_latent[:, 11:15]
                })

        self.data_logger.update(save_trajectory=save_trajectory, **log_data)


class RealLogger():

    def __init__(self, env):

        self.env = env

        self.with_zed = self.env.deploy_config.env.depth_cam
        self.with_pcl = self.env.deploy_config.env.pcl
        self.with_tactile = self.env.deploy_config.env.tactile
        self.with_ext_cam = self.env.deploy_config.env.ext_cam
        self.with_hand = self.env.deploy_config.env.hand
        self.with_arm = self.env.deploy_config.env.arm

        log_items = {
            'arm_joints_shape': NUM_JOINTS,
            'eef_pos_shape': POS_SIZE + QUAT_SIZE,
            'socket_pos_shape': POS_SIZE + QUAT_SIZE,
            'noisy_socket_pos_shape': POS_SIZE + QUAT_SIZE,
            'action_shape': ACT_SIZE,
            'target_shape': ACT_SIZE,
            'obs_hist_shape': env.full_config.task.env.numObservations,
            'obs_hist_stud_shape': env.full_config.task.env.numObsStudent,
            'contact_shape': env.full_config.task.env.num_points,
            # 'latent_shape': 8,  # env.deploy_config.network.merge_mlp.units[-1],
            'ft_shape': env.ft_data.shape[-1],
            'priv_obs_shape': env.full_config.train.ppo.priv_info_dim,
            'plug_pos_error_shape': POS_SIZE + QUAT_SIZE,
            'plug_pos_shape': POS_SIZE + QUAT_SIZE,
            'plug_hand_pos_shape': POS_SIZE + QUAT_SIZE,
            'img_shape': env.image_buf.shape[1:],
            'rgb_shape': env.rgb_buf.shape[1:],
            'seg_shape': env.seg_buf.shape[1:],
            'pcl_shape': env.pcl.shape[1:],
            'tactile_shape': env.tactile_record_imgs.shape[1:],
        }

        log_folder = env.deploy_config.data_logger.base_folder

        self.data_logger_init = lambda x: DataLoggerReal(env.num_envs,
                                                         env.max_episode_length,
                                                         env.device,
                                                         os.path.join(log_folder,
                                                                      env.deploy_config.data_logger.sub_folder),
                                                         env.deploy_config.data_logger.total_trajectories,
                                                         save_trajectory=env.deploy_config.data_logger.collect_data,
                                                         **log_items)

        self.data_logger = None

    def log_trajectory_data(self, action, latent, done, save_trajectory=True):
        # Pull everything from the env

        eef_pos = torch.cat((self.env.fingertip_centered_pos.clone(),
                             self.env.fingertip_centered_quat.clone()), dim=-1)

        socket_pos = torch.cat((self.env.socket_pos.clone(),
                                self.env.identity_quat.clone()), dim=-1)

        noisy_socket_pos = torch.cat((self.env.noisy_gripper_goal_pos.clone(),
                                      self.env.noisy_gripper_goal_quat.clone()), dim=-1)

        ft = self.env.ft_data.clone()
        # priv_obs = self.env.states_buf.clone()

        log_data = {
            'arm_joints': self.env.arm_dof_pos.clone(),
            'eef_pos': eef_pos,
            'socket_pos': socket_pos,
            'noisy_socket_pos': noisy_socket_pos,
            'action': action,
            # 'latent': latent,
            # 'priv_obs': priv_obs,
            'done': done.squeeze(0),
            'ft': ft
        }

        if self.with_ext_cam:
            plug_pos = torch.cat((self.env.plug_pos.clone(),
                                  self.env.plug_quat.clone()), dim=-1)

            plug_hand_pos = torch.cat((self.env.plug_hand_pos.clone(),
                                       self.env.plug_hand_quat.clone()), dim=-1)

            plug_pos_error = torch.cat((self.env.plug_pos_error.clone(),
                                        self.env.plug_quat_error.clone()), dim=-1)

            log_data.update({
                'plug_pos': plug_pos,
                'plug_hand_pos': plug_hand_pos,
                'plug_pos_error': plug_pos_error,
            })

        if self.with_zed:
            log_data.update({
                'img': self.env.image_buf.clone(),
                'seg': self.env.seg_buf.clone(),
                'rgb': self.env.rgb_buf.clone()
            })

        if self.with_tactile:
            log_data.update({
                'tactile': self.env.tactile_record_imgs.clone()
            })

        if self.with_pcl:
            log_data.update({
                'pcl': self.env.pcl.clone()
            })
        self.data_logger.update(save_trajectory=save_trajectory, **log_data)
