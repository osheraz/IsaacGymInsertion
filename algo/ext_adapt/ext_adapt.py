# --------------------------------------------------------
# now its our turn.
# https://arxiv.org/abs/todo
# Copyright (c) 2024 Osher & Co.
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
import cv2
import torch
import numpy as np
from termcolor import cprint

from algo.ppo.experience import StudentBuffer
from algo.models.transformer.runner import Runner as Student
from algo.models.models_split import ActorCriticSplit as ActorCritic
from algo.models.running_mean_std import RunningMeanStd
from isaacgyminsertion.utils.misc import AverageScalarMeter
from isaacgyminsertion.utils.misc import add_to_fifo, multi_gpu_aggregate_stats

from tensorboardX import SummaryWriter
import torch.distributed as dist
import imageio
import pickle
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import json


# torch.autograd.set_detect_anomaly(True)

def log_test_result(best_loss, cur_loss, best_reward, cur_reward, steps, success_rate, log_file='test_results.yaml'):
    def convert_to_serializable(val):
        if isinstance(val, torch.Tensor):
            return val.item()
        return val

    log_data = {
        'best_loss': convert_to_serializable(best_loss),
        'cur_loss': convert_to_serializable(cur_loss),
        'best_reward': convert_to_serializable(best_reward),
        'cur_reward': convert_to_serializable(cur_reward),
        'steps': convert_to_serializable(steps),
        'success_rate': convert_to_serializable(success_rate),
        'timestamp': datetime.now().isoformat()
    }

    # Check if the log file exists
    if os.path.exists(log_file):
        # Load existing log
        with open(log_file, 'r') as f:
            if log_file.endswith('.yaml'):
                existing_data = yaml.safe_load(f) or []
            else:  # assuming json
                existing_data = json.load(f)
    else:
        existing_data = []

    # Append new log entry
    existing_data.append(log_data)

    # Save the updated log
    with open(log_file, 'w') as f:
        if log_file.endswith('.yaml'):
            yaml.dump(existing_data, f)
        else:  # assuming json
            json.dump(existing_data, f, indent=4)

    # Create a figure comparing the values
    steps_list = [entry['steps'] for entry in existing_data]
    best_loss_list = [entry['best_loss'] for entry in existing_data]
    cur_loss_list = [entry['cur_loss'] for entry in existing_data]
    best_reward_list = [entry['best_reward'] for entry in existing_data]
    cur_reward_list = [entry['cur_reward'] for entry in existing_data]
    success_rate_list = [entry['success_rate'] for entry in existing_data]

    plt.figure(figsize=(10, 6))

    # Plot losses
    plt.subplot(3, 1, 1)
    plt.plot(steps_list, best_loss_list, label='Best Loss')
    plt.plot(steps_list, cur_loss_list, label='Current Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss over Steps')
    plt.legend()

    # Plot rewards and success rate
    plt.subplot(3, 1, 2)
    plt.plot(steps_list, best_reward_list, label='Best Reward')
    plt.plot(steps_list, cur_reward_list, label='Current Reward')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title('Rewards over Steps')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(steps_list, success_rate_list, label='Success Rate')
    plt.xlabel('Steps')
    plt.ylabel('Success')
    plt.title('Success Rate over Steps')
    plt.legend()

    plt.tight_layout()

    # Save the figure
    figure_path = os.path.join(os.path.dirname(log_file), 'test_results_plot.png')
    plt.savefig(figure_path)
    plt.close()

    print(f"Log and plot saved. Plot saved at: {figure_path}")


def replace_nan_with_zero(tensor):
    nan_mask = torch.isnan(tensor)
    if nan_mask.any():
        # print(f'NaN found at tensor')
        tensor[nan_mask] = 0.0
    return tensor


def display_obs(depth, seg, pcl, tactile, ax=None):

    env_id = 2

    if tactile is not None:
        tactile = tactile[env_id, 0, ...].reshape(3, 32, 64).cpu().detach().numpy()
        horizontal_stack = np.hstack([tactile[i] for i in range(3)])
        cv2.imshow('Tactile Sequence ', horizontal_stack)

    if depth is not None:
        depth = depth[env_id, 0, ...].reshape(1, 54, 96)
        dp = depth.cpu().detach().numpy()
        dp = np.transpose(dp, (1, 2, 0))
        cv2.imshow('Depth Sequence', dp)

    if seg is not None:
        seg = seg[env_id, 0, ...].reshape(1, 54, 96)
        sg = seg.cpu().detach().numpy()
        sg = np.transpose(sg, (1, 2, 0))
        cv2.imshow('Seg Sequence', sg)

    if pcl is not None:
        pcl = pcl.cpu().detach().numpy()

        N = pcl.shape[1] // 2

        ax.scatter(pcl[env_id, :N, 0],
                   pcl[env_id, :N, 1],
                   pcl[env_id, :N, 2], color='k', s=2)
        ax.scatter(pcl[env_id, N:, 0],
                   pcl[env_id, N:, 1],
                   pcl[env_id, N:, 2], color='g', s=2)
        # ax.set_xlim([0.3, 0.65])
        plt.pause(0.0001)
        ax.cla()

    cv2.waitKey(1)


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

        self.full_config = full_config
        self.task_config = full_config.task
        self.network_config = full_config.train.network
        self.train_config = full_config.offline_train
        self.ppo_config = full_config.train.ppo
        self.only_bc = self.train_config.only_bc

        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        self.obs_shape = (self.task_config.env.numObservations * self.task_config.env.numObsHist,)
        self.obs_stud_shape = (self.task_config.env.numObsStudent * self.task_config.env.numObsStudentHist,)

        self.max_agent_steps = self.ppo_config['max_agent_steps']

        self.actions_num = self.task_config.env.numActions
        # ---- Obs Info (student)----
        self.obs_info = self.ppo_config["obs_info"]
        self.tactile_info = self.ppo_config["tactile_info"]
        self.img_info = self.ppo_config["img_info"]
        self.seg_info = self.ppo_config["seg_info"]
        self.pcl_info = self.ppo_config["pcl_info"]

        self.priv_info = self.ppo_config['priv_info']
        self.priv_info_dim = self.ppo_config['priv_info_dim']
        self.gt_contacts_info = self.ppo_config['compute_contact_gt']
        self.num_contacts_points = self.ppo_config['num_points']

        # ---- Model ----
        agent_config = {
            'actor_units': self.network_config.mlp.units,
            'actions_num': self.actions_num,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'input_shape': self.obs_shape,
            'priv_info_dim': self.priv_info_dim,
            'priv_info': self.priv_info,
            "gt_contacts_info": self.gt_contacts_info,
            "only_contact": self.ppo_config['only_contact'],
            "contacts_mlp_units": self.network_config.contact_mlp.units,
            "num_contact_points": self.num_contacts_points,
            "shared_parameters": self.ppo_config.shared_parameters,
            "full_config": full_config,
            "vt_policy": False,
        }

        self.agent = ActorCritic(agent_config)
        self.agent.to(self.device)
        self.agent.eval()

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.priv_mean_std = RunningMeanStd((self.priv_info_dim,)).to(self.device)
        self.priv_mean_std.eval()

        student_cfg = self.full_config
        student_cfg.offline_train.model.use_tactile = self.tactile_info
        student_cfg.offline_train.model.use_seg = self.seg_info
        student_cfg.offline_train.model.use_lin = self.obs_info
        student_cfg.offline_train.model.use_img = self.img_info
        student_cfg.offline_train.model.use_pcl = self.pcl_info

        self.stud_obs_mean_std = RunningMeanStd(self.obs_stud_shape).to(self.device)
        self.stud_obs_mean_std.train()

        # self.pcl_channel = 3
        # if self.task_config.env.merge_goal_pcl:
        #     self.pcl_channel += 3
        # if self.task_config.env.merge_socket_pcl:
        #     self.pcl_channel += 3

        self.pcl_mean_std = RunningMeanStd((3,)).to(self.device)
        self.pcl_mean_std.train()

        self.student = Student(student_cfg)

        self.stats = None

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
        self.horizon_length = self.ppo_config['horizon_length']
        self.batch_size = self.horizon_length * self.num_actors
        self.mini_epochs_num = self.ppo_config['mini_epochs']
        self.minibatch_size = self.batch_size // self.mini_epochs_num
        assert self.batch_size % self.minibatch_size == 0

        student_shapes = {'img': self.env.img_queue.shape[1:] if self.img_info else None,
                          'seg': self.env.seg_queue.shape[1:] if self.seg_info else None,
                          'tactile': self.env.tactile_queue.shape[1:] if self.tactile_info else None,
                          'student_obs': self.obs_stud_shape[0] if self.obs_info else None,
                          'pcl': self.env.pcl_queue.shape[1:] if self.pcl_info else None}

        self.storage = StudentBuffer(self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size,
                                     self.obs_shape[0], self.actions_num, self.priv_info_dim,
                                     student_shapes, self.device)

        self.mean_eps_reward = AverageScalarMeter(window_size=100)
        self.mean_eps_length = AverageScalarMeter(window_size=100)
        self.mean_eps_success = AverageScalarMeter(window_size=100)

        self.latent_scale = self.train_config.train.latent_scale
        self.action_scale = self.train_config.train.action_scale

        self.best_rewards = -10000
        self.best_loss = 10000
        self.test_success = 0
        self.best_success = 0
        self.cur_reward = self.best_rewards
        self.cur_loss = self.best_loss
        self.agent_steps = 0

        # ---- Optim ----
        # TODO should we retrain the policy?
        for name, p in self.agent.named_parameters():
            p.requires_grad = False

        self.optim = torch.optim.Adam(self.student.model.parameters(), lr=3e-4)

        def recurse(module, indent=0):
            print("  " * indent + f"{module.__class__.__name__}:")
            for name, child in module.named_children():
                if not name.isdigit():
                    print("  " * (indent + 1) + f"{name}:")
                recurse(child, indent + 2)
            for name, param in module.named_parameters(recurse=False):
                status = " (requires_grad=True)" if param.requires_grad else " (requires_grad=False)"
                print("  " * (indent + 1) + f"{name}: {param.__class__.__name__}{status}")

        recurse(self.student.model)
        print('--------')

        # ---- Training Misc
        batch_size = self.num_actors
        self.step_reward = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        # self.step_reward = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.step_length = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.step_success = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        # ---- Rollout Videos ----
        self.it = 0
        self.log_video_every = self.env.cfg_task.env.record_video_every
        self.log_ft_every = self.env.cfg_task.env.record_ft_every

        self.last_recording_it = 0
        self.last_recording_it_ft = 0
        # ---- Data Logger ----
        if self.env.cfg_task.data_logger.collect_data:
            from algo.ppo.experience import SimLogger
            self.data_logger = SimLogger(env=self.env)

        self.display_obs = self.env.cfg_task.external_cam.display

        if self.display_obs and self.pcl_info:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
        else:
            self.ax = None

    def set_eval(self):
        self.agent.eval()
        self.running_mean_std.eval()
        self.priv_mean_std.eval()

    def set_student_eval(self):

        self.student.model.eval()
        if self.train_config.model.transformer.load_tact:
            self.student.tact.eval()
        # if not self.train_config.from_offline:
        #     if self.stud_obs_mean_std:
        #         self.stud_obs_mean_std.eval()
        # if not self.train_config.from_offline:
        #     if self.pcl_mean_std:
        #         self.pcl_mean_std.eval()

    def set_student_train(self):

        self.student.model.train()
        if self.train_config.model.transformer.load_tact:
            self.student.tact.train()
        if not self.train_config.from_offline:
            if self.stud_obs_mean_std:
                self.stud_obs_mean_std.train()
            if self.pcl_mean_std:
                self.pcl_mean_std.train()

    def update_student_alpha(self, steps, max_steps=1e6, init_alpha=0.01, final_alpha=1.0):

        alpha = min(init_alpha + (final_alpha - init_alpha) * (steps / max_steps), 1.0)
        self.student.model.alpha = alpha
        cprint(f'Update model alpha: {alpha}', 'blue', attrs=['bold'])

    def process_obs(self, obs, obj_id=2, socket_id=3, distinct=True):

        student_obs = obs['student_obs'] if self.obs_info else None
        tactile = obs['tactile'] if self.tactile_info else None
        img = obs['img'] if self.img_info else None
        seg = obs['seg'] if self.seg_info else None
        pcl = obs['pcl'] if self.pcl_info else None

        if self.seg_info:
            valid_mask = ((seg == obj_id) | (seg == socket_id)).float()
            seg = seg * valid_mask if distinct else valid_mask

            if self.img_info:
                img = img * valid_mask

            # obj_mask = (seg == obj_id).float()
            # socket_mask = (seg == socket_id).float()
            # socket_depth = (img * socket_mask).sum(dim=2, keepdim=True) /(socket_mask.sum(dim=2, keepdim=True) + 1e-6)
            # relative_depth = (img - socket_depth) * obj_mask
            # img = relative_depth * valid_mask

        if self.pcl_info:
            # [B, T, N*3] to [B, T*N*3] to [B, T*N, 3]
            if self.display_obs:
                pcl_to_display = pcl.reshape((obs['pcl'].shape[0], -1, 3))
            pcl = self.pcl_mean_std(pcl.reshape(-1, 3)).reshape((obs['pcl'].shape[0], -1, 3))

        if student_obs is not None:
            if self.stats is not None and self.train_config.from_offline:
                eef_pos = student_obs[:, :9]
                socket_pos = student_obs[:, 9:12]
                action = student_obs[:, 12:]
                eef_pos = (eef_pos - self.stats["mean"]['eef_pos_rot6d']) / self.stats["std"]['eef_pos_rot6d']
                socket_pos = (socket_pos - self.stats["mean"]["socket_pos"][:3]) / self.stats["std"]["socket_pos"][:3]
                student_obs = torch.cat([eef_pos, socket_pos, action], dim=-1)

            elif not self.train_config.from_offline:
                student_obs = self.stud_obs_mean_std(student_obs)
            else:
                assert NotImplementedError

        student_dict = {
            'student_obs': student_obs,
            'tactile': tactile,
            'img': img,
            'seg': seg,
            'pcl': pcl,
        }

        if self.display_obs:
            display_obs(img, seg, pcl_to_display, tactile, ax=self.ax)

        return student_dict

    def test_log(self, total_steps=1e9, trials_per_noise=10):
        """
        Test the model with varying levels of point cloud noise, run each configuration multiple times,
        log and plot the mean and std success rates, and plot the results.
        """

        noise_levels = np.linspace(0.00, 0.01, 10).tolist()

        # Helper: Add noise to point cloud
        def add_noise_to_pcl(pcl, noise_level):
            noise = torch.randn_like(pcl) * noise_level
            return pcl + noise

        # Helper: Save success rates log with mean and std
        def save_success_rate_log(noise_results, log_file):
            timestamp = datetime.now().isoformat()
            data = {'timestamp': timestamp, 'results': noise_results}
            with open(log_file, 'w') as f:
                json.dump(data, f, indent=4)  # Save as JSON
            print(f"Success rate log saved to: {log_file}")

        # Helper: Plot mean and std success rates
        def plot_success_rate_with_std(noise_results, plot_file):
            noise_levels = list(noise_results.keys())
            means = [noise_results[noise_level]['mean'] for noise_level in noise_levels]
            stds = [noise_results[noise_level]['std'] for noise_level in noise_levels]

            plt.figure(figsize=(8, 6))
            plt.plot(noise_levels, means, marker='o', linestyle='-', color='b', label='Mean Success Rate')
            plt.fill_between(noise_levels, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                             alpha=0.2, label='±1 Std Dev')

            plt.xlabel('Noise Level')
            plt.ylabel('Success Rate')
            plt.title('Success Rate vs. Point Cloud Noise (Mean ±1 Std)')
            plt.grid(True)
            plt.legend()
            plt.savefig(plot_file)
            plt.close()
            print(f"Success rate plot saved to: {plot_file}")

        noise_results = {}  # Store mean and std success rates for each noise level
        self.env.reset(reset_at_success=False, reset_at_fails=False)

        reset_at_success = False
        for noise_level in noise_levels:
            print(f"Testing with point cloud noise level: {noise_level}")

            success_rates = []  # Track success rates across trials

            for trial in range(trials_per_noise):
                print(f"Trial {trial + 1} for Noise Level {noise_level}")

                # Reset environment at the start of each trial
                obs_dict = self.env.reset(reset_at_success=reset_at_success, reset_at_fails=False)

                self.set_eval()
                self.set_student_eval()
                steps = 0
                num_success, total_dones = 0, 0

                while steps < self.env.max_episode_length:
                    steps += 1

                    # Process observations with noise
                    def process_obs_with_noise(obs):
                        if obs['pcl'] is not None:
                            obs['pcl'] = add_noise_to_pcl(obs['pcl'], noise_level)
                        prep_obs = self.process_obs(obs)
                        return prep_obs

                    prep_obs = process_obs_with_noise(obs_dict)

                    student_dict = {
                        'student_obs': prep_obs.get('student_obs'),
                        'tactile': prep_obs.get('tactile'),
                        'img': prep_obs.get('img'),
                        'seg': prep_obs.get('seg'),
                        'pcl': prep_obs.get('pcl'),
                    }

                    # Student model forward pass
                    latent, _ = self.student.predict(student_dict, requires_grad=False)

                    # Agent model forward pass
                    if not self.only_bc:
                        input_dict = {
                            'obs': self.running_mean_std(obs_dict['obs']),
                            'latent': latent,
                        }
                        mu, latent = self.agent.act_inference(input_dict)
                    else:
                        mu = latent

                    mu = torch.clamp(mu, -1.0, 1.0)

                    # Perform an environment step
                    obs_dict, r, done, info = self.env.step(mu)

                    # Track success if we reach the end of the episode
                    if self.env.progress_buf[0] == self.env.max_episode_length - 1 and not reset_at_success:
                        num_success = self.env.success_reset_buf[done.nonzero()].sum().item()
                        total_dones = len(done.nonzero())
                        success_rate = num_success / total_dones if total_dones > 0 else 0
                        print(f'Success Rate for Trial {trial + 1}: {success_rate}')
                        success_rates.append(success_rate)
                        break

            # Calculate and store the mean and std success rate for this noise level
            mean_success_rate = np.mean(success_rates)
            std_success_rate = np.std(success_rates)
            noise_results[noise_level] = {'mean': mean_success_rate, 'std': std_success_rate}

            print(f'Mean Success Rate for Noise {noise_level}: {mean_success_rate}, Std Dev: {std_success_rate}')

        # Define log and plot file paths
        log_where = '/home/roblab30/osher3_workspace/src/isaacgym/python/IsaacGymInsertion/isaacgyminsertion/outputs/new_comp'
        log_file = os.path.join(log_where, 'noise_success_log.json')  # Save as JSON
        plot_file = os.path.join(log_where, 'noise_success_plot.png')

        # Save results to a log file and generate a plot
        save_success_rate_log(noise_results, log_file)
        plot_success_rate_with_std(noise_results, plot_file)

        return noise_results

    def test(self, total_steps=1e9):

        save_trajectory = self.env.cfg_task.data_logger.collect_data
        if save_trajectory:
            if self.data_logger.data_logger is None:
                self.data_logger.data_logger = self.data_logger.data_logger_init(None)
            else:
                self.data_logger.data_logger.reset()

        self.set_eval()
        self.set_student_eval()

        steps = 0
        reset_at_success = False
        obs_dict = self.env.reset(reset_at_success=reset_at_success, reset_at_fails=False)
        total_dones, num_success = 0, 0

        while steps < total_steps:

            steps += 1
            # self.log_video()
            # self.it += 1

            prep_obs = self.process_obs(obs_dict)

            student_dict = {
                'student_obs': prep_obs['student_obs'] if 'student_obs' in prep_obs else None,
                'tactile': prep_obs['tactile'] if 'tactile' in prep_obs else None,
                'img': prep_obs['img'] if 'img' in prep_obs else None,
                'seg': prep_obs['seg'] if 'seg' in prep_obs else None,
                'pcl': prep_obs['pcl'] if 'pcl' in prep_obs else None,
            }

            # student pass
            latent, _ = self.student.predict(student_dict, requires_grad=False)

            if not self.only_bc:
                input_dict = {
                    'obs': self.running_mean_std(obs_dict['obs']),
                    'latent': latent,
                }

                mu, latent = self.agent.act_inference(input_dict)
            else:
                mu = latent

            mu = torch.clamp(mu, -1.0, 1.0)

            obs_dict, r, done, info = self.env.step(mu)

            if save_trajectory:
                self.data_logger.log_trajectory_data(mu, latent, done, save_trajectory=save_trajectory)

            if self.env.progress_buf[0] == self.env.max_episode_length - 1 and not reset_at_success:
                num_success = self.env.success_reset_buf[done.nonzero()].sum()
                total_dones = len(done.nonzero())
                success_rate2 = num_success / total_dones
                success_rate = torch.mean(self.env.success_reset_buf * 1.0).item()
                self.test_success = success_rate
                print('[Test] success rate:', success_rate, success_rate2)
                log_test_result(best_loss=self.best_loss,
                                cur_loss=self.cur_loss,
                                best_reward=self.best_rewards,
                                cur_reward=self.cur_reward,
                                steps=self.agent_steps,
                                success_rate=success_rate,
                                log_file=os.path.join(self.nn_dir, f'log.json'))
            else:
                num_success = self.env.test_reset_buf.sum().item()
                total_dones = len(self.env.test_reset_buf)
                success_rate = num_success / total_dones

            if (steps == self.env.max_episode_length - 1 or success_rate >= 1.0) and reset_at_success:
                self.test_success = success_rate
                print(f'[Test2] success rate: {success_rate}, max_steps: {self.env.progress_buf.max()}')
                log_test_result(best_loss=self.best_loss,
                                cur_loss=self.cur_loss,
                                best_reward=self.best_rewards,
                                cur_reward=self.cur_reward,
                                steps=self.agent_steps,
                                success_rate=success_rate,
                                log_file=os.path.join(self.nn_dir, f'log.json'))
                break

        if self.test_success > self.best_success and self.agent_steps > 1e5:
            cprint(f'saved model at {self.agent_steps} Success {self.test_success:.2f}', 'green', attrs=['bold'])
            prev_best_ckpt = os.path.join(self.nn_dir, f'best_succ_{self.best_success:.2f}.pth')
            if os.path.exists(prev_best_ckpt):
                os.remove(prev_best_ckpt)
                os.remove(prev_best_ckpt.replace('.pth', '_stud.pth'))
            self.best_success = self.test_success
            self.save(os.path.join(self.nn_dir, f'best_succ_{self.best_success :.2f}'))

        return num_success, total_dones

    def play_steps(self):

        for n in range(self.horizon_length):
            # if not self.multi_gpu or (self.multi_gpu and self.rank == 0):
            self.log_video()
            self.it += 1

            with torch.no_grad():
                # Collect samples
                n_obs = self.running_mean_std(self.obs['obs'])
                n_priv_info = self.priv_mean_std(self.obs['priv_info'])

                res_dict = self.agent.full_act({'obs': n_obs, 'priv_info': n_priv_info})

                prep_obs = self.process_obs(self.obs)

                student_dict = {
                    'student_obs': prep_obs['student_obs'] if 'student_obs' in prep_obs else None,
                    'tactile': prep_obs['tactile'] if 'tactile' in prep_obs else None,
                    'img': prep_obs['img'] if 'img' in prep_obs else None,
                    'seg': prep_obs['seg'] if 'seg' in prep_obs else None,
                    'pcl': prep_obs['pcl'] if 'pcl' in prep_obs else None,
                }

                latent, _ = self.student.predict(student_dict, requires_grad=False)

                if not self.only_bc:
                    student_actions, _ = self.agent.act_inference({
                        'obs': n_obs,
                        'latent': latent
                    })
                else:
                    student_actions = latent

            # n indicates already normalized
            if self.obs_info:
                self.storage.update_data('n_student_obs', n, student_dict['student_obs'])
            if self.seg_info:
                # num_envs, num_timesteps, C*H*W)
                self.storage.update_data('n_img', n, student_dict['img'])
                self.storage.update_data('n_seg', n, student_dict['seg'])
            if self.tactile_info:
                # num_envs, num_fingers, num_timesteps, C*H*W)
                self.storage.update_data('n_tactile', n, student_dict['tactile'])
            if self.pcl_info:
                self.storage.update_data('n_pcl', n, student_dict['pcl'].reshape(*self.env.pcl_queue.shape))

            # already normalized
            self.storage.update_data('n_obs', n, n_obs)
            self.storage.update_data('n_priv_info', n, n_priv_info)
            self.storage.update_data('latent_gt', n, res_dict['latent_gt'])
            self.storage.update_data('teacher_actions', n, res_dict['actions'])
            self.storage.update_data('student_actions', n, student_actions)

            # do env step
            if self.agent_steps < 1e6 and not self.tactile_info:
                actions = torch.clamp(res_dict['actions'], -1.0, 1.0)
            # elif self.agent_steps < 5e4 and self.tactile_info:
            #     actions = torch.clamp(res_dict['actions'], -1.0, 1.0)
            else:
                def get_beta(max_steps=3e6, start_value=1.0, end_value=0.0):
                    beta = max(end_value, start_value - (start_value - end_value) * (self.agent_steps / max_steps))
                    return beta

                beta = get_beta()
                if torch.rand(1).item() < beta:
                    actions = torch.clamp(res_dict['actions'], -1.0, 1.0)
                else:
                    actions = torch.clamp(student_actions, -1.0, 1.0)

            self.obs, rewards, self.dones, infos = self.env.step(actions)

            rewards = rewards.unsqueeze(1)
            # update dones and rewards after env step
            if self.ppo_config['value_bootstrap'] and 'time_outs' in infos:
                # bootstrapping from "the value function". (reduced variance, but almost fake reward?)
                shaped_rewards = 0.01 * rewards.clone()
                shaped_rewards += self.ppo_config['gamma'] * res_dict['values'] * infos['time_outs'].unsqueeze(
                    1).float()
            else:
                shaped_rewards = rewards.clone()
            self.storage.update_data('rewards', n, shaped_rewards)

            self.step_reward += rewards
            self.step_success += infos['successes']
            self.step_length += 1

            done_indices = self.dones.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])
            self.mean_eps_success.update(self.step_success[done_indices])

            assert isinstance(infos, dict), 'Info Should be a Dict'
            self.extra_info = {}
            for k, v in infos.items():
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            not_dones = 1.0 - self.dones.float()
            self.step_reward = self.step_reward * not_dones.unsqueeze(1)
            self.step_length = self.step_length * not_dones
            self.step_success = self.step_success * not_dones

        self.agent_steps = (
            (self.agent_steps + self.batch_size)
            if not self.multi_gpu
            else self.agent_steps + self.batch_size * self.rank_size
        )

        self.storage.prepare_training()

    def train_epoch(self):

        _t = time.time()
        self.set_student_eval()
        self.play_steps()  # collect data
        self.set_student_train()
        _last_t = time.time()
        loss_latent_fn = torch.nn.MSELoss()
        loss_action_fn = torch.nn.MSELoss()

        latent_losses, action_losses = [], []

        for _ in range(0, self.mini_epochs_num):

            for i in range(len(self.storage)):

                batched_obs = self.storage[i]

                student_dict = {
                    'student_obs': batched_obs['n_student_obs'] if 'n_student_obs' in batched_obs else None,
                    'tactile': batched_obs['n_tactile'] if 'n_tactile' in batched_obs else None,
                    'img': batched_obs['n_img'] if 'n_img' in batched_obs else None,
                    'seg': batched_obs['n_seg'] if 'n_seg' in batched_obs else None,
                    'pcl': batched_obs['n_pcl'].reshape(batched_obs['n_pcl'].shape[0], -1, 3) if 'n_pcl' in batched_obs
                    else None,
                }

                # student pass (obs already normalized)
                latent, _ = self.student.predict(student_dict, requires_grad=True)

                if not self.only_bc:
                    # act with student latent
                    mu, _ = self.agent.act_with_grad({
                        'obs': batched_obs['n_obs'],
                        'latent': latent
                    })

                    loss_latent = loss_latent_fn(latent, batched_obs['latent_gt'].detach())
                else:
                    # pure behaviour cloning
                    mu = latent
                    loss_latent = torch.zeros(1, device=self.device)

                # loss_action = loss_action_fn(torch.clamp(mu, -1, 1),
                #                              torch.clamp(batched_obs['teacher_actions'].detach(), -1, 1))
                weights = torch.ones(6, device=self.device)
                weights[2] = 0.1
                loss_action = (torch.clamp(mu, -1, 1) - torch.clamp(batched_obs['teacher_actions'].detach(), -1,
                                                                    1)) ** 2

                loss_action = torch.sum(loss_action * weights).mean()

                # squared_diff = (torch.clamp(mu, -1, 1) - torch.clamp(batched_obs['teacher_actions'].detach(), -1,
                #                                                      1)) ** 2
                # weighted_squared_diff = squared_diff * weights
                # loss_action = weighted_squared_diff.mean()

                self.optim.zero_grad()
                loss = (self.action_scale * loss_action)  # + (self.latent_scale * loss_latent)
                loss.backward()

                latent_losses.append(loss_latent)
                action_losses.append(loss_action)

                if self.multi_gpu:
                    # batch all_reduce ops: see https://github.com/entity-neural-network/incubator/pull/220
                    all_grads_list = []

                    for param in filter(lambda p: p.requires_grad, self.student.model.parameters()):
                        if param.grad is not None:
                            all_grads_list.append(param.grad.view(-1))
                    all_grads = torch.cat(all_grads_list)
                    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                    offset = 0
                    for param in filter(lambda p: p.requires_grad, self.student.model.parameters()):
                        if param.grad is not None:
                            param.grad.data.copy_(
                                all_grads[offset: offset + param.numel()].view_as(
                                    param.grad.data
                                )
                                / self.rank_size
                            )
                            offset += param.numel()

                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.student.model.parameters()), 0.5)

                self.optim.step()

                torch.cuda.empty_cache()

        return action_losses, latent_losses

    def train(self):
        _t = time.time()
        _last_t = time.time()
        update_alpha = 1e4
        test_every = 5e5 if not self.tactile_info else 5e5
        self.update_alpha_every = 0
        self.epoch_num = 0
        self.next_test_step = test_every
        reset_at_success = True
        self.obs = self.env.reset(reset_at_success=reset_at_success, reset_at_fails=True)
        self.agent_steps = (self.batch_size if not self.multi_gpu else self.batch_size * self.rank_size)
        if self.multi_gpu:
            torch.cuda.set_device(self.rank)
            print("====================> broadcasting parameters, rank:", self.rank)
            model_params = [self.student.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.student.model.load_state_dict(model_params[0])
        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            action_losses, latent_losses = self.train_epoch()
            self.storage.data_dict = None

            if self.multi_gpu:
                action_losses, latent_losses = multi_gpu_aggregate_stats([action_losses, latent_losses])
                mean_rewards, mean_lengths, mean_success = multi_gpu_aggregate_stats(
                    [torch.Tensor([self.mean_eps_length.get_mean()]).float().to(self.device),
                     torch.Tensor([self.mean_eps_length.get_mean()]).float().to(self.device),
                     torch.Tensor([self.mean_eps_success.get_mean()]).float().to(self.device), ])
                for k, v in self.extra_info.items():
                    if type(v) is not torch.Tensor:
                        v = torch.Tensor([v]).float().to(self.device)
                    self.extra_info[k] = multi_gpu_aggregate_stats(v[None].to(self.device))
            else:
                mean_rewards = self.mean_eps_reward.get_mean()
                mean_lengths = self.mean_eps_length.get_mean()
                mean_success = self.mean_eps_success.get_mean()
                a_loss = torch.mean(torch.stack(action_losses)).item()
                l_loss = torch.mean(torch.stack(latent_losses)).item()

            if not self.multi_gpu or (self.multi_gpu and self.rank == 0):
                all_fps = self.agent_steps / (time.time() - _t)
                last_fps = self.batch_size / (time.time() - _last_t)
                _last_t = time.time()
                info_string = f'ExtAdapt: Agent Steps: {int(self.agent_steps // 1e3):04}K | FPS: {all_fps:.1f} | ' \
                              f'Last FPS: {last_fps:.1f} | ' \
                              f'Best Reward: {self.best_rewards:.2f} | ' \
                              f'Cur Reward: {mean_rewards:.2f} | ' \
                              f'Best Loss: {self.best_loss:.2f} | ' \
                              f'act_loss: {a_loss:.2f} | ' \
                              f'ext_loss: {l_loss:.2f}'
                self.cur_reward = mean_rewards
                self.cur_loss = a_loss
                print(info_string)

                if self.agent_steps >= self.next_test_step:
                    cprint(f'Disabling resets and evaluating', 'blue', attrs=['bold'])
                    self.test(total_steps=self.env.cfg_task.rl.max_episode_length)
                    self.obs = self.env.reset(reset_at_success=reset_at_success, reset_at_fails=True)
                    self.set_student_train()
                    self.next_test_step += test_every
                    cprint(f'Resume training', 'blue', attrs=['bold'])
                    cprint(f'saved model at {self.agent_steps}', 'green', attrs=['bold'])
                    self.save(os.path.join(self.nn_dir, f'last'))

                if a_loss < self.best_loss and self.agent_steps > 1e5:
                    cprint(f'saved model at {self.agent_steps} Loss {a_loss:.2f}', 'green', attrs=['bold'])
                    prev_best_ckpt = os.path.join(self.nn_dir, f'best_loss_{self.best_loss:.2f}.pth')
                    if os.path.exists(prev_best_ckpt):
                        os.remove(prev_best_ckpt)
                        os.remove(prev_best_ckpt.replace('.pth', '_stud.pth'))
                    self.best_loss = a_loss
                    self.save(os.path.join(self.nn_dir, f'best_loss_{self.best_loss :.2f}'))

                if mean_rewards > self.best_rewards and self.agent_steps >= 1e5 and mean_rewards != 0.0:
                    cprint(f"save current best reward: {mean_rewards:.2f}", 'green', attrs=['bold'])
                    prev_best_ckpt = os.path.join(self.nn_dir, f"best_reward_{self.best_rewards:.2f}.pth")
                    if os.path.exists(prev_best_ckpt):
                        os.remove(prev_best_ckpt)
                        os.remove(prev_best_ckpt.replace('.pth', '_stud.pth'))
                    self.best_rewards = mean_rewards
                    self.save(os.path.join(self.nn_dir, f'best_reward_{mean_rewards:.2f}'))

                if self.tactile_info and self.agent_steps > self.update_alpha_every:
                    self.update_student_alpha(steps=self.agent_steps)
                    self.update_alpha_every += update_alpha

                self.success_rate = mean_success

        print('max steps achieved')

    def train_single(self):

        _t = time.time()
        _last_t = time.time()
        loss_latent_fn = torch.nn.MSELoss()
        loss_action_fn = torch.nn.MSELoss()

        obs_dict = self.env.reset()
        self.agent_steps += self.batch_size
        cprint('Starting to train student', 'green', attrs=['bold'])
        test_every = 1e6
        self.next_test_step = test_every
        while self.agent_steps <= 1e9:
            # self.log_video()
            # self.it += 1

            # teacher pass
            mu_gt, e_gt = self.agent.act_inference({
                'obs': self.running_mean_std(obs_dict['obs']),
                'priv_info': self.priv_mean_std(obs_dict['priv_info']),
            })

            prep_obs = self.process_obs(obs_dict)

            student_dict = {
                'student_obs': prep_obs['student_obs'] if 'student_obs' in prep_obs else None,
                'tactile': prep_obs['tactile'] if 'tactile' in prep_obs else None,
                'img': prep_obs['img'] if 'img' in prep_obs else None,
                'seg': prep_obs['seg'] if 'seg' in prep_obs else None,
                'pcl': prep_obs['pcl'] if 'pcl' in prep_obs else None,
            }

            # student pass
            latent, _ = self.student.predict(student_dict, requires_grad=True)

            if not self.only_bc:
                # act with student latent
                mu, _ = self.agent.act_with_grad({
                    'obs': self.running_mean_std(obs_dict['obs']),
                    'latent': latent
                })

                loss_latent = loss_latent_fn(latent, e_gt.detach())
            else:
                # pure behaviour cloning
                mu = latent
                loss_latent = torch.zeros(1, device=self.device)

            loss_action = loss_action_fn(mu, mu_gt.detach())

            self.optim.zero_grad()
            loss = (self.action_scale * loss_action) + (self.latent_scale * loss_latent)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.student.model.parameters()), 0.5)

            for param in filter(lambda p: p.requires_grad, self.student.model.parameters()):
                if param.grad is not None:
                    param.grad = replace_nan_with_zero(param.grad)

            self.optim.step()
            torch.cuda.empty_cache()

            mu = mu.detach()

            if self.agent_steps < 4000:
                mu = mu_gt

            mu = torch.clamp(mu, -1.0, 1.0)

            obs_dict, r, done, info = self.env.step(mu)  # online

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

            if self.agent_steps >= self.next_test_step and self.task_config.reset_at_success:
                cprint(f'Disabling resets and evaluating', 'blue', attrs=['bold'])
                self.test(total_steps=self.env.cfg_task.rl.max_episode_length)
                self.set_student_train()
                cprint(f'Resume training', 'blue', attrs=['bold'])
                self.next_test_step += test_every
                # if self.agent_steps % 1e7 == 0:
                cprint(f'saved model at {self.agent_steps}', 'green', attrs=['bold'])
                self.save(os.path.join(self.nn_dir, f'last'))

            mean_rewards = self.mean_eps_reward.get_mean()
            if mean_rewards > self.best_rewards:
                self.best_rewards = mean_rewards
                if not self.task_config.reset_at_success:
                    self.save(os.path.join(self.nn_dir, f'best'))
                    cprint('saved new best', 'green', attrs=['bold'])
            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'ExtAdapt: Agent Steps: {int(self.agent_steps // 1e3):04}K | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Best Reward: {self.best_rewards:.2f} | ' \
                          f'Cur Reward: {mean_rewards:.2f} | ' \
                          f'Cur Loss: {loss.item():.2f}  '

            cprint(info_string)

    def log_tensorboard(self):
        self.writer.add_scalar('episode_rewards/step', self.mean_eps_reward.get_mean(), self.agent_steps)
        self.writer.add_scalar('episode_lengths/step', self.mean_eps_length.get_mean(), self.agent_steps)
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f'{k}/frame', v, self.agent_steps)

    def log_wandb(self):
        # TODO
        pass

    def restore_train(self, fn, restore_student=False, phase=None):
        checkpoint = torch.load(fn)
        cprint(f'Restoring train with teacher: {fn}', 'red', attrs=['bold'])
        self.agent.load_state_dict(checkpoint['model'])
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        self.set_eval()

        if restore_student:
            stud_fn = fn.replace('stage1_nn/last.pth', 'stage2_nn/last_stud.pth')
            self.restore_student(stud_fn, from_offline=self.train_config.from_offline, phase=phase)

    def restore_test(self, fn):
        # load teacher for gt label, should act with student
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.agent.load_state_dict(checkpoint['model'])
        self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])

        stud_fn = fn.replace('stage1_nn/last.pth', 'stage2_nn/last_stud.pth')

        self.restore_student(stud_fn, from_offline=self.train_config.from_offline, phase=1)
        self.set_eval()
        self.set_student_eval()

    def restore_student(self, fn, from_offline=False, phase=None):

        if from_offline:
            if phase == 2:
                checkpoint = torch.load(fn, map_location=self.device)
                self.student.model.load_state_dict(checkpoint['student'])
                cprint(f'Restoring student from: {fn}',
                       'red', attrs=['bold'])
            else:
                checkpoint = torch.load(self.train_config.train.student_ckpt_path, map_location=self.device)
                self.student.model.load_state_dict(checkpoint)

                cprint(f'Restoring student from: {self.train_config.train.student_ckpt_path}',
                       'red', attrs=['bold'])

            if self.train_config.model.transformer.load_tact:
                self.student.load_tact_model(self.train_config.model.transformer.tact_path)

            cprint(f'Using offline stats from: {self.train_config.train.normalize_file}',
                   'red', attrs=['bold'])
            del self.stud_obs_mean_std
            self.stats = {'mean': {}, 'std': {}}
            stats = pickle.load(open(self.train_config.train.normalize_file, "rb"))
            for key in stats['mean'].keys():
                self.stats['mean'][key] = torch.tensor(stats['mean'][key], device=self.device)
                self.stats['std'][key] = torch.tensor(stats['std'][key], device=self.device)
        else:
            cprint(f'Restoring student from: {fn}', 'blue', attrs=['bold'])
            self.stats = None
            checkpoint = torch.load(fn)
            self.stud_obs_mean_std.load_state_dict(checkpoint['stud_obs_mean_std'])
            self.pcl_mean_std.load_state_dict(checkpoint['pcl_mean_std'])
            cprint(f'stud_obs_mean_std: {self.stud_obs_mean_std.running_mean}', 'green', attrs=['bold'])
            cprint(f'pcl_mean_std: {self.pcl_mean_std.running_mean}', 'green', attrs=['bold'])

            cprint(f'Non-strict student loading!', 'red', attrs=['bold'])
            self.student.model.load_state_dict(checkpoint['student'], strict=False)
            if phase == 3:
                optim_params = []
                for name, p in self.student.model.named_parameters():
                    if 'tac' in name or 'new' in name:
                        optim_params.append(p)
                    else:
                        p.requires_grad = False

                self.optim = torch.optim.Adam(optim_params, lr=1e-3, weight_decay=1e-6)
                cprint(f'phase three (only new decoder)', 'red', attrs=['bold'])
                for name, param in self.student.model.named_parameters():
                    if param.requires_grad:
                        cprint(f"Layer: {name} | Requires Grad: {param.requires_grad}", 'red', attrs=['bold'])

    def save(self, name):

        weights = {
            'model': self.agent.state_dict(),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.priv_mean_std:
            weights['priv_mean_std'] = self.priv_mean_std.state_dict()

        torch.save(weights, f'{name}.pth')

        stud_weights = {
            'student': self.student.model.state_dict()
        }
        if not self.train_config.from_offline:
            if self.stud_obs_mean_std:
                stud_weights['stud_obs_mean_std'] = self.stud_obs_mean_std.state_dict()
            if self.pcl_mean_std:
                stud_weights['pcl_mean_std'] = self.pcl_mean_std.state_dict()
        torch.save(stud_weights, f'{name}_stud.pth')

    def log_video(self):

        if self.it == 0:
            self.env.start_recording()
            self.last_recording_it = self.it
            self.env.start_recording_ft()
            self.last_recording_it_ft = self.it
            return

        frames = self.env.get_complete_frames()
        ft_frames = self.env.get_ft_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            self.env.pause_recording_ft()

            if len(frames) < 20:
                self.env.start_recording()
                self.last_recording_it = self.it
                self.env.start_recording_ft()
                self.last_recording_it_ft = self.it
                return
            video_dir = os.path.join(self.output_dir, 'videos1')
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self._write_video(frames, ft_frames, f"{video_dir}/{self.it:05d}.mp4", frame_rate=30)
            print(f"LOGGING VIDEO {self.it:05d}.mp4")

            ft_dir = os.path.join(self.output_dir, 'ft')
            if not os.path.exists(ft_dir):
                os.makedirs(ft_dir)
            self._write_ft(ft_frames, f"{ft_dir}/{self.it:05d}")
            # self.create_line_and_image_animation(frames, ft_frames, f"{ft_dir}/{self.it:05d}_line.mp4")

            self.env.start_recording()
            self.last_recording_it = self.it

            self.env.start_recording_ft()
            self.last_recording_it_ft = self.it

    def _write_video(self, frames, ft_frames, output_loc, frame_rate):
        writer = imageio.get_writer(output_loc, mode='I', fps=frame_rate)
        for i in range(len(frames)):
            frame = np.uint8(frames[i])
            x, y = 30, 100
            for item in ft_frames[i].tolist():
                cv2.putText(frame, str(round(item, 3)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            cv2.LINE_AA)
                y += 30  # Move down to the next line
            frame = np.uint8(frame)
            writer.append_data(frame)
        writer.close()

    def _write_ft(self, data, output_loc):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(np.array(data)[:, :3])
        plt.xlabel('time')
        # plt.ylim([-0.25, 0.25])
        plt.ylabel('force')
        plt.savefig(f'{output_loc}_force.png')
        plt.close()
