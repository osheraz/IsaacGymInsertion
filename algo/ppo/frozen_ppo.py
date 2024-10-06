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
import matplotlib
from termcolor import cprint
from isaacgyminsertion.tasks.factory_tactile.factory_utils import RotationTransformer

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from algo.ppo.experience import ExperienceBuffer
from algo.models.models_split import ActorCriticSplit as ActorCritic
from algo.models.running_mean_std import RunningMeanStd

from isaacgyminsertion.utils.misc import AverageScalarMeter
from isaacgyminsertion.utils.misc import add_to_fifo, multi_gpu_aggregate_stats
from tqdm import tqdm
from tensorboardX import SummaryWriter
# import wandb
import json
import yaml


def log_test_result(best_reward, cur_reward, steps, success_rate, log_file='test_results.yaml'):
    def convert_to_serializable(val):
        if isinstance(val, torch.Tensor):
            return val.item()
        return val

    log_data = {
        'best_reward': convert_to_serializable(best_reward),
        'cur_reward': convert_to_serializable(cur_reward),
        'steps': convert_to_serializable(steps),
        'success_rate': convert_to_serializable(success_rate),
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
    best_reward_list = [entry['best_reward'] for entry in existing_data]
    cur_reward_list = [entry['cur_reward'] for entry in existing_data]
    success_rate_list = [entry['success_rate'] for entry in existing_data]

    plt.figure(figsize=(10, 6))

    # Plot rewards and success rate
    plt.subplot(2, 1, 1)
    plt.plot(steps_list, best_reward_list, label='Best Reward')
    plt.plot(steps_list, cur_reward_list, label='Current Reward')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title('Rewards over Steps')
    plt.legend()

    plt.subplot(2, 1, 2)
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


class PPO(object):
    def __init__(self, env, output_dif, full_config):

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
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        self.actions_num = self.task_config.env.numActions
        self.obs_shape = (self.task_config.env.numObservations * self.task_config.env.numObsHist,)

        # ---- Tactile Info ---
        self.vt_policy = False

        # ---- Priv Info ----
        self.priv_info = self.ppo_config['priv_info']
        self.priv_info_dim = self.ppo_config['priv_info_dim']
        self.gt_contacts_info = self.ppo_config['compute_contact_gt']
        self.only_contact = self.ppo_config['only_contact']
        self.num_contacts_points = self.ppo_config['num_points']
        self.priv_info_embed_dim = self.network_config.priv_mlp.units[-1]
        # ---- Model ----
        net_config = {
            'actor_units': self.network_config.mlp.units,
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'priv_info_dim': self.priv_info_dim,
            'priv_info': self.priv_info,
            "gt_contacts_info": self.gt_contacts_info,
            "only_contact": self.only_contact,
            "contacts_mlp_units": self.network_config.contact_mlp.units,
            "num_contact_points": self.num_contacts_points,
            "shared_parameters": self.ppo_config.shared_parameters,
            "full_config": self.full_config,
            "vt_policy": self.vt_policy,
        }

        self.rot_tf = RotationTransformer(from_rep='matrix', to_rep='rotation_6d')

        self.model = ActorCritic(net_config)
        self.model.to(self.device)

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.priv_mean_std = RunningMeanStd((self.priv_info_dim,)).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)

        if self.vt_policy:
            self.obs_stud_shape = (18 * 1,)
            self.stud_obs_mean_std = RunningMeanStd(self.obs_stud_shape).to(self.device)
            self.stud_obs_mean_std.train()

        if env is not None and not full_config.offline_training:
            # ---- Output Dir ----
            self.output_dir = output_dif
            self.nn_dir = os.path.join(self.output_dir, 'stage1_nn')
            self.tb_dif = os.path.join(self.output_dir, 'stage1_tb')
            os.makedirs(self.nn_dir, exist_ok=True)
            os.makedirs(self.tb_dif, exist_ok=True)
            # ---- Tensorboard Logger ----
            self.extra_info = {}
            writer = SummaryWriter(self.tb_dif)
            self.writer = writer

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

        # ---- Rollout Videos ----
        self.it = 0
        self.log_video_every = self.task_config.env.record_video_every

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
                                        self.num_contacts_points,
                                        self.vt_policy,
                                        self.device, )

        # ---- Data Logger ----
        if env is not None and (self.env.cfg_task.data_logger.collect_data or self.full_config.offline_training_w_env):
            from algo.ppo.experience import SimLogger
            self.data_logger = SimLogger(env=self.env)

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        # get shape
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.current_success = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.best_rewards = -10000
        self.test_success = 0
        self.best_success = 0
        self.cur_reward = self.best_rewards
        self.success_rate = 0

        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0

        # ---- wandb
        # wandb.init()

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls, grad_norms, returns_list):
        self.writer.add_scalar('performance/RLTrainFPS', self.agent_steps / self.rl_train_time, self.agent_steps)
        self.writer.add_scalar('performance/EnvStepFPS', self.agent_steps / self.data_collect_time, self.agent_steps)

        if not self.multi_gpu:
            self.writer.add_scalar('losses/actor_loss', torch.mean(torch.stack(a_losses)).item(), self.agent_steps)
            self.writer.add_scalar('losses/bounds_loss', torch.mean(torch.stack(b_losses)).item(), self.agent_steps)
            self.writer.add_scalar('losses/critic_loss', torch.mean(torch.stack(c_losses)).item(), self.agent_steps)
            self.writer.add_scalar('losses/entropy', torch.mean(torch.stack(entropies)).item(), self.agent_steps)
            self.writer.add_scalar('info/kl', torch.mean(torch.stack(kls)).item(), self.agent_steps)
            self.writer.add_scalar("info/grad_norms", torch.mean(torch.stack(grad_norms)).item(), self.agent_steps)
        else:
            self.writer.add_scalar('losses/actor_loss', torch.mean(a_losses).item(), self.agent_steps)
            self.writer.add_scalar('losses/bounds_loss', torch.mean(b_losses).item(), self.agent_steps)
            self.writer.add_scalar('losses/critic_loss', torch.mean(c_losses).item(), self.agent_steps)
            self.writer.add_scalar('losses/entropy', torch.mean(entropies).item(), self.agent_steps)
            self.writer.add_scalar('info/kl', torch.mean(kls).item(), self.agent_steps)
            self.writer.add_scalar("info/grad_norms", torch.mean(grad_norms).item(), self.agent_steps)

        self.writer.add_scalar('info/last_lr', self.last_lr, self.agent_steps)
        self.writer.add_scalar('info/e_clip', self.e_clip, self.agent_steps)

        self.writer.add_scalar("info/returns_list", torch.mean(torch.stack(returns_list)).item(), self.agent_steps)

        # wandb.log({
        #     'losses/actor_loss': torch.mean(torch.stack(a_losses)).item(),
        #     'losses/bounds_loss': torch.mean(torch.stack(b_losses)).item(),
        #     'losses/critic_loss': torch.mean(torch.stack(c_losses)).item(),
        #     'losses/entropy': torch.mean(torch.stack(entropies)).item(),
        #     'info/last_lr': self.last_lr,
        #     'info/e_clip': self.e_clip,
        #     'info/kl': torch.mean(torch.stack(kls)).item(),
        #     'info/grad_norms': torch.mean(torch.stack(grad_norms)).item(),
        #     'info/returns_list': torch.mean(torch.stack(returns_list)).item(),
        #     'performance/RLTrainFPS': self.agent_steps / self.rl_train_time,
        #     'performance/EnvStepFPS': self.agent_steps / self.data_collect_time,
        # })

        for k, v in self.extra_info.items():
            self.writer.add_scalar(f'{k}', v, self.agent_steps)
        #     wandb.log({
        #         f'{k}': v,
        #     })

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
            self.priv_mean_std.eval()
            if self.vt_policy:
                self.stud_obs_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
            self.priv_mean_std.train()
            if self.vt_policy:
                self.stud_obs_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    def model_act(self, obs_dict):

        processed_obs = self.running_mean_std(obs_dict['obs'])
        processed_priv = self.priv_mean_std(obs_dict['priv_info'])

        input_dict = {
            'obs': processed_obs,
            'priv_info': processed_priv,
        }

        if 'latent' in obs_dict and obs_dict['latent'] is not None:
            input_dict['latent'] = obs_dict['latent']

        if 'contacts' in obs_dict and self.gt_contacts_info:
            input_dict['contacts'] = obs_dict['contacts']

        if 'img' in obs_dict and self.vt_policy:
            input_dict['img'] = obs_dict['img']
            input_dict['seg'] = obs_dict['seg']
            input_dict['student_obs'] = self.stud_obs_mean_std(obs_dict['student_obs'])

        res_dict = self.model.act(input_dict)
        res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict

    def train(self):
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset(reset_at_success=False, reset_at_fails=True)
        test_every = 10e6
        self.next_test_step = test_every

        self.agent_steps = (self.batch_size if not self.multi_gpu else self.batch_size * self.rank_size)
        if self.multi_gpu:
            torch.cuda.set_device(self.rank)
            print("====================> broadcasting parameters, rank:", self.rank)
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])
        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            a_losses, c_losses, b_losses, entropies, kls, grad_norms, returns_list = self.train_epoch()
            self.storage.data_dict = None
            if self.multi_gpu:
                agg_stats = multi_gpu_aggregate_stats([a_losses, b_losses, c_losses, entropies, kls, grad_norms])
                a_losses, b_losses, c_losses, entropies, kls, grad_norms = agg_stats
                mean_rewards, mean_lengths, mean_success = multi_gpu_aggregate_stats(
                    [torch.Tensor([self.episode_rewards.get_mean()]).float().to(self.device),
                     torch.Tensor([self.episode_lengths.get_mean()]).float().to(self.device),
                     torch.Tensor([self.episode_success.get_mean()]).float().to(self.device), ])
                for k, v in self.extra_info.items():
                    if type(v) is not torch.Tensor:
                        v = torch.Tensor([v]).float().to(self.device)
                    self.extra_info[k] = multi_gpu_aggregate_stats(v[None].to(self.device))
            else:
                mean_rewards = self.episode_rewards.get_mean()
                mean_lengths = self.episode_lengths.get_mean()
                mean_success = self.episode_success.get_mean()

            if not self.multi_gpu or (self.multi_gpu and self.rank == 0):
                all_fps = self.agent_steps / (time.time() - _t)
                last_fps = self.batch_size / (time.time() - _last_t)
                _last_t = time.time()
                info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                              f'Last FPS: {last_fps:.1f} | ' \
                              f'Collect Time: {self.data_collect_time / 60:.1f} min | ' \
                              f'Train RL Time: {self.rl_train_time / 60:.1f} min | ' \
                              f"Best Reward: {self.best_rewards:.2f} | " \
                              f'Cur Reward: {mean_rewards:.2f} | ' \
                              f'Priv info: {self.full_config.train.ppo.priv_info} | ' \
                              f'Ext Contact: {self.full_config.task.env.compute_contact_gt}'
                self.cur_reward = mean_rewards
                print(info_string)

                self.write_stats(a_losses, c_losses, b_losses, entropies, kls, grad_norms, returns_list)
                self.writer.add_scalar('episode_rewards/step', mean_rewards, self.agent_steps)
                self.writer.add_scalar('episode_lengths/step', mean_lengths, self.agent_steps)
                self.writer.add_scalar('mean_success/step', mean_success, self.agent_steps)

                if self.agent_steps >= self.next_test_step:
                    cprint(f'Disabling resets and evaluating', 'blue', attrs=['bold'])
                    self.test(total_steps=self.env.cfg_task.rl.max_episode_length)
                    self.obs = self.env.reset(reset_at_success=False, reset_at_fails=True)
                    self.set_train()
                    self.next_test_step += test_every
                    cprint(f'Resume training', 'blue', attrs=['bold'])
                    cprint(f'saved model at {self.agent_steps}', 'green', attrs=['bold'])
                    self.save(os.path.join(self.nn_dir, f'last'))

                # if self.save_freq > 0:
                #     if self.epoch_num % self.save_freq == 0:
                        # checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}M_reward_{mean_rewards:.2f}'
                        # self.save(os.path.join(self.nn_dir, checkpoint_name))
                        # self.save(os.path.join(self.nn_dir, 'last.pth'))
                if mean_rewards > self.best_rewards and self.agent_steps >= self.save_best_after and mean_rewards != 0.0:
                    print(f"save current best reward: {mean_rewards:.2f}")
                    prev_best_ckpt = os.path.join(self.nn_dir, f"best_reward_{self.best_rewards:.2f}.pth")
                    if os.path.exists(prev_best_ckpt):
                        os.remove(prev_best_ckpt)
                    self.best_rewards = mean_rewards
                    self.save(os.path.join(self.nn_dir, f"best_reward_{mean_rewards:.2f}"))
                self.success_rate = mean_success

        print('max steps achieved')

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.priv_mean_std:
            weights['priv_mean_std'] = self.priv_mean_std.state_dict()
        if self.value_mean_std:
            weights['value_mean_std'] = self.value_mean_std.state_dict()

        if self.vt_policy:
            if self.stud_obs_mean_std:
                weights['stud_obs_mean_std'] = self.stud_obs_mean_std.state_dict()

        torch.save(weights, f'{name}.pth')

    def restore_train(self, fn, **kwargs):
        cprint(f"Restore teacher from {fn}", 'red', attrs=['bold'])

        if not fn:
            return
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint['model'])
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        if self.vt_policy:
            self.stud_obs_mean_std.load_state_dict(checkpoint['stud_obs_mean_std'])

    def restore_test(self, fn):
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
            if self.vt_policy:
                self.stud_obs_mean_std.load_state_dict(checkpoint['stud_obs_mean_std'])

    def play_latent_step(self, obs_dict):
        processed_obs = self.running_mean_std(obs_dict['obs'])
        input_dict = {
            'obs': processed_obs,
            'latent': obs_dict['latent'],
        }
        action, latent = self.model.act_with_grad(input_dict)
        return action, latent

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()  # collect data
        self.data_collect_time += (time.time() - _t)
        # update network
        _t = time.time()
        self.set_train()
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls, grad_norms = [], [], []
        returns_list = []
        continue_training = True
        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            approx_kl_divs = []

            for i in range(len(self.storage)):

                if self.vt_policy:
                    value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                        returns, actions, obs, priv_info, contacts, img, seg, student_obs = self.storage[i]
                else:
                    value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                        returns, actions, obs, priv_info, contacts = self.storage[i]

                obs = self.running_mean_std(obs)
                priv_info = self.priv_mean_std(priv_info)

                batch_dict = {
                    'prev_actions': actions,
                    'obs': obs,
                    'priv_info': priv_info,
                    'contacts': contacts,
                }

                if self.vt_policy:
                    batch_dict['img'] = img
                    batch_dict['seg'] = seg
                    batch_dict['student_obs'] = self.stud_obs_mean_std(student_obs)

                res_dict = self.model(batch_dict)
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                a_loss = torch.max(-surr1, -surr2)
                # critic loss
                value_pred_clipped = value_preds + (values - value_preds).clamp(-self.e_clip, self.e_clip)
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                c_loss = torch.max(value_losses, value_losses_clipped)
                # bounded loss
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_max(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(-mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = 0
                a_loss, c_loss, entropy, b_loss = [torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]

                rl_loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
                loss = rl_loss

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)
                    log_ratio = action_log_probs - old_action_log_probs
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                kl = kl_dist
                ep_kls.append(kl)
                entropies.append(entropy)

                # print(returns[0], kl_dist)

                # if approx_kl_div > (1.5 * self.kl_threshold):
                #     continue_training = False
                #     print(f"Early stopping at step due to reaching max kl: {approx_kl_div:.2f}")
                #     break

                self.optimizer.zero_grad()
                loss.backward()

                if self.multi_gpu:
                    # batch all_reduce ops: see https://github.com/entity-neural-network/incubator/pull/220
                    all_grads_list = []
                    for param in self.model.parameters():
                        if param.grad is not None:
                            all_grads_list.append(param.grad.view(-1))
                    all_grads = torch.cat(all_grads_list)
                    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                    offset = 0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.data.copy_(
                                all_grads[offset: offset + param.numel()].view_as(
                                    param.grad.data
                                )
                                / self.rank_size
                            )
                            offset += param.numel()

                grad_norms.append(torch.norm(
                    torch.cat([p.reshape(-1) for p in self.model.parameters()])))

                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()

                a_losses.append(a_loss)
                c_losses.append(c_loss)
                returns_list.append(returns)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

                del loss
                del res_dict
                torch.cuda.empty_cache()

            av_kls = torch.mean(torch.stack(ep_kls))
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.rank_size
            kls.append(av_kls)

            # self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())

            if self.multi_gpu:
                lr_tensor = torch.tensor([self.last_lr], device=self.device)
                dist.broadcast(lr_tensor, 0)
                lr = lr_tensor.item()
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
            else:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.last_lr

            # if not continue_training:
            #     break

        self.rl_train_time += (time.time() - _t)
        return a_losses, c_losses, b_losses, entropies, kls, grad_norms, returns_list

    def play_steps(self):

        for n in range(self.horizon_length):
            if not self.multi_gpu or (self.multi_gpu and self.rank == 0):
                self.log_video()
                self.it += 1

            res_dict = self.model_act(self.obs)
            self.storage.update_data('obses', n, self.obs['obs'])
            self.storage.update_data('priv_info', n, self.obs['priv_info'])

            if 'img' in self.obs and self.vt_policy:
                self.storage.update_data('student_obs', n, self.obs['student_obs'])
                self.storage.update_data('img', n, self.obs['img'].squeeze())
                self.storage.update_data('seg', n, self.obs['seg'].squeeze())

            if 'contacts' in self.obs:
                self.storage.update_data('contacts', n, self.obs['contacts'])

            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])

            # do env step
            actions = torch.clamp(res_dict['actions'], -1.0, 1.0)
            self.obs, rewards, self.dones, infos = self.env.step(actions)

            rewards = rewards.unsqueeze(1)
            # update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            if self.value_bootstrap and 'time_outs' in infos:
                # bootstrapping from "the value function". (reduced variance, but almost fake reward?)
                shaped_rewards = 0.01 * rewards.clone()
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
            else:
                shaped_rewards = rewards.clone()
            self.storage.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_success += infos['successes']
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])
            self.episode_success.update(self.current_success[done_indices])

            assert isinstance(infos, dict), 'Info Should be a Dict'
            self.extra_info = {}
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            self.current_success = self.current_success * not_dones

        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']

        # self.agent_steps += self.batch_size
        self.agent_steps = (
            (self.agent_steps + self.batch_size)
            if not self.multi_gpu
            else self.agent_steps + self.batch_size * self.rank_size
        )
        self.storage.computer_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        returns = self.storage.data_dict['returns']
        values = self.storage.data_dict['values']
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns

    def test(self, milestone=100, total_steps=1e9):

        save_trajectory = self.env.cfg_task.data_logger.collect_data
        if save_trajectory:
            if self.data_logger.data_logger is None:
                self.data_logger.data_logger = self.data_logger.data_logger_init(None)
            else:
                self.data_logger.data_logger.reset()

        self.set_eval()

        steps = 0
        total_dones, num_success = 0, 0
        self.obs = self.env.reset(reset_at_success=False, reset_at_fails=False)

        while save_trajectory or (steps < total_steps):
            # log video during test
            steps += 1
            # self.log_video()
            # self.it += 1

            obs_dict = {
                'obs': self.running_mean_std(self.obs['obs']),
                'priv_info': self.priv_mean_std(self.obs['priv_info']),
            }

            action, latent = self.model.act_inference(obs_dict)
            action = torch.clamp(action, -1.0, 1.0)

            self.obs, r, done, info = self.env.step(action)

            # num_success += self.env.success_reset_buf[done.nonzero()].sum()

            # logging data
            if save_trajectory:
                self.data_logger.log_trajectory_data(action, None, done, save_trajectory=save_trajectory)
                total_dones += len(done.nonzero())
                if total_dones >= milestone:
                    print('[Test] success rate:', num_success / total_dones)
                    milestone += 100

            if self.env.progress_buf[0] == self.env.max_episode_length - 1:
                num_success = self.env.success_reset_buf[done.nonzero()].sum()
                total_dones = len(done.nonzero())
                success_rate = num_success / total_dones
                self.test_success = success_rate
                print('[Test] success rate:', success_rate)
                log_test_result(best_reward=self.best_rewards,
                                cur_reward=self.cur_reward,
                                steps=self.agent_steps,
                                success_rate=success_rate,
                                log_file=os.path.join(self.nn_dir, f'log.json'))

        if self.test_success > self.best_success and self.agent_steps > 1e5:
            cprint(f'saved model at {self.agent_steps} Success {self.test_success:.2f}', 'green', attrs=['bold'])
            prev_best_ckpt = os.path.join(self.nn_dir, f'best_succ_{self.best_success:.2f}.pth')
            if os.path.exists(prev_best_ckpt):
                os.remove(prev_best_ckpt)
            self.best_success = self.test_success
            self.save(os.path.join(self.nn_dir, f'best_succ_{self.best_success :.2f}'))

        # print('[LastTest] success rate:', num_success / total_dones)
        return num_success, total_dones

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
        plt.plot(np.array(data)[:, :])
        plt.xlabel('time')
        # plt.ylim([-0.25, 0.25])
        plt.ylabel('action')
        plt.savefig(f'{output_loc}_action.png')
        plt.close()

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


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()


# from https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py
class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr
