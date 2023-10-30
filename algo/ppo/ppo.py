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
        action_space = self.env.action_space
        self.actions_num = action_space.shape[0]
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        print("OBS", self.obs_shape)
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
            'actor_units': self.network_config.mlp.units,
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'extrin_adapt': self.extrin_adapt,
            'priv_info_dim': self.priv_info_dim,
            'priv_info': self.priv_info,
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
            'finger_normalized_forces_shape': self.env.finger_normalized_forces.shape[-1],
            'plug_heights_shape': self.env.plug_heights.shape[-1],
            'obs_hist_shape': self.env.obs_buf.shape[-1],
            'priv_obs_shape': self.env.states_buf.shape[-1],
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

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls, grad_norms):
        self.writer.add_scalar('performance/RLTrainFPS', self.agent_steps / self.rl_train_time, self.agent_steps)
        self.writer.add_scalar('performance/EnvStepFPS', self.agent_steps / self.data_collect_time, self.agent_steps)

        self.writer.add_scalar('losses/actor_loss', torch.mean(torch.stack(a_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/bounds_loss', torch.mean(torch.stack(b_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/critic_loss', torch.mean(torch.stack(c_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/entropy', torch.mean(torch.stack(entropies)).item(), self.agent_steps)

        self.writer.add_scalar('info/last_lr', self.last_lr, self.agent_steps)
        self.writer.add_scalar('info/e_clip', self.e_clip, self.agent_steps)
        self.writer.add_scalar('info/kl', torch.mean(torch.stack(kls)).item(), self.agent_steps)
        self.writer.add_scalar("info/grad_norms", torch.mean(torch.stack(grad_norms)).item(), self.agent_steps)

        for k, v in self.extra_info.items():
            self.writer.add_scalar(f'{k}', v, self.agent_steps)

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

    def train(self):
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()
        self.agent_steps = self.batch_size

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            a_losses, c_losses, b_losses, entropies, kls, grad_norms = self.train_epoch()
            self.storage.data_dict = None

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Collect Time: {self.data_collect_time / 60:.1f} min | ' \
                          f'Train RL Time: {self.rl_train_time / 60:.1f} min | ' \
                          f'Mean Reward: {self.best_rewards:.2f}'
            print(info_string)
            self.write_stats(a_losses, c_losses, b_losses, entropies, kls, grad_norms)
            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            mean_success = self.episode_success.get_mean()
            self.writer.add_scalar('episode_rewards/step', mean_rewards, self.agent_steps)
            self.writer.add_scalar('episode_lengths/step', mean_lengths, self.agent_steps)
            self.writer.add_scalar('mean_success/step', mean_success, self.agent_steps)

            checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}M_reward_{mean_rewards:.2f}'

            if self.save_freq > 0:
                if self.epoch_num % self.save_freq == 0:
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                    self.save(os.path.join(self.nn_dir, 'last'))
            self.best_rewards = mean_rewards
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
        torch.save(weights, f'{name}.pth')

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint['model'])
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])

    def restore_test(self, fn):
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])

    def log_trajectory_data(self, action, latent, done):

        eef_pos = torch.cat(self.env.pose_world_to_robot_base(self.env.fingertip_centered_pos.clone(), self.env.fingertip_centered_quat.clone()), dim=-1)
        plug_pos = torch.cat(self.env.pose_world_to_robot_base(self.env.plug_pos.clone(), self.env.plug_quat.clone()), dim=-1)
        socket_pos = torch.cat(self.env.pose_world_to_robot_base(self.env.socket_pos.clone(), self.env.socket_quat.clone()), dim=-1)
        noisy_socket_pos = torch.cat(self.env.pose_world_to_robot_base(self.env.noisy_gripper_goal_pos.clone(), self.env.noisy_gripper_goal_quat.clone()), dim=-1)
        rigid_physics_params = self.env.rigid_physics_params.clone()
        plug_socket_pos_error = self.env.plug_socket_pos_error.clone()
        plug_socket_quat_error = self.env.plug_socket_quat_error.clone()

        finger_normalized_forces = self.env.finger_normalized_forces.clone()
        plug_heights = self.env.plug_heights.clone()

        obs_hist = self.env.obs_buf.clone()
        priv_obs = self.env.states_buf.clone()

        log_data = {
            'arm_joints': self.env.arm_dof_pos,
            'eef_pos': eef_pos,
            'socket_pos': socket_pos,
            'noisy_socket_pos': noisy_socket_pos,
            'plug_pos': plug_pos,
            'action': action,
            'target': self.env.targets,
            'tactile': self.env.tactile_imgs,
            'latent': latent,
            'rigid_physics_params': rigid_physics_params,
            'plug_socket_pos_error': plug_socket_pos_error,
            'plug_socket_quat_error': plug_socket_quat_error,
            'finger_normalized_forces': finger_normalized_forces,
            'plug_heights': plug_heights,
            'obs_hist': obs_hist,
            'priv_obs': priv_obs,
            'done': done
        }

        self.data_logger.update(**log_data)

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

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += (time.time() - _t)
        # update network
        _t = time.time()
        self.set_train()
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls, grad_norms = [], [], []
        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.storage)):
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                returns, actions, obs, priv_info = self.storage[i]

                obs = self.running_mean_std(obs)
                priv_info = self.priv_mean_std(priv_info)

                batch_dict = {
                    'prev_actions': actions,
                    'obs': obs,
                    'priv_info': priv_info,
                }
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

                loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

                self.optimizer.zero_grad()
                loss.backward()
                grad_norms.append(torch.norm(
                    torch.cat([p.reshape(-1) for p in self.model.parameters()])))

                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls))
            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.last_lr
            kls.append(av_kls)

        self.rl_train_time += (time.time() - _t)
        return a_losses, c_losses, b_losses, entropies, kls, grad_norms

    # TODO move all of this logging to an utils\misc folder
    def _write_video(self, frames, output_loc, frame_rate):
        writer = imageio.get_writer(output_loc, mode='I', fps=frame_rate)
        # out = cv2.VideoWriter(output_loc, self.fourcc, frame_rate, (240, 360))
        for i in range(len(frames)):
            frame = np.uint8(frames[i])
            writer.append_data(frame)
            # cv2.imshow('frame', frames[i])
            # cv2.waitKey(0)
            # out.write(frames[i])
        writer.close()
        # out.release()
        # cv2.destroyAllWindows()

    def _write_ft(self, data, output_loc):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(np.array(data)[:, :3])
        plt.xlabel('time')
        plt.ylim([-0.25, 0.25])
        plt.ylabel('force')
        plt.savefig(f'{output_loc}_force.png')
        plt.close()
        plt.figure(figsize=(8, 6))
        plt.plot(np.array(data)[:, 3:])
        plt.xlabel('time')
        plt.ylabel('torque')
        plt.savefig(f'{output_loc}_torque.png')
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
            self._write_video(frames, f"{video_dir}/{self.it:05d}.mp4", frame_rate=30)
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

    def play_steps(self):

        for n in range(self.horizon_length):
            self.log_video()
            self.it += 1

            res_dict = self.model_act(self.obs)
            self.storage.update_data('obses', n, self.obs['obs'])
            self.storage.update_data('priv_info', n, self.obs['priv_info'])

            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])

            # do env step
            actions = torch.clamp(res_dict['actions'], -1.0, 1.0)
            self.obs, rewards, self.dones, infos = self.env.step(actions)

            rewards = rewards.unsqueeze(1)
            # update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            shaped_rewards = 0.01 * rewards.clone()
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
            self.storage.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])
            # self.episode_success.update(?) TODO..

            assert isinstance(infos, dict), 'Info Should be a Dict'
            self.extra_info = {}
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']

        self.agent_steps += self.batch_size
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
