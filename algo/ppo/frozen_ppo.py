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
# from algo.models.models import ActorCritic
from algo.models.models_split import ActorCriticSplit as ActorCritic
from algo.models.running_mean_std import RunningMeanStd

from isaacgyminsertion.utils.misc import AverageScalarMeter
from isaacgyminsertion.utils.misc import add_to_fifo, multi_gpu_aggregate_stats
from tqdm import tqdm
from tensorboardX import SummaryWriter
# import wandb
from torchvision import transforms

from scipy.spatial.transform import Rotation


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
        # self.observation_space = self.env.observation_space
        # self.obs_shape = self.observation_space.shape
        self.obs_shape = (self.task_config.env.numObservations * self.task_config.env.numObsHist,)
        # print("OBS", self.obs_shape)

        # ---- Tactile Info ---
        self.tactile_info = self.ppo_config["tactile_info"]
        self.tactile_seq_length = self.network_config.tactile_encoder.tactile_seq_length
        self.tactile_input_dim = [self.network_config.tactile_encoder.img_width,
                                  self.network_config.tactile_encoder.img_height,
                                  self.network_config.tactile_encoder.num_channels]
        if self.task_config.tactile.half_image:
            self.tactile_input_dim[0] = self.tactile_input_dim[0] // 2
        self.mlp_tactile_info_dim = self.network_config.tactile_mlp.units[0]
        self.tactile_hist_dim = (self.network_config.tactile_encoder.tactile_seq_length, 3, *self.tactile_input_dim)
        # ---- ft Info ---
        self.ft_info = self.ppo_config["ft_info"]
        self.ft_seq_length = self.ppo_config["ft_seq_length"]
        self.ft_input_dim = self.ppo_config["ft_input_dim"]
        self.ft_info_dim = self.ft_input_dim * self.ft_seq_length
        # ---- Priv Info ----
        self.priv_info = self.ppo_config['priv_info']
        self.priv_info_dim = self.ppo_config['priv_info_dim']
        self.extrin_adapt = self.ppo_config['extrin_adapt']
        self.gt_contacts_info = self.ppo_config['compute_contact_gt']
        self.only_contact = self.ppo_config['only_contact']
        self.num_contacts_points = self.ppo_config['num_points']
        self.priv_info_embed_dim = self.network_config.priv_mlp.units[-1]
        # ---- Obs Info (student)----
        self.obs_info = self.ppo_config["obs_info"]
        # ---- Hand Info (joints)----
        self.hand_info = self.ppo_config["hand_info"]
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
            "gt_contacts_info": self.gt_contacts_info,
            "only_contact": self.only_contact,
            "contacts_mlp_units": self.network_config.contact_mlp.units,
            "num_contact_points": self.num_contacts_points,
            "tactile_info": self.tactile_info,
            "mlp_tactile_input_shape": self.mlp_tactile_info_dim,
            "mlp_tactile_units": self.network_config.tactile_mlp.units,
            'tactile_input_dim': self.tactile_input_dim,
            'tactile_seq_length': self.tactile_seq_length,
            "tactile_encoder_embed_dim": self.network_config.tactile_mlp.units[0],
            "shared_parameters": self.ppo_config.shared_parameters,
            "merge_units": self.network_config.merge_mlp.units,
            "hand_mlp_units": self.network_config.hand_mlp.units,
            "hand_info": self.hand_info
            # "physics_mlp_units": self.network_config.physics_mlp.units,
            # 'body_mlp_units': self.network_config.body_mlp.units,
        }

        self.model = ActorCritic(net_config)
        self.model.to(self.device)

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.priv_mean_std = RunningMeanStd((self.priv_info_dim,)).to(self.device)

        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
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
        self.log_ft_every = self.task_config.env.record_ft_every

        self.last_recording_it = 0
        self.last_recording_it_ft = 0

        self.episode_rewards = AverageScalarMeter(2000)
        self.episode_lengths = AverageScalarMeter(2000)
        self.episode_success = AverageScalarMeter(2000)

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

        self.writer.add_scalar('losses/actor_loss', torch.mean(torch.stack(a_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/bounds_loss', torch.mean(torch.stack(b_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/critic_loss', torch.mean(torch.stack(c_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/entropy', torch.mean(torch.stack(entropies)).item(), self.agent_steps)

        self.writer.add_scalar('info/last_lr', self.last_lr, self.agent_steps)
        self.writer.add_scalar('info/e_clip', self.e_clip, self.agent_steps)
        self.writer.add_scalar('info/kl', torch.mean(torch.stack(kls)).item(), self.agent_steps)
        self.writer.add_scalar("info/grad_norms", torch.mean(torch.stack(grad_norms)).item(), self.agent_steps)
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
            # wandb.log({
            #     f'{k}': v,
            # })

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

        if 'latent' in obs_dict and obs_dict['latent'] is not None:
            input_dict['latent'] = obs_dict['latent']

        if 'contacts' in obs_dict and self.gt_contacts_info:
            input_dict['contacts'] = obs_dict['contacts']

        # if 'tactile_hist' in obs_dict and self.tactile_info:
        #     input_dict['tactile_hist'] = obs_dict['tactile_hist']

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
            a_losses, c_losses, b_losses, entropies, kls, grad_norms, returns_list = self.train_epoch()
            self.storage.data_dict = None

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Collect Time: {self.data_collect_time / 60:.1f} min | ' \
                          f'Train RL Time: {self.rl_train_time / 60:.1f} min | ' \
                          f'Priv info: {self.full_config.train.ppo.priv_info} | ' \
                          f'Extrinsic Contact: {self.full_config.task.env.compute_contact_gt}'
            #   f'Mean Reward: {self.best_rewards:.2f} | ' \
            print(info_string)
            self.write_stats(a_losses, c_losses, b_losses, entropies, kls, grad_norms, returns_list)
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

    def play_latent_step(self, obs_dict):
        processed_obs = self.running_mean_std(obs_dict['obs'])
        input_dict = {
            'obs': processed_obs,
            'latent': obs_dict['latent'],
        }
        action, latent = self.model.act_inference(input_dict)
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
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                    returns, actions, obs, priv_info, contacts = self.storage[i]

                obs = self.running_mean_std(obs)
                priv_info = self.priv_mean_std(priv_info)

                batch_dict = {
                    'prev_actions': actions,
                    'obs': obs,
                    'priv_info': priv_info,
                    'contacts': contacts,
                    # 'tactile_hist': tactile_hist
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

                if approx_kl_div > (1.5 * self.kl_threshold):
                    continue_training = False
                    print(f"Early stopping at step due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.optimizer.zero_grad()
                loss.backward()
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
            # self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.last_lr
            kls.append(av_kls)

            if not continue_training:
                break

        self.rl_train_time += (time.time() - _t)
        return a_losses, c_losses, b_losses, entropies, kls, grad_norms, returns_list

    # TODO move all of this logging to an utils\misc folder
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

    def play_steps(self):

        for n in range(self.horizon_length):
            self.log_video()
            self.it += 1

            res_dict = self.model_act(self.obs)
            self.storage.update_data('obses', n, self.obs['obs'])
            self.storage.update_data('priv_info', n, self.obs['priv_info'])
            # self.storage.update_data('tactile_hist', n, self.obs['tactile_hist'])
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

    def test(self, get_latent=None, normalize_dict=None, milestone=100):
        # this will test either the student or the teacher model.
        # (check if the student model can be tested within models)

        self.set_eval()

        action, latent, done = None, None, None
        object_ori = None

        save_trajectory = self.env.cfg_task.data_logger.collect_data
        offline_test = self.full_config.offline_training_w_env

        # convert normalizing measures to torch and add to device
        if normalize_dict is not None:
            stud_norm_dict = {
                'mean': {},
                'std': {}
            }
            for key in normalize_dict['mean'].keys():
                stud_norm_dict['mean'][key] = torch.tensor(normalize_dict['mean'][key], device=self.device)
                stud_norm_dict['std'][key] = torch.tensor(normalize_dict['std'][key], device=self.device)

        # reset all envs
        self.obs = self.env.reset()

        # logging initial data only if one of the above is true
        if save_trajectory or offline_test:
            if self.data_logger.data_logger is None:
                self.data_logger.data_logger = self.data_logger.data_logger_init(None)
            else:
                self.data_logger.data_logger.reset()
            if not save_trajectory:
                # record initial data for latent inference (not needed if recording trajectory data, TODO: check why?)
                self.data_logger.log_trajectory_data(action, latent, done, save_trajectory=save_trajectory)

        self.env_ids = torch.arange(self.env.num_envs).view(-1, 1)
        total_dones, num_success = 0, 0
        total_env_runs = self.full_config.offline_train.train.test_episodes

        while save_trajectory or (total_dones < total_env_runs):
            # log video during test
            self.log_video()
            # getting data from data logger
            latent = None
            if offline_test:
                if get_latent is not None:
                    # Making data for the latent prediction from student model
                    tactile, img, lin_input = self.get_last_student_obs(self.data_logger.data_logger.get_data(),
                                                                               stud_norm_dict)
                    self.display_tactile_images(tactile.clone())
                    # getting the latent data from the student model
                    if self.full_config.offline_train.model.transformer.full_sequence:
                        latent = get_latent(tactile, img, lin_input)[self.env_ids,
                                 self.env.progress_buf.view(-1, 1), :].squeeze(1)
                    else:
                        latent = get_latent(tactile, img, lin_input)

                    # Adding the latent to the obs_dict (if present test with student, else test with teacher)
                    # If we predict the error directly with the model
                    # original_plug_pos_error = (latent[:,:3] * stud_norm_dict["std"]["plug_pos_error"]) + \
                    #                            stud_norm_dict["mean"]["plug_pos_error"]
                    # original_plug_quat_error = (latent[:, 3:] * stud_norm_dict["std"]["plug_quat_error"]) + \
                    #                            stud_norm_dict["mean"]["plug_quat_error"]
                    #
                    # latent = torch.cat((original_plug_pos_error, original_plug_quat_error), dim=1)

            obs_dict = {
                'obs': self.running_mean_std(self.obs['obs']),
                'priv_info': self.priv_mean_std(self.obs['priv_info']),
                'contacts': self.obs['contacts'] if 'contacts' in self.obs else None,
                'latent': latent,
            }
            action, latent = self.model.act_inference(obs_dict)
            action = torch.clamp(action, -1.0, 1.0)
            self.obs, r, done, info = self.env.step(action)

            num_success += self.env.success_reset_buf[done.nonzero()].sum()

            # logging data
            if save_trajectory or offline_test:
                self.data_logger.log_trajectory_data(action, None, done, save_trajectory=save_trajectory)
                total_dones += len(done.nonzero())
                if total_dones > milestone:
                    print('[Test] success rate:', num_success / total_dones)
                    milestone += 100

        print('[LastTest] success rate:', num_success / total_dones)
        return num_success, total_dones

    def display_tactile_images(self, tactile):
        for t in range(tactile.shape[1]):  # Iterate through the sequence of images
            # Extract the images for all fingers at time step 't' and adjust dimensions for display.
            fingers_images = tactile[0, t]  # This selects all fingers at time 't', shape is [F, C, W, H]

            # Prepare the images for concatenation and display
            imgs = []
            for f in range(fingers_images.shape[0]):
                img = fingers_images[f]
                img = img.cpu().detach()
                # Move tensor to CPU and detach from its computation history
                # Inverse normalize the image
                img = transforms.functional.normalize(
                    img, [-0.5 / 0.5] * img.shape[0], [1 / 0.5] * img.shape[0], inplace=False)
                img = img.numpy()  # Convert to numpy
                img = np.transpose(img, (1, 2, 0))  # Reorder dimensions to [W, H, C]
                # Scale the image data to [0, 255] and convert to uint8
                img = (img * 255).astype(np.uint8)
                if img.shape[2] == 1:  # If grayscale, convert to BGR for consistent OpenCV display
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                imgs.append(img)

            concatenated_img = np.concatenate(imgs, axis=1)  # Concatenate images horizontally
            cv2.imshow('Tactile Sequence', concatenated_img)  # Display the concatenated image
            cv2.waitKey(1)

    def get_last_student_obs(self, data, normalize_dict):
        # cnn input
        self.to_torch = lambda x: torch.from_numpy(x).float()

        # E, T, F, C, W, H = tactile.shape
        # tactile = tactile.reshape(E, T, F*C, W, H)
        tactile = data["tactile"]
        img = data["img"]

        plug_pos_error = data["plug_pos_error"]
        plug_quat_error = data["plug_quat_error"]

        eef_pos = data['eef_pos']
        hand_joints = data['hand_joints']

        # ori = data["plug_hand_quat"].cpu().numpy().squeeze(0) + 1e-8
        # object_ori = Rotation.from_quat(ori).as_euler('xyz')
        # object_ori = np.hstack((np.sin(object_ori[:, 0:1]), np.cos(object_ori[:, 0:1]),
        #                    np.sin(object_ori[:, 1:2]), np.cos(object_ori[:, 1:2]),))
        # object_ori = self.to_torch(object_ori).unsqueeze(0)

        action = data["action"]

        if normalize_dict is not None:
            eef_pos = (eef_pos - normalize_dict["mean"]["eef_pos"]) / normalize_dict["std"]["eef_pos"]
            hand_joints = (hand_joints - normalize_dict["mean"]["hand_joints"]) / normalize_dict["std"]["hand_joints"]
            plug_pos_error = (plug_pos_error - normalize_dict["mean"]["plug_pos_error"]) / normalize_dict["std"][
                "plug_pos_error"]
            plug_quat_error = (plug_quat_error - normalize_dict["mean"]["plug_quat_error"]) / normalize_dict["std"][
                "plug_quat_error"]

        lin_input = torch.cat([
            # eef_pos,
            hand_joints,
            # action
        ], dim=-1)

        if self.full_config.offline_train.model.transformer.full_sequence:
            return tactile, img, lin_input

        sequence_length = self.full_config.offline_train.model.transformer.sequence_length
        # Adjust tactile and lin_input based on the sequence length and progress buffer
        tactile_adjusted = self.get_last_sequence(tactile, self.env.progress_buf, sequence_length)
        img_adjusted = self.get_last_sequence(img, self.env.progress_buf, sequence_length)
        lin_input_adjusted = self.get_last_sequence(lin_input, self.env.progress_buf, sequence_length)
        # ori_adjusted = self.get_last_sequence(object_ori, self.env.progress_buf, sequence_length)

        return tactile_adjusted, img_adjusted, lin_input_adjusted

    def get_last_sequence(self, input_tensor, progress_buf, sequence_length):
        """
        Adjusts the input tensor to the desired sequence length by padding with zeros if the
        actual sequence length is less than the desired, or by selecting the most recent
        entries up to the desired sequence length if the actual sequence is longer.

        Parameters:
        - input_tensor: A tensor with shape [E, T, ...], where E is the number of environments,
          and T is the temporal dimension.
        - progress_buf: A tensor or an array indicating the actual sequence length for each environment.
        - sequence_length: The desired sequence length.

        Returns:
        - A tensor adjusted to [E, sequence_length, ...].
        """

        E, _, *other_dims = input_tensor.shape
        adjusted_tensor = torch.zeros(E, sequence_length, *other_dims, device=input_tensor.device,
                                      dtype=input_tensor.dtype)

        for env_id in range(E):

            actual_seq_len = progress_buf[env_id]
            if actual_seq_len < sequence_length:
                # If the sequence is shorter, pad the beginning with zeros
                start_pad = sequence_length - actual_seq_len
                adjusted_tensor[env_id, start_pad:, :] = input_tensor[env_id, :actual_seq_len, :]
            else:
                # If the sequence is longer, take the most recent `sequence_length` entries
                adjusted_tensor[env_id, :, :] = input_tensor[env_id, actual_seq_len - sequence_length:actual_seq_len, :]

        return adjusted_tensor


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
