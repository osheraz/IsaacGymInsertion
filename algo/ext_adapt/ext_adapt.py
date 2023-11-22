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
import imageio


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
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        self.obs_shape = (self.task_config.env.numObservations * self.task_config.env.numObsHist,)
        # self.obs_shape = (self.task_config.env.numObservations,)
        self.actions_num = self.task_config.env.numActions
        # ---- Tactile Info ---
        self.tactile_info = self.ppo_config["tactile_info"]
        self.tactile_seq_length = self.network_config.tactile_decoder.tactile_seq_length
        self.mlp_tactile_info_dim = self.network_config.tactile_mlp.units[0]
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
        self.gt_contacts_info = self.ppo_config['compute_contact_gt']
        self.num_contacts_points = self.ppo_config['num_points']
        self.priv_info_embed_dim = self.network_config.priv_mlp.units[-1]
        # ---- Obs Info (student)----co
        self.obs_info = self.ppo_config["obs_info"]
        self.student_obs_input_shape = self.ppo_config['student_obs_input_shape']
        # ---- Model ----
        net_config = {
            'actor_units': self.network_config.mlp.units,
            'actions_num': self.actions_num,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'input_shape': self.obs_shape,
            'extrin_adapt': self.extrin_adapt,
            'priv_info_dim': self.priv_info_dim,
            'priv_info': self.priv_info,
            "ft_input_shape": self.ft_info_dim,
            "ft_info": self.ft_info,
            "ft_units": self.network_config.ft_mlp.units,
            "obs_units": self.network_config.obs_mlp.units,
            "obs_info": self.obs_info,
            'student_obs_input_shape': self.student_obs_input_shape,
            "gt_contacts_info": self.gt_contacts_info,
            "contacts_mlp_units": self.network_config.contact_mlp.units,
            "num_contact_points": self.num_contacts_points,

            "tactile_info": self.tactile_info,
            "mlp_tactile_input_shape": self.mlp_tactile_info_dim,
            'tactile_input_dim': self.tactile_input_dim,
            "mlp_tactile_units": self.network_config.tactile_mlp.units,
            'tactile_seq_length': self.tactile_seq_length,
            "tactile_decoder_embed_dim": self.network_config.tactile_mlp.units[0],
            "shared_parameters": self.ppo_config.shared_parameters,

            "merge_units": self.network_config.merge_mlp.units
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
        self.running_mean_std_stud = RunningMeanStd((self.student_obs_input_shape,)).to(self.device)
        self.running_mean_std_stud.train()
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
        self.mean_eps_reward = AverageScalarMeter(window_size=50)
        self.mean_eps_length = AverageScalarMeter(window_size=50)
        self.mean_eps_success = AverageScalarMeter(window_size=50)

        self.best_rewards = -10000
        self.agent_steps = 0

        # ---- Optim ----
        print('Training the following layers:')
        adapt_params = []
        for name, p in self.model.named_parameters():
            if 'tactile_decoder' in name or 'tactile_mlp' in name or 'obs_mlp' in name or 'merge_mlp' in name or 'tactile_decoder_m' in name:
                adapt_params.append(p)
                print(name)
            else:
                p.requires_grad = False

        self.optim = torch.optim.Adam(adapt_params, lr=0.001)
        # ---- Training Misc
        self.internal_counter = 0
        self.latent_loss_stat = 0
        self.loss_stat_cnt = 0
        batch_size = self.num_actors
        self.step_reward = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.step_length = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

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

    def set_eval(self):
        self.model.eval()
        self.running_mean_std.eval()
        self.priv_mean_std.eval()
        # self.ft_mean_std.eval()
        self.running_mean_std_stud.eval()

    def test(self):

        if self.env.cfg_task.data_logger.collect_data:
            if self.data_logger.data_logger is None:
                self.data_logger.data_logger = self.data_logger.data_logger_init(None)
            else:
                self.data_logger.data_logger.reset()

        self.set_eval()
        obs_dict = self.env.reset()

        while True:

            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
                'priv_info': self.priv_mean_std(obs_dict['priv_info']),
                'contacts': obs_dict['contacts'].detach(),
                'tactile_hist': obs_dict['tactile_hist'].detach(),
                'student_obs': self.running_mean_std_stud(obs_dict['student_obs'].detach()),
            }
            mu, latent = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            if self.env.cfg_task.data_logger.collect_data:
                self.data_logger.log_trajectory_data(mu, latent, done)

    def train(self):
        _t = time.time()
        _last_t = time.time()
        loss_fn = torch.nn.MSELoss()
        obs_dict = self.env.reset()
        self.agent_steps += self.batch_size
        while self.agent_steps <= 1e9:
            self.log_video()
            self.it += 1
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
                'priv_info': self.priv_mean_std(obs_dict['priv_info']),
                'student_obs': self.running_mean_std_stud(obs_dict['student_obs']),
                'tactile_hist': obs_dict['tactile_hist'],
            }
            mu, _, _, e, e_gt = self.model._actor_critic(input_dict)
            # loss = ((e - e_gt.detach()) ** 2).mean()
            loss = loss_fn(e, e_gt.detach())
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
                self.save(os.path.join(self.nn_dir, f'last'))
            if self.agent_steps % 1e4 == 0:
                self.save(os.path.join(self.nn_dir, f'{self.agent_steps // 1e4}0k'))

            mean_rewards = self.mean_eps_reward.get_mean()
            self.best_rewards = mean_rewards
            if mean_rewards > self.best_rewards:
                self.save(os.path.join(self.nn_dir, f'best'))
                cprint('saved new best')
                # self.best_rewards = mean_rewards

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Mean Best: {self.best_rewards:.2f} | ' \
                          f'GAN: {self.ppo_config.sim2real}'

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

    def restore_test(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.model.load_state_dict(checkpoint['model'])
        # self.ft_mean_std.load_state_dict(checkpoint['ft_mean_std'])
        self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        self.running_mean_std_stud.load_state_dict(checkpoint['running_mean_std_stud'])

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.running_mean_std_stud:
            weights['running_mean_std_stud'] = self.running_mean_std_stud.state_dict()
        if self.priv_mean_std:
            weights['priv_mean_std'] = self.priv_mean_std.state_dict()
        if self.ft_mean_std:
            weights['ft_mean_std'] = self.ft_mean_std.state_dict()
        torch.save(weights, f'{name}.pth')

    def log_video(self):
        if self.it == 0:
            self.env.start_recording()
            print("START RECORDING")
            self.last_recording_it = self.it
            self.env.start_recording_ft()
            print("START FT RECORDING")
            self.last_recording_it_ft = self.it
            return

        frames = self.env.get_complete_frames()
        ft_frames = self.env.get_ft_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            self.env.pause_recording_ft()

            if len(frames) < 20:
                self.env.start_recording()
                print("START RECORDING")
                self.last_recording_it = self.it
                self.env.start_recording_ft()
                print("START FT RECORDING")
                self.last_recording_it_ft = self.it
                return
            video_dir = os.path.join(self.output_dir, 'videos1')
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self._write_video(frames, f"{video_dir}/{self.it:05d}.mp4", frame_rate=30)
            print("LOGGING VIDEO")

            ft_dir = os.path.join(self.output_dir, 'ft')
            if not os.path.exists(ft_dir):
                os.makedirs(ft_dir)
            self._write_ft(ft_frames, f"{ft_dir}/{self.it:05d}")
            # self.create_line_and_image_animation(frames, ft_frames, f"{ft_dir}/{self.it:05d}_line.mp4")

            print("LOGGING FT")

            self.env.start_recording()
            print("START RECORDING")
            self.last_recording_it = self.it

            self.env.start_recording_ft()
            print("START FT RECORDING")
            self.last_recording_it_ft = self.it

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
        # todo convert it to gif with same rate as video
        # todo split into 2 plot, 1 for the fore a
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