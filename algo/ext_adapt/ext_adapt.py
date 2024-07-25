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
import cv2
import torch
import numpy as np
from termcolor import cprint

from algo.ppo.experience import ExperienceBuffer
from algo.models.transformer.runner import Runner as Student
from algo.models.models_split import ActorCriticSplit as ActorCritic
from algo.models.running_mean_std import RunningMeanStd
from isaacgyminsertion.utils.misc import AverageScalarMeter
from tensorboardX import SummaryWriter
import torch.distributed as dist
import imageio
import pickle


# torch.autograd.set_detect_anomaly(True)

def replace_nan_with_zero(tensor):
    nan_mask = torch.isnan(tensor)
    if nan_mask.any():
        # print(f'NaN found at tensor')
        tensor[nan_mask] = 0.0
    return tensor


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

        self.actions_num = self.task_config.env.numActions
        # ---- Obs Info (student)----
        self.obs_info = self.ppo_config["obs_info"]
        self.tactile_info = self.ppo_config["tactile_info"]
        self.img_info = self.ppo_config["img_info"]
        self.seg_info = self.ppo_config["seg_info"]
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

        self.stud_obs_mean_std = RunningMeanStd((student_cfg.offline_train.model.linear.input_size,)).to(self.device)
        self.stud_obs_mean_std.train()

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
        self.batch_size = self.num_actors
        self.mean_eps_reward = AverageScalarMeter(window_size=50)
        self.mean_eps_length = AverageScalarMeter(window_size=50)
        self.mean_eps_success = AverageScalarMeter(window_size=50)

        self.latent_scale = self.train_config.train.latent_scale
        self.action_scale = self.train_config.train.action_scale

        self.best_rewards = -10000
        self.agent_steps = 0

        # ---- Optim ----
        # TODO should we retrain the policy?
        for name, p in self.agent.named_parameters():
            p.requires_grad = False

        self.optim = torch.optim.Adam(self.student.model.parameters(), lr=1e-4)

        # ---- Training Misc
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
        self.agent.eval()
        self.running_mean_std.eval()
        self.priv_mean_std.eval()

    def set_student_eval(self):

        self.student.model.eval()
        if self.train_config.model.transformer.load_tact:
            self.student.tact.eval()
        if self.stud_obs_mean_std:
            self.stud_obs_mean_std.eval()

    def process_obs(self, obs, obj_id=2, socket_id=3, display=False):

        student_obs = obs['student_obs'] if 'student_obs' in obs else None
        tactile = obs['tactile'] if 'tactile' in obs else None
        img = obs['img'] if 'img' in obs else None
        seg = obs['seg'] if 'seg' in obs else None

        seg = ((seg == obj_id) | (seg == socket_id)).float()
        img = img * seg.unsqueeze(2)

        eef_pos = student_obs[:, :9]
        socket_pos = student_obs[:, 9:]

        if student_obs is not None:
            if self.stats is not None:

                eef_pos = (eef_pos - self.stats["mean"]['eef_pos_rot6d']) / self.stats["std"]['eef_pos_rot6d']
                socket_pos = (socket_pos - self.stats["mean"]["socket_pos"][:3]) / self.stats["std"]["socket_pos"][:3]
                student_obs = torch.cat([eef_pos, socket_pos], dim=-1)
            else:
                student_obs = self.stud_obs_mean_std(student_obs)

        student_dict = {
            'student_obs': student_obs,
            'tactile': tactile,
            'img': img,
            'seg': seg,
        }

        if display:
            self.display_obs(img, seg)

        return student_dict

    def display_obs(self, depth, seg):

        depth = depth[:, -1:, ...]
        seg = seg[:, -1:, ...]

        for t in range(depth.shape[1]):  # Iterate through the sequence of images
            # Extract the images for all fingers at time step 't' and adjust dimensions for display.

            d = depth[0, t].cpu().detach().numpy()
            d = np.transpose(d, (1, 2, 0))
            cv2.imshow('Depth Sequence', d + 0.5)  # Display the concatenated image

            d = seg[t].cpu().detach().numpy()
            d = np.transpose(d, (1, 2, 0))
            cv2.imshow('Seg Sequence', d)  # Display the concatenated image

            cv2.waitKey(1)

    def test(self):

        if self.env.cfg_task.data_logger.collect_data:
            if self.data_logger.data_logger is None:
                self.data_logger.data_logger = self.data_logger.data_logger_init(None)
            else:
                self.data_logger.data_logger.reset()

        self.set_eval()
        self.set_student_eval()

        obs_dict = self.env.reset()
        total_dones, num_success = 0, 0

        while True:

            prep_obs = self.process_obs(obs_dict)

            student_dict = {
                'student_obs': prep_obs['student_obs'] if 'student_obs' in prep_obs else None,
                'tactile': prep_obs['tactile'] if 'tactile' in prep_obs else None,
                'img': prep_obs['img'] if 'img' in prep_obs else None,
                'seg': prep_obs['seg'] if 'seg' in prep_obs else None,
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

            if self.env.cfg_task.data_logger.collect_data:
                self.data_logger.log_trajectory_data(mu, latent, done)

            if self.env.progress_buf[0] == self.env.max_episode_length - 1:
                num_success += self.env.success_reset_buf[done.nonzero()].sum()
                total_dones += len(done.nonzero())
                print('[Test] success rate:', num_success / total_dones)

    def train(self):

        _t = time.time()
        _last_t = time.time()
        loss_latent_fn = torch.nn.MSELoss()
        loss_action_fn = torch.nn.MSELoss()

        obs_dict = self.env.reset()
        self.agent_steps += self.batch_size

        while self.agent_steps <= 1e9:
            self.log_video()
            self.it += 1

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
                # pure behaviour cloning output
                mu = latent
                loss_latent = torch.zeros(1, device=self.device)

            loss_action = loss_action_fn(mu, mu_gt.detach())

            self.optim.zero_grad()
            loss = (self.action_scale * loss_action) + (self.latent_scale * loss_latent)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.model.parameters(), 0.5)

            for param in self.student.model.parameters():
                if param.grad is not None:
                    param.grad = replace_nan_with_zero(param.grad)
            self.optim.step()

            mu = mu.detach()
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

            if self.agent_steps % 1e3 == 0:
                self.save(os.path.join(self.nn_dir, f'last'))
            if self.agent_steps % 1e4 == 0:
                self.save(os.path.join(self.nn_dir, f'{self.agent_steps // 1e4}0k'))

            mean_rewards = self.mean_eps_reward.get_mean()
            if mean_rewards > self.best_rewards:
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, f'best'))
                cprint('saved new best')
                # self.best_rewards = mean_rewards

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'ExtAdapt: Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Best Reward: {self.best_rewards:.2f} | ' \
                          f'Cur Reward: {mean_rewards:.2f} | ' \

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
        cprint('Restoring train with teacher')
        cprint('careful, using non-strict matching', 'red', attrs=['bold'])
        self.agent.load_state_dict(checkpoint['model'], strict=False)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        self.set_eval()

        if self.train_config.from_offline:
            stud_fn = fn.replace('stage1_nn/last.pth', 'stage2_nn/last_stud.pth')
            self.restore_student(stud_fn, from_offline=self.train_config.from_offline)

    def restore_test(self, fn):
        # load teacher for gt label, should act with student
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.agent.load_state_dict(checkpoint['model'])
        self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])

        stud_fn = fn.replace('stage1_nn/last.pth', 'stage2_nn/last_stud.pth')

        self.restore_student(stud_fn, from_offline=self.train_config.from_offline)
        self.set_eval()
        self.set_student_eval()

    def restore_student(self, fn, from_offline=False):

        if from_offline:
            cprint(f'Using offline stats from: {self.train_config.train.normalize_file}', 'red', attrs=['bold'])
            cprint(f'Restoring student from: {self.train_config.train.student_ckpt_path}', 'red', attrs=['bold'])
            checkpoint = torch.load(self.train_config.train.student_ckpt_path, map_location=self.device)
            self.student.model.load_state_dict(checkpoint)
            if self.train_config.model.transformer.load_tact:
                self.student.load_tact_model(self.train_config.model.transformer.tact_path)
            self.stats = {'mean': {}, 'std': {}}
            stats = pickle.load(open(self.train_config.train.normalize_file, "rb"))
            for key in stats['mean'].keys():
                self.stats['mean'][key] = torch.tensor(stats['mean'][key], device=self.device)
                self.stats['std'][key] = torch.tensor(stats['std'][key], device=self.device)
        else:
            cprint(f'Restoring student from: {fn}', 'blue', attrs=['bold'])
            checkpoint = torch.load(fn)
            self.stud_obs_mean_std.load_state_dict(checkpoint['stud_obs_mean_std'])
            self.student.model.load_state_dict(checkpoint['student'])


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
        if self.stud_obs_mean_std:
            stud_weights['stud_obs_mean_std'] = self.stud_obs_mean_std.state_dict()

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
