##### !/usr/bin/env /home/osher/Desktop/isaacgym/venv/bin/python3
# --------------------------------------------------------
# now its our turn.
# https://arxiv.org/abs/todo
# Copyright (c) 2023 Osher & Co.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import rospy
# from algo.models.models import ActorCritic
from algo.models.models_split import ActorCriticSplit as ActorCritic
from algo.models.running_mean_std import RunningMeanStd
import torch
import os
import hydra
import cv2
from isaacgyminsertion.utils import torch_jit_utils
import isaacgyminsertion.tasks.factory_tactile.factory_control as fc
from isaacgyminsertion.tasks.factory_tactile.factory_utils import quat2R
from time import time
import numpy as np
import threading
import omegaconf
from matplotlib import pyplot as plt

from scipy.spatial.transform import Rotation as R
import json
import trimesh
import open3d as o3d
from matplotlib import pyplot as plt
# from tf.transformations import quaternion_matrix, identity_matrix, quaternion_from_matrix
from tqdm import tqdm


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


class HardwarePlayer(object):
    def __init__(self, output_dir, full_config):

        self.num_envs = 1
        self.deploy_config = full_config.deploy
        self.full_config = full_config

        # Overwriting action scales for the controller
        self.max_episode_length = self.deploy_config.rl.max_episode_length
        self.pos_scale_deploy = self.deploy_config.rl.pos_action_scale
        self.rot_scale_deploy = self.deploy_config.rl.rot_action_scale
        self.gripper_action_scale_deploy = self.deploy_config.rl.gripper_action_scale

        self.pos_scale = self.full_config.task.rl.pos_action_scale
        self.rot_scale = self.full_config.task.rl.rot_action_scale
        self.gripper_action_scale = self.full_config.task.rl.gripper_action_scale
        self.hand_action = self.full_config.task.env.hand_action

        self.device = self.full_config["rl_device"]
        self.episode_length = torch.zeros((1, 1), device=self.device, dtype=torch.float)

        # ---- build environment ----
        self.num_observations = self.full_config.task.env.numObservations
        self.num_obs_stud = self.full_config.task.env.numObsStudent
        self.obs_shape = (self.full_config.task.env.numObservations,)
        self.obs_stud_shape = (self.full_config.task.env.numObsStudent,)

        self.num_actions = self.full_config.task.env.numActions
        self.num_targets = self.full_config.task.env.numTargets
        self.num_contact_points = self.full_config.task.env.num_points

        # ---- Tactile Info ---
        self.tactile_info = self.full_config.train.ppo.tactile_info
        self.tactile_seq_length = self.full_config.train.network.tactile_encoder.tactile_seq_length
        self.tactile_info_dim = self.full_config.train.network.tactile_mlp.units[0]
        self.mlp_tactile_info_dim = self.full_config.train.network.tactile_mlp.units[0]
        self.tactile_input_dim = (self.full_config.train.network.tactile_encoder.img_width,
                                  self.full_config.train.network.tactile_encoder.img_height,
                                  self.full_config.train.network.tactile_encoder.num_channels)

        # ---- ft Info --- currently ft isn't supported
        # self.ft_info = self.full_config.train.ppo.ft_info
        # self.ft_seq_length = self.full_config.train.ppo.ft_seq_length
        # self.ft_input_dim = self.full_config.train.ppo.ft_input_dim
        # self.ft_info_dim = self.ft_input_dim * self.ft_seq_length

        # ---- Priv Info ----
        self.priv_info = self.full_config.train.ppo.priv_info
        self.priv_info_dim = self.full_config.train.ppo.priv_info_dim
        self.extrin_adapt = self.full_config.train.ppo.extrin_adapt
        # ---- Obs Info (student)----
        self.obs_info = self.full_config.train.ppo.obs_info
        self.student_obs_input_shape = self.full_config.train.ppo.student_obs_input_shape

        self.gt_contacts_info = self.full_config.train.ppo.compute_contact_gt

        self.tact_hist_len = self.full_config.task.env.tactile_history_len
        self.img_hist_len = self.full_config.task.env.img_history_len

        self.external_cam = self.full_config.task.external_cam.external_cam
        self.res = [self.full_config.task.external_cam.cam_res.w, self.full_config.task.external_cam.cam_res.h]
        self.cam_type = self.full_config.task.external_cam.cam_type
        self.save_im = self.full_config.task.external_cam.save_im
        self.near_clip = self.full_config.task.external_cam.near_clip
        self.far_clip = self.full_config.task.external_cam.far_clip
        self.dis_noise = self.full_config.task.external_cam.dis_noise
        self.rel_act_angles = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.act_angles = torch.tensor([0., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        net_config = {
            'actor_units': self.full_config.train.network.mlp.units,
            'actions_num': self.num_actions,
            'priv_mlp_units': self.full_config.train.network.priv_mlp.units,
            'input_shape': self.obs_shape,
            'extrin_adapt': self.extrin_adapt,
            'priv_info_dim': self.priv_info_dim,
            'priv_info': self.priv_info,
            "tactile_info": self.tactile_info,
            "obs_info": self.obs_info,
            'student_obs_input_shape': self.student_obs_input_shape,
            "mlp_tactile_input_shape": self.mlp_tactile_info_dim,
            # "ft_input_shape": self.ft_info_dim,
            # "ft_info": self.ft_info,
            "mlp_tactile_units": self.full_config.train.network.tactile_mlp.units,
            "tactile_encoder_embed_dim": self.full_config.train.network.tactile_mlp.units[0],
            "ft_units": self.full_config.train.network.ft_mlp.units,
            'tactile_input_dim': self.tactile_input_dim,
            'tactile_seq_length': self.tactile_seq_length,
            "merge_units": self.full_config.train.network.merge_mlp.units,
            "obs_units": self.full_config.train.network.obs_mlp.units,
            "num_contact_points": self.num_contact_points,
            # Added contact options
            "gt_contacts_info": self.gt_contacts_info,
            "only_contact": self.full_config.train.ppo.only_contact,
            "contacts_mlp_units": self.full_config.train.network.contact_mlp.units,
            'shared_parameters': False
        }

        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.running_mean_std_stud = RunningMeanStd((self.student_obs_input_shape,)).to(self.device)
        self.running_mean_std_stud.eval()

        self.priv_mean_std = RunningMeanStd((self.priv_info_dim,)).to(self.device)
        self.priv_mean_std.eval()

        # ---- Output Dir ----
        self.output_dir = output_dir
        self.dp_dir = os.path.join(self.output_dir, 'deploy')
        os.makedirs(self.dp_dir, exist_ok=True)

        self.cfg_tactile = full_config.task.tactile

        asset_info_path = '../../../assets/factory/yaml/factory_asset_info_insertion.yaml'
        self.asset_info_insertion = hydra.compose(config_name=asset_info_path)
        self.asset_info_insertion = self.asset_info_insertion['']['']['']['']['']['']['']['']['']['assets']['factory'][
            'yaml']

        self.extrinsic_contact = None

    def restore(self, fn):
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if 'running_mean_std_stud' in checkpoint:
            self.running_mean_std_stud.load_state_dict(checkpoint['running_mean_std_stud'])
        else:
            print('Adaptation module without observation')
        if 'priv_mean_std' in checkpoint:
            print('Policy with priv info')
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        else:
            print('Policy without priv info')
        self.model.load_state_dict(checkpoint['model'])
        self.set_eval()

    def set_eval(self):
        self.model.eval()
        self.running_mean_std.eval()
        self.running_mean_std_stud.eval()
        self.priv_mean_std.eval()

    def _initialize_grasp_poses(self, gp='yellow_round_peg_2in'):

        self.initial_grasp_poses = np.load(f'initial_grasp_data/{gp}.npz')

        self.total_init_poses = self.initial_grasp_poses['socket_pos'].shape[0]
        self.init_dof_pos = torch.zeros((self.total_init_poses, 15))
        self.init_dof_pos = self.init_dof_pos[:, :7]
        dof_pos = self.initial_grasp_poses['dof_pos'][:, :7]

        from tqdm import tqdm
        print("Loading init grasping poses for:", gp)
        for i in tqdm(range(self.total_init_poses)):
            self.init_dof_pos[i] = torch.from_numpy(dof_pos[i])

    def _create_asset_info(self):

        subassembly = self.deploy_config.desired_subassemblies[0]
        components = list(self.asset_info_insertion[subassembly])
        rospy.logwarn('Parameters load for: {} --- >  {}'.format(components[0], components[1]))

        self.plug_height = self.asset_info_insertion[subassembly][components[0]]['length']
        self.socket_height = self.asset_info_insertion[subassembly][components[1]]['height']
        if any('rectangular' in sub for sub in components):
            self.plug_depth = self.asset_info_insertion[subassembly][components[0]]['width']
            self.plug_width = self.asset_info_insertion[subassembly][components[0]]['depth']
            self.socket_width = self.asset_info_insertion[subassembly][components[1]]['width']
            self.socket_depth = self.asset_info_insertion[subassembly][components[1]]['depth']
        else:
            self.plug_width = self.asset_info_insertion[subassembly][components[0]]['diameter']
            self.socket_width = self.asset_info_insertion[subassembly][components[1]]['diameter']

    def _pose_world_to_hand_base(self, pos, quat, as_matrix=True):
        """Convert pose from world frame to robot base frame."""

        info = self.env.get_info_for_control()
        ee_pose = info['ee_pose']

        self.fingertip_centered_pos = torch.tensor(ee_pose[:3], device=self.device, dtype=torch.float).unsqueeze(0)
        self.fingertip_centered_quat = torch.tensor(ee_pose[3:], device=self.device, dtype=torch.float).unsqueeze(0)

        torch_pi = torch.tensor(
            np.pi,
            dtype=torch.float32,
            device=self.device
        )
        rotation_quat_x = torch_jit_utils.quat_from_angle_axis(torch_pi, torch.tensor([1, 0, 0], dtype=torch.float32,
                                                                                      device=self.device)).repeat(
            (self.num_envs, 1))
        rotation_quat_z = torch_jit_utils.quat_from_angle_axis(-torch_pi * 0.5,
                                                               torch.tensor([0, 0, 1], dtype=torch.float32,
                                                                            device=self.device)).repeat(
            (self.num_envs, 1))

        q_rotated = torch_jit_utils.quat_mul(rotation_quat_x, self.fingertip_centered_quat.clone())
        q_rotated = torch_jit_utils.quat_mul(rotation_quat_z, q_rotated)

        robot_base_transform_inv = torch_jit_utils.tf_inverse(
            q_rotated, self.fingertip_centered_pos.clone()
        )
        quat_in_robot_base, pos_in_robot_base = torch_jit_utils.tf_combine(
            robot_base_transform_inv[0], robot_base_transform_inv[1], quat, pos
        )

        if as_matrix:
            return pos_in_robot_base, quat2R(quat_in_robot_base).reshape(1, -1)
        else:
            return pos_in_robot_base, quat_in_robot_base

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        r_prox = 1.0
        r_dist = 0.5
        self.r_act = 2.0

        max_f = 5.0  # mapping coeff between act_force to tendon tension or somthing like that
        k1 = 0.5  # spring coeff (
        k2 = 1.0  # spring coeff (harder to move the distal finger)
        d1 = d2 = 6.0  # damping coeff

        self.R = torch.tensor([[r_prox, 0., 0.],
                               [r_dist, 0., 0.],
                               [0., r_prox, 0.],
                               [0., r_dist, 0.],
                               [0., 0., r_prox],
                               [0., 0., r_dist]], device=self.device).repeat((self.num_envs, 1, 1))

        self.Q = torch.tensor([[max_f, 0., 0.],
                               [0., max_f, 0.],
                               [0., 0., max_f]], device=self.device).repeat((self.num_envs, 1, 1))

        self.K = torch.diag(torch.tensor([k1, k2, k1, k2, k1, k2], device=self.device)).repeat((self.num_envs, 1, 1))
        self.D = torch.diag(torch.tensor([d1, d2, d1, d2, d1, d2], device=self.device)).repeat((self.num_envs, 1, 1))

        self.done = torch.zeros((1, 1), device=self.device, dtype=torch.bool)
        self.inserted = torch.zeros((1, 1), device=self.device, dtype=torch.bool)

        # Gripper pointing down w.r.t the world frame
        gripper_goal_euler = torch.tensor(self.full_config.task.randomize.fingertip_midpoint_rot_initial,
                                          device=self.device).unsqueeze(0)

        self.gripper_goal_quat = torch_jit_utils.quat_from_euler_xyz(gripper_goal_euler[:, 0],
                                                                     gripper_goal_euler[:, 1],
                                                                     gripper_goal_euler[:, 2])

        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0)

        self.plug_grasp_pos_local = self.plug_height * 0.5 * torch.tensor([0.0, 0.0, 1.0],
                                                                          device=self.device).unsqueeze(0)
        self.plug_grasp_quat_local = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0)

        self.plug_tip_pos_local = self.plug_height * torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0)
        self.socket_tip_pos_local = self.socket_height * torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0)

        self.actions = torch.zeros((1, self.num_actions), device=self.device)
        self.targets = torch.zeros((1, self.full_config.task.env.numTargets), device=self.device)
        self.contacts = torch.zeros((1, self.num_contact_points), device=self.device)
        self.prev_targets = torch.zeros((1, self.full_config.task.env.numTargets), dtype=torch.float,
                                        device=self.device)

        # Keep track of history
        self.obs_queue = torch.zeros((self.num_envs,
                                      self.full_config.task.env.numObsHist * self.num_observations),
                                     dtype=torch.float, device=self.device)
        self.obs_stud_queue = torch.zeros((self.num_envs,
                                           self.full_config.task.env.numObsStudentHist * self.num_obs_stud),
                                          dtype=torch.float, device=self.device)
        self.contact_points_hist = torch.zeros((self.num_envs, self.full_config.task.env.num_points * 1),
                                               dtype=torch.float, device=self.device)

        # tactile buffers
        self.num_channels = self.cfg_tactile.encoder.num_channels
        self.width = self.cfg_tactile.encoder.width // 2 if self.cfg_tactile.half_image else self.cfg_tactile.encoder.width
        self.height = self.cfg_tactile.encoder.height

        # tactile buffers
        self.tactile_imgs = torch.zeros(
            (1, 3,  # left, right, bottom
             self.num_channels, self.width, self.height),
            device=self.device,
            dtype=torch.float,
        )

        # Way too big tensor.
        self.tactile_queue = torch.zeros(
            (1, self.tactile_seq_length, 3,  # left, right, bottom
             self.num_channels, self.width, self.height),
            device=self.device,
            dtype=torch.float,
        )

        self.img_queue = torch.zeros(
            (1, self.img_hist_len, 1 if self.cam_type == 'd' else 3, self.res[1], self.res[0]),
            device=self.device,
            dtype=torch.float,
        )
        self.image_buf = torch.zeros(1, 1 if self.cam_type == 'd' else 3, self.res[1], self.res[0]).to(
            self.device)

        self.ft_data = torch.zeros((1, 6), device=self.device, dtype=torch.float)
        # self.ft_queue = torch.zeros((1, self.ft_seq_length, 6), device=self.device, dtype=torch.float)

        self.obs_buf = torch.zeros((1, self.obs_shape[0]), device=self.device, dtype=torch.float)
        self.obs_student_buf = torch.zeros((1, self.obs_stud_shape[0]), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros((1, self.priv_info_dim), device=self.device, dtype=torch.float)

    def _set_socket_pose(self, pos):

        self.socket_pos = torch.tensor(pos, device=self.device).unsqueeze(0)

    def _set_plug_pose(self, pos):

        self.plug_pos = torch.tensor(pos, device=self.device).unsqueeze(0)

    def _update_socket_pose(self):
        """ Update the noisy estimation of the socket pos"""

        socket_obs_pos_noise = 2 * (
                torch.rand((1, 3), dtype=torch.float32, device=self.device)
                - 0.5
        )
        socket_obs_pos_noise = socket_obs_pos_noise @ torch.diag(
            torch.tensor(
                self.full_config.task.env.socket_pos_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.noisy_socket_pos = torch.zeros_like(
            self.socket_pos, dtype=torch.float32, device=self.device
        )

        self.noisy_socket_pos[:, 0] = self.socket_pos[:, 0] + socket_obs_pos_noise[:, 0]
        self.noisy_socket_pos[:, 1] = self.socket_pos[:, 1] + socket_obs_pos_noise[:, 1]
        self.noisy_socket_pos[:, 2] = self.socket_pos[:, 2] + socket_obs_pos_noise[:, 2]

        # Add observation noise to socket rot
        socket_rot_euler = torch.zeros(
            (1, 3), dtype=torch.float32, device=self.device
        )
        socket_obs_rot_noise = 2 * (
                torch.rand((1, 3), dtype=torch.float32, device=self.device)
                - 0.5
        )
        socket_obs_rot_noise = socket_obs_rot_noise @ torch.diag(
            torch.tensor(
                self.full_config.task.env.socket_rot_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        socket_obs_rot_euler = socket_rot_euler + socket_obs_rot_noise
        self.noisy_socket_quat = torch_jit_utils.quat_from_euler_xyz_deploy(
            socket_obs_rot_euler[:, 0],
            socket_obs_rot_euler[:, 1],
            socket_obs_rot_euler[:, 2],
        )

        # Compute observation noise on socket
        (
            self.noisy_gripper_goal_quat,
            self.noisy_gripper_goal_pos,
        ) = torch_jit_utils.tf_combine_deploy(
            self.noisy_socket_quat,
            self.noisy_socket_pos,
            self.gripper_goal_quat,
            self.socket_tip_pos_local,
        )

    def _update_plug_pose(self):

        plug_pos = self.env.tracker.get_obj_pos()
        plug_rpy = self.env.tracker.get_obj_rpy()

        self.plug_pos = torch.tensor(plug_pos, device=self.device, dtype=torch.float).unsqueeze(0)
        plug_rpy = torch.tensor(plug_rpy, device=self.device, dtype=torch.float).unsqueeze(0)
        self.plug_quat = torch_jit_utils.quat_from_euler_xyz_deploy(plug_rpy[:, 0], plug_rpy[:, 1], plug_rpy[:, 2])

        self.plug_pos_error, self.plug_quat_error = fc.get_pose_error_deploy(
            fingertip_midpoint_pos=self.plug_pos,
            fingertip_midpoint_quat=self.plug_quat,
            ctrl_target_fingertip_midpoint_pos=self.socket_pos,
            ctrl_target_fingertip_midpoint_quat=self.identity_quat,
            jacobian_type='geometric',
            rot_error_type='quat')

        self.plug_hand_pos, self.plug_hand_quat = self._pose_world_to_hand_base(self.plug_pos,
                                                                                self.plug_quat,
                                                                                as_matrix=False)
        if self.gt_contacts_info:
            self.contacts[0, :] = torch.tensor(self.env.tracker.extrinsic_contact).to(self.device)

    def compute_observations(self, display_image=False, with_priv=False, with_tactile=False, with_img=False):

        obses = self.env.get_obs()

        arm_joints = obses['joints']
        ee_pose = obses['ee_pose']
        ft = obses['ft']
        self.ft_data = torch.tensor(ft, device=self.device).unsqueeze(0)
        self.arm_dof_pos = torch.tensor(arm_joints, device=self.device).unsqueeze(0)
        self.fingertip_centered_pos = torch.tensor(ee_pose[:3], device=self.device, dtype=torch.float).unsqueeze(0)
        self.fingertip_centered_quat = torch.tensor(ee_pose[3:], device=self.device, dtype=torch.float).unsqueeze(0)

        # self.ft_queue[:, 1:] = self.ft_queue[:, :-1].clone().detach()
        # self.ft_queue[:, 0, :] = self.ft_data

        if with_tactile:
            left, right, bottom = obses['frames']

            # Cutting by half
            if self.cfg_tactile.half_image:
                w = left.shape[0]
                left = left[:w // 2, :, :]
                right = right[:w // 2, :, :]
                bottom = bottom[:w // 2, :, :]

            # Resizing to encoder size
            left = cv2.resize(left, (self.height, self.width), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (self.height, self.width), interpolation=cv2.INTER_AREA)
            bottom = cv2.resize(bottom, (self.height, self.width), interpolation=cv2.INTER_AREA)

            if display_image:
                cv2.imshow("Hand View\tLeft\tRight\tMiddle", np.concatenate((left, right, bottom), axis=1))
                cv2.waitKey(1)

            if self.num_channels == 3:
                self.tactile_imgs[0, 0] = to_torch(left).permute(2, 0, 1).to(self.device)
                self.tactile_imgs[0, 1] = to_torch(right).permute(2, 0, 1).to(self.device)
                self.tactile_imgs[0, 2] = to_torch(bottom).permute(2, 0, 1).to(self.device)

                # self.tactile_imgs[0, 0] = torch_jit_utils.rgb_transform(left).to(self.device)
                # self.tactile_imgs[0, 1] = torch_jit_utils.rgb_transform(right).to(self.device)
                # self.tactile_imgs[0, 2] = torch_jit_utils.rgb_transform(bottom).to(self.device)

            else:
                self.tactile_imgs[0, 0] = to_torch(
                    cv2.cvtColor(left.astype('float32'), cv2.COLOR_BGR2GRAY)).to(self.device)
                self.tactile_imgs[0, 1] = to_torch(
                    cv2.cvtColor(right.astype('float32'), cv2.COLOR_BGR2GRAY)).to(self.device)
                self.tactile_imgs[0, 2] = to_torch(
                    cv2.cvtColor(bottom.astype('float32'), cv2.COLOR_BGR2GRAY)).to(self.device)

            self.tactile_queue[:, 1:] = self.tactile_queue[:, :-1].clone().detach()
            self.tactile_queue[:, 0, :] = self.tactile_imgs

        if with_img:

            img = obses['img']
            self.image_buf[0] = to_torch(img).permute(2, 0, 1).to(self.device)
            self.img_queue[:, 1:] = self.img_queue[:, :-1].clone().detach()
            self.img_queue[:, 0, ...] = self.image_buf

            if display_image:
                cv2.imshow("Depth Image", img.transpose(1, 2, 0) + 0.5)
                cv2.waitKey(1)

        # some-like taking a new socket pose measurement
        # self._update_socket_pose()

        # eef_pos = torch.cat((self.fingertip_centered_pos,
        #                      quat2R(self.fingertip_centered_quat).reshape(1, -1)), dim=-1)

        # noisy_delta_pos = self.noisy_gripper_goal_pos - self.fingertip_centered_pos

        # obs = torch.cat([eef_pos,
        #                  self.actions,
        #                  ], dim=-1)

        obs = torch.cat([
            self.rel_act_angles,  # 3
            self.identity_quat,  # 4
            self.actions[:, -3:],  # 3
        ], dim=-1)

        self.obs_queue[:, :-self.num_observations] = self.obs_queue[:, self.num_observations:]
        self.obs_queue[:, -self.num_observations:] = obs

        obs_tensors_student = torch.cat([self.actions,  # 6
                                         # self.actions[:, -3:],  # 3
                                         ], dim=-1)

        self.obs_stud_queue[:, :-self.num_obs_stud] = self.obs_stud_queue[:, self.num_obs_stud:]
        self.obs_stud_queue[:, -self.num_obs_stud:] = obs_tensors_student

        self.obs_buf = self.obs_queue.clone()  # torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)
        self.obs_student_buf = self.obs_stud_queue.clone()  # shape = (num_envs, num_observations_student)

        obs_dict = {'obs': self.obs_buf.clone(),
                    'student_obs': self.obs_student_buf.clone()}

        if with_tactile:
            obs_dict['tactile'] = self.tactile_queue.clone()
        if with_img:
            obs_dict['img'] = self.img_queue.clone()

        if with_priv:
            # Compute privileged info (gt_error + contacts)
            self._update_plug_pose()

            state_tensors = [
                self.plug_hand_pos,  # 3
                self.plug_hand_quat,  # 4
                # self.plug_pos_error,  # 3
                # self.plug_quat_error,  # 4
            ]
            self.states_buf = torch.cat(state_tensors, dim=-1)  # shape = (num_envs, num_states)
            obs_dict['priv_info'] = self.states_buf.clone()

        return obs_dict

    def _update_reset_buf(self):

        plug_hand_xy_distance = torch.norm(self.plug_hand_pos[:, :2])

        quat_diff = torch_jit_utils.quat_mul(self.plug_quat, torch_jit_utils.quat_conjugate(self.identity_quat))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
        goal_resets = torch.abs(rot_dist) <= self.deploy_config.rl.goal_min_rot

        self.inserted = goal_resets
        is_too_far = (plug_hand_xy_distance > 0.08)

        roll, pitch, yaw = torch_jit_utils.get_euler_xyz(self.plug_quat.clone())
        roll[roll > np.pi] -= 2 * np.pi
        pitch[pitch > np.pi] -= 2 * np.pi
        is_too_rotated = (torch.abs(roll) > 0.7) | (torch.abs(pitch) > 0.7)

        timeout = (self.episode_length >= self.max_episode_length)

        self.done = is_too_far | timeout | self.inserted | is_too_rotated

        if self.done[0, 0].item():
            print('[Reset] ~',
                  "Far-away" if is_too_far[0].item() else "",
                  "Timeout" if timeout.item() else "",
                  "Inserted" if self.inserted.item() else "", )

    def reset(self):

        # start_joint_pos = self.deploy_config.common_poses.start_pos
        # Move above plug
        # self.env.move_to_joint_values(start_joint_pos, wait=True)
        self.env.release()
        self.env.arm.move_manipulator.scale_vel(scale_vel=0.1, scale_acc=0.1)
        # Align and grasp
        # input('Hand is about to grasp - please set the object')
        # Create an event to signal when input has been received
        input_event = threading.Event()

        def get_user_input(event):
            input('Hand is about to grasp - please set the object')
            event.set()  # Signal that input has been received

        # Start the user input thread
        # input_thread = threading.Thread(target=get_user_input, args=(input_event,))
        # input_thread.start()
        #
        # # Wait for the input event
        # while not input_event.is_set():
        #     rospy.sleep(0.1)  # Small sleep to avoid busy waiting

        self.env.randomize_grasp()

        # self.env.arm.move_manipulator.scale_vel(scale_vel=0.1, scale_acc=0.1)
        # self.env.set_random_init_error(self.socket_pos)

        # self.env.arm.move_manipulator.scale_vel(scale_vel=0.02, scale_acc=0.02)
        print('Starting Insertion')

        self.done[...] = False
        self.episode_length[...] = 0.
        self.env.arm.move_manipulator.scale_vel(scale_vel=0.02, scale_acc=0.02)
        self.inserted[...] = False

    def _move_arm_to_desired_pose(self, desired_pos=None, desired_rot=None, by_moveit=True):
        """Move gripper to desired pose."""

        info = self.env.get_info_for_control()
        ee_pose = info['ee_pose']

        if desired_pos is None:
            self.ctrl_target_fingertip_centered_pos = torch.tensor(ee_pose[:3], device=self.device).unsqueeze(0)
        else:
            self.ctrl_target_fingertip_centered_pos = torch.tensor(desired_pos, device=self.device).unsqueeze(0)

        if desired_rot is None:
            self.ctrl_target_fingertip_centered_quat = torch.tensor(ee_pose[3:], device=self.device).unsqueeze(0)
            # ctrl_target_fingertip_centered_euler = torch.tensor(self.full_config.task.env.fingertip_midpoint_rot_initial,
            #                                                     device=self.device).unsqueeze(0)

            # self.ctrl_target_fingertip_centered_quat = torch_jit_utils.quat_from_euler_xyz_deploy(
            #     ctrl_target_fingertip_centered_euler[:, 0],
            #     ctrl_target_fingertip_centered_euler[:, 1],
            #     ctrl_target_fingertip_centered_euler[:, 2])
        else:
            self.ctrl_target_fingertip_centered_quat = torch.tensor(desired_rot, device=self.device).unsqueeze(0)

        if by_moveit:
            pose_target = torch.cat((self.ctrl_target_fingertip_centered_pos, self.ctrl_target_fingertip_centered_quat),
                                    dim=-1).cpu().detach().numpy().squeeze().tolist()
            self.env.move_to_pose(pose_target, wait=True)

        else:
            cfg_ctrl = {'num_envs': 1,
                        'jacobian_type': 'geometric'}

            for _ in range(3):
                info = self.env.get_info_for_control()
                ee_pose = info['ee_pose']

                self.fingertip_centered_pos = torch.tensor(ee_pose[:3], device=self.device,
                                                           dtype=torch.float).unsqueeze(0)
                self.fingertip_centered_quat = torch.tensor(ee_pose[3:], device=self.device,
                                                            dtype=torch.float).unsqueeze(0)

                # Dealing with -180 + 180
                # self.ctrl_target_fingertip_centered_quat = torch.mul(torch.abs(self.ctrl_target_fingertip_centered_quat),
                #                                                      torch.sign(self.fingertip_centered_quat))

                pos_error, axis_angle_error = fc.get_pose_error_deploy(
                    fingertip_midpoint_pos=self.fingertip_centered_pos,
                    fingertip_midpoint_quat=self.fingertip_centered_quat,
                    ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_centered_pos,
                    ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_centered_quat,
                    jacobian_type=cfg_ctrl['jacobian_type'],
                    rot_error_type='axis_angle')

                delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
                actions = torch.zeros((1, self.num_actions), device=self.device)
                actions[:, :6] = delta_hand_pose
                # Apply the action, keep fingers in the same status
                self.apply_action(actions=actions, do_scale=False, do_clamp=False, wait=True, regulize_force=False,
                                  by_moveit=True, by_vel=False)

    def update_and_apply_action(self, actions, wait=True, by_moveit=True, by_vel=False):

        self.actions = actions.clone().to(self.device)
        self.actions[:, :6] = 0.

        delta_targets = torch.cat([
            self.actions[:, :3] @ torch.diag(torch.tensor(self.pos_scale, device=self.device)),  # 3
            self.actions[:, 3:6] @ torch.diag(torch.tensor(self.rot_scale, device=self.device)),  # 3
            self.actions[:, 6:9] @ torch.diag(torch.tensor(self.gripper_action_scale, device=self.device))  # 3
        ], dim=-1).clone()

        # Update targets
        self.targets = self.prev_targets + delta_targets
        self.prev_targets[:] = self.targets.clone()

        self.apply_action(self.actions, wait=wait, by_moveit=by_moveit, by_vel=by_vel)

    def apply_action(self, actions, do_scale=True, do_clamp=False, regulize_force=False, wait=True, by_moveit=True,
                     by_vel=False):

        # Apply the action
        if regulize_force:
            ft = torch.tensor(self.env.get_ft(), device=self.device, dtype=torch.float).unsqueeze(0)
            condition_mask = torch.abs(ft[:, 2]) > 2.0
            actions[:, 2] = torch.where(condition_mask, torch.clamp(actions[:, 2], min=0.0), actions[:, 2])
            # actions = torch.where(torch.abs(ft) > 1.5, torch.clamp(actions, min=0.0), actions)
            print("Regularized Actions:", np.round(actions[0][:3].cpu().numpy(), 4))

        if do_clamp:
            actions = torch.clamp(actions, -1.0, 1.0)
        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.pos_scale_deploy, device=self.device))
        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.rot_scale_deploy, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_jit_utils.quat_from_angle_axis_deploy(angle, axis)
        if self.deploy_config.rl.clamp_rot:
            rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > self.deploy_config.rl.clamp_rot_thresh,
                                           rot_actions_quat,
                                           torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device))

        self.ctrl_target_fingertip_centered_quat = torch_jit_utils.quat_mul_deploy(rot_actions_quat,
                                                                                   self.fingertip_centered_quat)

        if self.hand_action:
            gripper_actions = actions[:, 6:9]
            # gripper_actions[:, :2] = 1.0

            if do_scale:
                gripper_actions = gripper_actions @ torch.diag(
                    torch.tensor(self.gripper_action_scale_deploy, device=self.device))

            act_angles = torch.tensor(self.env.get_hand_motor_state(), device=self.device, dtype=torch.float).unsqueeze(0)
            self.act_angles = act_angles + gripper_actions  # add the action
            self.rel_act_angles += gripper_actions
            self.gripper_actions = gripper_actions

            # tendon_forces = self.Q @ self.act_angles.unsqueeze(-1)  # linear mapping between act_angle to tension
            # self.act_torque = self.R @ tendon_forces - self.K @ self.gripper_dof_pos[:, idx].unsqueeze(-1)
            # self.act_torque = self.act_torque.squeeze(-1)  # sum torque@each joint

        # self.generate_arm_ctrl_signals(wait=wait, by_moveit=by_moveit, by_vel=by_vel)
        self.generate_hand_ctrl_signals(wait=wait)

    def generate_hand_ctrl_signals(self, wait=True):

        target_motor = self.act_angles.cpu().detach().numpy().squeeze().tolist()
        try:
            self.env.hand.set_gripper_motors(target_motor)
        except Exception as e:
            print(f'failed to reach {target_motor}')

    def generate_arm_ctrl_signals(self, wait=True, by_moveit=True, by_vel=False):

        ctrl_info = self.env.get_info_for_control()

        self.fingertip_centered_pos = torch.tensor(ctrl_info['ee_pose'][:3],
                                                   device=self.device, dtype=torch.float).unsqueeze(0)
        self.fingertip_centered_quat = torch.tensor(ctrl_info['ee_pose'][3:],
                                                    device=self.device, dtype=torch.float).unsqueeze(0)

        fingertip_centered_jacobian_tf = torch.tensor(ctrl_info['jacob'],
                                                      device=self.device).unsqueeze(0)

        arm_dof_pos = torch.tensor(ctrl_info['joints'], device=self.device).unsqueeze(0)

        cfg_ctrl = {'num_envs': 1,
                    'jacobian_type': 'geometric',
                    'ik_method': 'dls'
                    }

        self.ctrl_target_dof_pos = fc.compute_dof_pos_target_deploy(
            cfg_ctrl=cfg_ctrl,
            arm_dof_pos=arm_dof_pos,
            fingertip_midpoint_pos=self.fingertip_centered_pos,
            fingertip_midpoint_quat=self.fingertip_centered_quat,
            jacobian=fingertip_centered_jacobian_tf,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_centered_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_centered_quat,
            ctrl_target_gripper_dof_pos=0,
            device=self.device)

        # clamp_values = [
        #     (-3.14159265359, 3.14159265359),  # Joint 1
        #     (-1.57079632679, 1.57079632679),  # Joint 2
        #     (-1.57079632679, 2.35619449019),  # Joint 3
        #     (-3.14159265359, 3.14159265359),  # Joint 4
        #     (-1.57079632679, 1.57079632679),  # Joint 5
        #     (-3.14159265359, 3.14159265359)   # Joint 6
        #     (-3.14159265359, 3.14159265359)   # Joint 7
        # ]
        #
        # for i in range(7):
        #     self.ctrl_target_dof_pos[:, i] = torch.clamp(self.ctrl_target_dof_pos[:, i], *clamp_values[i])
        # self.ctrl_target_dof_pos = self.ctrl_target_dof_pos[:, :6]

        target_joints = self.ctrl_target_dof_pos.cpu().detach().numpy().squeeze().tolist()
        try:
            self.env.move_to_joint_values(target_joints, wait=wait, by_moveit=by_moveit, by_vel=by_vel)
        except Exception as e:
            print(f'failed to reach {target_joints}')

    def deploy(self):

        # self._initialize_grasp_poses()
        from algo.deploy.env.env import ExperimentEnv
        rospy.init_node('DeployEnv')
        self.env = ExperimentEnv(with_tactile=False, with_zed=False)
        self.env.arm.move_manipulator.scale_vel(scale_vel=0.004, scale_acc=0.004)

        rospy.logwarn('Finished setting the env, lets play.')

        hz = 30
        ros_rate = rospy.Rate(hz)

        self._create_asset_info()
        self._acquire_task_tensors()

        # ---- Data Logger ----
        if self.deploy_config.data_logger.collect_data:
            from algo.ppo.experience import RealLogger
            data_logger = RealLogger(env=self)
            data_logger.data_logger = data_logger.data_logger_init(None)

        true_socket_pose = self.deploy_config.common_poses.socket_pos
        self._set_socket_pose(pos=true_socket_pose)

        self.env.arm.move_manipulator.scale_vel(scale_vel=0.5, scale_acc=0.5)
        self.env.move_to_init_state()

        self.reset()
        # rospy.sleep(2.0)

        num_episodes = 5
        cur_episode = 0

        while cur_episode < num_episodes:

            # Bias the ft sensor
            # self.env.arm.calib_robotiq()
            # rospy.sleep(2.0)
            # self.env.arm.calib_robotiq()

            obses = self.compute_observations(with_priv=True)
            filtered_obses = {key: value for key, value in obses.items() if value is not None}

            obs = filtered_obses.get('obs')
            obs_stud = filtered_obses.get('student_obs')
            tactile = filtered_obses.get('tactile')
            priv = filtered_obses.get('priv_info')
            # obs, obs_stud, tactile, priv = obses['obs'], obses['student_obs'], obses['tactile'], obses['priv_info']

            self._update_reset_buf()

            for i in range(self.full_config.task.env.numObsHist):
                pass

            while not self.done[0, 0]:

                obs = self.running_mean_std(obs.clone())
                # obs_stud = self.running_mean_std_stud(obs_stud.clone())
                priv = self.priv_mean_std(priv.clone())

                input_dict = {
                    'obs': obs,
                    'tactile_hist': tactile,
                    'priv_info': priv,
                }

                action, latent = self.model.act_inference(input_dict)
                action = torch.clamp(action, -1.0, 1.0)

                self.update_and_apply_action(action, wait=False, by_moveit=False, by_vel=True)

                ros_rate.sleep()
                self._update_reset_buf()
                self.episode_length += 1

                if self.deploy_config.data_logger.collect_data:
                    data_logger.log_trajectory_data(action, latent, self.done.clone())

                if self.done[0, 0]:
                    cur_episode += 1
                    break

                # Compute next observation
                obses = self.compute_observations(with_priv=True)
                filtered_obses = {key: value for key, value in obses.items() if value is not None}

                obs = filtered_obses.get('obs')
                obs_stud = filtered_obses.get('student_obs')
                tactile = filtered_obses.get('tactile')
                priv = filtered_obses.get('priv_info')

            self.env.arm.move_manipulator.stop_motion()
            self.reset()
