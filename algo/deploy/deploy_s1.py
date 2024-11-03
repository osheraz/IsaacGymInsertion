##### !/usr/bin/env /home/osher/Desktop/isaacgym/venv/bin/python3
# --------------------------------------------------------
# now its our turn.
# https://arxiv.org/abs/todo
# Copyright (c) 2024 Osher & Co.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import cv2
import hydra
import numpy as np
import rospy
import torch

import isaacgyminsertion.tasks.factory_tactile.factory_control as fc
from algo.models.models_split import ActorCriticSplit as ActorCritic
from algo.models.running_mean_std import RunningMeanStd
from isaacgyminsertion.tasks.factory_tactile.factory_utils import quat2R, RotationTransformer
from isaacgyminsertion.utils import torch_jit_utils


def to_torch(x, dtype=torch.float, device='cuda:0'):
    return torch.tensor(x, dtype=dtype, device=device)


class HardwarePlayer:
    def __init__(self, full_config):

        self.num_envs = 1
        self.deploy_config = full_config.deploy
        self.full_config = full_config

        # Overwriting action scales for the controller
        self.max_episode_length = self.deploy_config.rl.max_episode_length
        self.pos_scale_deploy = self.deploy_config.rl.pos_action_scale
        self.rot_scale_deploy = self.deploy_config.rl.rot_action_scale

        self.pos_scale = self.full_config.task.rl.pos_action_scale
        self.rot_scale = self.full_config.task.rl.rot_action_scale

        self.device = self.full_config["rl_device"]
        self.episode_length = torch.zeros((1, 1), device=self.device, dtype=torch.float)

        # ---- build environment ----
        self.num_observations = self.full_config.task.env.numObservations
        self.num_obs_stud = self.full_config.task.env.numObsStudent
        self.obs_shape = (self.full_config.task.env.numObservations,)
        self.obs_stud_shape = (self.full_config.task.env.numObsStudent,)

        self.num_actions = self.full_config.task.env.numActions
        self.num_targets = self.full_config.task.env.numTargets

        # ---- Priv Info ----
        self.priv_info = self.full_config.train.ppo.priv_info
        self.priv_info_dim = self.full_config.train.ppo.priv_info_dim
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

        net_config = {
            'actor_units': self.full_config.train.network.mlp.units,
            'actions_num': self.num_actions,
            'priv_mlp_units': self.full_config.train.network.priv_mlp.units,
            'input_shape': self.obs_shape,
            'priv_info_dim': self.priv_info_dim,
            'priv_info': self.priv_info,
            "obs_info": self.obs_info,
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

        self.cfg_tactile = full_config.task.tactile
        asset_info_path = '../../assets/factory/yaml/factory_asset_info_insertion.yaml'

        # asset_info_path = '../../../assets/factory/yaml/factory_asset_info_insertion.yaml'
        self.asset_info_insertion = hydra.compose(config_name=asset_info_path)
        # self.asset_info_insertion = self.asset_info_insertion['']['']['']['']['']['']['']['']['']['assets']['factory'][
        #     'yaml']
        self.asset_info_insertion = self.asset_info_insertion['']['']['']['']['']['']['assets']['factory']['yaml']
        self.rot_tf = RotationTransformer()
        self.rot_tf_back = RotationTransformer(to_rep='quaternion', from_rep='rotation_6d',)
        self.stud_tf = RotationTransformer(from_rep='matrix', to_rep='rotation_6d')

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

    def _pose_world_to_hand_base(self, pos, quat, to_rep=None):
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
        rotation_quat_x = torch_jit_utils.quat_from_angle_axis(torch_pi,
                          torch.tensor([1, 0, 0], dtype=torch.float32, device=self.device)).repeat((self.num_envs, 1))
        rotation_quat_z = torch_jit_utils.quat_from_angle_axis(-torch_pi * 0.5,
                          torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)).repeat((self.num_envs, 1))

        q_rotated = torch_jit_utils.quat_mul(rotation_quat_x, self.fingertip_centered_quat.clone())
        q_rotated = torch_jit_utils.quat_mul(rotation_quat_z, q_rotated)

        robot_base_transform_inv = torch_jit_utils.tf_inverse(
            q_rotated, self.fingertip_centered_pos.clone()
        )
        quat_in_robot_base, pos_in_robot_base = torch_jit_utils.tf_combine(
            robot_base_transform_inv[0], robot_base_transform_inv[1], quat, pos
        )

        if to_rep == 'matrix':
            return pos_in_robot_base, quat2R(quat_in_robot_base).reshape(self.num_envs, -1)
        elif to_rep == 'rot6d':
            return pos_in_robot_base, self.rot_tf.forward(quat_in_robot_base)
        else:
            return pos_in_robot_base, quat_in_robot_base

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        self.done = torch.zeros((1, 1), device=self.device, dtype=torch.bool)
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
        self.width = self.cfg_tactile.encoder.width // 2 if self.cfg_tactile.crop_roi else self.cfg_tactile.encoder.width
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
            (1, self.tact_hist_len, 3,  # left, right, bottom
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

        self.seg_queue = torch.zeros(
            (1, self.img_hist_len, self.res[1], self.res[0]),device=self.device, dtype=torch.float,
        )
        self.seg_buf = torch.zeros(1, self.res[1], self.res[0]).to(self.device)

        self.ft_data = torch.zeros((1, 6), device=self.device, dtype=torch.float)

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
        self.socket_obs_pos_noise = socket_obs_pos_noise @ torch.diag(
            torch.tensor(
                self.deploy_config.env.socket_pos_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.noisy_socket_pos = torch.zeros_like(
            self.socket_pos, dtype=torch.float32, device=self.device
        )

        self.noisy_socket_pos[:, 0] = self.socket_pos[:, 0] + self.socket_obs_pos_noise[:, 0]
        self.noisy_socket_pos[:, 1] = self.socket_pos[:, 1] + self.socket_obs_pos_noise[:, 1]
        self.noisy_socket_pos[:, 2] = self.socket_pos[:, 2] + self.socket_obs_pos_noise[:, 2]

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
                self.deploy_config.env.socket_rot_obs_noise,
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
                                                                                self.plug_quat)
        if self.gt_contacts_info:
            self.contacts[0, :] = torch.tensor(self.env.tracker.extrinsic_contact).to(self.device)

    def compute_observations(self, with_tactile=True, with_img=False, display_image=True, with_priv=False):

        obses = self.env.get_obs()
        # some-like taking a new socket pose measurement
        self._update_socket_pose()

        arm_joints = obses['joints']
        ee_pose = obses['ee_pose']
        ft = obses['ft']

        self.ft_data = torch.tensor(ft, device=self.device).unsqueeze(0)
        self.arm_dof_pos = torch.tensor(arm_joints, device=self.device).unsqueeze(0)
        self.fingertip_centered_pos = torch.tensor(ee_pose[:3], device=self.device, dtype=torch.float).unsqueeze(0)
        self.fingertip_centered_quat = torch.tensor(ee_pose[3:], device=self.device, dtype=torch.float).unsqueeze(0)

        if with_tactile:

            left, right, bottom = obses['frames']

            if self.cfg_tactile.crop_roi:
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
            else:
                self.tactile_imgs[0, 0] = to_torch(cv2.cvtColor(left.astype('float32'), cv2.COLOR_BGR2GRAY)).to(self.device)
                self.tactile_imgs[0, 1] = to_torch(cv2.cvtColor(right.astype('float32'), cv2.COLOR_BGR2GRAY)).to(self.device)
                self.tactile_imgs[0, 2] = to_torch(cv2.cvtColor(bottom.astype('float32'), cv2.COLOR_BGR2GRAY)).to(self.device)

            self.tactile_queue[:, 1:] = self.tactile_queue[:, :-1].clone().detach()
            self.tactile_queue[:, 0, :] = self.tactile_imgs

        if with_img:
            img = obses['img']
            self.image_buf[0] = to_torch(img).to(self.device)
            self.img_queue[:, 1:] = self.img_queue[:, :-1].clone().detach()
            self.img_queue[:, 0, ...] = self.image_buf

            seg = obses['seg']
            self.seg_buf[0] = to_torch(seg).to(self.device)
            self.seg_queue[:, 1:] = self.seg_queue[:, :-1].clone().detach()
            self.seg_queue[:, 0, ...] = self.seg_buf

            if display_image:
                cv2.imshow("Depth Image", img.transpose(1, 2, 0) + 0.5)
                cv2.waitKey(1)

        # eef_pos = torch.cat((self.fingertip_centered_pos,
        #                      quat2R(self.fingertip_centered_quat).reshape(1, -1)), dim=-1)
        eef_pos = torch.cat((self.fingertip_centered_pos,
                             self.rot_tf.forward(self.fingertip_centered_quat)), dim=-1)

        noisy_delta_pos = self.noisy_gripper_goal_pos - self.fingertip_centered_pos

        obs = torch.cat([eef_pos,
                         self.actions,
                         noisy_delta_pos
                         ], dim=-1)

        self.obs_queue[:, :-self.num_observations] = self.obs_queue[:, self.num_observations:]
        self.obs_queue[:, -self.num_observations:] = obs
        self.obs_buf = self.obs_queue.clone()  # torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)

        eef_stud = torch.cat((self.fingertip_centered_pos, quat2R(self.fingertip_centered_quat).reshape(1, -1)), dim=-1)
        # fix bug
        eef_stud = torch.cat((self.fingertip_centered_pos,
                              self.stud_tf.forward(eef_stud[:, 3:].reshape(eef_stud.shape[0], 3, 3))), dim=1)

        obs_tensors_student = torch.cat([eef_stud,  # 6
                                         self.socket_pos,  # 3
                                         self.actions
                                         ], dim=-1)

        self.obs_stud_queue[:, :-self.num_obs_stud] = self.obs_stud_queue[:, self.num_obs_stud:]
        self.obs_stud_queue[:, -self.num_obs_stud:] = obs_tensors_student

        self.obs_student_buf = self.obs_stud_queue.clone()  # shape = (num_envs, num_observations_student)

        if not with_priv:
            return self.obs_buf, self.obs_student_buf, self.tactile_queue
        else:
            # Compute privileged info (gt_error + contacts)
            self._update_plug_pose()

            state_tensors = [
                self.plug_hand_pos,  # 3
                self.plug_hand_quat,  # 4
                self.plug_pos_error,  # # 3
                self.plug_quat_error,  # 4
            ]

            self.states_buf = torch.cat(state_tensors, dim=-1)  # shape = (num_envs, num_states)

            return self.obs_buf, self.obs_student_buf, self.tactile_queue, self.states_buf

    def _update_reset_buf(self):

        plug_socket_xy_distance = torch.norm(self.plug_pos_error[:, :2])

        is_very_close_xy = plug_socket_xy_distance < 0.005
        is_bellow_surface = -self.plug_pos_error[:, 2] < 0.007

        self.inserted = is_very_close_xy & is_bellow_surface
        is_too_far = (plug_socket_xy_distance > 0.08) | (self.fingertip_centered_pos[:, 2] > 0.125)

        timeout = (self.episode_length >= self.max_episode_length)

        self.done = is_too_far | timeout | self.inserted

        if self.done[0, 0].item():
            print('Reset because ',
                  "far away" if is_too_far[0].item() else "",
                  "timeoout" if timeout.item() else "",
                  "inserted" if self.inserted.item() else "", )

    def reset(self):

        joints_above_socket = self.deploy_config.common_poses.joints_above_socket
        joints_above_plug = self.deploy_config.common_poses.joints_above_plug

        # Move above the socket
        self.env.arm.move_manipulator.scale_vel(scale_vel=0.1, scale_acc=0.1)
        self.env.move_to_joint_values(joints_above_socket, wait=True)
        # Set random init error
        self.env.set_random_init_error(self.socket_pos)
        # If not inserted, return plug?
        if not self.inserted:
            self.env.arm.move_manipulator.scale_vel(scale_vel=0.5, scale_acc=0.5)
            self.env.move_to_joint_values(joints_above_plug, wait=True)
            self.env.arm.move_manipulator.scale_vel(scale_vel=0.1, scale_acc=0.1)
            self.env.align_and_release(init_plug_pose=[0.4103839067235552, 0.17531695171951858, 0.008])

            self.grasp_and_init()

        # self.env.randomize_grasp()

        self.env.arm.move_manipulator.scale_vel(scale_vel=0.02, scale_acc=0.02)
        self.inserted[...] = False
        self.done[...] = False
        self.episode_length[...] = 0.

    def grasp_and_init(self):

        joints_above_socket = self.deploy_config.common_poses.joints_above_socket
        joints_above_plug = self.deploy_config.common_poses.joints_above_plug
        # Move above plug
        self.env.move_to_joint_values(joints_above_plug, wait=True)
        self.env.arm.move_manipulator.scale_vel(scale_vel=0.1, scale_acc=0.1)
        # Align and grasp
        self.env.align_and_grasp()
        self.env.arm.move_manipulator.scale_vel(scale_vel=0.5, scale_acc=0.5)
        # Move up a bit
        self.env.move_to_joint_values(joints_above_plug, wait=True)
        # Move above the socket
        self.env.move_to_joint_values(joints_above_socket, wait=True)

        # Set random init error
        self.env.arm.move_manipulator.scale_vel(scale_vel=0.1, scale_acc=0.1)
        self.env.set_random_init_error(self.socket_pos)

        self.env.arm.move_manipulator.scale_vel(scale_vel=0.02, scale_acc=0.02)
        print('Starting Insertion')

        self.done[...] = False
        self.episode_length[...] = 0.

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

        delta_targets = torch.cat([
            self.actions[:, :3] @ torch.diag(torch.tensor(self.pos_scale, device=self.device)),  # 3
            self.actions[:, 3:6] @ torch.diag(torch.tensor(self.rot_scale, device=self.device))  # 3
        ], dim=-1).clone()

        # Update targets
        self.targets = self.prev_targets + delta_targets
        self.prev_targets[:] = self.targets.clone()

        self.apply_action(self.actions, wait=wait, by_moveit=by_moveit, by_vel=by_vel)

    def apply_action(self, actions, do_scale=True, do_clamp=False, regulize_force=True, wait=True, by_moveit=True,
                     by_vel=False):

        # Apply the action
        if regulize_force:
            ft = torch.tensor(self.env.get_ft(), device=self.device, dtype=torch.float).unsqueeze(0)
            condition_mask = torch.abs(ft[:, 2]) > 2.0
            actions[:, 2] = torch.where(condition_mask, torch.clamp(actions[:, 2], min=0.0), actions[:, 2])
            # actions = torch.where(torch.abs(ft) > 1.5, torch.clamp(actions, min=0.0), actions)
            print("Error:", np.round(self.plug_pos_error[0].cpu().numpy(), 4))
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

        self.generate_ctrl_signals(wait=wait, by_moveit=by_moveit, by_vel=by_vel)

    def generate_ctrl_signals(self, wait=True, by_moveit=True, by_vel=False):

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
        except:
            print(f'failed to reach {target_joints}')

    def deploy(self):

        # self._initialize_grasp_poses()
        from algo.deploy.env.env import ExperimentEnv
        rospy.init_node('DeployEnv')
        self.env = ExperimentEnv()
        self.env.arm.move_manipulator.scale_vel(scale_vel=0.004, scale_acc=0.004)

        rospy.logwarn('Finished setting the env, lets play.')

        hz = 20
        ros_rate = rospy.Rate(hz)

        self._create_asset_info()
        self._acquire_task_tensors()

        self.deploy_config.data_logger.collect_data = True
        # ---- Data Logger ----
        if self.deploy_config.data_logger.collect_data:
            from algo.ppo.experience import RealLogger
            data_logger = RealLogger(env=self)
            data_logger.data_logger = data_logger.data_logger_init(None)

        true_socket_pose = self.deploy_config.common_poses.socket_pos
        self._set_socket_pose(pos=true_socket_pose)

        self.env.arm.move_manipulator.scale_vel(scale_vel=0.5, scale_acc=0.5)
        self.env.move_to_init_state()

        self.grasp_and_init()
        rospy.sleep(2.0)

        num_episodes = 5
        cur_episode = 0

        while cur_episode < num_episodes:

            # Bias the ft sensor
            self.env.arm.calib_robotiq()

            obs, obs_stud, tactile, priv = self.compute_observations(with_priv=True)
            self._update_reset_buf()

            for i in range(self.full_config.task.env.numObsHist):
                pass

            while not self.done[0, 0]:

                obs = self.running_mean_std(obs.clone())
                priv = self.priv_mean_std(priv.clone())

                input_dict = {
                    'obs': obs,
                    'priv_info': priv,
                }

                action, latent = self.model.act_inference(input_dict)
                action = torch.clamp(action, -1.0, 1.0)

                # start_time = time()
                self.update_and_apply_action(action, wait=False, by_moveit=False, by_vel=True)
                # print("Actions:", np.round(action[0].cpu().numpy(), 3), "\tFPS: ", 1.0 / (time() - start_time))

                ros_rate.sleep()
                self._update_reset_buf()
                self.episode_length += 1

                if self.deploy_config.data_logger.collect_data:
                    data_logger.log_trajectory_data(action, latent, self.done.clone())

                if self.done[0, 0]:
                    cur_episode += 1
                    break

                # Compute next observation
                obs, obs_stud, tactile, priv = self.compute_observations(with_priv=True)

            self.env.arm.move_manipulator.stop_motion()
            self.reset()
