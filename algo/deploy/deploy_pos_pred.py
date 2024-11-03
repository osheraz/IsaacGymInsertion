##### !/usr/bin/env /home/osher/Desktop/isaacgym/venv/bin/python3
# --------------------------------------------------------
# now its our turn.
# https://arxiv.org/abs/todo
# Copyright (c) 2024 Osher & Co.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import rospy
from algo.models.running_mean_std import RunningMeanStd
import torch
import os
import hydra
import cv2
from isaacgyminsertion.utils import torch_jit_utils
import isaacgyminsertion.tasks.factory_tactile.factory_control as fc
from algo.models.transformer.tactile_runner import Runner as TAgent
from scipy.spatial.transform import Rotation

import numpy as np
from isaacgyminsertion.tasks.factory_tactile.factory_utils import quat2R, RotationTransformer

from matplotlib import pyplot as plt


def to_torch(x, dtype=torch.float, device='cuda:0'):
    return torch.tensor(x, dtype=dtype, device=device)


class HardwarePlayer:
    def __init__(self, full_config):

        self.f_right = None
        self.f_left = None
        self.f_bottom = None
        self.num_envs = 1
        self.grasp_flag = True

        self.deploy_config = full_config.deploy
        self.full_config = full_config
        self.tact_config = full_config.offline_train

        # Overwriting action scales for the controller
        self.max_episode_length = self.deploy_config.rl.max_episode_length

        self.device = self.full_config["rl_device"]
        self.episode_length = torch.zeros((1, 1), device=self.device, dtype=torch.float)
        self.actions = torch.zeros((1, 6), device=self.device, dtype=torch.float)
        self.latent = torch.zeros((1, 8), device=self.device, dtype=torch.float)

        self.tactile_seq_length = self.tact_config.model.transformer.sequence_length
        self.img_hist_len = self.tact_config.model.transformer.sequence_length
        self.external_cam = self.full_config.task.external_cam.external_cam
        self.res = [self.full_config.task.external_cam.cam_res.w, self.full_config.task.external_cam.cam_res.h]
        self.cam_type = self.full_config.task.external_cam.cam_type
        self.near_clip = self.full_config.task.external_cam.near_clip
        self.far_clip = self.full_config.task.external_cam.far_clip
        self.dis_noise = self.full_config.task.external_cam.dis_noise

        self.cfg_tactile = full_config.task.tactile

        self.cfg_tactile = full_config.task.tactile
        asset_info_path = '../../assets/factory/yaml/factory_asset_info_insertion.yaml'
        # asset_info_path = '../../../assets/factory/yaml/factory_asset_info_insertion.yaml'
        self.asset_info_insertion = hydra.compose(config_name=asset_info_path)
        self.asset_info_insertion = self.asset_info_insertion['']['']['']['']['']['']['assets']['factory']['yaml']
        self.rot_tf = RotationTransformer()

    def restore(self, fn):
        import pickle
        self.tact = TAgent(self.full_config)
        self.tact.load_model(self.tact_config.model.transformer.tact_path)
        self.tact.model.eval()
        self.stats = pickle.load(open(self.tact_config.train.normalize_file, "rb"))

    def compile_inference(self, precision="high"):
        torch.set_float32_matmul_precision(precision)
        self.tact.model.forward = torch.compile(torch.no_grad(self.tact.model.forward))

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

        torch_pi = torch.tensor(np.pi, dtype=torch.float32, device=self.device)
        rotation_quat_x = torch_jit_utils.quat_from_angle_axis(torch_pi,
                                                               torch.tensor([1, 0, 0], dtype=torch.float32,
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

        self.plug_hand_pos, self.plug_hand_quat = self._pose_world_to_hand_base(self.plug_pos, self.plug_quat)

    def compute_observations(self, with_tactile=True, with_img=True, display_image=True, with_priv=False, diff_tac=False):

        obses = self.env.get_obs()

        arm_joints = obses['joints']
        ee_pose = obses['ee_pose']
        ft = obses['ft']

        self.ft_data = torch.tensor(ft, device=self.device).unsqueeze(0)
        self.arm_dof_pos = torch.tensor(arm_joints, device=self.device).unsqueeze(0)
        self.fingertip_centered_pos = torch.tensor(ee_pose[:3], device=self.device, dtype=torch.float).unsqueeze(0)
        self.fingertip_centered_quat = torch.tensor(ee_pose[3:], device=self.device, dtype=torch.float).unsqueeze(0)

        eef_pos = torch.cat((self.fingertip_centered_pos,
                             quat2R(self.fingertip_centered_quat).reshape(1, -1)), dim=-1)

        if with_tactile:

            left, right, bottom = obses['frames']

            # Cutting by half
            if self.cfg_tactile.crop_roi:
                w = left.shape[0]
                left = left[:w // 2, :, :]
                right = right[:w // 2, :, :]
                bottom = bottom[:w // 2, :, :]
            # Resizing to encoder size
            left = cv2.resize(left, (self.height, self.width), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (self.height, self.width), interpolation=cv2.INTER_AREA)
            bottom = cv2.resize(bottom, (self.height, self.width), interpolation=cv2.INTER_AREA)

            if self.grasp_flag:
                self.f_left = left.copy()
                self.f_right = right.copy()
                self.f_bottom = bottom.copy()
                self.grasp_flag = False

            if diff_tac:
                left -= self.f_left
                right -= self.f_right
                bottom -= self.f_bottom

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
            self.image_buf[0] = to_torch(img).permute(2, 0, 1).to(self.device)

            if display_image:
                cv2.imshow("Depth Image", img.transpose(1, 2, 0) + 0.5)
                cv2.waitKey(1)

            self.img_queue[:, 1:] = self.img_queue[:, :-1].clone().detach()
            self.img_queue[:, 0, ...] = self.image_buf

        # some-like taking a new socket pose measurement
        self._update_socket_pose()

        obs = torch.cat([eef_pos], dim=-1)

        obs_dict = {'obs': obs}

        if with_tactile:
            obs_dict['tactile'] = self.tactile_queue.clone()
        if with_img:
            obs_dict['img'] = self.img_queue.clone()

        if with_priv:
            # Compute privileged info (gt_error + contacts)
            self._update_plug_pose()

            state_tensors = [
                self.plug_hand_pos,  # 3
                self.plug_hand_quat,
                # self.plug_pos_error,  # 3
                # self.plug_quat_error,  # 4
            ]

            obs_dict['priv_info'] = torch.cat(state_tensors, dim=-1)
            self.states_buf = torch.cat(state_tensors, dim=-1)  # shape = (num_envs, num_states)

        return obs_dict

    def _update_reset_buf(self):

        plug_socket_xy_distance = torch.norm(self.plug_pos_error[:, :2])

        is_very_close_xy = plug_socket_xy_distance < 0.005
        is_bellow_surface = -self.plug_pos_error[:, 2] < 0.005

        self.inserted = is_very_close_xy & is_bellow_surface
        is_too_far = (plug_socket_xy_distance > 0.08) | (self.fingertip_centered_pos[:, 2] > 0.125)

        timeout = (self.episode_length >= self.max_episode_length - 1)

        self.done = timeout  # | is_too_far |  self.inserted

        if self.done[0, 0].item():
            print('Reset because ',
                  "far away" if is_too_far[0].item() else "",
                  "timeoout" if timeout.item() else "",
                  "inserted" if self.inserted.item() else "", )

    def reset(self):

        joints_above_socket = self.deploy_config.common_poses.joints_above_socket
        joints_above_plug = self.deploy_config.common_poses.joints_above_plug

        # Move above the socket
        # self.env.arm.move_manipulator.scale_vel(scale_vel=0.1, scale_acc=0.1)
        # self.env.move_to_joint_values(joints_above_socket, wait=True)
        # # Set random init error
        # self.env.set_random_init_error(self.socket_pos)
        # If not inserted, return plug?
        if False:  # not self.inserted and False:

            self.env.arm.move_manipulator.scale_vel(scale_vel=0.5, scale_acc=0.5)
            self.env.move_to_joint_values(joints_above_plug, wait=True)
            self.env.arm.move_manipulator.scale_vel(scale_vel=0.1, scale_acc=0.1)
            self.env.align_and_release(init_plug_pose=[0.4103839067235552, 0.17531695171951858, 0.008])

            self.grasp_and_init()
        # self.env.randomize_grasp()
        self.env.release()
        self.env.grasp()
        self.env.arm.move_manipulator.scale_vel(scale_vel=0.02, scale_acc=0.02)
        self.inserted[...] = False
        self.done[...] = False
        self.episode_length[...] = 0.

    def grasp_and_init(self):

        skip = True
        joints_above_socket = self.deploy_config.common_poses.joints_above_socket
        joints_above_plug = self.deploy_config.common_poses.joints_above_plug
        # Move above plug
        self.env.move_to_joint_values(joints_above_plug, wait=True)
        self.env.arm.move_manipulator.scale_vel(scale_vel=0.1, scale_acc=0.1)
        # Align and grasp
        if not skip:
            self.env.align_and_grasp()

            self.env.arm.move_manipulator.scale_vel(scale_vel=0.5, scale_acc=0.5)
            # Move up a bit
            self.env.move_to_joint_values(joints_above_plug, wait=True)
            # Move above the socket
            self.env.move_to_joint_values(joints_above_socket, wait=True)

            # Set random init error
            self.env.arm.move_manipulator.scale_vel(scale_vel=0.1, scale_acc=0.1)
            self.env.set_random_init_error(self.socket_pos)
        else:
            self.env.grasp()

        self.env.arm.move_manipulator.scale_vel(scale_vel=0.02, scale_acc=0.02)
        print('Starting')
        self.grasp_flag = True

        self.done[...] = False
        self.episode_length[...] = 0.

    def deploy(self, test=False):

        # self._initialize_grasp_poses()
        from algo.deploy.env.env import ExperimentEnv
        rospy.init_node('DeployEnv')

        self.env = ExperimentEnv(with_zed=self.deploy_config.env.depth_cam,
                                 with_tactile=self.deploy_config.env.tactile,
                                 with_ext_cam=self.deploy_config.env.ext_cam,
                                 with_hand=self.deploy_config.env.hand,
                                 with_arm=self.deploy_config.env.arm)

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
        # self.env.move_to_init_state()

        self.grasp_and_init()

        num_episodes = self.deploy_config.data_logger.total_trajectories
        cur_episode = 0

        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title('POS')
        ax1.legend()
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title('RPY')
        ax2.legend()
        width = 0.35
        indices = np.arange(3)

        display = True

        while cur_episode < num_episodes:

            # TODO add a module that set init interaction with the socket

            obs_dict = self.compute_observations(with_priv=True, with_img=False, diff_tac=test)
            tactile = obs_dict['tactile']
            label = obs_dict['priv_info'].cpu().detach().numpy()[0]
            label0 = label.copy()
            sxyz, squat = label0[:3], label0[3:] + 1e-6
            seuler = Rotation.from_quat(squat).as_euler('xyz')

            self._update_reset_buf()

            for i in range(self.full_config.task.env.numObsHist):
                pass

            while not self.done[0, 0]:

                if test:
                    print(tactile.max(), tactile.min())
                    out = self.tact.get_latent(tactile).cpu().detach().numpy()
                    d_plug_pos, d_euler = out[:, :3], out[:, 3:]
                    d_plug_pos = (d_plug_pos * self.stats["std"]["plug_hand_pos_diff"]) + self.stats["mean"]["plug_hand_pos_diff"]
                    d_euler = (d_euler * self.stats["std"]["plug_hand_diff_euler"]) + self.stats["mean"]["plug_hand_diff_euler"]
                    d_pos_rpy = np.hstack((d_plug_pos, d_euler))[0]

                if display:
                    if test:
                        ax1.bar(indices - width / 2, d_pos_rpy[:3], width, label='d_pos_rpy')
                        ax2.bar(indices - width / 2, d_pos_rpy[3:], width, label='d_pos_rpy')
                    xyz, quat = label[:3], label[3:] + 1e-6
                    euler = Rotation.from_quat(quat).as_euler('xyz')
                    ax1.bar(indices + width / 2, xyz - sxyz, width, label='True POS')
                    ax2.bar(indices + width / 2, euler - seuler, width, label='True RPY')

                    ax1.set_ylim([-0.02, 0.02])
                    ax2.set_ylim([-0.5, 0.5])
                    plt.pause(0.00000000002)
                    ax1.cla()
                    ax2.cla()

                ros_rate.sleep()
                self._update_reset_buf()
                self.episode_length += 1

                if self.deploy_config.data_logger.collect_data:
                    data_logger.log_trajectory_data(self.actions, self.latent, self.done.clone())

                if self.done[0, 0]:
                    cur_episode += 1
                    break

                # Compute next observation
                obs_dict = self.compute_observations(with_priv=True, with_img=False, diff_tac=test)
                tactile = obs_dict['tactile']
                label = obs_dict['priv_info'].cpu().detach().numpy()[0]

            self.env.arm.move_manipulator.stop_motion()
            self.reset()

        self.env.release()