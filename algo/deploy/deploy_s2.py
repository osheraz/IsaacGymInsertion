##### !/usr/bin/env /home/osher/Desktop/isaacgym/venv/bin/python3
# --------------------------------------------------------
# now its our turn.
# https://arxiv.org/abs/todo
# Copyright (c) 2023 Osher & Co.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import rospy
import pickle
from algo.models.models_split import ActorCriticSplit as ActorCritic
from algo.models.running_mean_std import RunningMeanStd
import torch
import os
import hydra
import cv2
from isaacgyminsertion.utils import torch_jit_utils
import isaacgyminsertion.tasks.factory_tactile.factory_control as fc
from algo.models.transformer.runner import Runner as Student
import numpy as np
from termcolor import cprint
from isaacgyminsertion.tasks.factory_tactile.factory_utils import quat2R, RotationTransformer
import matplotlib.pyplot as plt


def to_torch(x, dtype=torch.float, device='cuda:0'):
    return torch.tensor(x, dtype=dtype, device=device)


def display_obs(depth, seg, pcl, ax=None):
    if depth is not None:
        depth = depth[0, 0, ...].reshape(1, 54, 96)
        dp = depth.cpu().detach().numpy()
        dp = np.transpose(dp, (1, 2, 0))
        cv2.imshow('Depth Sequence', dp)

    if seg is not None:
        seg = seg[0, 0, ...].reshape(1, 54, 96)
        sg = seg.cpu().detach().numpy()
        sg = np.transpose(sg, (1, 2, 0))
        cv2.imshow('Seg Sequence', sg)

    if pcl is not None:
        pcl = pcl.cpu().detach().numpy()
        ax.plot(pcl[0, :, 0],
                pcl[0, :, 1],
                pcl[0, :, 2], 'ko')
        plt.pause(0.0001)
        ax.cla()

    cv2.waitKey(1)


class HardwarePlayer:
    def __init__(self, full_config):

        self.f_right = None
        self.f_left = None
        self.f_bottom = None
        self.tactile_grasp_flag = True
        self.stats = None

        self.rot_tf = RotationTransformer()
        self.stud_tf = RotationTransformer(from_rep='matrix', to_rep='rotation_6d')

        self.num_envs = 1
        self.deploy_config = full_config.deploy
        self.full_config = full_config
        self.train_config = full_config.offline_train

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
        self.num_obs_stud = self.train_config.model.linear.input_size
        self.obs_shape = (self.full_config.task.env.numObservations,)
        self.obs_stud_shape = (self.full_config.task.env.numObsStudent,)

        self.num_actions = self.full_config.task.env.numActions
        self.num_targets = self.full_config.task.env.numTargets

        # ---- Hist Info ---
        self.tactile_seq_length = self.train_config.model.transformer.sequence_length
        self.img_hist_len = self.train_config.model.transformer.sequence_length
        self.stud_hist_len = self.train_config.model.transformer.sequence_length
        # ---- Priv Info ----
        self.priv_info = False
        self.priv_info_dim = self.full_config.train.ppo.priv_info_dim

        self.external_cam = self.full_config.task.external_cam.external_cam
        self.res = [self.full_config.task.external_cam.cam_res.w, self.full_config.task.external_cam.cam_res.h]
        self.res = [320, 180]
        self.cam_type = self.full_config.task.external_cam.cam_type
        self.near_clip = self.full_config.task.external_cam.near_clip
        self.far_clip = self.full_config.task.external_cam.far_clip
        self.dis_noise = self.full_config.task.external_cam.dis_noise

        agent_config = {
            'actor_units': self.full_config.train.network.mlp.units,
            'actions_num': self.num_actions,
            'priv_mlp_units': self.full_config.train.network.priv_mlp.units,
            'input_shape': self.obs_shape,
            'gt_contacts_info': False,
            'priv_info_dim': self.priv_info_dim,
            'priv_info': self.priv_info,
            "only_contact": self.full_config.train.ppo.only_contact,
            "contacts_mlp_units": self.full_config.train.network.contact_mlp.units,
            'shared_parameters': False,
            'full_config': self.full_config,
            'vt_policy': False,
        }

        self.agent = ActorCritic(agent_config)
        self.agent.to(self.device)
        self.agent.eval()

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.priv_mean_std = RunningMeanStd((self.priv_info_dim,)).to(self.device)
        self.priv_mean_std.eval()

        self.cfg_tactile = full_config.task.tactile
        asset_info_path = '../../assets/factory/yaml/factory_asset_info_insertion.yaml'
        # asset_info_path = '../../../assets/factory/yaml/factory_asset_info_insertion.yaml'
        self.asset_info_insertion = hydra.compose(config_name=asset_info_path)
        # self.asset_info_insertion = self.asset_info_insertion['']['']['']['']['']['']['']['']['']['assets']['factory'][
        #     'yaml']
        self.asset_info_insertion = self.asset_info_insertion['']['']['']['']['']['']['assets']['factory']['yaml']

        self.ppo_config = full_config.train.ppo
        self.obs_info = self.deploy_config.ppo.obs_info
        self.tactile_info = self.deploy_config.ppo.tactile_info
        self.img_info = self.deploy_config.ppo.img_info
        self.seg_info = self.deploy_config.ppo.seg_info
        self.pcl_info = self.deploy_config.ppo.pcl_info

        student_cfg = self.full_config
        student_cfg.offline_train.only_bc = True
        student_cfg.offline_train.model.use_tactile = self.tactile_info
        student_cfg.offline_train.model.use_seg = self.seg_info
        student_cfg.offline_train.model.use_lin = self.obs_info
        student_cfg.offline_train.model.use_img = self.img_info
        student_cfg.offline_train.model.use_pcl = self.pcl_info

        if self.pcl_info:
            self.pcl_mean_std = RunningMeanStd((3,)).to(self.device)
            self.pcl_mean_std.eval()

        self.student = Student(student_cfg)
        self.display_obs = False

        if self.display_obs and self.pcl_info:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
        else:
            self.ax = None

    def restore(self, fn):
        checkpoint = torch.load(fn)
        # self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if 'priv_mean_std' in checkpoint:
            print('Loading Policy with priv info')
            # self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        else:
            print('Loading Policy without priv info')
        # self.agent.load_state_dict(checkpoint['model'])

        # stud_fn = fn.replace('stage1_nn/last.pth', 'student/checkpoints/model_last.pt')
        stud_fn = fn.replace('stage1_nn/last.pth', 'stage2_nn/last_stud.pth')

        self.restore_student(stud_fn, from_offline=False, phase=1)

        self.set_eval()
        self.set_student_eval()

    def restore_student(self, fn, from_offline=False, phase=1):

        if from_offline:
            if phase == 2:
                checkpoint = torch.load(fn, map_location=self.device)
                self.student.model.load_state_dict(checkpoint['student'])
                cprint(f'Restoring student from: {fn}',
                       'red', attrs=['bold'])
            else:
                cprint(f'Restoring student from: {self.train_config.train.student_ckpt_path}', 'red', attrs=['bold'])
                checkpoint = torch.load(self.train_config.train.student_ckpt_path, map_location=self.device)
                self.student.model.load_state_dict(checkpoint)

            if self.train_config.model.transformer.load_tact:
                self.student.load_tact_model(self.train_config.model.transformer.tact_path)

            cprint(f'Using offline stats from: {self.train_config.train.normalize_file}', 'red', attrs=['bold'])
            self.stats = {'mean': {}, 'std': {}}
            stats = pickle.load(open(self.train_config.train.normalize_file, "rb"))
            for key in stats['mean'].keys():
                self.stats['mean'][key] = torch.tensor(stats['mean'][key], device=self.device)
                self.stats['std'][key] = torch.tensor(stats['std'][key], device=self.device)
        else:
            cprint(f'Restoring student from: {fn}', 'blue', attrs=['bold'])

            checkpoint = torch.load(fn, map_location=self.device)
            self.stud_obs_mean_std = RunningMeanStd((self.full_config.offline_train.model.linear.input_size,)).to(
                self.device)
            self.stud_obs_mean_std.load_state_dict(checkpoint['stud_obs_mean_std'])
            self.pcl_mean_std.load_state_dict(checkpoint['pcl_mean_std'])
            self.student.model.load_state_dict(checkpoint['student'])
            cprint(f'stud_obs_mean_std: {self.stud_obs_mean_std.running_mean}', 'green', attrs=['bold'])
            cprint(f'pcl_mean_std: {self.pcl_mean_std.running_mean}', 'green', attrs=['bold'])

    def compile_inference(self, precision="high"):
        torch.set_float32_matmul_precision(precision)
        self.agent.policy.forward = torch.compile(torch.no_grad(self.student.model.forward))
        self.agent.tact.model.forward = torch.compile(torch.no_grad(self.tact.model.forward))

    def set_eval(self):
        self.agent.eval()
        self.running_mean_std.eval()
        self.priv_mean_std.eval()

    def set_student_eval(self):

        self.student.model.eval()
        if self.train_config.model.transformer.load_tact:
            self.student.tact.eval()
        if not self.train_config.from_offline:
            if self.stud_obs_mean_std:
                self.stud_obs_mean_std.eval()
            if self.pcl_mean_std:
                self.pcl_mean_std.eval()

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

        self.actions = torch.zeros((1, self.num_actions), device=self.device)
        self.targets = torch.zeros((1, self.full_config.task.env.numTargets), device=self.device)
        self.prev_targets = torch.zeros((1, self.full_config.task.env.numTargets), dtype=torch.float,
                                        device=self.device)

        # Keep track of history
        self.obs_queue = torch.zeros((self.num_envs,
                                      self.full_config.task.env.numObsHist * self.num_observations),
                                     dtype=torch.float, device=self.device)
        self.obs_stud_queue = torch.zeros((self.num_envs, self.stud_hist_len, self.num_obs_stud),
                                          dtype=torch.float, device=self.device)
        # tactile buffers
        self.num_channels = self.cfg_tactile.encoder.num_channels
        self.width = self.train_config.tactile_width # // 2 if self.cfg_tactile.crop_roi else self.train_config.tactile_width
        self.height = self.train_config.tactile_height
        self.width_rec = 112
        self.height_rec = 224

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

        self.tactile_record_imgs = torch.zeros(
            (1, 3,  # left, right, bottom
             self.num_channels, self.width_rec, self.height_rec),
            device=self.device,
            dtype=torch.float,
        )

        self.img_queue = torch.zeros(
            (1, self.img_hist_len, self.res[1], self.res[0]),
            device=self.device,
            dtype=torch.float,
        )
        self.image_buf = torch.zeros(1, self.res[1], self.res[0]).to(self.device)
        self.rgb_buf = torch.zeros(1, 3, 480, 640).to(self.device)

        self.seg_queue = torch.zeros((1, self.img_hist_len, self.res[1], self.res[0]),
                                     device=self.device, dtype=torch.float,)
        self.seg_buf = torch.zeros(1, self.res[1], self.res[0]).to(self.device)

        num_points = self.full_config.task.env.num_points

        if self.full_config.task.env.merge_socket_pcl:
            num_points += self.full_config.task.env.num_points_socket

        if self.full_config.task.env.merge_goal_pcl:
            self.goal_pcl = torch.zeros((self.num_envs, self.full_config.task.env.num_points_goal, 3),
                                        device=self.device, dtype=torch.float)

            num_points += self.full_config.task.env.num_points_goal

        self.pcl_queue = torch.zeros((self.num_envs, self.full_config.task.env.numObsStudentHist, num_points, 3),
                                     dtype=torch.float, device=self.device)

        self.pcl = torch.zeros((self.num_envs, num_points, 3), device=self.device, dtype=torch.float)

        self.ft_data = torch.zeros((1, 6), device=self.device, dtype=torch.float)
        self.obs_buf = torch.zeros((1, self.obs_shape[0]), device=self.device, dtype=torch.float)

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
                self.deploy_config.env.socket_pos_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.noisy_socket_pos = torch.zeros_like(
            self.socket_pos, dtype=torch.float32, device=self.device
        )

        self.socket_tip = fc.translate_along_local_z(pos=self.socket_pos,
                                                     quat=self.identity_quat,
                                                     offset=self.socket_height,
                                                     device=self.device)

        self.noisy_socket_pos[:, 0] = self.socket_tip[:, 0] + socket_obs_pos_noise[:, 0]
        self.noisy_socket_pos[:, 1] = self.socket_tip[:, 1] + socket_obs_pos_noise[:, 1]
        self.noisy_socket_pos[:, 2] = self.socket_tip[:, 2] + socket_obs_pos_noise[:, 2]

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

        self.plug_hand_pos, self.plug_hand_quat = self._pose_world_to_hand_base(self.plug_pos, self.plug_quat)

    def compute_observations(self, with_tactile=True, with_img=False, with_pcl=False, display_image=True,
                             with_priv=False, with_stud=True, diff_tac=False, record=True):

        obses = self.env.get_obs()

        arm_joints = obses['joints']
        ee_pose = obses['ee_pose']
        ft = obses['ft']

        self.ft_data = torch.tensor(ft, device=self.device).unsqueeze(0)
        self.arm_dof_pos = torch.tensor(arm_joints, device=self.device).unsqueeze(0)
        self.fingertip_centered_pos = torch.tensor(ee_pose[:3], device=self.device, dtype=torch.float).unsqueeze(0)
        self.fingertip_centered_quat = torch.tensor(ee_pose[3:], device=self.device, dtype=torch.float).unsqueeze(0)

        # eef_pos = torch.cat((self.fingertip_centered_pos,
        #                      quat2R(self.fingertip_centered_quat).reshape(1, -1)), dim=-1)

        if with_tactile or record:

            left, right, bottom = obses['frames']
            # Cutting by half
            if self.cfg_tactile.crop_roi:
                w = left.shape[0]
                left = left[:w // 2, :, :]
                right = right[:w // 2, :, :]
                bottom = bottom[:w // 2, :, :]

            left_rec = cv2.resize(left.copy(), (self.height_rec, self.width_rec), interpolation=cv2.INTER_AREA)
            right_rec = cv2.resize(right.copy(), (self.height_rec, self.width_rec), interpolation=cv2.INTER_AREA)
            bottom_rec = cv2.resize(bottom.copy(), (self.height_rec, self.width_rec), interpolation=cv2.INTER_AREA)

            self.tactile_record_imgs[0, 0] = to_torch(cv2.cvtColor(left_rec.astype('float32'), cv2.COLOR_BGR2GRAY)).to(
                self.device)
            self.tactile_record_imgs[0, 1] = to_torch(cv2.cvtColor(right_rec.astype('float32'), cv2.COLOR_BGR2GRAY)).to(
                self.device)
            self.tactile_record_imgs[0, 2] = to_torch(cv2.cvtColor(bottom_rec.astype('float32'), cv2.COLOR_BGR2GRAY)).to(
                self.device)

            # Resizing to encoder size
            left = cv2.resize(left, (self.height, self.width), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (self.height, self.width), interpolation=cv2.INTER_AREA)
            bottom = cv2.resize(bottom, (self.height, self.width), interpolation=cv2.INTER_AREA)

            if self.tactile_grasp_flag:
                self.f_left = left.copy()
                self.f_right = right.copy()
                self.f_bottom = bottom.copy()
                self.tactile_grasp_flag = False

            if diff_tac:
                left -= self.f_left
                right -= self.f_right
                bottom -= self.f_bottom

            if self.num_channels == 3:
                self.tactile_imgs[0, 0] = to_torch(left).permute(2, 0, 1).to(self.device)
                self.tactile_imgs[0, 1] = to_torch(right).permute(2, 0, 1).to(self.device)
                self.tactile_imgs[0, 2] = to_torch(bottom).permute(2, 0, 1).to(self.device)
            else:
                self.tactile_imgs[0, 0] = to_torch(cv2.cvtColor(left.astype('float32'), cv2.COLOR_BGR2GRAY)).to(
                    self.device)
                self.tactile_imgs[0, 1] = to_torch(cv2.cvtColor(right.astype('float32'), cv2.COLOR_BGR2GRAY)).to(
                    self.device)
                self.tactile_imgs[0, 2] = to_torch(cv2.cvtColor(bottom.astype('float32'), cv2.COLOR_BGR2GRAY)).to(
                    self.device)

            self.tactile_queue[:, 1:] = self.tactile_queue[:, :-1].clone().detach()
            self.tactile_queue[:, 0, ...] = self.tactile_imgs

            # if display_image:
                # cv2.imshow("Hand View\tLeft\tRight\tMiddle", np.concatenate((left, right, bottom), axis=1))
                # cv2.waitKey(1)

        if with_img or record: # TODO: modify! just for the record
            img = obses['img']
            self.image_buf[0] = to_torch(img[0]).to(self.device)
            self.img_queue[:, 1:] = self.img_queue[:, :-1].clone().detach()
            self.img_queue[:, 0, ...] = self.image_buf

            seg = obses['seg']
            self.seg_buf[0] = to_torch(seg).to(self.device)
            self.seg_queue[:, 1:] = self.seg_queue[:, :-1].clone().detach()
            self.seg_queue[:, 0, ...] = self.seg_buf

            self.rgb_buf[0] = to_torch(obses['rgb']).permute(2, 0, 1).to(self.device)
            if display_image:
                cv2.imshow("Depth Image", img.transpose(1, 2, 0))
                cv2.waitKey(1)

        if with_pcl or record:
            pcl = obses['pcl']
            self.pcl[0] = to_torch(pcl).to(self.device)
            self.pcl_queue[:, 1:] = self.pcl_queue[:, :-1].clone().detach()
            self.pcl_queue[:, 0, ...] = self.pcl

        # some-like taking a new socket pose measurement
        self._update_socket_pose()

        eef_pos = torch.cat((self.fingertip_centered_pos,
                             self.rot_tf.forward(self.fingertip_centered_quat)), dim=-1)

        action = self.actions.clone()
        noisy_delta_pos = self.noisy_gripper_goal_pos - self.fingertip_centered_pos

        obs = torch.cat([eef_pos,
                         action,
                         # noisy_delta_pos
                         ], dim=-1)

        self.obs_queue[:, :-self.num_observations] = self.obs_queue[:, self.num_observations:]
        self.obs_queue[:, -self.num_observations:] = obs

        self.obs_buf = self.obs_queue.clone()  # shape = (num_envs, num_observations)

        obs_dict = {'obs': self.obs_buf}

        if with_stud:
            # eef_stud = torch.cat((self.fingertip_centered_pos, quat2R(self.fingertip_centered_quat).reshape(1, -1)),
            #                      dim=-1)
            # # fix bug
            # eef_stud = torch.cat((self.fingertip_centered_pos,
            #                       self.stud_tf.forward(eef_stud[:, 3:].reshape(eef_stud.shape[0], 3, 3))), dim=1)

            # if self.train_config.from_offline:
            #     eef_stud = (eef_stud - self.stats["mean"]["eef_pos_rot6d"]) / self.stats["std"]["eef_pos_rot6d"]
            #     socket_pos = (self.socket_pos - self.stats["mean"]["socket_pos"][:3]) / self.stats["std"]["socket_pos"][:3]

            obs_stud = torch.cat([eef_pos,
                                  action,
                                  ], dim=-1)

            self.obs_stud_queue[:, 1:] = self.obs_stud_queue[:, :-1].clone().detach()
            self.obs_stud_queue[:, 0, :] = obs_stud

            obs_dict['student_obs'] = self.obs_stud_queue.clone()

        if with_tactile:
            obs_dict['tactile'] = self.tactile_queue.clone()
        if with_img:
            obs_dict['img'] = self.img_queue.clone()
            obs_dict['seg'] = self.seg_queue.clone()

        if with_pcl:
            obs_dict['pcl'] = self.pcl_queue.clone()

        if with_priv:
            # Compute privileged info (gt_error + contacts)
            self._update_plug_pose()
            plug_hand_pos = self.plug_hand_pos.clone()
            plug_hand_quat = self.plug_hand_quat.clone()

            state_tensors = [
                plug_hand_pos,  # 3
                plug_hand_quat,  # 7
            ]
            obs_dict['priv_info'] = torch.cat(state_tensors, dim=-1)

        return obs_dict

    def _update_reset_buf(self):

        timeout = (self.episode_length >= self.max_episode_length - 1)
        self.done = timeout

        key = cv2.waitKey(1) & 0xFF  # Wait for a key press (non-blocking)
        if key == ord('q'):
            print("External reset triggered by keypress 'q'")
            self.done[0, 0] = True

        if self.done[0, 0].item():
            print('Reset because ',
                  "timeoout" if timeout.item() else "", )

    def reset(self):

        joints_above_socket = self.deploy_config.common_poses.joints_above_socket
        joints_above_plug = self.deploy_config.common_poses.joints_above_plug

        # Move above the socket
        self.env.arm.move_manipulator.scale_vel(scale_vel=0.1, scale_acc=0.1)
        self.env.move_to_joint_values(joints_above_socket, wait=True)
        # Set random init error
        self.env.set_random_init_error(self.socket_pos, with_tracker=False)
        # If not inserted, return plug?
        if False:  # not self.inserted and False:

            self.env.arm.move_manipulator.scale_vel(scale_vel=0.5, scale_acc=0.5)
            self.env.move_to_joint_values(joints_above_plug, wait=True)
            self.env.arm.move_manipulator.scale_vel(scale_vel=0.1, scale_acc=0.1)
            self.env.align_and_release(init_plug_pose=[0.4103839067235552, 0.17531695171951858, 0.008])

            self.grasp_and_init()

        # self.env.randomize_grasp()

        self.env.arm.move_manipulator.scale_vel(scale_vel=0.02, scale_acc=0.02)
        # self.inserted[...] = False
        self.done[...] = False
        self.episode_length[...] = 0.

    def grasp_and_init(self):

        skip = True
        joints_above_socket = self.deploy_config.common_poses.joints_above_socket
        joints_above_plug = self.deploy_config.common_poses.joints_above_plug
        # Move above plug

        if not skip:
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

        self.env.set_random_init_error(self.socket_pos, with_tracker=False)
        self.env.grasp()

        self.env.arm.move_manipulator.scale_vel(scale_vel=0.02, scale_acc=0.02)
        print('Starting Insertion')
        self.tactile_grasp_flag = True

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
            condition_mask = torch.abs(ft[:, 2]) > 3.0
            actions[:, 2] = torch.where(condition_mask, torch.clamp(actions[:, 2], min=0.0), actions[:, 2])
            # actions = torch.where(torch.abs(ft) > 1.5, torch.clamp(actions, min=0.0), actions)
            # print("Error:", np.round(self.plug_pos_error[0].cpu().numpy(), 4))
            print("Regularized Actions:", np.round(actions[0][:].cpu().numpy(), 4))

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

    def process_obs(self, obs, obj_id=2, socket_id=3, distinct=True, display=False):

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

        if self.pcl_info:
            pcl = self.pcl_mean_std(pcl.reshape(-1, 3)).reshape((obs['pcl'].shape[0], -1, 3))

        if student_obs is not None:
            if self.stats is not None and self.train_config.from_offline:
                assert NotImplementedError
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
            display_obs(img, seg, pcl, ax=self.ax)

        return student_dict

    def deploy(self):

        self._initialize_grasp_poses()
        from algo.deploy.env.env import ExperimentEnv
        rospy.init_node('DeployEnv')

        ext_cam = self.deploy_config.env.ext_cam
        pcl = self.deploy_config.env.pcl
        tactile = self.deploy_config.env.tactile

        self.env = ExperimentEnv(with_ext_cam=ext_cam,
                                 with_pcl=pcl,
                                 with_tactile=tactile)

        self.env.arm.move_manipulator.scale_vel(scale_vel=0.004, scale_acc=0.004)

        rospy.logwarn('Finished setting the env, lets play.')

        hz = 20
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

        self.grasp_and_init()

        num_episodes = self.deploy_config.data_logger.total_trajectories
        cur_episode = 0

        while cur_episode < num_episodes:

            # self.env.set_random_init_error(true_socket_pose=true_socket_pose)
            # self.env.grasp()  # little squeeze never hurts

            # Bias the ft sensor
            self.env.arm.calib_robotiq()
            self.env.arm.calib_robotiq()

            obs_dict = self.compute_observations(with_priv=self.priv_info,
                                                 with_tactile=self.tactile_info,
                                                 with_pcl=self.pcl_info,
                                                 with_img=self.img_info)

            prep_obs = self.process_obs(obs_dict)

            student_dict = {
                'student_obs': prep_obs['student_obs'] if 'student_obs' in prep_obs else None,
                'tactile': prep_obs['tactile'] if 'tactile' in prep_obs else None,
                'img': prep_obs['img'] if 'img' in prep_obs else None,
                'seg': prep_obs['seg'] if 'seg' in prep_obs else None,
                'pcl': prep_obs['pcl'] if 'pcl' in prep_obs else None,
            }

            self._update_reset_buf()

            for i in range(self.full_config.task.env.numObsHist):
                pass

            while not self.done[0, 0]:

                latent, _ = self.student.predict(student_dict, requires_grad=False)

                if not self.train_config.only_bc:

                    input_dict = {
                        'obs': self.running_mean_std(obs_dict['obs'].clone()),
                        'latent': latent
                    }

                    action, latent = self.agent.act_inference(input_dict)
                else:
                    action = latent

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
                obs_dict = self.compute_observations(with_priv=self.priv_info,
                                                     with_tactile=self.tactile_info,
                                                     with_pcl=self.pcl_info,
                                                     with_img=self.img_info)

                prep_obs = self.process_obs(obs_dict)

                student_dict = {
                    'student_obs': prep_obs['student_obs'] if 'student_obs' in prep_obs else None,
                    'tactile': prep_obs['tactile'] if 'tactile' in prep_obs else None,
                    'img': prep_obs['img'] if 'img' in prep_obs else None,
                    'seg': prep_obs['seg'] if 'seg' in prep_obs else None,
                    'pcl': prep_obs['pcl'] if 'pcl' in prep_obs else None,

                }

            self.env.arm.move_manipulator.stop_motion()
            self.reset()
