##### !/usr/bin/env /home/osher/Desktop/isaacgym/venv/bin/python3
# --------------------------------------------------------
# now its our turn.
# https://arxiv.org/abs/todo
# Copyright (c) 2023 Osher & Co.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import rospy
from algo.models.models import ActorCritic
from algo.models.running_mean_std import RunningMeanStd
import torch
import os
import hydra
import cv2
from isaacgyminsertion.utils import torch_jit_utils
import isaacgyminsertion.tasks.factory_tactile.factory_control as fc
from time import time
import numpy as np


class HardwarePlayer(object):
    def __init__(self, output_dir, full_config):

        self.deploy_config = full_config.deploy

        self.pos_scale = self.deploy_config.rl.pos_action_scale
        self.rot_scale = self.deploy_config.rl.rot_action_scale
        self.device = full_config["rl_device"]

        # ---- build environment ----
        self.obs_shape = (self.deploy_config.env.numObservations,)
        self.num_actions = self.deploy_config.env.numActions
        self.num_targets = self.deploy_config.env.numTargets

        # ---- Tactile Info ---
        self.tactile_info = self.deploy_config.ppo.tactile_info
        self.tactile_seq_length = self.deploy_config.ppo.tactile_seq_length
        self.tactile_info_dim = self.deploy_config.network.tactile_mlp.units[0]
        self.mlp_tactile_info_dim = self.deploy_config.network.tactile_mlp.units[0]
        self.tactile_input_dim = (self.deploy_config.network.tactile_decoder.img_width,
                                  self.deploy_config.network.tactile_decoder.img_height,
                                  self.deploy_config.network.tactile_decoder.num_channels)
        # ---- ft Info --- currently ft isn't supported
        self.ft_info = self.deploy_config.ppo.ft_info
        self.ft_seq_length = self.deploy_config.ppo.ft_seq_length
        self.ft_input_dim = self.deploy_config.ppo.ft_input_dim
        self.ft_info_dim = self.ft_input_dim * self.ft_seq_length
        # ---- Priv Info ----
        self.priv_info = self.deploy_config.ppo.priv_info
        self.priv_info_dim = self.deploy_config.ppo.priv_info_dim
        self.extrin_adapt = self.deploy_config.ppo.extrin_adapt
        # ---- Obs Info (student)----
        self.obs_info = self.deploy_config.ppo.obs_info
        self.student_obs_input_shape = self.deploy_config.ppo.student_obs_input_shape

        net_config = {
            'actor_units': self.deploy_config.network.mlp.units,
            'actions_num': self.num_actions,
            'priv_mlp_units': self.deploy_config.network.priv_mlp.units,
            'input_shape': self.obs_shape,
            'extrin_adapt': True,
            'priv_info_dim': self.priv_info_dim,
            'priv_info': True,
            "tactile_info": self.tactile_info,
            "obs_info": self.obs_info,
            'student_obs_input_shape': self.student_obs_input_shape,
            "mlp_tactile_input_shape": self.mlp_tactile_info_dim,
            "ft_input_shape": self.ft_info_dim,
            "ft_info": self.ft_info,
            "mlp_tactile_units": self.deploy_config.network.tactile_mlp.units,
            "tactile_decoder_embed_dim": self.deploy_config.network.tactile_mlp.units[0],
            "ft_units": self.deploy_config.network.ft_mlp.units,
            'tactile_input_dim': self.tactile_input_dim,
            'tactile_seq_length': self.tactile_seq_length,
        }

        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.running_mean_std_stud = RunningMeanStd((self.student_obs_input_shape,)).to(self.device)
        self.running_mean_std_stud.eval()
        # ---- Output Dir ----
        self.output_dir = output_dir
        self.dp_dir = os.path.join(self.output_dir, 'deploy')
        os.makedirs(self.dp_dir, exist_ok=True)

        tactile_info_path = '../allsight/experiments/conf/test.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_tactile = hydra.compose(config_name=tactile_info_path)['']['']['']['allsight']['experiments']['conf']

        asset_info_path = '../../assets/factory/yaml/factory_asset_info_insertion.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_insertion = hydra.compose(config_name=asset_info_path)
        self.asset_info_insertion = self.asset_info_insertion['']['']['']['']['']['']['assets']['factory'][
            'yaml']  # strip superfluous nesting

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

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        self.gripper_normal_quat = torch.tensor([-1 / 2 ** 0.5, -1 / 2 ** 0.5, 0.0, 0.0], device=self.device).unsqueeze(
            0)

        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0)

        self.plug_grasp_pos_local = self.plug_height * 0.5 * torch.tensor([0.0, 0.0, 1.0],
                                                                          device=self.device).unsqueeze(0)
        self.plug_grasp_quat_local = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0)

        self.plug_tip_pos_local = self.plug_height * torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0)
        self.socket_tip_pos_local = self.socket_height * torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0)

        self.actions = torch.zeros((1, self.num_actions), device=self.device)
        self.targets = torch.zeros((1, self.deploy_config.env.numTargets), device=self.device)
        self.prev_targets = torch.zeros((1, self.deploy_config.env.numTargets), dtype=torch.float, device=self.device)

        # Keep track of history
        self.arm_joint_queue = torch.zeros((1, self.deploy_config.env.obs_seq_length, 7), dtype=torch.float,
                                           device=self.device)
        self.arm_vel_queue = torch.zeros((1, self.deploy_config.env.obs_seq_length, 7), dtype=torch.float,
                                         device=self.device)
        self.actions_queue = torch.zeros((1, self.deploy_config.env.obs_seq_length, self.num_actions),
                                         dtype=torch.float, device=self.device)
        self.targets_queue = torch.zeros((1, self.deploy_config.env.obs_seq_length, self.num_targets),
                                         dtype=torch.float, device=self.device)
        self.eef_queue = torch.zeros((1, self.deploy_config.env.obs_seq_length, 7),
                                     dtype=torch.float, device=self.device)
        self.goal_noisy_queue = torch.zeros((1, self.deploy_config.env.obs_seq_length, 7),
                                            dtype=torch.float, device=self.device)

        # Bad, should queue the obs!
        self.arm_joint_queue_student = torch.zeros((1, self.deploy_config.env.stud_obs_seq_length, 7),
                                                   dtype=torch.float, device=self.device)
        self.arm_vel_queue_student = torch.zeros((1, self.deploy_config.env.stud_obs_seq_length, 7),
                                                 dtype=torch.float, device=self.device)
        self.actions_queue_student = torch.zeros((1, self.deploy_config.env.stud_obs_seq_length, self.num_actions),
                                                 dtype=torch.float, device=self.device)
        self.targets_queue_student = torch.zeros((1, self.deploy_config.env.stud_obs_seq_length, self.num_actions),
                                                 dtype=torch.float, device=self.device)
        self.eef_queue_student = torch.zeros((1, self.deploy_config.env.stud_obs_seq_length, 7),
                                             dtype=torch.float, device=self.device)
        self.goal_noisy_queue_student = torch.zeros((1, self.deploy_config.env.stud_obs_seq_length, 7),
                                                    dtype=torch.float, device=self.device)

        # tactile buffers
        self.tactile_imgs = torch.zeros(
            (1, 3,  # left, right, bottom
             self.cfg_tactile.decoder.width, self.cfg_tactile.decoder.height, 3),
            device=self.device,
            dtype=torch.float,
        )
        # Way too big tensor.
        self.tactile_queue = torch.zeros(
            (1, self.tactile_seq_length, 3,  # left, right, bottom
             self.cfg_tactile.decoder.width, self.cfg_tactile.decoder.height, 3),
            device=self.device,
            dtype=torch.float,
        )

        self.ft_queue = torch.zeros((1, self.ft_seq_length, 6), device=self.device, dtype=torch.float)

        self.obs_buf = torch.zeros((1, self.obs_shape[0]), device=self.device, dtype=torch.float)

    def _set_socket_pose(self, pos):

        self.socket_pos = torch.tensor(pos, device=self.device).unsqueeze(0)

    def _set_plug_pose(self, pos):

        self.plug_pos = torch.tensor(pos, device=self.device).unsqueeze(0)

    def _update_socket_pose(self):

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

        self.noisy_socket_pos[:, 0] = self.socket_pos[:, 0] + socket_obs_pos_noise[:, 0]
        self.noisy_socket_pos[:, 1] = self.socket_pos[:, 1] + socket_obs_pos_noise[:, 1]
        self.noisy_socket_pos[:, 2] = self.socket_pos[:, 2] + socket_obs_pos_noise[:, 2]

        self.noisy_gripper_goal_quat, self.noisy_gripper_goal_pos = torch_jit_utils.tf_combine_deploy(
            self.identity_quat,
            self.noisy_socket_pos,
            self.gripper_normal_quat,
            self.socket_tip_pos_local)

    def compute_observations(self):
        obses = self.env.get_obs()

        arm_joints = obses['joints']
        ee_pose = obses['ee_pose']
        left, right, bottom = obses['frames']
        ft = obses['ft']

        self.arm_joint_queue[:, 1:] = self.arm_joint_queue[:, :-1].clone().detach()
        self.arm_joint_queue[:, 0, :] = torch.tensor(arm_joints, device=self.device, dtype=torch.float)

        self.arm_joint_queue_student[:, 1:] = self.arm_joint_queue_student[:, :-1].clone().detach()
        self.arm_joint_queue_student[:, 0, :] = torch.tensor(arm_joints, device=self.device, dtype=torch.float)

        self.fingertip_centered_pos = torch.tensor(ee_pose[:3], device=self.device, dtype=torch.float).unsqueeze(0)
        self.fingertip_centered_quat = torch.tensor(ee_pose[3:], device=self.device, dtype=torch.float).unsqueeze(0)

        self.eef_queue[:, 1:] = self.eef_queue[:, :-1].clone().detach()
        self.eef_queue[:, 0, :] = torch.cat((self.fingertip_centered_pos.clone(),
                                             self.fingertip_centered_quat.clone()), dim=-1)

        self.eef_queue_student[:, 1:] = self.eef_queue_student[:, :-1].clone().detach()
        self.eef_queue_student[:, 0, :] = torch.cat((self.fingertip_centered_pos.clone(),
                                                     self.fingertip_centered_quat.clone()), dim=-1)

        left = cv2.resize(left, (self.cfg_tactile.decoder.width, self.cfg_tactile.decoder.height),
                          interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (self.cfg_tactile.decoder.width, self.cfg_tactile.decoder.height),
                           interpolation=cv2.INTER_AREA)
        bottom = cv2.resize(bottom, (self.cfg_tactile.decoder.width, self.cfg_tactile.decoder.height),
                            interpolation=cv2.INTER_AREA)

        self.tactile_imgs[0, 0] = torch_jit_utils.rgb_transform(left).to(self.device).permute(1, 2, 0)
        self.tactile_imgs[0, 1] = torch_jit_utils.rgb_transform(right).to(self.device).permute(1, 2, 0)
        self.tactile_imgs[0, 2] = torch_jit_utils.rgb_transform(bottom).to(self.device).permute(1, 2, 0)

        self.tactile_queue[:, 1:] = self.tactile_queue[:, :-1].clone().detach()
        self.tactile_queue[:, 0, :] = self.tactile_imgs

        self.ft_queue[:, 1:] = self.ft_queue[:, :-1].clone().detach()
        self.ft_queue[:, 0, :] = torch.tensor(ft, device=self.device, dtype=torch.float)

        obs_tensors = [
            self.arm_joint_queue.reshape(1, -1),  # 7 * hist
            self.eef_queue.reshape(1, -1),  # (envs, 7 * hist)
            self.goal_noisy_queue.reshape(1, -1),  # (envs, 7 * hist)
            self.actions_queue.reshape(1, -1),  # (envs, 6 * hist)
            self.targets_queue.reshape(1, -1),  # (envs, 6 * hist)
        ]

        obs_tensors_student = [
            self.arm_joint_queue_student.reshape(1, -1),  # 7 * hist
            self.eef_queue_student.reshape(1, -1),  # (envs, 7 * hist)
            self.goal_noisy_queue_student.reshape(1, -1),  # (envs, 7 * hist)
            self.actions_queue_student.reshape(1, -1),  # (envs, 6 * hist)
            self.targets_queue_student.reshape(1, -1),  # (envs, 6 * hist)
        ]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)
        self.obs_buf_stud = torch.cat(obs_tensors_student, dim=-1)

        return self.obs_buf, self.obs_buf_stud, self.tactile_queue

    def _move_arm_to_desired_pose(self, desired_pos, desired_rot=None):
        """Move gripper to desired pose."""

        self.ctrl_target_fingertip_centered_pos = torch.tensor(desired_pos, device=self.device).unsqueeze(0)

        if desired_rot is None:
            ctrl_target_fingertip_centered_euler = torch.tensor(self.deploy_config.env.fingertip_midpoint_rot_initial,
                                                                device=self.device).unsqueeze(0)

            self.ctrl_target_fingertip_centered_quat = torch_jit_utils.quat_from_euler_xyz_deploy(
                ctrl_target_fingertip_centered_euler[:, 0],
                ctrl_target_fingertip_centered_euler[:, 1],
                ctrl_target_fingertip_centered_euler[:, 2])
        else:
            self.ctrl_target_fingertip_centered_quat = torch.tensor(desired_rot, device=self.device).unsqueeze(0)

        cfg_ctrl = {'num_envs': 1,
                    'jacobian_type': 'geometric'}

        for _ in range(3):

            info = self.env.get_info_for_control()
            ee_pose = info['ee_pose']
            self.fingertip_centered_pos = torch.tensor(ee_pose[:3], device=self.device, dtype=torch.float).unsqueeze(0)
            self.fingertip_centered_quat = torch.tensor(ee_pose[3:], device=self.device, dtype=torch.float).unsqueeze(0)

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
            self.apply_action(actions=actions, do_scale=False, do_clamp=False, wait=True)


    def update_and_apply_action(self, actions, wait=True):

        self.actions = actions.clone().to(self.device)

        delta_targets = torch.cat([
            self.actions[:, :3] @ torch.diag(torch.tensor(self.pos_scale, device=self.device)),  # 3
            self.actions[:, 3:6] @ torch.diag(torch.tensor(self.rot_scale, device=self.device))  # 3
        ], dim=-1).clone()

        # Update targets
        self.targets = self.prev_targets + delta_targets

        # update the queue
        self.actions_queue[:, 1:] = self.actions_queue[:, :-1].clone().detach()
        self.actions_queue[:, 0, :] = self.actions

        self.actions_queue_student[:, 1:] = self.actions_queue_student[:, :-1].clone().detach()
        self.actions_queue_student[:, 0, :] = self.actions

        self.targets_queue[:, 1:] = self.targets_queue[:, :-1].clone().detach()
        self.targets_queue[:, 0, :] = self.targets
        self.targets_queue_student[:, 1:] = self.targets_queue_student[:, :-1].clone().detach()
        self.targets_queue_student[:, 0, :] = self.targets
        self.prev_targets[:] = self.targets.clone()

        # some-like taking a new socket pose measurement
        self._update_socket_pose()
        self.goal_noisy_queue[:, 1:] = self.goal_noisy_queue[:, :-1].clone().detach()
        self.goal_noisy_queue[:, 0, :] = torch.cat((self.noisy_gripper_goal_pos.clone(),
                                                    self.noisy_gripper_goal_quat.clone()),
                                                   dim=-1)
        self.goal_noisy_queue_student[:, 1:] = self.goal_noisy_queue_student[:, :-1].clone().detach()
        self.goal_noisy_queue_student[:, 0, :] = torch.cat((self.noisy_gripper_goal_pos.clone(),
                                                            self.noisy_gripper_goal_quat.clone()),
                                                           dim=-1)
        self.apply_action(self.actions, wait=wait)

    def apply_action(self, actions, do_scale=True, do_clamp=True, wait=True):

        # Apply the action
        if do_clamp:
            actions = torch.clamp(actions, -1.0, 1.0)
        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.pos_scale, device=self.device))
        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.rot_scale, device=self.device))

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

        self.generate_ctrl_signals(wait=wait)

    def generate_ctrl_signals(self, wait=True):

        ctrl_info = self.env.get_info_for_control()

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

        self.ctrl_target_dof_pos = self.ctrl_target_dof_pos[:, :7]
        target_joints = self.ctrl_target_dof_pos.cpu().detach().numpy().squeeze().tolist()

        try:
            self.env.move_to_joint_values(target_joints, wait=wait)
        except:
            print(f'failed to reach {target_joints}')

    def restore(self, fn):
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.model.load_state_dict(checkpoint['model'])

    def deploy(self):

        from algo.deploy.env.env import ExperimentEnv

        rospy.init_node('DeployEnv')
        self.env = ExperimentEnv()
        rospy.logwarn('Finished setting the env, lets play.')

        # Wait for connections.
        rospy.sleep(0.5)

        hz = 100
        ros_rate = rospy.Rate(hz)

        self._create_asset_info()
        self._acquire_task_tensors()
        self.env.move_to_init_state()

        true_socket_pose = [self.deploy_config.env.kuka_depth, 0.0, self.deploy_config.env.table_height]
        above_socket_pose = [x + y for x, y in zip(true_socket_pose, [0, 0, 0.1])]
        self._set_socket_pose(pos=true_socket_pose)

        true_plug_pose = [self.deploy_config.env.kuka_depth, 0.3, self.deploy_config.env.table_height]
        self._set_plug_pose(pos=true_plug_pose)
        above_plug_pose = [x + y for x, y in zip(true_plug_pose, [0, 0, 0.2])]
        self._move_arm_to_desired_pose([0.5, 0, 0.2])

        # self.env.move_to_joint_values(self.env.joints_true_socket_pos)

        # above_plug_pose = [x + y for x, y in zip(true_plug_pose, [0, 0, 0.1])]
        # plug_grasp_pose = [x + y for x, y in zip(true_plug_pose, [0, 0, 0.05])]

        # # Move & grasp the object
        # self._move_arm_to_desired_pose(above_plug_pose)
        # self._move_arm_to_desired_pose(plug_grasp_pose)
        # self.env.grasp()
        #
        # self._move_arm_to_desired_pose(above_socket_pose)
        # # self._move_arm_to_desired_pose(true_socket_pose)

        obs, obs_stud, tactile = self.compute_observations()

        # TODO? Should we fill the history buffs?
        for i in range(self.deploy_config.env.obs_seq_length):
            pass

        while True:
            obs = self.running_mean_std(obs.clone())

            input_dict = {
                'obs': obs,
                'student_obs': obs_stud,
                'tactile_hist': tactile
            }
            action, _ = self.model.act_inference(input_dict)
            start_time = time()
            self.update_and_apply_action(action, wait=True)
            print("FPS: ", 1.0 / (time() - start_time))  # FPS = 1 / time to process loop

            ros_rate.sleep()

            obs, obs_stud, tactile = self.compute_observations()



'''
[x 0.5, y 0.3,z 0.1]
  a1: 0.396711885929
  a2: 0.644319176674
  a3: 0.220347777009
  a4: -1.55624902248
  a5: -0.160987839103
  a6: 0.95887452364
  a7: -0.90379679203
---

    x: 0.501488964447
    y: 0.00795072908259
    z: 0.171816062996
  a1: -0.098761215806
  a2: 0.379355311394
  a3: 0.142728880048
  a4: -1.75525605679
  a5: -0.0600436516106
  a6: 1.0154761076
  a7: -1.50801086426

'''