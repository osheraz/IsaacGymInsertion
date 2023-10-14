# --------------------------------------------------------
# now its our turn.
# https://arxiv.org/abs/todo
# Copyright (c) 2023 Osher & Co.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------


import numpy as np
import rospy

from algo.models.models import ActorCritic
from algo.models.running_mean_std import RunningMeanStd
import torch
import os
import hydra
import cv2
from isaacgyminsertion.utils import torch_jit_utils
import isaacgyminsertion.tasks.factory_tactile.factory_control as fc


class HardwarePlayer(object):
    def __init__(self, output_dir, full_config):

        self.pos_scale = full_config.task.rl.pos_action_scale
        self.rot_scale = full_config.task.rl.rot_action_scale

        self.device = full_config["rl_device"]
        # ------
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        self.env_config = full_config.task.env
        self.full_config = full_config
        # ---- build environment ----
        self.obs_shape = (self.env_config.numObservations,)
        self.num_actions = self.env_config.numActions
        self.num_targets = self.env_config.numTargets

        # ---- Tactile Info ---
        self.tactile_info = self.ppo_config.tactile_info
        self.tactile_seq_length = self.ppo_config.tactile_seq_length
        self.tactile_info_dim = self.network_config.tactile_mlp.units[0]
        # ---- ft Info --- TODO currently we dont use ft
        self.ft_info = self.ppo_config.ft_info
        self.ft_seq_length = self.ppo_config.ft_seq_length
        self.ft_input_dim = self.ppo_config.ft_input_dim
        self.ft_info_dim = self.ft_input_dim * self.ft_seq_length
        # ---- Priv Info ----
        self.priv_info = self.ppo_config.priv_info
        self.priv_info_dim = self.ppo_config.priv_info_dim
        self.extrin_adapt = self.ppo_config.extrin_adapt

        net_config = {
            'actor_units': self.network_config.mlp.units,
            'actions_num': self.num_actions,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'input_shape': self.obs_shape,
            'extrin_adapt': True,
            'priv_info_dim': self.priv_info_dim,
            'priv_info': True,
            "tactile_info": self.tactile_info,
            "tactile_input_shape": self.tactile_info_dim,
            "ft_input_shape": self.ft_info_dim,
            "ft_info": self.ft_info,
            "tactile_units": self.network_config.tactile_mlp.units,
            "tactile_decoder_embed_dim": self.network_config.tactile_mlp.units[0],
            "ft_units": self.network_config.ft_mlp.units,
        }

        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()

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

    def _create_asset_info(self, i):

        subassembly = self.full_config.env.desired_subassemblies[i]
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
        self.targets = torch.zeros((1, self.env_config.numTargets), device=self.device)
        self.prev_targets = torch.zeros((1, self.env_config.numTargets), dtype=torch.float, device=self.device)

        # Keep track of history
        self.arm_joint_queue = torch.zeros((1, self.env_config.numObsHist, 7), dtype=torch.float, device=self.device)
        self.arm_vel_queue = torch.zeros((1, self.env_config.numObsHist, 7), dtype=torch.float, device=self.device)
        self.actions_queue = torch.zeros((1, self.env_config.numObsHist, self.num_actions),
                                         dtype=torch.float, device=self.device)
        self.targets_queue = torch.zeros((1, self.env_config.numObsHist, self.num_targets),
                                         dtype=torch.float, device=self.device)
        self.eef_queue = torch.zeros((1, self.env_config.numObsHist, 7),
                                     dtype=torch.float, device=self.device)
        self.goal_noisy_queue = torch.zeros((1, self.env_config.numObsHist, 7),
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

    def _update_socket_pose(self):

        # TODO verify that everything is w.r.t robot base..

        socket_obs_pos_noise = 2 * (
                torch.rand((1, 3), dtype=torch.float32, device=self.device)
                - 0.5
        )
        socket_obs_pos_noise = socket_obs_pos_noise @ torch.diag(
            torch.tensor(
                self.full_config.task.env.deploy.socket_pos_obs_noise,
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

        self.noisy_gripper_goal_quat, self.noisy_gripper_goal_pos = torch_jit_utils.tf_combine(
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

        self.fingertip_centered_pos = torch.tensor(ee_pose[:3], device=self.device, dtype=torch.float)
        self.fingertip_centered_quat = torch.tensor(ee_pose[3:], device=self.device, dtype=torch.float)

        self.eef_queue[:, 1:] = self.eef_queue[:, :-1].clone().detach()
        self.eef_queue[:, 0, :] = torch.cat((self.fingertip_centered_pos.clone(),
                                             self.fingertip_centered_quat.clone()), dim=-1)

        left = cv2.resize(left, (self.cfg_tactile.decoder.width, self.cfg_tactile.decoder.height),
                          interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (self.cfg_tactile.decoder.width, self.cfg_tactile.decoder.height),
                           interpolation=cv2.INTER_AREA)
        bottom = cv2.resize(bottom, (self.cfg_tactile.decoder.width, self.cfg_tactile.decoder.height),
                            interpolation=cv2.INTER_AREA)

        self.tactile_imgs[1, 0] = torch_jit_utils.img_transform(left).to(self.device).permute(1, 2, 0)
        self.tactile_imgs[1, 1] = torch_jit_utils.img_transform(right).to(self.device).permute(1, 2, 0)
        self.tactile_imgs[1, 2] = torch_jit_utils.img_transform(bottom).to(self.device).permute(1, 2, 0)

        self.tactile_queue[:, 1:] = self.tactile_queue[:, :-1].clone().detach()
        self.tactile_queue[:, 0, :] = self.tactile_imgs

        self.ft_queue[:, 1:] = self.ft_queue[:, :-1].clone().detach()
        self.ft_queue[:, 0, :] = torch.tensor(ft, device=self.device, dtype=torch.float)

        obs_tensors = [
            self.arm_joint_queue.reshape(1, -1),  # 7 * hist
            # self.arm_vel_queue.reshape(1, -1),
            self.eef_queue.reshape(1, -1),  # (envs, 7 * hist)
            self.goal_noisy_queue.reshape(1, -1),  # (envs, 7 * hist)
            self.actions_queue.reshape(1, -1),  # (envs, 6 * hist)
            self.targets_queue.reshape(1, -1),  # (envs, 6 * hist)
        ]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)

        return self.obs_buf

    def update_and_apply_action(self, actions):

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

        self.targets_queue[:, 1:] = self.targets_queue[:, :-1].clone().detach()
        self.targets_queue[:, 0, :] = self.targets
        self.prev_targets[:] = self.targets.clone()

        # some-like taking a new socket pose measurement
        self._update_socket_pose()
        self.goal_noisy_queue[:, 1:] = self.goal_noisy_queue[:, :-1].clone().detach()
        self.goal_noisy_queue[:, 0, :] = torch.cat((self.noisy_gripper_goal_pos.clone(),
                                                    self.noisy_gripper_goal_quat.clone()),
                                                   dim=-1)

        self.apply_action(actions)

    def apply_action(self, actions, do_scale=True):

        # Apply the action
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
        rot_actions_quat = torch_jit_utils.quat_from_angle_axis(angle, axis)
        if self.full_config.task.rl.clamp_rot:
            rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                           rot_actions_quat,
                                           torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device))
        self.ctrl_target_fingertip_centered_quat = torch_jit_utils.quat_mul(rot_actions_quat,
                                                                            self.fingertip_centered_quat)

        self.generate_ctrl_signals()

    def generate_ctrl_signals(self):

        ctrl_info = self.env.get_info_for_control()

        fingertip_centered_jacobian_tf = torch.tensor(ctrl_info['jacob'],
                                                      device=self.device).unsqueeze(0)

        arm_dof_pos = torch.tensor(ctrl_info['joints'], device=self.device).unsqueeze(0)

        cfg_ctrl = {'num_envs': 1,
                    'jacobian_type': 'geometric'}

        self.ctrl_target_dof_pos = fc.compute_dof_pos_target(
            cfg_ctrl=cfg_ctrl,
            arm_dof_pos=arm_dof_pos,
            fingertip_midpoint_pos=self.fingertip_centered_pos,
            fingertip_midpoint_quat=self.fingertip_centered_quat,
            jacobian=fingertip_centered_jacobian_tf,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_centered_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_centered_quat,
            ctrl_target_gripper_dof_pos=0,
            device=self.device).squeeze(0)

        self.env.move_to_joint_values(self.ctrl_target_dof_pos.cpu().detach().numpy().tolist())

    def restore(self, fn):
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.model.load_state_dict(checkpoint['model'])

    def deploy(self):

        from algo.deploy.env.env import ExperimentEnv
        # try to set up rospy
        rospy.init_node('DeployEnv')
        self.env = ExperimentEnv()
        rospy.logwarn('Finished setting the env, lets play.')

        # Wait for connections.
        rospy.sleep(0.5)

        hz = 60
        ros_rate = rospy.Rate(hz)

        self.env.move_to_init_state()
        self.env.grasp_object()

        # TODO index the real objects
        self._create_asset_info(1)
        self._acquire_task_tensors()
        true_pose = [0, 0, 0]
        self._set_socket_pose(pos=true_pose)

        obs = self.compute_observations()

        # TODO? Should we fill the history buffs?
        for i in range(self.env_config.numObsHist):
            pass

        while True:
            obs = self.running_mean_std(obs.clone())

            input_dict = {
                'obs': obs,
                'tactile_hist': self.tactile_queue,
            }

            action = self.model.act_inference(input_dict)
            self.update_and_apply_action(action)

            ros_rate.sleep()

            obs = self.compute_observations()
