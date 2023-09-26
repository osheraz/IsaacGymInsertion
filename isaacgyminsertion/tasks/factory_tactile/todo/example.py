# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: Class for insertion task.

Inherits insertion environment class and abstract task class (not enforced). Can be executed with
python train.py task=FactoryTaskInsertion

Only the environment is provided; training a successful RL policy is an open research problem left to the user.
"""

import hydra
import math
import omegaconf
import os
import torch
import time
import sys
import numpy as np

import isaacgym
from isaacgym import gymapi, gymtorch, torch_utils
import isaacgyminsertion.tasks.factory.factory_control as fc
from isaacgyminsertion.tasks.factory.factory_env_insertion import FactoryEnvInsertion
from isaacgyminsertion.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgyminsertion.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask
from isaacgyminsertion.utils.torch_jit_utils import *
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as R


def axis_angle2quat(axis_angle_error):
    angle = torch.norm(-axis_angle_error, p=2, dim=-1)
    axis = -axis_angle_error / angle.unsqueeze(-1)
    quat = quat_from_angle_axis(angle, axis)
    return quat


def matrix_to_euler_xyz(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to Euler angles in the XYZ convention.
    Args:
        rotation_matrix: tensor of shape (..., 3, 3).
    Returns:
        Euler angles as tensor of shape (..., 3).
    """
    # Ensure the input matrix is of shape (..., 3, 3)
    assert rotation_matrix.shape[-2:] == (3, 3), "Invalid shape of rotation matrix"

    # Extract matrix components for readability
    r11 = rotation_matrix[..., 0, 0]
    r21 = rotation_matrix[..., 1, 0]
    r31 = rotation_matrix[..., 2, 0]
    r32 = rotation_matrix[..., 2, 1]
    r33 = rotation_matrix[..., 2, 2]

    # Compute Euler angles
    theta_y = torch.asin(-r31)
    theta_z = torch.atan2(r32, r33)
    theta_x = torch.atan2(r21, r11)

    return torch.stack((theta_x, theta_y, theta_z), -1)


def quat2euler(quat):
    euler = get_euler_xyz(quat)
    euler_x, euler_y, euler_z = euler
    euler = torch.stack([euler_x, euler_y, euler_z], dim=-1)
    return euler


def quat2euler2(quat):
    M = quaternion_to_matrix(quat)
    euler = matrix_to_euler_xyz(M)
    return euler


class FactoryTaskInsertion(FactoryEnvInsertion, FactoryABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize task superclass."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.cfg = cfg
        self._get_task_yaml_params()

        self._acquire_task_tensors()
        self.parse_controller_spec()

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        if self.viewer != None:
            self._set_viewer_params()

        # if self.viewer != None:
        #    self._set_viewer_params()

        if self.cfg_base.mode.export_scene:
            self.export_scene(label='franka_task_insertion')

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

        asset_info_path = '../../assets/factory/yaml/factory_asset_info_insertion.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_insertion = hydra.compose(config_name=asset_info_path)
        self.asset_info_insertion = self.asset_info_insertion['']['']['']['']['']['']['assets']['factory'][
            'yaml']  # strip superfluous nesting

        ppo_path = 'train/FactoryTaskInsertionPPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        # pass

        ####################################### Custom 2 (from nut_bolt_screw) #######################################################
        #
        #
        #

        target_heights = self.cfg_base.env.table_height + self.socket_heights + self.plug_heights * 0.5
        self.target_pos = target_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))

        # Plug-Socket tensors

        # self.plug_base_pos_local = torch.tensor(self.socket_heights, device=self.device).repeat((self.num_envs, 1)) * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
        # socket_heights = self.socket_heights + self.socket_depths
        # self.socket_tip_pos_local = torch.tensor(socket_heights, device=self.device).repeat((self.num_envs, 1)) * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))

        self.plug_base_pos_local = self.socket_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1))
        self.socket_tip_pos_local = self.socket_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1))

        # Keypoint tensors
        self.keypoint_offsets = self._get_keypoint_offsets(
            self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
        self.keypoints_plug = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3), dtype=torch.float32,
                                          device=self.device)
        self.keypoints_socket = torch.zeros_like(self.keypoints_plug, device=self.device)

        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs,
                                                                                                        1)

        self.actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)

    def _refresh_task_tensors(self):

        """Refresh tensors."""

        # Compute pos of keypoints on gripper, nut, and bolt in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_plug[:, idx] = torch_jit_utils.tf_combine(self.plug_quat,
                                                                     self.plug_pos,
                                                                     self.identity_quat,
                                                                     (keypoint_offset + self.plug_base_pos_local))[1]
            self.keypoints_socket[:, idx] = torch_jit_utils.tf_combine(self.socket_quat,
                                                                       self.socket_pos,
                                                                       self.identity_quat,
                                                                       (keypoint_offset + self.socket_tip_pos_local))[1]

        #
        #
        #
        ##############################################################################################################################

    def _refresh_task_tensors(self):
        """Refresh tensors."""

        # pass

        ####################################### Custom 3 (from nut_bolt_screw) #######################################################
        #
        #
        #

        self.fingerpad_midpoint_pos = fc.translate_along_local_z(pos=self.finger_midpoint_pos,
                                                                 quat=self.hand_quat,
                                                                 offset=self.asset_info_franka_table.franka_finger_length - self.asset_info_franka_table.franka_fingerpad_length * 0.5,
                                                                 device=self.device)
        self.finger_plug_keypoint_dist = self._get_keypoint_dist()  # self._get_keypoint_dist(body='finger_plug')
        self.plug_keypoint_dist = self._get_keypoint_dist()  # self._get_keypoint_dist(body='plug')
        self.plug_dist_to_target = torch.norm(self.target_pos - self.plug_com_pos, p=2,
                                              dim=-1)  # distance between plug COM and target
        self.plug_dist_to_fingerpads = torch.norm(self.fingerpad_midpoint_pos - self.plug_com_pos, p=2,
                                                  dim=-1)  # distance between plug COM and midpoint between centers of fingerpads

        #
        #
        #
        ##############################################################################################################################

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains."""

        # print('Actions:', actions.size(), actions)
        # print("Current action:", actions)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]
        # self.actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
        ################################################# Custom 4 ########################################################################
        #
        #
        #

        self._apply_actions_as_ctrl_targets(actions=self.actions, ctrl_target_gripper_dof_pos=0., do_scale=True)

        #
        #
        #
        ###################################################################################################################################

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

        self.progress_buf[:] += 1

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        """Compute observations."""

        ################################## Custom 5 ####################################################################################
        # Shallow copies of tensors
        obs_tensors = [self.fingertip_midpoint_pos,
                       self.fingertip_midpoint_quat,
                       self.fingertip_midpoint_linvel,
                       self.fingertip_midpoint_angvel,
                       self.plug_com_pos,
                       self.plug_com_quat,
                       self.plug_com_linvel,
                       self.plug_com_angvel]

        if self.cfg_task.rl.add_obs_finger_force:
            obs_tensors += [self.left_finger_force, self.right_finger_force]

        obs_tensors = torch.cat(obs_tensors, dim=-1)
        self.obs_buf[:, :obs_tensors.shape[-1]] = obs_tensors  # shape = (num_envs, num_observations)
        ################################################################################################################################

        return self.obs_buf  # shape = (num_envs, num_observations)

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        ##########################  Custom 6 ############################################################################################
        #
        #
        #
        # Get successful and failed envs at current timestep
        curr_successes = self._get_curr_successes()
        curr_failures = self._get_curr_failures(curr_successes)

        self._update_reset_buf(curr_successes, curr_failures)
        self._update_rew_buf(curr_successes)
        #
        #
        #
        #################################################################################################################################

        # self._update_rew_buf()
        # self._update_reset_buf()

    '''
    def render_indicator(self):
        eef_pose = to_torch(self.rb_states[self.hand_idxs, :7])
        z_pose = to_torch([0, 0, 1., 0., 0., 1., 1.]).repeat((self.num_envs, 1))
        z_transformed = apply_transform_to_pose_local_torch(eef_pose, z_pose)
        z_transformed[:, :3] = to_torch([0.5, 0, 1])
        self.reset_indicator(z_transformed)

    def render(self):
        if not self.is_headless:
            # update viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            # indicator
            self.render_indicator()
    '''

    def _update_rew_buf(self, curr_successes):
        """Compute reward at current timestep."""

        # pass

        ########################### Custom 7 ###########################################################################################
        #
        #
        #
        keypoint_reward = -(self.plug_keypoint_dist + self.finger_plug_keypoint_dist)
        action_penalty = torch.norm(self.actions, p=2, dim=-1)
        self.rew_buf[
        :] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale - action_penalty * self.cfg_task.rl.action_penalty_scale + curr_successes * self.cfg_task.rl.success_bonus
        # print('Reward:', self.rew_buf[:])

        #
        #
        #
        ################################################################################################################################

    def _update_reset_buf(self, curr_successes, curr_failures):
        """Assign environments for reset if successful or failed."""

        # pass

        ########################### Custom 8 ###########################################################################################
        #
        #
        #

        self.reset_buf[:] = torch.logical_or(curr_successes, curr_failures)

        #
        #
        #
        ################################################################################################################################

    def refresh_and_acquire_tensors(self):
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self._acquire_env_tensors()

    def zero_velocities(self, env_ids):

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])
        # Set DOF state
        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))
        self.gym.simulate(self.sim)
        self.render()

    def calculate_grasp_pose(self, env_ids, plug_pos_target, plug_quat_target, device):
        grasp_pos = plug_pos_target - torch.tensor([0.0, 0.0, .005], device=device).repeat((len(env_ids), 1)).double()

        grasp_gripper_euler = torch.tensor([np.pi, .0, np.pi], device=device).repeat((len(env_ids), 1)).double()

        grasp_gripper_quat = torch_utils.quat_from_euler_xyz(grasp_gripper_euler[:, 0],
                                                             grasp_gripper_euler[:, 1],
                                                             grasp_gripper_euler[:, 2])

        if self.cfg_task.randomize.initial_state == 'random':
            grasp_gripper_quat = torch_utils.quat_mul(self.delta_q, grasp_gripper_quat)

        return grasp_pos, grasp_gripper_quat

    def track_pose_grasp(self, env_ids, num_envs, sim_steps, jacobian_type, franka_gripper_width, device, target_pos,
                         target_quat, grasp, Kp_magnif=0.):

        # Step sim and render
        iters = 0

        while ((torch.norm(target_pos[0, 0].double() - self.fingerpad_midpoint_pos[0, 0].double(), p=2) >= .001)
               or (torch.norm(target_pos[0, 1].double() - self.fingerpad_midpoint_pos[0, 1].double(), p=2) >= .001)
               or (torch.norm(target_pos[0, 2].double() - self.fingerpad_midpoint_pos[0, 2].double(), p=2) >= .001)
               or torch.rad2deg(quat_diff_rad(self.fingertip_midpoint_quat[0, :].unsqueeze(0),
                                              target_quat[0, :].unsqueeze(0))) >= 1):
            iters += 1

            self.refresh_and_acquire_tensors()

            ctrl_target_fingertip_midpoint_pos = target_pos
            ctrl_target_fingertip_midpoint_quat = target_quat

            self.ctrl_target_fingertip_midpoint_pos = ctrl_target_fingertip_midpoint_pos
            self.ctrl_target_fingertip_midpoint_quat = ctrl_target_fingertip_midpoint_quat

            s_t_pos_error, s_t_axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingerpad_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=jacobian_type,
                rot_error_type='axis_angle')

            s_t_delta_hand_pose = torch.cat((s_t_pos_error, s_t_axis_angle_error), dim=-1)
            s_t_actions = torch.zeros((num_envs, self.cfg_task.env.numActions), device=device)
            s_t_actions[:, :6] = s_t_delta_hand_pose

            self._apply_actions_as_ctrl_targets(actions=s_t_actions,
                                                ctrl_target_gripper_dof_pos=franka_gripper_width,
                                                do_scale=False)
            self.zero_velocities(env_ids)

        self.zero_velocities(env_ids)
        self.refresh_and_acquire_tensors()

    def track_pose(self, env_ids, num_envs, sim_steps, jacobian_type, franka_gripper_width, device, target_pos,
                   target_quat, th_pos, th_rot, Kp_magnif=0., grasp=False, ):

        # Step sim and render
        iters = 0
        self.refresh_and_acquire_tensors()
        while iters < 100 and (
                (torch.norm(target_pos[0, 0].double() - self.plug_pos[0, 0].double(), p=2) >= th_pos) or (
                torch.norm(target_pos[0, 1].double() - self.plug_com_pos[0, 1].double(), p=2) >= th_pos) or (
                        torch.norm(target_pos[0, 2].double() - self.plug_com_pos[0, 2].double(),
                                   p=2) >= th_pos) or torch.rad2deg(
                quat_diff_rad(self.plug_quat[0, :].unsqueeze(0), target_quat[0, :].unsqueeze(0))) >= th_rot):
            iters += 1

            self.refresh_and_acquire_tensors()

            # DeltaG, inv_DeltaG, DG_transl, DG_Quat = self.DG(num_envs, device, jacobian_type)
            inv_DeltaG_t, inv_DeltaG_quat = self.DG(num_envs, device, jacobian_type)

            ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat = self.apply_invDG_insert(
                num_envs=self.num_envs, target_pos=target_pos, target_quat=target_quat, inv_DeltaG_t=inv_DeltaG_t,
                inv_DeltaG_quat=inv_DeltaG_quat, device=device)
            # ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat = self.apply_invDG(num_envs = self.num_envs, target_pos = target_pos, target_quat = target_quat, DeltaG = DeltaG, inv_DeltaG = inv_DeltaG, device = device)

            self.ctrl_target_fingertip_midpoint_pos = ctrl_target_fingertip_midpoint_pos
            self.ctrl_target_fingertip_midpoint_quat = ctrl_target_fingertip_midpoint_quat

            s_t_pos_error, s_t_axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=jacobian_type,
                rot_error_type='axis_angle')

            s_t_delta_hand_pose = torch.cat((s_t_pos_error, s_t_axis_angle_error), dim=-1)
            s_t_actions = torch.zeros((num_envs, self.cfg_task.env.numActions), device=device)
            s_t_actions[:, :6] = s_t_delta_hand_pose

            self._apply_actions_as_ctrl_targets(actions=s_t_actions,
                                                ctrl_target_gripper_dof_pos=franka_gripper_width,
                                                do_scale=False)
            self.zero_velocities(env_ids)

        self.zero_velocities(env_ids)
        self.refresh_and_acquire_tensors()

    def track_pose_insert(self, env_ids, num_envs, sim_steps, jacobian_type, franka_gripper_width, device, target_pos,
                          target_quat, th_pos, th_rot, Kp_magnif=0., grasp=False, ):

        # Step sim and render
        iters = 0
        self.refresh_and_acquire_tensors()
        while iters <= 100 and (
                (torch.norm(target_pos[0, 0].double() - self.plug_pos[0, 0].double(), p=2) >= th_pos) or (
                torch.norm(target_pos[0, 1].double() - self.plug_pos[0, 1].double(), p=2) >= th_pos) or (
                        torch.norm(target_pos[0, 2].double() - self.plug_pos[0, 2].double(),
                                   p=2) >= th_pos) or torch.rad2deg(
                quat_diff_rad(self.plug_quat[0, :].unsqueeze(0), target_quat[0, :].unsqueeze(0))) >= th_rot):
            # time.sleep(.1)

            iters += 1

            self.refresh_and_acquire_tensors()

            inv_DeltaG_t, inv_DeltaG_quat = self.DG(num_envs, device, jacobian_type)
            ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat = self.apply_invDG_insert(
                num_envs=self.num_envs, target_pos=target_pos, target_quat=target_quat, inv_DeltaG_t=inv_DeltaG_t,
                inv_DeltaG_quat=inv_DeltaG_quat, device=device)

            print("DG_transl:", DG_transl[0])
            print("DG_Quat:", DG_Quat[0])
            print("DG_Euler:", torch.rad2deg(quat2euler(DG_Quat))[0])
            print("Frame Transformation Delta angle:", torch.rad2deg(
                quat_diff_rad(self.plug_quat[0, :].unsqueeze(0), self.fingertip_midpoint_quat[0, :].unsqueeze(0))))
            print("Rotational Distance:",
                  torch.rad2deg(quat_diff_rad(self.plug_quat[0, :].unsqueeze(0), target_quat[0, :].unsqueeze(0))))
            print("Current Plug Quat:", self.plug_quat[0])

            self.ctrl_target_fingertip_midpoint_pos = ctrl_target_fingertip_midpoint_pos
            self.ctrl_target_fingertip_midpoint_quat = ctrl_target_fingertip_midpoint_quat

            s_t_pos_error, s_t_axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=jacobian_type,
                rot_error_type='axis_angle')

            s_t_delta_hand_pose = torch.cat((s_t_pos_error, s_t_axis_angle_error), dim=-1)
            s_t_actions = torch.zeros((num_envs, self.cfg_task.env.numActions), device=device)
            s_t_actions[:, :6] = s_t_delta_hand_pose

            self._apply_actions_as_ctrl_targets(actions=s_t_actions,
                                                ctrl_target_gripper_dof_pos=franka_gripper_width,
                                                do_scale=False)
            self.zero_velocities(env_ids)

        self.zero_velocities(env_ids)
        self.refresh_and_acquire_tensors()

    def insert(self, env_ids, num_envs, sim_steps, jacobian_type, franka_gripper_width, device, target_pos,
               target_quat):

        # Step sim and render
        iters = 0
        self.refresh_and_acquire_tensors()
        curr_failures = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        while torch.norm(self.plug_pos[0, 2].double() - target_pos[0, 2].double(), p=2) > .001 or torch.rad2deg(
                quat_diff_rad(self.plug_quat[0, :].unsqueeze(0), target_quat[0, :].unsqueeze(0))) > 1:
            iters += 1

            self.refresh_and_acquire_tensors()

            if (torch.norm(self.plug_pos[0, 0].double() - target_pos[0, 0].double(), p=2) >= .001) or (
                    torch.norm(self.plug_pos[0, 1].double() - target_pos[0, 1].double(),
                               p=2) >= .001):  # or torch.rad2deg(quat_diff_rad(self.plug_quat[0, :].unsqueeze(0), target_quat[0, :].unsqueeze(0))) >= .1:

                # Mode 1: Try to correct the Object Pose
                print("!!!!!!!!!!Correction_Mode!!!!!!!!!!")
                correction_pos = self.initial_insertion_pos
                self.refresh_and_acquire_tensors()
                # if self.plug_pos[0,2] < self.initial_insertion_pos[0,2]:
                # correction_pos[:,2] = self.plug_pos[:,2] + .001 #self.plug_pos[:,2] + .005
                Kp_magnif = 1.0
                print("Cause of correction:")
                print("Translational cause x:",
                      torch.norm(self.plug_pos[0, 0].double() - target_pos[0, 0].double(), p=2) >= .001)
                print("Translational cause y:",
                      torch.norm(self.plug_pos[0, 1].double() - target_pos[0, 1].double(), p=2) >= .001)
                print("Plug Pos: ", self.plug_pos[0])
                print("Plug Quat: ", self.plug_quat[0])
                print("Target Correction pos: ", correction_pos[0])
                print("Target Correction quat: ", self.initial_insertion_quat[0])
                self.track_pose(target_pos=correction_pos, target_quat=self.initial_insertion_quat, env_ids=env_ids,
                                num_envs=self.num_envs, sim_steps=None, jacobian_type=self.cfg_ctrl['jacobian_type'],
                                franka_gripper_width=0.0, device=self.device, th_pos=0.01, th_rot=1,
                                Kp_magnif=Kp_magnif, grasp=False)

            else:
                # Mode 2: Try to insert the Object

                print("!!!!!!!!!!Insert_Mode!!!!!!!!!!")
                # In order to test:
                inv_DeltaG_t, inv_DeltaG_quat = self.DG(num_envs, device, jacobian_type)

                ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat = self.apply_invDG_insert(
                    num_envs=self.num_envs, target_pos=target_pos, target_quat=target_quat, inv_DeltaG_t=inv_DeltaG_t,
                    inv_DeltaG_quat=inv_DeltaG_quat, device=device)

                self.ctrl_target_fingertip_midpoint_pos = ctrl_target_fingertip_midpoint_pos
                self.ctrl_target_fingertip_midpoint_quat = ctrl_target_fingertip_midpoint_quat

                print("Before action:")
                print("Finger Pose:", self.fingertip_midpoint_pos[0], self.fingertip_midpoint_quat[0])
                print("Plug Pose:", self.plug_pos[0], self.plug_quat[0])

                s_t_pos_error, s_t_axis_angle_error = fc.get_pose_error(
                    fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                    fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                    ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                    ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                    jacobian_type=jacobian_type,
                    rot_error_type='axis_angle')

                s_t_delta_hand_pose = torch.cat((s_t_pos_error, s_t_axis_angle_error), dim=-1)
                s_t_actions = torch.zeros((num_envs, self.cfg_task.env.numActions), device=device)
                s_t_actions[:, :6] = s_t_delta_hand_pose

                self._apply_actions_as_ctrl_targets(actions=s_t_actions,
                                                    ctrl_target_gripper_dof_pos=franka_gripper_width,
                                                    do_scale=False)
                self.zero_velocities(env_ids)

                self.refresh_and_acquire_tensors()

                print("Torques:", self.dof_torque[0, :])
                print("After action:")
                print("Finger Pose:", self.fingertip_midpoint_pos[0], self.fingertip_midpoint_quat[0])
                print("Plug Pose:", self.plug_pos[0], self.plug_quat[0])

            curr_failures = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        self.zero_velocities(env_ids)
        self.refresh_and_acquire_tensors()

    def reset_idx(self, env_ids):
        sim_params = self.sim_params
        sim_params.physx.max_gpu_contact_pairs *= 1000
        # print("Max_Contact_Pairs:", max_gpu_contact_pairs)
        # time.sleep(100)
        """Reset specified environments."""
        successes_factors = []
        # hole_factors = [5.0, 4.0, 3.0, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
        hole_factors = [1.]
        # for hole_factor in hole_factors:
        for _ in range(1):
            self.hole_factor = hole_factor = 1.  # hole_factor
            Trials = 100
            successes = 0
            for tr in range(Trials):

                self._reset_franka(env_ids)
                self._reset_object(env_ids, hole_factor)
                self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps)
                self._open_gripper()
                self.refresh_and_acquire_tensors()

                self.socket_pos_2 = torch.tensor([0.000, 0.000, 0.4000], device=self.device).repeat(self.num_envs, 1)
                self.socket_euler_2 = torch.tensor([0.000, 0.000, 0.000], device=self.device).repeat(self.num_envs, 1)
                self.socket_quat_2 = torch_utils.quat_from_euler_xyz(self.socket_euler_2[:, 0],
                                                                     self.socket_euler_2[:, 1],
                                                                     self.socket_euler_2[:, 2])

                self.initial_insertion_pos = self.socket_pos_2 + torch.tensor([0.000, 0.000, 0.0200],
                                                                              device=self.device).repeat(self.num_envs,
                                                                                                         1)
                self.initial_insertion_quat = self.socket_quat_2

                self.target_pos = self.socket_pos_2.clone()
                self.target_quat = self.socket_quat_2.clone()

                self.initial_pos = self.fingertip_midpoint_pos.clone()
                self.initial_quat = self.fingertip_midpoint_quat.clone()

                self.refresh_and_acquire_tensors()

                self.initial_finger_midpoint_euler = get_euler_xyz(self.initial_quat)
                self.plug_euler = get_euler_xyz(self.plug_quat)
                self.socket_euler = get_euler_xyz(self.socket_quat_2)

                self.plug_pos_target = self.plug_com_pos.clone()
                self.plug_quat_target = self.socket_quat_2.clone()

                self.grasp_pos, self.grasp_quat = self.calculate_grasp_pose(env_ids=env_ids,
                                                                            plug_pos_target=self.plug_pos_target,
                                                                            plug_quat_target=self.plug_quat_target,
                                                                            device=self.device)

                self.refresh_and_acquire_tensors()
                # Move to the target grasp pose              
                self.track_pose_grasp(target_pos=self.grasp_pos, target_quat=self.grasp_quat, env_ids=env_ids,
                                      num_envs=self.num_envs, sim_steps=None,
                                      jacobian_type=self.cfg_ctrl['jacobian_type'],
                                      franka_gripper_width=self.asset_info_franka_table.franka_gripper_width_max,
                                      device=self.device, grasp=True)
                # Close gripper to grasp the object
                self._close_gripper(sim_steps=10)

                # Lift the object
                self.track_pose_grasp(target_pos=self.initial_pos, target_quat=self.initial_quat, env_ids=env_ids,
                                      num_envs=self.num_envs, sim_steps=None,
                                      jacobian_type=self.cfg_ctrl['jacobian_type'], franka_gripper_width=0.0,
                                      device=self.device, grasp=True)
                # Move to initial insertion pose above the hole

                # Phase 2
                self.track_pose(target_pos=self.initial_insertion_pos, target_quat=self.initial_insertion_quat,
                                env_ids=env_ids, num_envs=self.num_envs, sim_steps=None,
                                jacobian_type=self.cfg_ctrl['jacobian_type'], franka_gripper_width=0.0,
                                device=self.device, th_pos=0.001, th_rot=1, grasp=False)
                # Insert object in the hole
                # implement (2-mode control logic: in and out of hole with check)
                # Phase 3
                self.insert(target_pos=self.socket_pos_2, target_quat=self.socket_quat_2, env_ids=env_ids,
                            num_envs=self.num_envs, sim_steps=None, jacobian_type=self.cfg_ctrl['jacobian_type'],
                            franka_gripper_width=0.0, device=self.device)
                # Open gripper to release the object
                self._open_gripper()
                if self.plug_pos[0, 2] <= 0.401:
                    successes += 1
                print("Successes/Trials:", successes, tr + 1, successes / (tr + 1))
                # Lift the gripper
                self._lift_gripper(env_ids, franka_gripper_width=self.asset_info_franka_table.franka_gripper_width_max,
                                   lift_distance=0.05, sim_steps=100)
                # Zero velocities
                self.zero_velocities(env_ids)
                # Reset buffers
                self._reset_buffers(env_ids)

                # print('Sleep!')
                # time.sleep(10)

                # self.plug_euler = quat2euler(self.plug_quat)
                # self.socket_euler = quat2euler(self.socket_quat_2)
                # self.final_finger_midpoint_euler = get_euler_xyz(self.fingertip_midpoint_quat)

                # for j in range(10000):
                #    is_plug_close_to_socket = self._check_plug_close_to_socket()
                #    print('is_plug_close_to_socket: ', is_plug_close_to_socket)
                #    self.gym.simulate(self.sim)
                #    self.render()

                # self.refresh_and_acquire_tensors()

            mean_factor = successes / Trials
            print("Success_prcntg:", successes, Trials, successes / Trials)
            time.sleep(100)
            successes_factors.append(mean_factor)
        print("Successes_Factors:", hole_factors, successes_factors)

    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""

        self.dof_pos[env_ids] = torch.cat(
            (torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device),
             torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device),
             torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device)),
            dim=-1).unsqueeze(0).repeat((self.num_envs, 1))  # shape = (num_envs, num_dofs)
        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

    def _reset_object(self, env_ids, hole_factor):

        """Reset root states of nut and bolt."""

        # shape of root_pos = (num_envs, num_actors, 3)
        # shape of root_quat = (num_envs, num_actors, 4)
        # shape of root_linvel = (num_envs, num_actors, 3)
        # shape of root_angvel = (num_envs, num_actors, 3)

        if self.cfg_task.randomize.initial_state == 'random':
            self.root_pos[env_ids, self.plug_actor_id_env] = \
                torch.cat(((torch.rand((self.num_envs, 1),
                                       device=self.device) * 2.0 - 1.0) * self.cfg_task.randomize.plug_noise_xy,
                           self.cfg_task.randomize.plug_bias_y + (torch.rand((self.num_envs, 1),
                                                                             device=self.device) * 2.0 - 1.0) * self.cfg_task.randomize.plug_noise_xy,
                           torch.ones((self.num_envs, 1), device=self.device) * (
                                       self.cfg_base.env.table_height + self.cfg_task.randomize.plug_bias_z)), dim=1)

            # random_angle = torch.normal(torch.tensor(0.,device=self.device), torch.tensor(45.,device=self.device)).to(self.device).item()
            random_angle = 45
            self.delta_q = torch.tensor(
                [0., 0., torch.sin(torch.deg2rad(torch.tensor([random_angle / 2], device=self.device))),
                 torch.cos(torch.deg2rad(torch.tensor([random_angle / 2], device=self.device)))],
                device=self.device).repeat(len(env_ids), 1)
            self.root_euler = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
            self.root_quat[env_ids, self.plug_actor_id_env] = torch_utils.quat_from_euler_xyz(self.root_euler[:, 0],
                                                                                              self.root_euler[:, 1],
                                                                                              self.root_euler[:, 2])
            self.root_quat[env_ids, self.plug_actor_id_env] = torch_utils.quat_mul(self.delta_q, self.root_quat[
                env_ids, self.plug_actor_id_env])

        elif self.cfg_task.randomize.initial_state == 'goal':
            self.root_pos[env_ids, self.plug_actor_id_env] = torch.tensor([0.0, 0.0, self.cfg_base.env.table_height],
                                                                          device=self.device)

        self.root_linvel[env_ids, self.plug_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.plug_actor_id_env] = 0.0

        # Randomize root state of socket
        socket_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        socket_noise_xy = socket_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.socket_pos_xy_noise, dtype=torch.float32, device=self.device))
        self.root_pos[env_ids, self.socket_actor_id_env, 0] = self.cfg_task.randomize.socket_pos_xy_initial[0] + \
                                                              socket_noise_xy[env_ids, 0]
        self.root_pos[env_ids, self.socket_actor_id_env, 1] = self.cfg_task.randomize.socket_pos_xy_initial[1] + \
                                                              socket_noise_xy[env_ids, 1]
        self.root_pos[env_ids, self.socket_actor_id_env, 2] = self.cfg_base.env.table_height
        self.root_quat[env_ids, self.socket_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32,
                                                                         device=self.device).repeat(len(env_ids), 1)

        self.root_linvel[env_ids, self.socket_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.socket_actor_id_env] = 0.0

        plug_socket_actor_ids_sim = torch.cat((self.plug_actor_ids_sim[env_ids], self.socket_actor_ids_sim[env_ids]),
                                              dim=0)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(plug_socket_actor_ids_sim),
                                                     len(plug_socket_actor_ids_sim))

    def _reset_buffers(self, env_ids):
        """Reset buffers. """

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-.5, -.5, .7)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """Apply actions from policy as position/rotation targets or force/torque targets."""

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                rot_actions_quat,
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
            )

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        if self.cfg_ctrl['do_force_ctrl']:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if self.cfg_task.rl.unidirectional_force:
                force_actions[:, 2] = -(force_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

            self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        self.generate_ctrl_signals()

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5
        return keypoint_offsets

    def _get_keypoint_dist(self):
        """Get keypoint distances."""

        keypoint_dist = torch.sum(torch.norm(self.keypoints_socket - self.keypoints_plug, p=2, dim=-1), dim=-1)
        return keypoint_dist

    def _get_curr_successes(self):
        """Get success mask at current timestep."""

        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # If nut is close enough to target pos

        is_close = torch.where(self.plug_dist_to_target < torch.ones_like(curr_successes) * .005,
                               torch.ones_like(curr_successes),
                               torch.zeros_like(curr_successes))

        curr_successes = torch.logical_or(curr_successes, is_close)

        return curr_successes

    def _get_curr_failures(self, curr_successes):
        """Get failure mask at current timestep."""

        curr_failures = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # If max episode length has been reached
        # self.is_expired = torch.where(self.progress_buf[:] >= self.cfg_task.rl.max_episode_length,
        #                              torch.ones_like(curr_failures),
        #                              curr_failures)

        # If nut is too far from target pos
        self.is_far = torch.where(self.plug_dist_to_target > self.cfg_task.rl.far_error_thresh,
                                  torch.ones_like(curr_failures),
                                  curr_failures)

        # If nut has slipped (distance-based definition)
        self.is_slipped = \
            torch.where(
                self.plug_dist_to_fingerpads > self.asset_info_franka_table.franka_fingerpad_length * 0.5 + self.plug_heights.squeeze(
                    -1) * 0.5,
                torch.ones_like(curr_failures),
                curr_failures)
        self.is_slipped = torch.logical_and(self.is_slipped,
                                            torch.logical_not(curr_successes))  # ignore slip if successful

        # If nut has fallen (i.e., if nut XY pos has drifted from center of bolt and nut Z pos has drifted below top of bolt)

        self.is_fallen = torch.logical_and(
            torch.norm(self.plug_com_pos[:, 0:2], p=2, dim=-1) > torch.tensor(self.socket_widths,
                                                                              device=self.device).squeeze(-1) * 0.5,
            self.plug_com_pos[:, 2] < self.cfg_base.env.table_height
            + torch.tensor(self.socket_heights, device=self.device).squeeze(-1)
            + torch.tensor(self.socket_heights, device=self.device).squeeze(-1)
            + torch.tensor(self.plug_heights, device=self.device).squeeze(-1) * 0.5)

        curr_failures = torch.logical_or(curr_failures, self.is_far)
        curr_failures = torch.logical_or(curr_failures, self.is_slipped)
        curr_failures = torch.logical_or(curr_failures, self.is_fallen)

        return curr_failures

    def _close_gripper(self, sim_steps=20):
        """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""
        self._move_gripper_to_dof_pos(gripper_dof_pos=0.0, sim_steps=sim_steps)

    def _open_gripper(self, sim_steps=20):
        """Fully open gripper using controller. Called outside RL loop (i.e., after last step of episode)."""
        self._move_gripper_to_dof_pos(gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
                                      sim_steps=sim_steps)

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps=20):
        """Move gripper fingers to specified DOF position using controller."""

        delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
                                      device=self.device)  # No hand motion
        self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)

        # Step sim
        for _ in range(sim_steps):
            self.render()
            self.gym.simulate(self.sim)

    def _lift_gripper(self, env_ids, franka_gripper_width=0.0, lift_distance=0.3, sim_steps=20):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        target_pos = self.fingertip_midpoint_pos.clone() + torch.tensor([0.0, 0.0, lift_distance],
                                                                        device=self.device).repeat(self.num_envs, 1)
        self.ctrl_target_fingertip_midpoint_pos = target_pos
        self.ctrl_target_fingertip_midpoint_quat = self.fingertip_midpoint_quat.clone()

        # Step sim
        iters = 0
        while iters < sim_steps and torch.norm(
                self.ctrl_target_fingertip_midpoint_pos[:, 2] - self.fingertip_midpoint_pos[:, 2], p=2) >= 0.001:
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()
            self._acquire_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(actions, franka_gripper_width, do_scale=False)

            self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])
            # Set DOF state
            multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
            # print("Qs:", self.dof_state[0, :])
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                  len(multi_env_ids_int32))

            self.render()
            self.gym.simulate(self.sim)

            iters += 1

    def _check_plug_close_to_socket(self):
        """Check if plug is close to socket."""

        keypoint_dist = torch.norm(self.keypoints_socket - self.keypoints_plug, p=2, dim=-1)

        is_plug_close_to_socket = torch.where(torch.sum(keypoint_dist, dim=-1) < self.cfg_task.rl.close_error_thresh,
                                              torch.ones_like(self.progress_buf),
                                              torch.zeros_like(self.progress_buf))

        return is_plug_close_to_socket

    def _randomize_gripper_pose(self, env_ids, sim_steps):
        """Move gripper to random pose."""

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = \
            torch.tensor([0.0, 0.0, self.cfg_base.env.table_height], device=self.device) \
            + torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(
            self.num_envs, 1)

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                                                            device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2])

        # Step sim and render
        for _ in range(sim_steps):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(actions=actions,
                                                ctrl_target_gripper_dof_pos=0.0,
                                                # self.asset_info_franka_table.franka_gripper_width_max,
                                                do_scale=False)

            self.gym.simulate(self.sim)
            self.render()

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        # Set DOF state
        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

    def DG(self, num_envs, device, jacobian_type):

        inv_Pose_Gripper_quat, inv_Pose_Gripper_t = tf_inverse(self.fingertip_midpoint_quat,
                                                               self.fingertip_midpoint_pos)

        DeltaG_quat, DeltaG_t = tf_combine(inv_Pose_Gripper_quat, inv_Pose_Gripper_t, self.plug_quat, self.plug_pos)
        inv_DeltaG_quat, inv_DeltaG_t = tf_inverse(DeltaG_quat, DeltaG_t)

        return inv_DeltaG_t, inv_DeltaG_quat

    def apply_invDG_insert(self, num_envs, target_pos, target_quat, inv_DeltaG_t, inv_DeltaG_quat, device):

        target_Pose_Gripper_quat, target_Pose_Gripper_t = tf_combine(target_quat, target_pos, inv_DeltaG_quat,
                                                                     inv_DeltaG_t)

        return target_Pose_Gripper_t, target_Pose_Gripper_quat
