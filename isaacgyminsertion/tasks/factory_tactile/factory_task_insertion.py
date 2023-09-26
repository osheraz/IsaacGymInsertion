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
python train.py task=FactoryTaskInsertionTactile

Only the environment is provided; training a successful RL policy is an open research problem left to the user.
"""

import hydra
import omegaconf
import time

from isaacgym import gymapi, gymtorch
from isaacgyminsertion.tasks.factory_tactile.factory_env_insertion import FactoryEnvInsertionTactile
from isaacgyminsertion.tasks.factory_tactile.factory_schema_class_task import FactoryABCTask
from isaacgyminsertion.tasks.factory_tactile.factory_schema_config_task import FactorySchemaConfigTask
import isaacgyminsertion.tasks.factory_tactile.factory_control as fc
from isaacgyminsertion.tasks.factory_tactile.factory_utils import *
from isaacgyminsertion.utils import torch_jit_utils

torch.set_printoptions(sci_mode=False)


class FactoryTaskInsertionTactile(FactoryEnvInsertionTactile, FactoryABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize task superclass."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.cfg = cfg
        self._get_task_yaml_params()

        self._acquire_task_tensors()
        self.parse_controller_spec()

        # if self.cfg_task.sim.disable_gravity:
        #     self.disable_gravity()

        if self.viewer is not None:
            self._set_viewer_params()

        if self.cfg_base.mode.export_scene:
            self.export_scene(label='kuka_task_insertion')

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

        ppo_path = 'train/FactoryTaskInsertionTactilePPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        plug_grasp_heights = self.socket_heights + self.plug_heights * 1.1
        self.plug_grasp_pos_local = plug_grasp_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1))
        self.plug_grasp_quat_local = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(
            self.num_envs, 1)
        # Compute pose of plug grasping frame
        self.plug_grasp_quat, self.plug_grasp_pos = torch_jit_utils.tf_combine(self.plug_quat,
                                                                               self.plug_pos,
                                                                               self.plug_grasp_quat_local,
                                                                               self.plug_grasp_pos_local)

        # From place
        self.socket_base_pos_local = self.socket_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1))
        self.socket_tip_pos_local = self.socket_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1))

        # Keypoint tensors
        self.keypoint_offsets = self._get_keypoint_offsets(
            self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
        self.keypoints_gripper = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3), dtype=torch.float32,
                                             device=self.device)
        self.keypoints_plug = torch.zeros_like(self.keypoints_gripper, device=self.device)
        self.keypoints_socket = torch.zeros_like(self.keypoints_gripper, device=self.device)

        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs,
                                                                                                        1)
        self.actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)

    def _refresh_task_tensors(self, update_tactile=True):
        """Refresh tensors."""

        self.refresh_base_tensors()
        self.refresh_env_tensors()

        self.plug_grasp_quat, self.plug_grasp_pos = torch_jit_utils.tf_combine(self.plug_quat,
                                                                               self.plug_pos,
                                                                               self.plug_grasp_quat_local,
                                                                               self.plug_grasp_pos_local)

        # Compute pos of keypoints on gripper, socket, and plug in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_gripper[:, idx] = torch_jit_utils.tf_combine(self.fingertip_midpoint_quat,
                                                                        self.fingertip_midpoint_pos,
                                                                        self.identity_quat,
                                                                        keypoint_offset.repeat(self.num_envs, 1))[1]
            self.keypoints_plug[:, idx] = torch_jit_utils.tf_combine(self.plug_quat,
                                                                     self.plug_pos,
                                                                     self.identity_quat,
                                                                     keypoint_offset.repeat(self.num_envs, 1))[1]
            self.keypoints_socket[:, idx] = torch_jit_utils.tf_combine(self.socket_quat,
                                                                       self.socket_pos,
                                                                       self.identity_quat,
                                                                       (keypoint_offset + self.socket_tip_pos_local))[1]

        if update_tactile:

            left_finger_pose = pose_vec_to_mat(
                torch.cat((self.left_finger_pos, self.left_finger_quat), axis=1)).cpu().numpy()
            right_finger_pose = pose_vec_to_mat(
                torch.cat((self.right_finger_pos, self.right_finger_quat), axis=1)).cpu().numpy()
            middle_finger_pose = pose_vec_to_mat(
                torch.cat((self.middle_finger_pos, self.middle_finger_quat), axis=1)).cpu().numpy()
            object_pose = pose_vec_to_mat(torch.cat((self.plug_pos, self.plug_quat), axis=1)).cpu().numpy()

            for e in range(self.num_envs):

                self.tactile_handles[e][0].update_pose_given_sim_pose(left_finger_pose[e], object_pose[e])
                self.tactile_handles[e][1].update_pose_given_sim_pose(right_finger_pose[e], object_pose[e])
                self.tactile_handles[e][2].update_pose_given_sim_pose(middle_finger_pose[e], object_pose[e])

                tactile_imgs, height_maps = [], []

                for n in range(3):
                    tactile_img, height_map, _ = self.tactile_handles[e][n].render(object_pose[e])
                    tactile_imgs.append(tactile_img)
                    height_maps.append(height_map)

            env_to_show = 0
            self.tactile_handles[env_to_show][0].updateGUI(tactile_imgs, height_maps)

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

            # If it fail to lift, reset
            lift_success = self._check_lift_success(height_multiple=1.0)
            env_ids = lift_success[lift_success == 0].int()

            if len(env_ids):
                self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]
        self._apply_actions_as_ctrl_targets(actions=self.actions,
                                            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
                                            do_scale=True)

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

        self.progress_buf[:] += 1

        # In this policy, episode length is constant
        # is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        """Compute observations."""

        # Shallow copies of tensors
        obs_tensors = [self.fingertip_midpoint_pos,
                       self.fingertip_midpoint_quat,
                       self.fingertip_midpoint_linvel,
                       self.fingertip_midpoint_angvel,
                       self.plug_pos,
                       self.plug_quat,
                       self.socket_pos,
                       self.socket_quat]

        if self.cfg_task.rl.add_obs_socket_tip_pos:
            obs_tensors += [self.socket_tip_pos_local]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)

        return self.obs_buf  # shape = (num_envs, num_observations)

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        self._update_rew_buf()
        self._update_reset_buf()

    def _update_rew_buf(self):
        """Compute reward at current timestep."""
        keypoint_reward = -self._get_keypoint_dist()
        action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale

        self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                          - action_penalty * self.cfg_task.rl.action_penalty_scale

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

        if is_last_step:
            # Check if plug is picked up and above table
            lift_success = self._check_lift_success(height_multiple=3.0)
            self.rew_buf[:] += lift_success * self.cfg_task.rl.success_bonus
            self.extras['successes'] = torch.mean(lift_success.float())

            # Check if plug is close enough to socket
            is_plug_close_to_socket = self._check_plug_close_to_socket()
            self.rew_buf[:] += is_plug_close_to_socket * self.cfg_task.rl.success_bonus
            self.extras['successes'] += torch.mean(is_plug_close_to_socket.float())

    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""
        # If max episode length has been reached
        self.reset_buf[:] = torch.where(self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1,
                                        torch.ones_like(self.reset_buf),
                                        self.reset_buf)


    def reset_idx(self, env_ids):
        """Reset specified environments."""

        for _ in range(1):
            self._reset_kuka(env_ids)
            self._reset_object(env_ids)
            self._zero_velocities(env_ids)
            self._refresh_task_tensors()

            # Move arm to grasp pose
            self._move_arm_to_desired_pose(env_ids, self.plug_grasp_pos.clone(),
                                           sim_steps=5 * self.cfg_task.env.num_gripper_move_sim_steps)
            self._zero_velocities(env_ids)
            self._refresh_task_tensors()

            # Grasp ~ todo add randomization
            self._close_gripper(env_ids, self.cfg_task.env.num_gripper_close_sim_steps)
            self._refresh_task_tensors()

            # Lift
            self._lift_gripper(env_ids, self.ctrl_target_gripper_dof_pos)
            self._refresh_task_tensors()

            # Move arm above the socket
            self._move_arm_to_desired_pose(env_ids, self.above_socket_pos.clone(),
                                           sim_steps=5 * self.cfg_task.env.num_gripper_move_sim_steps)
            self._refresh_task_tensors()

            # Isidoros @ todo oracle insertion

            # inv_Pose_Gripper_quat, inv_Pose_Gripper_t = tf_inverse(self.fingertip_midpoint_quat,
            #                                                        self.fingertip_midpoint_pos)
            #
            # DeltaG_quat, DeltaG_t = tf_combine(inv_Pose_Gripper_quat,
            #                                    inv_Pose_Gripper_t,
            #                                    self.plug_quat,
            #                                    self.plug_pos)
            #
            # inv_DeltaG_quat, inv_DeltaG_t = tf_inverse(DeltaG_quat, DeltaG_t)
            #
            # target_Pose_Gripper_quat, target_Pose_Gripper_t = tf_combine(self.socket_quat,
            #                                                              self.socket_tip,
            #                                                              inv_DeltaG_quat,
            #                                                              inv_DeltaG_t)

            # Insert
            self._move_arm_to_desired_pose(env_ids, self.socket_tip.clone(),
                                           sim_steps=5 * self.cfg_task.env.num_gripper_move_sim_steps)
            self._refresh_task_tensors()

            self._reset_buffers(env_ids)

    def _reset_kuka(self, env_ids):
        """Reset DOF states and DOF targets of kuka."""

        # shape of dof_pos = (num_envs, num_dofs)
        # shape of dof_vel = (num_envs, num_dofs)

        # Initialize kuka to middle of joint limits, plus joint noise
        self.dof_pos[env_ids, :7] = torch.tensor(self.cfg_task.randomize.kuka_arm_initial_dof_pos,
                                                 device=self.device).unsqueeze(0).repeat((self.num_envs, 1))

        self.dof_pos[
            env_ids, list(self.dof_dict.values()).index('base_to_finger_1_1')] = self.cfg_task.env.openhand.base_angle
        self.dof_pos[
            env_ids, list(self.dof_dict.values()).index('base_to_finger_2_1')] = -self.cfg_task.env.openhand.base_angle

        self.dof_pos[env_ids, list(self.dof_dict.values()).index(
            'finger_1_1_to_finger_1_2')] = self.cfg_task.env.openhand.proximal_open
        self.dof_pos[env_ids, list(self.dof_dict.values()).index(
            'finger_2_1_to_finger_2_2')] = self.cfg_task.env.openhand.proximal_open
        self.dof_pos[env_ids, list(self.dof_dict.values()).index(
            'base_to_finger_3_2')] = self.cfg_task.env.openhand.proximal_open

        self.dof_pos[env_ids, list(self.dof_dict.values()).index(
            'finger_1_2_to_finger_1_3')] = self.cfg_task.env.openhand.distal_open
        self.dof_pos[env_ids, list(self.dof_dict.values()).index(
            'finger_2_2_to_finger_2_3')] = self.cfg_task.env.openhand.distal_open
        self.dof_pos[env_ids, list(self.dof_dict.values()).index(
            'finger_3_2_to_finger_3_3')] = self.cfg_task.env.openhand.distal_open

        # Stabilize!
        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.dof_torque[env_ids] = 0.0  # shape = (num_envs, num_dofs)

        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.ctrl_target_gripper_dof_pos[env_ids] = self.dof_pos[env_ids, 7:]

        multi_env_ids_int32 = self.kuka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        self._simulate_and_refresh()

    def _reset_object(self, env_ids):
        """Reset root state of plug."""

        # shape of root_pos = (num_envs, num_actors, 3)
        # shape of root_quat = (num_envs, num_actors, 4)
        # shape of root_linvel = (num_envs, num_actors, 3)
        # shape of root_angvel = (num_envs, num_actors, 3)

        # Randomize root state of plug
        plug_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        plug_noise_xy = plug_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.plug_pos_xy_noise, device=self.device))

        self.root_pos[env_ids, self.plug_actor_id_env, 0] = self.cfg_task.randomize.plug_pos_xy_initial[0] \
                                                            + plug_noise_xy[env_ids, 0]
        self.root_pos[env_ids, self.plug_actor_id_env, 1] = self.cfg_task.randomize.plug_pos_xy_initial[1] \
                                                            + plug_noise_xy[env_ids, 1]
        self.root_pos[
            env_ids, self.plug_actor_id_env, 2] = self.cfg_base.env.table_height  # - self.socket_heights.squeeze(-1)

        self.root_quat[env_ids, self.plug_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32,
                                                                       device=self.device).repeat(len(env_ids), 1)

        # Stabilize plug
        self.root_linvel[env_ids, self.plug_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.plug_actor_id_env] = 0.0

        plug_actor_ids_sim_int32 = self.plug_actor_ids_sim.to(dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(plug_actor_ids_sim_int32[env_ids]),
                                                     len(plug_actor_ids_sim_int32[env_ids]))

        self._simulate_and_refresh()

    def _move_arm_to_desired_pose(self, env_ids, desired_pos, desired_rot=None, sim_steps=30):
        """Move gripper to desired pose."""

        # Set target pos above object
        self.ctrl_target_fingertip_midpoint_pos = desired_pos

        # self.ctrl_target_fingertip_midpoint_pos = self.ctrl_target_fingertip_midpoint_pos.repeat(self.num_envs, 1)

        # Set target rot
        if desired_rot is None:
            ctrl_target_fingertip_midpoint_euler = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                                                                device=self.device).unsqueeze(0).repeat(self.num_envs,
                                                                                                        1)

            self.ctrl_target_fingertip_midpoint_quat = torch_jit_utils.quat_from_euler_xyz(
                ctrl_target_fingertip_midpoint_euler[:, 0],
                ctrl_target_fingertip_midpoint_euler[:, 1],
                ctrl_target_fingertip_midpoint_euler[:, 2])
        else:
            self.ctrl_target_fingertip_midpoint_quat = desired_rot

        # Step sim and render
        for _ in range(sim_steps):
            self._simulate_and_refresh()

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

            # Apply the action, dont move the figners
            self.ctrl_target_dof_pos[env_ids, 7:] = self.ctrl_target_gripper_dof_pos  # 0.
            self._apply_actions_as_ctrl_targets(actions=actions,
                                                ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
                                                do_scale=False)

        # Stabilize
        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids, :])
        self.dof_torque[env_ids, :] = torch.zeros_like(self.dof_torque[env_ids, :])

        # Set DOF state
        multi_env_ids_int32 = self.kuka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        self._simulate_and_refresh()

    def _simulate_and_refresh(self):
        """Simulate one step, refresh tensors, and render results."""

        self.gym.simulate(self.sim)
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.render()

    def _reset_buffers(self, env_ids):
        """Reset buffers. """

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-1.0, -1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """Apply actions from policy as position/rotation targets."""

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
        rot_actions_quat = torch_jit_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                           rot_actions_quat,
                                           torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,
                                                                                                         1))
        self.ctrl_target_fingertip_midpoint_quat = torch_jit_utils.quat_mul(rot_actions_quat,
                                                                            self.fingertip_midpoint_quat)

        if self.cfg_ctrl['do_force_ctrl']:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

            self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

        # TODO should be changed to delta as well
        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos  # directly putting desired gripper_pos

        self.generate_ctrl_signals()

    def _open_gripper(self, env_ids, sim_steps=20):
        """Fully open gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        gripper_dof_pos = 0 * self.gripper_dof_pos.clone()
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'base_to_finger_1_1') - 7] = self.cfg_task.env.openhand.base_angle
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'base_to_finger_2_1') - 7] = -self.cfg_task.env.openhand.base_angle

        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'finger_1_1_to_finger_1_2') - 7] = self.cfg_task.env.openhand.proximal_open
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'finger_2_1_to_finger_2_2') - 7] = self.cfg_task.env.openhand.proximal_open
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'base_to_finger_3_2') - 7] = self.cfg_task.env.openhand.proximal_open

        self._move_gripper_to_dof_pos(env_ids=env_ids, gripper_dof_pos=gripper_dof_pos, sim_steps=sim_steps)
        self.ctrl_target_gripper_dof_pos = gripper_dof_pos

    def _close_gripper(self, env_ids, sim_steps=20):
        """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        gripper_dof_pos = 0 * self.gripper_dof_pos.clone()
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'base_to_finger_1_1') - 7] = self.cfg_task.env.openhand.base_angle
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'base_to_finger_2_1') - 7] = -self.cfg_task.env.openhand.base_angle

        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'finger_1_1_to_finger_1_2') - 7] = self.cfg_task.env.openhand.proximal_close
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'finger_2_1_to_finger_2_2') - 7] = self.cfg_task.env.openhand.proximal_close
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'base_to_finger_3_2') - 7] = self.cfg_task.env.openhand.proximal_close

        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'finger_1_2_to_finger_1_3') - 7] = self.cfg_task.env.openhand.distal_close
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'finger_2_2_to_finger_2_3') - 7] = self.cfg_task.env.openhand.distal_close
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'finger_3_2_to_finger_3_3') - 7] = self.cfg_task.env.openhand.distal_close

        self._move_gripper_to_dof_pos(env_ids=env_ids, gripper_dof_pos=gripper_dof_pos, sim_steps=sim_steps)

        self.ctrl_target_gripper_dof_pos = gripper_dof_pos

    def _move_gripper_to_dof_pos(self, env_ids, gripper_dof_pos, sim_steps=20):
        """Move gripper fingers to specified DOF position using controller."""

        delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
                                      device=self.device)  # no arm motion
        self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)

        # Step sim
        for _ in range(sim_steps):
            self.render()
            self._refresh_task_tensors()
            self.gym.simulate(self.sim)

        self._refresh_task_tensors()

    def _lift_gripper(self, env_ids, gripper_dof_pos, lift_distance=0.2, sim_steps=20):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        delta_hand_pose = torch.zeros([self.num_envs, 6], device=self.device)
        delta_hand_pose[env_ids, 2] = lift_distance  # lift along z

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)
            self.render()
            self._refresh_task_tensors()
            self.gym.simulate(self.sim)

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5
        return keypoint_offsets

    def _get_keypoint_dist(self):
        """Get keypoint distances."""

        keypoint_dist = torch.sum(torch.norm(self.keypoints_socket - self.keypoints_plug, p=2, dim=-1), dim=-1)
        return keypoint_dist

    def _check_lift_success(self, height_multiple):
        """Check if plug is above table by more than specified multiple times height of plug."""

        lift_success = torch.where(
            self.plug_pos[:, 2] > self.cfg_base.env.table_height + self.plug_heights.squeeze(-1) * height_multiple,
            torch.ones((self.num_envs,), device=self.device),
            torch.zeros((self.num_envs,), device=self.device))

        return lift_success

    def _check_plug_close_to_socket(self):
        """Check if plug is close to socket."""

        keypoint_dist = torch.norm(self.keypoints_socket - self.keypoints_plug, p=2, dim=-1)

        is_plug_close_to_socket = torch.where(torch.sum(keypoint_dist, dim=-1) < self.cfg_task.rl.close_error_thresh,
                                              torch.ones_like(self.progress_buf),
                                              torch.zeros_like(self.progress_buf))

        return is_plug_close_to_socket

    def _zero_velocities(self, env_ids):

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])
        # Set DOF state
        multi_env_ids_int32 = self.kuka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        self.root_linvel[env_ids, self.plug_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.plug_actor_id_env] = 0.0

        plug_actor_ids_sim_int32 = self.plug_actor_ids_sim.to(dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(plug_actor_ids_sim_int32[env_ids]),
                                                     len(plug_actor_ids_sim_int32[env_ids]))

        self.gym.simulate(self.sim)
        self.render()
