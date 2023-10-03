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

import torch

from isaacgym import gymapi, gymtorch
from isaacgyminsertion.tasks.factory_tactile.factory_env_insertion import FactoryEnvInsertionTactile
from isaacgyminsertion.tasks.factory_tactile.factory_schema_class_task import FactoryABCTask
from isaacgyminsertion.tasks.factory_tactile.factory_schema_config_task import FactorySchemaConfigTask
import isaacgyminsertion.tasks.factory_tactile.factory_control as fc
from isaacgyminsertion.tasks.factory_tactile.factory_utils import *
from isaacgyminsertion.utils import torch_jit_utils
from multiprocessing import Process, Queue, Manager

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

        self.ft_hist_len = self.cfg_task.env.ft_history_len
        self.tact_hist_len = self.cfg_task.env.tactile_history_len

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        self.plug_grasp_pos_local = self.plug_heights * 0.5 * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1))
        self.plug_grasp_quat_local = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(
            self.num_envs, 1)

        self.socket_tip_pos_local = self.socket_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1))
        self.gripper_normal_quat = (torch.tensor([-1 / 2 ** 0.5, -1 / 2 ** 0.5, 0.0, 0.0],
                                                 device=self.device).unsqueeze(0).repeat(self.num_envs, 1))

        # Keypoint tensors
        self.keypoint_offsets = self._get_keypoint_offsets(
            self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
        self.keypoints_plug = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3),
                                          dtype=torch.float32,
                                          device=self.device, )
        self.keypoints_socket = torch.zeros_like(self.keypoints_plug, device=self.device)

        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs,
                                                                                                        1)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.targets = torch.zeros((self.num_envs, self.cfg_task.env.numTargets), device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.cfg_task.env.numTargets), dtype=torch.float,
                                        device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.cfg_task.env.numTargets), dtype=torch.float,
                                       device=self.device)

        # Keep track of history
        self.actions_queue = torch.zeros((self.num_envs, self.cfg_task.env.numObsHist, self.num_actions),
                                         dtype=torch.float, device=self.device)
        self.targets_queue = torch.zeros((self.num_envs, self.cfg_task.env.numObsHist, self.num_actions),
                                         dtype=torch.float, device=self.device)

        # tactile buffers
        self.tactile_imgs = torch.zeros(
            (self.num_envs, len(self.fingertips),
             self.cfg_tactile.tacto.width, self.cfg_tactile.tacto.height, 3),
            device=self.device,
            dtype=torch.float,
        )
        self.tactile_queue = torch.zeros(
            (self.num_envs, self.tact_hist_len, len(self.fingertips),
             self.cfg_tactile.tacto.width, self.cfg_tactile.tacto.height, 3),
            device=self.device,
            dtype=torch.float,
        )

        self.ft_queue = torch.zeros((self.num_envs, self.ft_hist_len, 6), device=self.device, dtype=torch.float)


    def _refresh_task_tensors(self, update_tactile=True):
        """Refresh tensors."""

        self.refresh_base_tensors()
        self.refresh_env_tensors()

        self.plug_grasp_quat, self.plug_grasp_pos = torch_jit_utils.tf_combine(self.plug_quat,
                                                                               self.plug_pos,
                                                                               self.plug_grasp_quat_local,
                                                                               self.plug_grasp_pos_local)

        # Add observation noise to socket pos
        self.noisy_socket_pos = torch.zeros_like(
            self.socket_pos, dtype=torch.float32, device=self.device
        )
        socket_obs_pos_noise = 2 * (
                torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
                - 0.5
        )
        socket_obs_pos_noise = socket_obs_pos_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.env.socket_pos_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.noisy_socket_pos[:, 0] = self.socket_pos[:, 0] + socket_obs_pos_noise[:, 0]
        self.noisy_socket_pos[:, 1] = self.socket_pos[:, 1] + socket_obs_pos_noise[:, 1]
        self.noisy_socket_pos[:, 2] = self.socket_pos[:, 2] + socket_obs_pos_noise[:, 2]

        # Add observation noise to desired_rot
        desired_rot_euler = torch.tensor([-np.pi / 2, -np.pi / 2, 0], device=self.device).unsqueeze(0).repeat(
            self.num_envs, 1)

        desired_rot_noise = 2 * (
                torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
                - 0.5
        )
        socket_obs_rot_noise = desired_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.env.socket_rot_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        desired_obs_rot_euler = desired_rot_euler + socket_obs_rot_noise
        self.noisy_desired_quat = torch_jit_utils.quat_from_euler_xyz(
            desired_obs_rot_euler[:, 0],
            desired_obs_rot_euler[:, 1],
            desired_obs_rot_euler[:, 2],
        )

        self.noisy_gripper_goal_quat, self.noisy_gripper_goal_pos = torch_jit_utils.tf_combine(
            self.identity_quat,
            self.noisy_socket_pos,
            self.gripper_normal_quat,
            self.socket_tip_pos_local)

        # Compute pos of keypoints on gripper, socket, and plug in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
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

            self._update_tactile(left_finger_pose, right_finger_pose, middle_finger_pose, object_pose)

            # groups = 8 if self.num_envs > 8 else 1
            # offset = np.ceil(self.num_envs / groups).astype(int)
            #
            # processes = []
            # results = []
            #
            # with Manager() as manager:
            #     for g in range(groups):
            #         env_start = offset * g
            #         env_end = np.clip(offset * (g + 1), 0, self.num_envs)
            #
            #         queue = manager.Queue()
            #         process = Process(
            #             target=self._update_tactile,
            #             args=(
            #                 left_finger_pose,
            #                 right_finger_pose,
            #                 middle_finger_pose,
            #                 object_pose,
            #                 env_start,
            #                 queue,
            #             ),
            #         )
            #         process.start()
            #         processes.append(process)
            #         results.append(queue)
            #
            #     for process in processes:
            #         process.join()

    def _update_tactile(self, left_finger_pose, right_finger_pose, middle_finger_pose, object_pose, offset=None,
                        queue=None, display_viz=False):

        tactile_imgs, height_maps = [], []

        for e in range(self.num_envs):

            self.tactile_handles[e][0].update_pose_given_sim_pose(left_finger_pose[e], object_pose[e])
            self.tactile_handles[e][1].update_pose_given_sim_pose(right_finger_pose[e], object_pose[e])
            self.tactile_handles[e][2].update_pose_given_sim_pose(middle_finger_pose[e], object_pose[e])

            tactile_imgs_per_env, height_maps_per_env = [], []

            for n in range(3):
                tactile_img, height_map, _ = self.tactile_handles[e][n].render(object_pose[e])
                tactile_imgs_per_env.append(tactile_img)
                height_maps_per_env.append(height_map)

            tactile_imgs.append(tactile_imgs_per_env)
            height_maps.append(height_maps_per_env)

        self.tactile_imgs = torch.tensor(tactile_imgs, dtype=torch.float32, device=self.device)

        if display_viz:
            env_to_show = 0
            self.tactile_handles[env_to_show][0].updateGUI(tactile_imgs[env_to_show], height_maps[env_to_show])

        # if queue is not None:
        #     queue.put((tactile_imgs, offset))

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

            # If it fail to lift, reset
            # lift_success = self._check_lift_success(height_multiple=0.2)
            # env_ids = lift_success[lift_success == 0].int()

            # if len(env_ids):
            #     self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]
        delta_targets = torch.cat([
            self.actions[:, :3] @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device)),  # 3
            self.actions[:, 3:6] @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))  # 3
        ], dim=-1).clone()

        # Update targets
        self.targets = self.prev_targets + delta_targets

        # update the queue
        self.actions_queue[:, 1:] = self.actions_queue[:, :-1].clone().detach()
        self.actions_queue[:, 0, :] = self.actions

        self.targets_queue[:, 1:] = self.targets_queue[:, :-1].clone().detach()
        self.targets_queue[:, 0, :] = self.targets

        self._apply_actions_as_ctrl_targets(actions=self.actions,
                                            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
                                            do_scale=True)


        self.prev_actions[:] = self.actions.clone()
        self.prev_targets[:] = self.targets.clone()

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

        self.progress_buf[:] += 1
        self.randomize_buf[:] += 1

        # In this policy, episode length is constant
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

        if self.viewer and True:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                targetx = (self.socket_tip[i] + quat_apply(self.socket_quat[i],
                                                           to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                targety = (self.socket_tip[i] + quat_apply(self.socket_quat[i],
                                                           to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                targetz = (self.socket_tip[i] + quat_apply(self.socket_quat[i],
                                                           to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.socket_tip[i].cpu().numpy()  # + self.goal_displacement_tensor.cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])

                objectx = (self.plug_com_pos[i] + quat_apply(self.plug_quat[i], to_torch([1, 0, 0],
                                                                                         device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.plug_com_pos[i] + quat_apply(self.plug_quat[i], to_torch([0, 1, 0],
                                                                                         device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.plug_com_pos[i] + quat_apply(self.plug_quat[i], to_torch([0, 0, 1],
                                                                                         device=self.device) * 0.2)).cpu().numpy()

                p0 = self.plug_com_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

    def compute_observations(self):
        """Compute observations."""

        # Compute tactile
        self.tactile_queue[:, 1:] = self.tactile_queue[:, :-1].clone().detach()
        self.tactile_queue[:, 0, :] = self.tactile_imgs
        #
        self.ft_queue[:, 1:] = self.ft_queue[:, :-1].clone().detach()
        self.ft_queue[:, 0, :] = 0.1 * self.ft_sensor_tensor.clone()

        # Compute Observation and state at current timestep
        delta_pos = self.socket_tip - self.fingertip_centered_pos
        noisy_delta_pos = self.noisy_gripper_goal_pos - self.fingertip_centered_pos

        # Define observations (for actor)
        obs_tensors = [
            self.arm_dof_pos,  # 7
            self.pose_world_to_robot_base(self.fingertip_centered_pos, self.fingertip_centered_quat)[0],  # 3
            self.pose_world_to_robot_base(self.fingertip_centered_pos, self.fingertip_centered_quat)[1],  # 4
            self.pose_world_to_robot_base(self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat)[0],  # 3
            self.pose_world_to_robot_base(self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat)[1],  # 4
            noisy_delta_pos,  # 3
            self.actions_queue.reshape(self.actions_queue.shape[0], -1),  # numObsHist, 6
            self.targets_queue.reshape(self.targets_queue.shape[0], -1),  # numObsHist, 6
        ]  # TODO, may need to add history of other obs inputs

        # Define state (for teacher)
        # Physics states
        mass = [self.gym.get_actor_rigid_body_properties(e, p)[0].mass for e, p in zip(self.envs, self.plug_handles)]
        friction = [self.gym.get_actor_rigid_shape_properties(e, p)[0].friction for e, p in
                    zip(self.envs, self.plug_handles)]
        physics_params = torch.transpose(to_torch([mass, friction]), 0, 1)

        state_tensors = [
            self.arm_dof_pos,  # 7
            self.arm_dof_vel,  # 7
            self.pose_world_to_robot_base(self.fingertip_centered_pos, self.fingertip_centered_quat)[0],  # 3
            self.pose_world_to_robot_base(self.fingertip_centered_pos, self.fingertip_centered_quat)[1],  # 4
            self.fingertip_centered_linvel,  # 3
            self.fingertip_centered_angvel,  # 3
            self.pose_world_to_robot_base(self.fingertip_midpoint_pos, self.fingertip_midpoint_quat)[0],  # 3
            self.pose_world_to_robot_base(self.fingertip_midpoint_pos, self.fingertip_midpoint_quat)[1],  # 4
            self.fingertip_midpoint_linvel,  # 3
            self.fingertip_midpoint_angvel,  # 3
            self.pose_world_to_robot_base(self.socket_tip, self.gripper_normal_quat)[0],  # 3
            self.pose_world_to_robot_base(self.socket_tip, self.gripper_normal_quat)[1],  # 4
            delta_pos,  # 3
            self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[0],  # 3
            self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[1],  # 4
            noisy_delta_pos - delta_pos,  # 3
            physics_params,  # 2
        ]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)
        self.states_buf = torch.cat(state_tensors, dim=-1)  # shape = (num_envs, num_states)

        return self.obs_buf  # shape = (num_envs, num_observations)

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        self._update_rew_buf()
        self._update_reset_buf()

    def _update_rew_buf(self):
        """Compute reward at current timestep."""
        keypoint_reward = -self._get_keypoint_dist()
        action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale
        plug_ori_penalty = torch.norm(self.plug_quat - self.identity_quat, p=2, dim=-1)
        is_plug_oriented = plug_ori_penalty < 0.1

        dist_plug_socket = torch.norm(self.plug_pos - self.socket_pos, p=2, dim=-1)

        self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                          - action_penalty * self.cfg_task.rl.action_penalty_scale \
                          - plug_ori_penalty * self.cfg_task.rl.orientation_penalty_scale

        is_plug_engaged_w_socket = self._check_plug_engaged_w_socket()

        is_plug_inserted_in_socket = self._check_plug_inserted_in_socket()
        self.time_complete_task[self.time_complete_task == 0] = is_plug_inserted_in_socket * self.progress_buf[
            self.time_complete_task == 0
        ]

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

        if is_last_step:
            engagement_reward_scale = self._get_engagement_reward_scale(is_plug_engaged_w_socket,
                                                                        self.cfg_task.rl.success_height_thresh)
            self.rew_buf[:] += (engagement_reward_scale * self.cfg_task.rl.engagement_bonus)

            self.rew_buf[:] += is_plug_inserted_in_socket * self.cfg_task.rl.success_bonus

            self.extras['successes'] += torch.mean(is_plug_inserted_in_socket.float())
            self.extras["engaged_w_socket"] = torch.mean(is_plug_engaged_w_socket.float())
            self.extras["plug_oriented"] = torch.mean(is_plug_oriented.float())
            self.extras["successes"] = torch.mean(is_plug_inserted_in_socket.float())
            self.extras["dist_plug_socket"] = torch.mean(dist_plug_socket)
            self.extras["keypoint_reward"] = torch.mean(keypoint_reward.abs())
            self.extras["action_penalty"] = torch.mean(action_penalty)
            self.extras["mug_quat_penalty"] = torch.mean(plug_ori_penalty)
            self.extras["steps"] = torch.mean(self.progress_buf.float())
            self.extras["mean_time_complete_task"] = torch.mean(
                self.time_complete_task.float()
            )
            a = self.time_complete_task.float() * is_plug_inserted_in_socket
            self.extras["time_success_task"] = a.sum() / torch.where(a > 0)[0].shape[0]

        # # resets due to misbehavior TODO
        # ones = torch.ones_like(self.reset_buf)
        # die = torch.zeros_like(self.reset_buf)
        # die = torch.where(keypoint_reward > 5.0, ones, die)
        # die = torch.where(root_positions[..., 2] < 0.5, ones, die)
        # reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""
        # If max episode length has been reached
        self.reset_buf[:] = torch.where(self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1,
                                        torch.ones_like(self.reset_buf),
                                        self.reset_buf)

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

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
        self._zero_velocities(env_ids)

        # Lift
        self._lift_gripper(env_ids, self.ctrl_target_gripper_dof_pos)
        self._refresh_task_tensors()
        self._zero_velocities(env_ids)

        # Move arm above the socket
        self._move_arm_to_desired_pose(env_ids, self.above_socket_pos.clone(),
                                       sim_steps=5 * self.cfg_task.env.num_gripper_move_sim_steps)
        self._refresh_task_tensors()

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

        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids].clone()
        self.ctrl_target_gripper_dof_pos[env_ids] = self.dof_pos[env_ids, 7:]

        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos.clone()
        self.ctrl_target_fingertip_centered_quat = self.fingertip_centered_quat.clone()

        multi_env_ids_int32 = self.kuka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Set DOF torque
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_torque),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

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
        self.ctrl_target_fingertip_centered_pos = desired_pos.clone()

        # Set target rot
        if desired_rot is None:
            ctrl_target_fingertip_centered_euler = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                                                                device=self.device).unsqueeze(0).repeat(self.num_envs,
                                                                                                        1)

            self.ctrl_target_fingertip_centered_quat = torch_jit_utils.quat_from_euler_xyz(
                ctrl_target_fingertip_centered_euler[:, 0],
                ctrl_target_fingertip_centered_euler[:, 1],
                ctrl_target_fingertip_centered_euler[:, 2])
        else:
            self.ctrl_target_fingertip_centered_quat = desired_rot

        # Step sim and render
        # while torch.norm(pos_error, p=2, dim=-1) > 0.001 and torch.norm(axis_angle_error, p=2, dim=-1) > 0.001:
        for _ in range(sim_steps):
            self._simulate_and_refresh()

            # NOTE: midpoint is calculated based on the midpoint between the actual gripper finger pos,
            # and centered is calculated with the assumption that the gripper fingers are perfectly closed at center.
            # since the fingertips are underactuated, thus we cant know the true pose

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_centered_pos,
                fingertip_midpoint_quat=self.fingertip_centered_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_centered_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_centered_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[:, :6] = delta_hand_pose

            # Apply the action, keep fingers in the same status
            self.ctrl_target_dof_pos[env_ids, 7:] = self.ctrl_target_gripper_dof_pos
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
        self.time_complete_task = torch.zeros_like(self.progress_buf)

        # Reset history
        self.ft_queue[env_ids] = 0
        self.tactile_queue[env_ids] = 0
        self.actions_queue[env_ids] *= 0
        self.targets_queue[env_ids] *= 0
        self.prev_targets[env_ids] *= 0
        self.prev_actions[env_ids] *= 0

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
        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos + pos_actions

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
        self.ctrl_target_fingertip_centered_quat = torch_jit_utils.quat_mul(rot_actions_quat,
                                                                            self.fingertip_centered_quat)

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
            self._simulate_and_refresh()

    def _lift_gripper(self, env_ids, gripper_dof_pos, lift_distance=0.2, sim_steps=20):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        delta_hand_pose = torch.zeros([self.num_envs, 6], device=self.device)
        delta_hand_pose[env_ids, 2] = lift_distance  # lift along z

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)
            self._simulate_and_refresh()

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5
        return keypoint_offsets

    def _get_keypoint_dist(self):
        """Get keypoint distances."""

        keypoint_dist = torch.sum(torch.norm(self.keypoints_socket - self.keypoints_plug, p=2, dim=-1), dim=-1)
        return keypoint_dist

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

        self._simulate_and_refresh()

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

    def _check_plug_inserted_in_socket(self):
        """Check if plug is inserted in socket."""

        # Check if plug is within threshold distance of assembled state
        is_plug_below_insertion_height = (
                self.plug_pos[:, 2] < self.socket_pos[:, 2] + self.cfg_task.rl.success_height_thresh
        )

        # Check if plug is close to socket
        # NOTE: This check addresses edge case where plug is within threshold distance of
        # assembled state, but plug is outside socket
        is_plug_close_to_socket = self._check_plug_close_to_socket()

        # Combine both checks
        is_plug_inserted_in_socket = torch.logical_and(
            is_plug_below_insertion_height, is_plug_close_to_socket
        )

        return is_plug_inserted_in_socket

    def _check_plug_engaged_w_socket(self):
        """Check if plug is engaged with socket."""

        # Check if base of plug is below top of socket
        # NOTE: In assembled state, plug origin is coincident with socket origin;
        # thus plug pos must be offset to compute actual pos of base of plug
        is_plug_below_engagement_height = (
                self.plug_pos[:, 2] + self.cfg_task.env.socket_base_height < self.socket_tip[:, 2]
        )

        # Check if plug is close to socket
        # NOTE: This check addresses edge case where base of plug is below top of socket,
        # but plug is outside socket
        is_plug_close_to_socket = self._check_plug_close_to_socket()

        # Combine both checks
        is_plug_engaged_w_socket = torch.logical_and(
            is_plug_below_engagement_height, is_plug_close_to_socket
        )

        return is_plug_engaged_w_socket

    def _get_engagement_reward_scale(self, is_plug_engaged_w_socket, success_height_thresh):
        """Compute scale on reward. If plug is not engaged with socket, scale is zero.
        If plug is engaged, scale is proportional to distance between plug and bottom of socket."""

        # Set default value of scale to zero
        reward_scale = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        # For envs in which plug and socket are engaged, compute positive scale
        engaged_idx = np.argwhere(is_plug_engaged_w_socket.cpu().numpy().copy()).squeeze()
        height_dist = self.plug_pos[engaged_idx, 2] - self.socket_pos[engaged_idx, 2]
        # NOTE: Edge case: if success_height_thresh is greater than 0.1,
        # denominator could be negative
        reward_scale[engaged_idx] = 1.0 / ((height_dist - success_height_thresh) + 0.1)

        return reward_scale

    def step(self, actions):
        super().step(actions)
        self.obs_dict['tactile_hist'] = self.tactile_queue.to(self.rl_device)
        self.obs_dict['ft_hist'] = self.ft_queue.to(self.rl_device)
        self.obs_dict['priv_info'] = self.obs_dict['states']
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        super().reset()
        self.obs_dict['priv_info'] = self.obs_dict['states']
        self.obs_dict['tactile_hist'] = self.tactile_queue.to(self.rl_device)
        self.obs_dict['ft_hist'] = self.ft_queue.to(self.rl_device)
        return self.obs_dict
