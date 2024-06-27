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

from isaacgym import gymapi, gymtorch
from isaacgyminsertion.tasks.factory_tactile.factory_env_insertion import FactoryEnvInsertionTactile
from isaacgyminsertion.tasks.factory_tactile.schema.factory_schema_class_task import FactoryABCTask
from isaacgyminsertion.tasks.factory_tactile.schema.factory_schema_config_task import FactorySchemaConfigTask
import isaacgyminsertion.tasks.factory_tactile.factory_control as fc
from isaacgyminsertion.tasks.factory_tactile.factory_utils import *
from isaacgyminsertion.utils import torch_jit_utils
from tqdm import tqdm
import cv2
from hyperopt import fmin, hp, tpe, space_eval

torch.set_printoptions(sci_mode=False)


class FactoryTaskOptimizeTactile(FactoryEnvInsertionTactile, FactoryABCTask):

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

        self.total_grasps = self.cfg_task.grasp_logger.total_grasp
        self.grasps_save_ctr = 0
        self.total_init_grasp_count = 0
        self.pbar = tqdm(total=self.total_grasps)

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

        # 0.9 2in, 1.5in cylinder
        # 0.7 triangle

        self.plug_grasp_pos_local = self.plug_heights * 0.9 * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1))
        self.plug_grasp_quat_local = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(
            self.num_envs, 1)

        self.socket_tip_pos_local = self.socket_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1))

        # Keypoint tensors
        self.keypoint_offsets = self._get_keypoint_offsets(
            self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
        self.keypoints_plug = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3),
                                          dtype=torch.float32,
                                          device=self.device, )
        self.keypoints_socket = torch.zeros_like(self.keypoints_plug, device=self.device)

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
            (self.num_envs, len(self.fingertips),  # left, right, bottom
             self.cfg_tactile.decoder.width, self.cfg_tactile.decoder.height, 3),
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )
        self.depth_maps = torch.zeros(
            (self.num_envs, len(self.fingertips),  # left, right, bottom
             self.cfg_tactile.decoder.width, self.cfg_tactile.decoder.height),
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )
        # Way too big tensor.
        self.tactile_queue = torch.zeros(
            (self.num_envs, self.tact_hist_len, len(self.fingertips),  # left, right, bottom
             self.cfg_tactile.decoder.width, self.cfg_tactile.decoder.height, 3),
            device=self.device,
            dtype=torch.float,
        )

        self.ft_queue = torch.zeros((self.num_envs, self.ft_hist_len, 6), device=self.device, dtype=torch.float)

        self.gt_extrinsic_contact = torch.zeros((self.num_envs, self.cfg_task.env.num_points),
                                                device=self.device, dtype=torch.float)

    def _refresh_task_tensors(self, update_tactile=False):
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
                                                                       keypoint_offset.repeat(self.num_envs, 1))[1]

        if update_tactile and self.cfg_task.env.tactile:
            # left_finger_poses = pose_vec_to_mat(torch.cat((self.left_finger_pos,
            #                                               self.left_finger_quat), axis=1)).cpu().numpy()
            # right_finger_poses = pose_vec_to_mat(torch.cat((self.right_finger_pos,
            #                                                self.right_finger_quat), axis=1)).cpu().numpy()
            # middle_finger_poses = pose_vec_to_mat(torch.cat((self.middle_finger_pos,
            #                                                 self.middle_finger_quat), axis=1)).cpu().numpy()
            # object_pose = pose_vec_to_mat(torch.cat((self.plug_pos, self.plug_quat), axis=1)).cpu().numpy()

            left_allsight_poses = torch.cat((self.left_finger_pos, self.left_finger_quat), dim=-1)
            right_allsight_poses = torch.cat((self.right_finger_pos, self.right_finger_quat), dim=-1)
            middle_allsight_poses = torch.cat((self.middle_finger_pos, self.middle_finger_quat), dim=-1)
            object_poses = torch.cat((self.plug_pos, self.plug_quat), dim=-1)

            tf = np.eye(4)
            tf[0:3, 0:3] = euler_angles_to_matrix(
                euler_angles=torch.tensor([[3.14159265359, 0, 0]]), convention="XYZ"
            ).numpy()

            left_finger_poses = xyzquat_to_tf_numpy(left_allsight_poses.cpu().numpy())
            # left_finger_poses = left_finger_poses @ np.array(
            #     np.linalg.inv(
            #         [
            #             [0.0, -1.0, 0.0, 0.0],
            #             [0.0, 0.0, 1.0, 0.0],
            #             [-1.0, 0.0, 0.0, 0.0],
            #             [0.0, 0.0, 0.0, 1.0],
            #         ]
            #     )
            # )
            left_finger_poses = left_finger_poses @ tf

            right_finger_poses = xyzquat_to_tf_numpy(right_allsight_poses.cpu().numpy())
            # right_finger_poses = right_finger_poses @ np.array(
            #     np.linalg.inv(
            #         [
            #             [0.0, -1.0, 0.0, 0.0],
            #             [0.0, 0.0, 1.0, 0.0],
            #             [-1.0, 0.0, 0.0, 0.0],
            #             [0.0, 0.0, 0.0, 1.0],
            #         ]
            #     )
            # )
            right_finger_poses = right_finger_poses @ tf

            middle_finger_poses = xyzquat_to_tf_numpy(middle_allsight_poses.cpu().numpy())
            # middle_finger_poses = middle_finger_poses @ np.array(
            #     np.linalg.inv(
            #         [
            #             [0.0, -1.0, 0.0, 0.0],
            #             [0.0, 0.0, 1.0, 0.0],
            #             [-1.0, 0.0, 0.0, 0.0],
            #             [0.0, 0.0, 0.0, 1.0],
            #         ]
            #     )
            # )
            middle_finger_poses = middle_finger_poses @ tf

            object_pose = xyzquat_to_tf_numpy(object_poses.cpu().numpy())

            self._update_tactile(left_finger_poses, right_finger_poses, middle_finger_poses, object_pose)

            # groups = 8 if self.num_envs > 8 else 1
            # offset = np.ceil(self.num_envs / groups).astype(int)
            # #
            # processes = []
            # results = []
            # #
            # with Manager() as manager:
            #     for g in range(groups):
            #         env_start = offset * g
            #         env_end = np.clip(offset * (g + 1), 0, self.num_envs)

            #         left_finger_pose_g = left_finger_pose[env_start:env_end]
            #         right_finger_pose_g = right_finger_pose[env_start:env_end]
            #         middle_finger_pose_g = middle_finger_pose[env_start:env_end]
            #         object_pose_g = object_pose[env_start:env_end]

            #         tactile_handles_g = self.tactile_handles[env_start:env_end]
            #         print(left_finger_pose_g.shape, right_finger_pose_g.shape, middle_finger_pose_g.shape, object_pose_g.shape, len(tactile_handles_g))
            #         queue = manager.Queue()
            #         process = Process(
            #             target=self._update_tactile,
            #             args=(
            #                 tactile_handles_g,
            #                 left_finger_pose_g,
            #                 right_finger_pose_g,
            #                 middle_finger_pose_g,
            #                 object_pose_g,
            #                 env_start,
            #                 queue,
            #             ),
            #         )
            #         process.start()
            #         processes.append(process)
            #         results.append(queue)

            #     for process in processes:
            #         process.join()
            #     # self.tactile_imgs
            #     for queue in results:
            #         left, right, middle, left_depth, left_right, left_middle, offset = queue.get()
            #         print(offset, offset + len(left))
            #         self.tactile_imgs[offset: offset + len(left), 0, :] = torch.from_numpy(left).clone().detach().to(self.device)
            #         self.tactile_imgs[offset: offset + len(right), 1, :] = torch.from_numpy(right).clone().detach().to(self.device)
            #         self.tactile_imgs[offset: offset + len(middle), 2, :] = torch.from_numpy(middle).clone().detach().to(self.device)
            #         self.depth_maps[offset: offset + len(left), 0, :] = torch.from_numpy(left_depth).clone().detach().to(self.device)
            #         self.depth_maps[offset: offset + len(right), 1, :] = torch.from_numpy(left_right).clone().detach().to(self.device)
            #         self.depth_maps[offset: offset + len(middle), 2, :] = torch.from_numpy(left_middle).clone().detach().to(self.device)

            # self.digit_left_buf[
            #     offset : offset + len(left), 0, :
            # ] = torch.tensor(left)
            # self.digit_right_buf[
            #     offset : offset + len(right), 0, :
            # ] = torch.tensor(right)

            # self.tactile_imgs = torch.tensor(tactile_imgs, dtype=torch.float32, device=self.device)
            # for queue in results:
            #     imgs, offest = queue.get()
            #     print(imgs.shape)
            #     self.tactile_imgs = torch.tensor(imgs, dtype=torch.float32, device=self.device)

    def _update_tactile(self, left_finger_pose, right_finger_pose, middle_finger_pose, object_pose,
                        offset=None, queue=None, display_viz=False):

        tactile_imgs_list, height_maps = [], []  # only for display.

        for e in range(self.num_envs):

            # plug_file = self.asset_info_insertion[self.envs_asset[e]['subassembly'] ][self.envs_asset[e]['components'][0]]['urdf_path']
            # plug_file += '_subdiv_3x.obj' if 'rectangular' in plug_file else '.obj'
            # mesh_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'factory', 'mesh',
            #                          'factory_insertion')
            # self.tactile_handles.append([allsight_renderer(self.cfg_tactile,
            #                                                os.path.join(mesh_root, plug_file), randomize=False,
            #                                                finger_idx=i) for i in range(len(self.fingertips))])

            self.tactile_handles[e][0].update_pose_given_sim_pose(left_finger_pose[e], object_pose[e])
            self.tactile_handles[e][1].update_pose_given_sim_pose(right_finger_pose[e], object_pose[e])
            self.tactile_handles[e][2].update_pose_given_sim_pose(middle_finger_pose[e], object_pose[e])

            tactile_imgs_per_env, height_maps_per_env = [], []

            for n in range(3):
                tactile_img, height_map = self.tactile_handles[e][n].render(object_pose[e])
                resized_img = cv2.resize(tactile_img, (self.cfg_tactile.decoder.width,
                                                       self.cfg_tactile.decoder.height), interpolation=cv2.INTER_AREA)
                # resized_depth
                self.tactile_imgs[e, n] = torch_jit_utils.rgb_transform(resized_img).to(self.device).permute(1, 2, 0)
                self.depth_maps[e, n] = torch.tensor(height_map).to(self.device)
                tactile_imgs_per_env.append(tactile_img)
                height_maps_per_env.append(height_map)

            height_maps.append(height_maps_per_env)
            tactile_imgs_list.append(tactile_imgs_per_env)

        # self.tactile_imgs = torch.tensor(tactile_imgs_list, dtype=torch.float32, device=self.device)

        if display_viz:
            env_to_show = 0
            self.tactile_handles[env_to_show][0].updateGUI(tactile_imgs_list[env_to_show],
                                                           height_maps[env_to_show])

        if queue is not None:
            queue.put((tactile_imgs_list, offset))

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]
        # print('actions', self.actions)
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

                p0 = self.socket_pos[i].cpu().numpy()
                # print('socket tip', p0[2], targetx)
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

        self._render_headless()

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
            # self.arm_dof_pos,  # 7
            self.pose_world_to_robot_base(self.fingertip_centered_pos, self.fingertip_centered_quat)[0],  # 3
            self.pose_world_to_robot_base(self.fingertip_centered_pos, self.fingertip_centered_quat)[1],  # 4
            self.pose_world_to_robot_base(self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat)[0],  # 3
            self.pose_world_to_robot_base(self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat)[1],  # 4
            # noisy_delta_pos,  # 3 TODO, need to think about that one.
            self.actions_queue.reshape(self.num_envs, -1),  # numObsHist, 6 --> flatten
            self.targets_queue.reshape(self.num_envs, -1),  # numObsHist, 6 --> flatten
        ]  # TODO, may need to add history of other obs inputs

        # Define state (for teacher)
        # Physics states Priv
        mass = [self.gym.get_actor_rigid_body_properties(e, p)[0].mass for e, p in zip(self.envs, self.plug_handles)]
        friction = [self.gym.get_actor_rigid_shape_properties(e, p)[0].friction for e, p in
                    zip(self.envs, self.plug_handles)]
        physics_params = torch.transpose(to_torch([mass, friction]), 0, 1)

        state_tensors = [
            # self.arm_dof_pos,  # 7
            # self.arm_dof_vel,  # 7
            # self.pose_world_to_robot_base(self.fingertip_centered_pos, self.fingertip_centered_quat)[0],  # 3
            # self.pose_world_to_robot_base(self.fingertip_centered_pos, self.fingertip_centered_quat)[1],  # 4
            self.fingertip_centered_linvel,  # 3
            self.fingertip_centered_angvel,  # 3
            # self.pose_world_to_robot_base(self.fingertip_midpoint_pos, self.fingertip_midpoint_quat)[0],  # 3
            # self.pose_world_to_robot_base(self.fingertip_midpoint_pos, self.fingertip_midpoint_quat)[1],  # 4
            # self.fingertip_midpoint_linvel,  # 3
            # self.fingertip_midpoint_angvel,  # 3
            self.pose_world_to_robot_base(self.socket_tip, self.gripper_normal_quat)[0],  # 3
            self.pose_world_to_robot_base(self.socket_tip, self.gripper_normal_quat)[1],  # 4
            # delta_pos,  # 3
            self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[0],  # 3
            self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[1],  # 4
            # noisy_delta_pos - delta_pos,  # 3
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

        keypoint_reward = self._get_keypoint_dist()
        action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale
        plug_ori_penalty = torch.norm(self.plug_quat - self.identity_quat, p=2, dim=-1)
        is_plug_oriented = plug_ori_penalty < self.cfg_task.rl.orientation_threshold

        self.dist_plug_socket = torch.norm(self.plug_pos - self.socket_pos, p=2, dim=-1)

        self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                          + plug_ori_penalty * self.cfg_task.rl.orientation_penalty_scale
        #   + action_penalty * self.cfg_task.rl.action_penalty_scale \

        # print(keypoint_reward[0], self.rew_buf[0])

        is_plug_engaged_w_socket = self._check_plug_engaged_w_socket()

        engagement_reward_scale = self._get_engagement_reward_scale(is_plug_engaged_w_socket,
                                                                    self.cfg_task.rl.success_height_thresh)

        is_plug_inserted_in_socket = self._check_plug_inserted_in_socket()
        # print(is_plug_inserted_in_socket[0])
        self.time_complete_task[self.time_complete_task == 0] = (is_plug_inserted_in_socket * self.progress_buf)[
            self.time_complete_task == 0
            ]

        # In this policy, episode length is constant across all envs todo why?
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

        # if is_last_step:
        #     # engagement_reward_scale = self._get_engagement_reward_scale(is_plug_engaged_w_socket,
        #     #                                                             self.cfg_task.rl.success_height_thresh)
        #     # self.rew_buf[:] += (engagement_reward_scale * self.cfg_task.rl.engagement_bonus)

        #     # self.rew_buf[:] += is_plug_inserted_in_socket * self.cfg_task.rl.success_bonus

        #     self.extras['successes'] += torch.mean(is_plug_inserted_in_socket.float())
        #     self.extras["engaged_w_socket"] = torch.mean(is_plug_engaged_w_socket.float())
        #     self.extras["plug_oriented"] = torch.mean(is_plug_oriented.float())
        #     self.extras["successes"] = torch.mean(is_plug_inserted_in_socket.float())
        #     self.extras["dist_plug_socket"] = torch.mean(self.dist_plug_socket)
        #     self.extras["keypoint_reward"] = torch.mean(keypoint_reward.abs())
        #     self.extras["action_penalty"] = torch.mean(action_penalty)
        #     self.extras["mug_quat_penalty"] = torch.mean(plug_ori_penalty)
        #     self.extras["steps"] = torch.mean(self.progress_buf.float())
        #     self.extras["mean_time_complete_task"] = torch.mean(
        #         self.time_complete_task.float()
        #     )
        #     a = self.time_complete_task.float() * is_plug_inserted_in_socket
        #     self.extras["time_success_task"] = a.sum() / torch.where(a > 0)[0].shape[0]

        # TODO update reward function to reset at insertion

    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""

        # print('before', self.reset_buf)

        # If it fail to lift, reset
        # lift_success = self._check_lift_success(height_multiple=0.1)
        # self.reset_buf[:] = torch.where(lift_success[:] == 0,
        #                                 torch.ones_like(self.reset_buf),
        #                                 self.reset_buf)

        # print('lift success', lift_success, self.reset_buf)

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1,
                                        torch.ones_like(self.reset_buf),
                                        self.reset_buf)

        # print('max episode length', self.reset_buf)

        # check is object is grasped and reset if not
        d = torch.norm(self.fingertip_midpoint_pos - self.plug_pos, p=2, dim=-1)
        is_not_grasped = d >= self.cfg_task.env.plug_grasp_threshold
        self.reset_buf[is_not_grasped] = 1

        # print('object is grasped', self.reset_buf)

        # If plug is too far from socket pos
        self.reset_buf[:] = torch.where(self.dist_plug_socket > self.cfg_task.rl.far_error_thresh,
                                        torch.ones_like(self.reset_buf),
                                        self.reset_buf)

        # print('plug is too far', self.reset_buf)

    def _initialize_trajectories(self, gp='test_files/traj/'):

        from glob import glob

        all_paths = glob(f'{gp}/*.npz')
        print('Total trajectories:', len(all_paths))

        self.num_paths = len(all_paths)
        self.arm_joints_real = torch.zeros((len(all_paths), 1999, 7))
        self.eef_pose_real = torch.zeros((len(all_paths), 1999, 7))
        self.actions_real = torch.zeros((len(all_paths), 1999, 6))

        for i, p in enumerate(all_paths):
            data = np.load(p, allow_pickle=True)['arr_0'][()]  # saved without keys :X

            self.arm_joints_real[i] = torch.tensor(data['real_joints']).to(self.device)
            self.eef_pose_real[i] = torch.tensor(data['real_pose']).to(self.device)
            self.actions_real[i] = torch.tensor(data['real_actions']).to(self.device)

            # well...
            self.eef_pose_real[i, :, 3] = -1 * torch.abs(self.eef_pose_real[i, :, 3])
            self.eef_pose_real[i, :, 4] = 1 * torch.abs(self.eef_pose_real[i, :, 4])

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        if self.randomize and False:
            self.apply_randomizations(self.randomization_params)

        self._initialize_trajectories()

        def objective(params):
            """
            objective function to be minimized
            :param params: PID values
            :return: loss


            """
            '''
            {'joint_deriv_gains_1': 96,
             'joint_deriv_gains_2': 72,
             'joint_deriv_gains_3': 95,
             'joint_deriv_gains_4': 73,
             'joint_deriv_gains_5': 26,
             'joint_deriv_gains_6': 61,
             'joint_deriv_gains_7': 27,
             'joint_prop_gains_1': 275,
             'joint_prop_gains_2': 245,
             'joint_prop_gains_3': 58,
             'joint_prop_gains_4': 240,
             'joint_prop_gains_5': 208,
             'joint_prop_gains_6': 181,
             'joint_prop_gains_7': 190}
            
            '''

            cfg_ctrl = {'joint_prop_gains': [params['joint_prop_gains_1'] if 'joint_prop_gains_1' in params else 290,
                                             params['joint_prop_gains_2'] if 'joint_prop_gains_2' in params else 208,
                                             params['joint_prop_gains_3'] if 'joint_prop_gains_3' in params else 145,
                                             params['joint_prop_gains_4'] if 'joint_prop_gains_4' in params else 145,
                                             params['joint_prop_gains_5'] if 'joint_prop_gains_5' in params else 232,
                                             params['joint_prop_gains_6'] if 'joint_prop_gains_6' in params else 204,
                                             params['joint_prop_gains_7'] if 'joint_prop_gains_7' in params else 176],
                        'joint_deriv_gains': [params['joint_deriv_gains_1'] if 'joint_deriv_gains_1' in params else 81,
                                              params['joint_deriv_gains_2'] if 'joint_deriv_gains_2' in params else 80,
                                              params['joint_deriv_gains_3'] if 'joint_deriv_gains_3' in params else 40,
                                              params['joint_deriv_gains_4'] if 'joint_deriv_gains_4' in params else 43,
                                              params['joint_deriv_gains_5'] if 'joint_deriv_gains_5' in params else 75,
                                              params['joint_deriv_gains_6'] if 'joint_deriv_gains_6' in params else 82,
                                              params['joint_deriv_gains_7'] if 'joint_deriv_gains_7' in params else 47]
                        }


            self.cfg_ctrl['joint_prop_gains'] = torch.tensor(cfg_ctrl['joint_prop_gains'], device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['joint_deriv_gains'] = torch.tensor(cfg_ctrl['joint_deriv_gains'], device=self.device).repeat((self.num_envs, 1))

            prop_gains = torch.cat((torch.tensor(cfg_ctrl['joint_prop_gains']).repeat((self.num_envs, 1)).to(self.device),
                                    self.cfg_ctrl['gripper_prop_gains']), dim=-1).to('cpu')
            deriv_gains = torch.cat((torch.tensor(cfg_ctrl['joint_deriv_gains']).repeat((self.num_envs, 1)).to(self.device),
                                     self.cfg_ctrl['gripper_deriv_gains']), dim=-1).to('cpu')

            print(f"Current params:", params)

            for env_ptr, kuka_handle, prop_gain, deriv_gain in zip(self.envs,
                                                                   self.kuka_handles,
                                                                   prop_gains,
                                                                   deriv_gains):
                kuka_dof_props = self.gym.get_actor_dof_properties(env_ptr, kuka_handle)
                kuka_dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
                kuka_dof_props['stiffness'] = prop_gain
                kuka_dof_props['damping'] = deriv_gain
                self.gym.set_actor_dof_properties(env_ptr, kuka_handle, kuka_dof_props)

            self._simulate_and_refresh()

            # self.cfg_task.rl.pos_action_scale = [params['pos_scale_x'], params['pos_scale_y'], params['pos_scale_z']]
            # self.cfg_task.rl.rot_action_scale = [params['rot_scale_r'], params['rot_scale_p'], params['rot_scale_y']]

            idx_list = np.arange(0, min(self.num_paths, self.num_envs))
            idx_list = np.random.randint(0, self.num_paths, size=(min(self.num_paths, self.num_envs), 1))

            kuka_joints_to_mimic = torch.zeros((len(env_ids), 1999, 15))
            eef_pose_to_mimic = torch.zeros((len(env_ids), 1999, 7))
            action_to_apply = torch.zeros((len(env_ids), 1999, 6))

            # Each env will try to mimic a random recorded trajectory
            for i in range(len(env_ids)):
                kuka_joints_to_mimic[i][:, :7] = self.arm_joints_real[idx_list[i]]  # idx_list[i]
                eef_pose_to_mimic[i] = self.eef_pose_real[idx_list[i]]
                action_to_apply[i] = self.actions_real[idx_list[i]]

            # first starting joint state
            self._reset_kuka(env_ids, new_pose=kuka_joints_to_mimic[env_ids.cpu().numpy(), 0, :])
            # self._simulate_and_refresh()

            sim_joints = torch.zeros((len(env_ids), 1999, 7))
            sim_pose = torch.zeros((len(env_ids), 1999, 7))

            up_to = 1000
            for j in range(up_to):

                self._apply_actions_as_ctrl_targets(actions=torch.tensor(action_to_apply[:, j, :]).to(self.device),
                                                    ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
                                                    do_scale=True)
                self._simulate_and_refresh()

                sim_joints[:, j, :] = self.arm_dof_pos.clone()[:, :7]
                sim_pose[:, j, :] = torch.cat((self.fingertip_centered_pos.clone(),
                                               self.fingertip_centered_quat.clone()), dim=-1)

            # Between which joints to compare
            joints = np.sort(np.unique([int(key[-1]) - 1 for key in params]))

            loss1 = torch.sum((sim_pose[:, :up_to, :3] - eef_pose_to_mimic[:, :up_to, :3]) ** 2).cpu().numpy().item()
            loss2 = torch.sum((sim_joints[:, :up_to, joints] - kuka_joints_to_mimic[:, :up_to, joints]) ** 2).cpu().numpy().item()

            # plots
            display = False
            if display:

                sim_joints = sim_joints[0, :up_to, joints].cpu().numpy()
                kuka_joints = kuka_joints_to_mimic[0, :up_to, joints].cpu().numpy()

                import matplotlib.pyplot as plt
                ax = [plt.subplot(len(joints), 1, i + 1) for i in range(len(joints))]

                for j in range(len(ax)):
                    ax[j].plot(kuka_joints[:, j], color='r', marker='o', markersize=1, label='real')
                    ax[j].plot(sim_joints[:, j], color='b', marker='o',  markersize=1, label='sim')

                plt.legend()
                plt.title(f"Total Error: {loss2} \n")
                plt.show()

                sim_pose = sim_pose[0, :up_to, :].cpu().numpy()
                eef_pose_to_mimic = eef_pose_to_mimic[0, :up_to, :].cpu().numpy()
                ax = plt.figure(figsize=(12, 12)).add_subplot(projection='3d')
                ax.plot(sim_pose[:, 0], sim_pose[:, 1], zs=sim_pose[:, 2], marker='o', color='b',  markersize=1,
                        label='sim')
                ax.plot(eef_pose_to_mimic[:, 0], eef_pose_to_mimic[:, 1], zs=eef_pose_to_mimic[:, 2], marker='o',
                        color='r', markersize=1,   label='real')

                ax.set_xlabel('$X$', fontsize=20, rotation=150)
                ax.set_ylabel('$Y$', fontsize=20, rotation=150)
                ax.set_zlabel('$Z$', fontsize=30, rotation=60)
                plt.legend()
                plt.title(f"Total Error: {loss1} \n")

                plt.show()

                ax = [plt.subplot(1, 3, i + 1) for i in range(3)]
                ax[0].plot(sim_pose[:, 0], sim_pose[:, 1] , marker='o', color='b',  markersize=1,label='sim')
                ax[0].plot(eef_pose_to_mimic[:, 0], eef_pose_to_mimic[:, 1], marker='o', color='r', markersize=1,label='real')
                ax[0].set_xlabel('$X$', fontsize=10, rotation=150)
                ax[0].set_ylabel('$Y$', fontsize=10, rotation=150)
                ax[1].plot(sim_pose[:, 0], sim_pose[:, 2] , marker='o', color='b',  markersize=1,label='sim')
                ax[1].plot(eef_pose_to_mimic[:, 0], eef_pose_to_mimic[:, 2], marker='o', color='r', markersize=1,label='real')
                ax[1].set_xlabel('$X$', fontsize=10, rotation=150)
                ax[1].set_ylabel('$Z$', fontsize=10, rotation=150)
                ax[2].plot(sim_pose[:, 1], sim_pose[:, 2] , marker='o', color='b',  markersize=1,label='sim')
                ax[2].plot(eef_pose_to_mimic[:, 1], eef_pose_to_mimic[:, 2], marker='o', color='r', markersize=1,label='real')
                ax[2].set_xlabel('$Y$', fontsize=10, rotation=150)
                ax[2].set_ylabel('$Z$', fontsize=10, rotation=150)
                plt.legend()
                plt.show()

            print(f"Total Error: {loss2} \n")

            save = True
            if save:
                import os
                from datetime import datetime
                data_path = '/home/osher/Downloads/osher'

                dict_to_save = {'sim_joints': sim_joints.cpu().numpy().squeeze(),
                                'sim_pose': sim_pose.cpu().numpy().squeeze(),
                                'sim_actions': action_to_apply.cpu().numpy().squeeze()
                                }

                np.savez_compressed(os.path.join(data_path, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.npz'),
                                    dict_to_save)

            return loss2

        # space = {
        #     "pos_scale_x": hp.uniform("pos_scale_x", 0.0, 0.01),
        #     "pos_scale_y": hp.uniform("pos_scale_y", 0.0, 0.01),
        #     "pos_scale_z": hp.uniform("pos_scale_z", 0.0, 0.01),
        #     "rot_scale_r": hp.uniform("rot_scale_r", 0.0, 0.001),
        #     "rot_scale_p": hp.uniform("rot_scale_p", 0.0, 0.001),
        #     "rot_scale_y": hp.uniform("rot_scale_y", 0.0, 0.01),
        # }

        space = {
            "joint_prop_gains_1": hp.randint("joint_prop_gains_1", 200, 400),
            "joint_prop_gains_2": hp.randint("joint_prop_gains_2", 200, 400),
            "joint_prop_gains_3": hp.randint("joint_prop_gains_3", 50, 250),
            "joint_prop_gains_4": hp.randint("joint_prop_gains_4", 50, 250),
            "joint_prop_gains_5": hp.randint("joint_prop_gains_5", 50, 250),
            "joint_prop_gains_6": hp.randint("joint_prop_gains_6", 50, 250),
            "joint_prop_gains_7": hp.randint("joint_prop_gains_7", 50, 250),
            "joint_deriv_gains_1": hp.randint("joint_deriv_gains_1", 5, 100),
            "joint_deriv_gains_2": hp.randint("joint_deriv_gains_2", 5, 100),
            "joint_deriv_gains_3": hp.randint("joint_deriv_gains_3", 5, 100),
            "joint_deriv_gains_4": hp.randint("joint_deriv_gains_4", 5, 100),
            "joint_deriv_gains_5": hp.randint("joint_deriv_gains_5", 5, 100),
            "joint_deriv_gains_6": hp.randint("joint_deriv_gains_6", 5, 100),
            "joint_deriv_gains_7": hp.randint("joint_deriv_gains_7", 5, 100),
        }

        # TPE algo based on bayesian optimization
        algo = tpe.suggest
        # spark_trials = SparkTrials()
        best_result = fmin(
            fn=objective,
            space=space,
            algo=algo,
            max_evals=1000)

        print(f"Best params: \n")
        print(space_eval(space, best_result))

        import yaml
        with open('./best_params_joint_1000.yaml', 'w') as outfile:
            yaml.dump(space_eval(space, best_result), outfile, default_flow_style=False)

        print("Finished")

        exit()

    def _reset_kuka(self, env_ids, new_pose=None):
        """Reset DOF states and DOF targets of kuka."""

        # shape of dof_pos = (num_envs, num_dofs)
        # shape of dof_vel = (num_envs, num_dofs)

        self.dof_pos[env_ids, :] = new_pose.to(device=self.device)  # .repeat((len(env_ids), 1))

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

        # Simulate one step to apply changes
        # self._simulate_and_refresh()

    def _reset_object(self, env_ids):
        """Reset root state of plug."""

        # Randomize root state of plug
        plug_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        plug_noise_xy = plug_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.plug_pos_xy_noise, device=self.device))

        self.root_pos[env_ids, self.plug_actor_id_env, 0] = self.cfg_task.randomize.plug_pos_xy_initial[0] \
                                                            + plug_noise_xy[env_ids, 0]
        self.root_pos[env_ids, self.plug_actor_id_env, 1] = self.cfg_task.randomize.plug_pos_xy_initial[1] \
                                                            + plug_noise_xy[env_ids, 1]
        self.root_pos[env_ids, self.plug_actor_id_env, 2] = self.cfg_base.env.table_height

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

        # Randomize root state of socket
        socket_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)
        socket_noise_xy = socket_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.socket_pos_xy_noise, device=self.device))

        socket_noise_z_mag = (self.cfg_task.randomize.socket_pos_z_noise_bounds[1]
                              - self.cfg_task.randomize.socket_pos_z_noise_bounds[0])
        socket_noise_z = (socket_noise_z_mag * torch.rand((len(env_ids)), dtype=torch.float32, device=self.device)
                          + self.cfg_task.randomize.socket_pos_z_noise_bounds[0]
                          )

        self.root_pos[env_ids, self.socket_actor_id_env, 0] = self.cfg_task.randomize.socket_pos_xy_initial[0] \
                                                              + socket_noise_xy[env_ids, 0]
        self.root_pos[env_ids, self.socket_actor_id_env, 1] = self.cfg_task.randomize.socket_pos_xy_initial[1] \
                                                              + socket_noise_xy[env_ids, 1]

        self.root_pos[env_ids, self.socket_actor_id_env, 2] = self.cfg_base.env.table_height + socket_noise_z

        self.root_quat[env_ids, self.socket_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32,
                                                                         device=self.device).repeat(len(env_ids), 1)

        # Stabilize socket
        self.root_linvel[env_ids, self.socket_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.socket_actor_id_env] = 0.0

        socket_actor_ids_sim_int32 = self.socket_actor_ids_sim.to(dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(socket_actor_ids_sim_int32[env_ids]),
                                                     len(socket_actor_ids_sim_int32[env_ids]))

        self._simulate_and_refresh()

    def _move_arm_to_desired_pose(self, env_ids, desired_pos, desired_rot=None, sim_steps=30):
        """Move gripper to desired pose."""

        # Set target pos above object
        # print('here 5 1 1')
        self.ctrl_target_fingertip_centered_pos[env_ids] = desired_pos[env_ids].clone()

        # Set target rot
        # print('here 5 1 2')

        if desired_rot is None:
            ctrl_target_fingertip_centered_euler = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                                                                device=self.device).unsqueeze(0).repeat(len(env_ids),
                                                                                                        1)

            self.ctrl_target_fingertip_centered_quat[env_ids] = torch_jit_utils.quat_from_euler_xyz(
                ctrl_target_fingertip_centered_euler[:, 0],
                ctrl_target_fingertip_centered_euler[:, 1],
                ctrl_target_fingertip_centered_euler[:, 2])
        else:
            self.ctrl_target_fingertip_centered_quat[env_ids] = desired_rot[env_ids]

        # Step sim and render
        for sim_id in range(sim_steps):
            # print(sim_id, sim_steps)
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

            # print('here 5 1 3 2')

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[:, :6] = delta_hand_pose
            # actions[:, :6] = 

            # Apply the action, keep fingers in the same status
            self.ctrl_target_dof_pos[:, 7:] = self.ctrl_target_gripper_dof_pos
            self._apply_actions_as_ctrl_targets(actions=actions,
                                                ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
                                                do_scale=False)
            # print('here 5 1 3 3')

        # Stabilize
        # self.dof_vel[env_ids, :] = 0. # torch.zeros_like(self.dof_vel[env_ids, :])
        # self.dof_torque[env_ids, :] = 0. # torch.zeros_like(self.dof_torque[env_ids, :])
        # print('here 5 1 4')
        # Set DOF state
        multi_env_ids_int32 = self.kuka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))
        # print('here 5 1 5')
        self._simulate_and_refresh()

    def _simulate_and_refresh(self):
        """Simulate one step, refresh tensors, and render results."""

        # print('here 5 1 3 3 1')
        self.gym.simulate(self.sim)
        # print('here 5 1 3 3 2')
        # self.refresh_base_tensors()
        # self.refresh_env_tensors()
        self.render()
        # print('here 5 1 3 3 3')
        self._refresh_task_tensors()
        # print('here 5 1 3 3 4')

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

        # TODO should be changed to delta as well ?
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

        gripper_dof_pos = self.gripper_dof_pos.clone()
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'base_to_finger_1_1') - 7] = self.cfg_task.env.openhand.base_angle
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'base_to_finger_2_1') - 7] = -self.cfg_task.env.openhand.base_angle

        gripper_proximal_close_noise = np.random.uniform(0.0, 0.01, 3)
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'finger_1_1_to_finger_1_2') - 7] = self.cfg_task.env.openhand.proximal_close + \
                                                               gripper_proximal_close_noise[0]
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'finger_2_1_to_finger_2_2') - 7] = self.cfg_task.env.openhand.proximal_close + \
                                                               gripper_proximal_close_noise[1]
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'base_to_finger_3_2') - 7] = self.cfg_task.env.openhand.proximal_close + \
                                                         gripper_proximal_close_noise[2]

        gripper_distal_close_noise = np.random.uniform(0., 0.005, 3)
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'finger_1_2_to_finger_1_3') - 7] = self.cfg_task.env.openhand.distal_close + \
                                                               gripper_distal_close_noise[0]
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'finger_2_2_to_finger_2_3') - 7] = self.cfg_task.env.openhand.distal_close + \
                                                               gripper_distal_close_noise[1]
        gripper_dof_pos[env_ids,
                        list(self.dof_dict.values()).index(
                            'finger_3_2_to_finger_3_3') - 7] = self.cfg_task.env.openhand.distal_close + \
                                                               gripper_distal_close_noise[2]

        self.ctrl_target_gripper_dof_pos = gripper_dof_pos

        self._move_gripper_to_dof_pos(env_ids=env_ids, gripper_dof_pos=gripper_dof_pos, sim_steps=sim_steps)

    def _move_gripper_to_dof_pos(self, env_ids, gripper_dof_pos, sim_steps=20):
        """Move gripper fingers to specified DOF position using controller."""

        # Step sim
        for _ in range(sim_steps):
            delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
                                          device=self.device)  # no arm motion
            self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)
            self._simulate_and_refresh()

    def _check_grasped(self):
        normalize_forces = lambda x: torch.norm(x, dim=-1) > 0.1

        l1 = normalize_forces(self.left_finger_force.clone()),
        l2 = normalize_forces(self.right_finger_force.clone()),
        l3 = normalize_forces(self.middle_finger_force.clone()),

        grasped = torch.norm(self.fingertip_midpoint_pos - self.plug_pos, p=2,
                             dim=-1) <= self.cfg_task.env.plug_grasp_threshold

        return l1 and l2 and l3

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

        # print(self.keypoints_plug[0], self.keypoints_socket[0])

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

        # print('plug height', self.plug_pos[:, 2], self.cfg_base.env.table_height + self.plug_heights.squeeze(-1) * height_multiple)
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
        self.obs_dict['contacts'] = self.gt_extrinsic_contact.to(self.rl_device)
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        super().reset()
        self.obs_dict['priv_info'] = self.obs_dict['states']
        self.obs_dict['tactile_hist'] = self.tactile_queue.to(self.rl_device)
        self.obs_dict['ft_hist'] = self.ft_queue.to(self.rl_device)
        self.obs_dict['contacts'] = self.gt_extrinsic_contact.to(self.rl_device)
        return self.obs_dict
