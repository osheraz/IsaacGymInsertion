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
import os
import torch

from isaacgym import gymapi, gymtorch
from isaacgyminsertion.tasks.factory_tactile.factory_env_insertion import FactoryEnvInsertionTactile
from isaacgyminsertion.tasks.factory_tactile.factory_schema_class_task import FactoryABCTask
from isaacgyminsertion.tasks.factory_tactile.factory_schema_config_task import FactorySchemaConfigTask
import isaacgyminsertion.tasks.factory_tactile.factory_control as fc
from isaacgyminsertion.tasks.factory_tactile.factory_utils import *
from isaacgyminsertion.utils import torch_jit_utils
from multiprocessing import Process, Queue, Manager
import cv2
# from isaacgyminsertion.allsight.experiments.allsight_render import allsight_renderer
from isaacgyminsertion.allsight.tacto_allsight_wrapper.util.util import tensor2im

torch.set_printoptions(sci_mode=False)


class FactoryTaskInsertionTactile(FactoryEnvInsertionTactile, FactoryABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize task superclass."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.cfg = cfg
        self._get_task_yaml_params()

        self._acquire_task_tensors()
        self.parse_controller_spec()

        self.temp_ctr = 0

        if self.viewer is not None:
            self._set_viewer_params()

        if self.cfg_base.mode.export_scene:
            self.export_scene(label='kuka_task_insertion')

        if self.cfg_tactile.sim2real:
            from isaacgyminsertion.allsight.experiments.models import networks, pre_process
            # That model only trained with 224 as inputs
            opt = {
                "preprocess": "resize_and_crop",
                "crop_size": 224,
                "load_size": 224,
                "no_flip": True,
            }

            self.transform = pre_process.get_transform(opt=opt)

            self.model_G = networks.define_G(input_nc=3,
                                             output_nc=3,
                                             ngf=64,
                                             netG="resnet_9blocks",
                                             norm="instance",
                                             )

            path_to_g = os.path.join(os.path.dirname(__file__), '..', '..',
                                     f"allsight/experiments/models/GAN/{self.cfg_tactile.model_G}")

            self.model_G.load_state_dict(torch.load(path_to_g))
            self.model_G.to(self.device)
            self.model_G.eval()

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

        ppo_path = 'train/FactoryTaskInsertionTactilePPOv2.yaml'  # relative to Gym's Hydra search path (cfg dir)
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

        self.plug_tip_pos_local = self.plug_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1))
        self.socket_tip_pos_local = self.socket_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1))

        # Gripper pointing down w.r.t the world frame
        gripper_goal_euler = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                                          device=self.device).unsqueeze(0).repeat((self.num_envs, 1))

        self.gripper_goal_quat = torch_jit_utils.quat_from_euler_xyz(gripper_goal_euler[:, 0],
                                                                     gripper_goal_euler[:, 1],
                                                                     gripper_goal_euler[:, 2])

        # Keypoint tensors
        self.keypoint_offsets = self._get_keypoint_offsets(
            self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
        self.keypoints_plug = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3), dtype=torch.float32,
                                          device=self.device, )
        self.keypoints_socket = torch.zeros_like(self.keypoints_plug, device=self.device)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.targets = torch.zeros((self.num_envs, self.cfg_task.env.numTargets), device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.cfg_task.env.numTargets), dtype=torch.float,
                                        device=self.device)

        # Keep track of history
        self.obs_queue =  torch.zeros((self.num_envs, self.cfg_task.env.numObsHist * self.num_observations),
                                        dtype=torch.float, device=self.device)
        self.arm_joint_queue = torch.zeros((self.num_envs, self.cfg_task.env.numObsHist, 7),
                                           dtype=torch.float, device=self.device)
        self.arm_vel_queue = torch.zeros((self.num_envs, self.cfg_task.env.numObsHist, 7),
                                         dtype=torch.float, device=self.device)
        self.actions_queue = torch.zeros((self.num_envs, self.cfg_task.env.numObsHist, self.num_actions),
                                         dtype=torch.float, device=self.device)
        self.targets_queue = torch.zeros((self.num_envs, self.cfg_task.env.numObsHist, self.num_actions),
                                         dtype=torch.float, device=self.device)
        self.eef_queue = torch.zeros((self.num_envs, self.cfg_task.env.numObsHist, 12),
                                     dtype=torch.float, device=self.device)
        self.goal_noisy_queue = torch.zeros((self.num_envs, self.cfg_task.env.numObsHist, 3),
                                            dtype=torch.float, device=self.device)
        self.delta_pos_noisy_queue = torch.zeros((self.num_envs, self.cfg_task.env.numObsHist, 3),
                                            dtype=torch.float, device=self.device)

        # Bad, should queue the obs!
        
        self.arm_joint_queue_student = torch.zeros((self.num_envs, self.cfg_task.env.numObsStudentHist, 7),
                                                   dtype=torch.float, device=self.device)
        self.arm_vel_queue_student = torch.zeros((self.num_envs, self.cfg_task.env.numObsStudentHist, 7),
                                                 dtype=torch.float, device=self.device)
        self.actions_queue_student = torch.zeros((self.num_envs, self.cfg_task.env.numObsStudentHist, self.num_actions),
                                                 dtype=torch.float, device=self.device)
        self.targets_queue_student = torch.zeros((self.num_envs, self.cfg_task.env.numObsStudentHist, self.num_actions),
                                                 dtype=torch.float, device=self.device)
        self.eef_queue_student = torch.zeros((self.num_envs, self.cfg_task.env.numObsStudentHist, 12),
                                             dtype=torch.float, device=self.device)
        self.goal_noisy_queue_student = torch.zeros((self.num_envs, self.cfg_task.env.numObsStudentHist, 3),
                                                    dtype=torch.float, device=self.device)

        # tactile buffers
        self.num_channels = self.cfg_tactile.decoder.num_channels
        self.width = self.cfg_tactile.decoder.width // 2 if self.cfg_tactile.half_image else self.cfg_tactile.decoder.width
        self.height = self.cfg_tactile.decoder.height

        self.tactile_imgs = torch.zeros(
            (self.num_envs, len(self.fingertips),  # left, right, bottom
             self.width, self.height, self.num_channels),
            device=self.device,
            dtype=torch.float,
        )
        # Way too big tensor.
        self.tactile_queue = torch.zeros(
            (self.num_envs, self.tact_hist_len, len(self.fingertips),  # left, right, bottom
             self.width, self.height, self.num_channels),
            device=self.device,
            dtype=torch.float,
        )

        self.ft_queue = torch.zeros((self.num_envs, self.ft_hist_len, 6), device=self.device, dtype=torch.float)

        # reset tensors
        self.timeout_reset_buf = torch.zeros_like(self.reset_buf)
        self.degrasp_buf = torch.zeros_like(self.reset_buf)
        self.far_from_goal_buf = torch.zeros_like(self.reset_buf)
        self.success_reset_buf = torch.zeros_like(self.reset_buf)

        # state tensors
        self.plug_hand_pos, self.plug_hand_quat = torch.zeros((self.num_envs, 3),
                                                              device=self.device), torch.zeros(
            (self.num_envs, 4), device=self.device)
        self.rigid_physics_params = torch.zeros((self.num_envs, 6), device=self.device,
                                                dtype=torch.float)  # TODO: Take num_params to config
        self.finger_normalized_forces = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        self.gt_extrinsic_contact = torch.zeros((self.num_envs, self.cfg_task.env.num_points),
                                                device=self.device, dtype=torch.float)

        # reward tensor
        self.reward_log_buf = torch.zeros_like(self.rew_buf)

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

        # Add observation noise to socket rot
        socket_rot_euler = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        socket_obs_rot_noise = 2 * (
                torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
                - 0.5
        )
        socket_obs_rot_noise = socket_obs_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.env.socket_rot_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        socket_obs_rot_euler = socket_rot_euler + socket_obs_rot_noise
        self.noisy_socket_quat = torch_jit_utils.quat_from_euler_xyz(
            socket_obs_rot_euler[:, 0],
            socket_obs_rot_euler[:, 1],
            socket_obs_rot_euler[:, 2],
        )

        # Compute observation noise on socket
        (
            self.noisy_gripper_goal_quat,
            self.noisy_gripper_goal_pos,
        ) = torch_jit_utils.tf_combine(
            self.noisy_socket_quat,
            self.noisy_socket_pos,
            self.gripper_goal_quat,
            self.socket_tip_pos_local,
        )

        # Compute pos of keypoints on gripper, socket, and plug in world frame
        socket_tip_pos_local = self.socket_tip_pos_local.clone()
        socket_tip_pos_local[:, 2] -= self.socket_heights.view(-1)
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_plug[:, idx] = torch_jit_utils.tf_combine(self.plug_quat,
                                                                     self.plug_pos,
                                                                     self.identity_quat,
                                                                     (keypoint_offset * self.socket_heights))[1]
            self.keypoints_socket[:, idx] = torch_jit_utils.tf_combine(self.socket_quat,
                                                                       self.socket_pos,
                                                                       self.identity_quat,
                                                                       (
                                                                                   keypoint_offset * self.socket_heights) + socket_tip_pos_local)[
                1]
            
        # fingertip forces
        e = 0.9 if self.cfg_task.env.smooth_force else 0
        normalize_forces = lambda x: (torch.clamp(torch.norm(x, dim=-1), 0, 50) / 50).view(-1)
        self.finger_normalized_forces[:, 0] = (1 - e) * normalize_forces(
            self.left_finger_force.clone()) + e * self.finger_normalized_forces[:, 0]
        self.finger_normalized_forces[:, 1] = (1 - e) * normalize_forces(
            self.right_finger_force.clone()) + e * self.finger_normalized_forces[:, 1]
        self.finger_normalized_forces[:, 2] = (1 - e) * normalize_forces(
            self.middle_finger_force.clone()) + e * self.finger_normalized_forces[:, 2]

        if update_tactile and self.cfg_task.env.tactile:
            left_allsight_poses = torch.cat((self.left_finger_pos, self.left_finger_quat), dim=-1)
            right_allsight_poses = torch.cat((self.right_finger_pos, self.right_finger_quat), dim=-1)
            middle_allsight_poses = torch.cat((self.middle_finger_pos, self.middle_finger_quat), dim=-1)
            object_poses = torch.cat((self.plug_pos, self.plug_quat), dim=-1)

            tf = np.eye(4)
            tf[0:3, 0:3] = euler_angles_to_matrix(
                euler_angles=torch.tensor([[3.14159265359, 0, 0]]), convention="XYZ"
            ).numpy()

            left_finger_poses = xyzquat_to_tf_numpy(left_allsight_poses.cpu().numpy())

            left_finger_poses = left_finger_poses @ tf

            right_finger_poses = xyzquat_to_tf_numpy(right_allsight_poses.cpu().numpy())

            right_finger_poses = right_finger_poses @ tf

            middle_finger_poses = xyzquat_to_tf_numpy(middle_allsight_poses.cpu().numpy())

            middle_finger_poses = middle_finger_poses @ tf

            object_pose = xyzquat_to_tf_numpy(object_poses.cpu().numpy())

            self._update_tactile(left_finger_poses, right_finger_poses, middle_finger_poses, object_pose)

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
            #
            #
            #     for queue in results:
            #         imgs, offest = queue.get()
            #         print(imgs)
            #         self.tactile_imgs = torch.tensor(imgs, dtype=torch.float32, device=self.device)

    def _update_tactile(self, left_finger_pose, right_finger_pose, middle_finger_pose, object_pose,
                        offset=None, queue=None):

        tactile_imgs_list, height_maps = [], []  # only for display.
        finger_normalized_forces = self.finger_normalized_forces.clone()

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
                if self.cfg_task.env.tactile_wrt_force:
                    force = 50 * finger_normalized_forces[e, n].cpu().detach().numpy()
                else:
                    force = 20

                tactile_img, height_map = self.tactile_handles[e][n].render(object_pose[e], force)

                if self.cfg_tactile.sim2real:
                    # Reducing FPS by half
                    color_tensor = self.transform(tactile_img).unsqueeze(0).to(self.device)
                    tactile_img = self.model_G(color_tensor)
                    tactile_img = tensor2im(tactile_img)

                # Pulled subtract here cuz of the GAN
                if self.cfg_tactile.diff:
                    tactile_img = self.tactile_handles[e][n].remove_bg(tactile_img, self.tactile_handles[e][n].bg_img)
                    tactile_img *= self.tactile_handles[e][n].mask

                # Cutting by half
                if self.cfg_tactile.half_image:
                    w = tactile_img.shape[0]
                    tactile_img = tactile_img[:w // 2]
                    height_map = height_map[:w // 2]

                # Resizing to decoder size
                resized_img = cv2.resize(tactile_img, (self.height, self.width), interpolation=cv2.INTER_AREA)

                if self.num_channels == 3:
                    self.tactile_imgs[e, n] = torch_jit_utils.rgb_transform(resized_img).to(self.device).permute(1, 2, 0)
                else:
                    resized_img = cv2.cvtColor(resized_img.astype('float32'), cv2.COLOR_BGR2GRAY)
                    self.tactile_imgs[e, n] = torch_jit_utils.gray_transform(resized_img).to(self.device).permute(1, 2, 0)

                tactile_imgs_per_env.append(tactile_img)
                height_maps_per_env.append(height_map)

            height_maps.append(height_maps_per_env)
            tactile_imgs_list.append(tactile_imgs_per_env)

        if self.cfg_task.env.tactile_display_viz and self.cfg_task.env.tactile:
            env_to_show = 0
            self.tactile_handles[env_to_show][0].updateGUI(tactile_imgs_list[env_to_show],
                                                           height_maps[env_to_show])

        if queue is not None:
            queue.put((tactile_imgs_list, offset))

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains."""

        self.prev_actions[:] = self.actions.clone()
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

        # test actions for whenever we want to see some axis motion
        # self.actions[:, :] = 0.
        # self.actions[:, 0] = 1.
        # self.actions[:, 1] = 1.
        # self.actions[:, 2] = 1.

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

        self.goal_noisy_queue[:, 1:] = self.goal_noisy_queue[:, :-1].clone().detach()
        self.goal_noisy_queue[:, 0, :] = self.noisy_gripper_goal_pos.clone()
        # self.goal_noisy_queue[:, 0, :] = torch.cat(self.pose_world_to_robot_base(self.noisy_gripper_goal_pos.clone(),
        #                                                                          self.noisy_gripper_goal_quat.clone()),
        #                                            dim=-1)
        # Bad ,  should queue the obs
        self.actions_queue_student[:, 1:] = self.actions_queue_student[:, :-1].clone().detach()
        self.actions_queue_student[:, 0, :] = self.actions

        self.targets_queue_student[:, 1:] = self.targets_queue_student[:, :-1].clone().detach()
        self.targets_queue_student[:, 0, :] = self.targets

        self.goal_noisy_queue_student[:, 1:] = self.goal_noisy_queue_student[:, :-1].clone().detach()
        self.goal_noisy_queue_student[:, 0, :] = self.noisy_gripper_goal_pos.clone()
        # self.goal_noisy_queue_student[:, 0, :] = torch.cat(
        #     self.pose_world_to_robot_base(self.noisy_gripper_goal_pos.clone(),
        #                                   self.noisy_gripper_goal_quat.clone()), dim=-1)

        self._apply_actions_as_ctrl_targets(actions=self.actions,
                                            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
                                            do_scale=True)

        self.prev_targets[:] = self.targets.clone()

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

        self.progress_buf[:] += 1
        self.randomize_buf[:] += 1
        # print('progress_buf', self.progress_buf[0])

        # In this policy, episode length is constant
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

        update_tactile = False
        if (self.temp_ctr % self.cfg_task.tactile.tactile_freq) == 0:
            update_tactile = True
        self.temp_ctr += 1

        self._refresh_task_tensors(update_tactile=update_tactile)
        self.compute_observations()
        self.compute_reward()

        if self.viewer or True:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            rotate_vec = lambda q, x: quat_apply(q, to_torch(x, device=self.device) * 0.2).cpu().numpy()
            num_envs = 1
            for i in range(num_envs):
                
                actions = self.actions[i, :].clone().cpu().numpy()
                keypoints = self.keypoints_plug[i].clone().cpu().numpy()
                quat = self.plug_quat[i, :]

                for j in range(self.cfg_task.rl.num_keypoints):
                    ob = keypoints[j]
                    targetx = ob + rotate_vec(quat, [actions[0], 0, 0])
                    targety = ob + rotate_vec(quat, [0, actions[1], 0])
                    targetz = ob + rotate_vec(quat, [0, 0, actions[2]])

                    self.gym.add_lines(self.viewer, self.envs[i], 1,
                                       [ob[0], ob[1], ob[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1,
                                       [ob[0], ob[1], ob[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1,
                                       [ob[0], ob[1], ob[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])
                # print(keypoints)

            #     for j in range(self.cfg_task.rl.num_keypoints):
            #         ob = keypoints[j]
            #         targetx = ob + rotate_vec(quat, [1, 0, 0])
            #         targety = ob + rotate_vec(quat, [0, 1, 0])
            #         targetz = ob + rotate_vec(quat, [0, 0, 1])

            #         self.gym.add_lines(self.viewer, self.envs[i], 1,
            #                            [ob[0], ob[1], ob[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
            #         self.gym.add_lines(self.viewer, self.envs[i], 1,
            #                            [ob[0], ob[1], ob[2], targety[0], targety[1], targety[2]], [0.85, 0.1, 0.1])
            #         self.gym.add_lines(self.viewer, self.envs[i], 1,
            #                            [ob[0], ob[1], ob[2], targetz[0], targetz[1], targetz[2]], [0.85, 0.1, 0.1])

            # for i in range(num_envs):
            #     keypoints = self.keypoints_socket[i].clone().cpu().numpy()
            #     quat = self.socket_quat[i, :]
            #     # print(keypoints)

            #     for j in range(self.cfg_task.rl.num_keypoints):
            #         ob = keypoints[j]
            #         targetx = ob + rotate_vec(quat, [1, 0, 0])
            #         targety = ob + rotate_vec(quat, [0, 1, 0])
            #         targetz = ob + rotate_vec(quat, [0, 0, 1])

            #         self.gym.add_lines(self.viewer, self.envs[i], 1,
            #                            [ob[0], ob[1], ob[2], targetx[0], targetx[1], targetx[2]], [0.1, 0.85, 0.1])
            #         self.gym.add_lines(self.viewer, self.envs[i], 1,
            #                            [ob[0], ob[1], ob[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
            #         self.gym.add_lines(self.viewer, self.envs[i], 1,
            #                            [ob[0], ob[1], ob[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.85, 0.1])

        self._render_headless()

    def compute_observations(self):
        """Compute observations."""
        # update the queue
        # self.arm_vel_queue[:, 1:] = self.arm_vel_queue[:, :-1].clone().detach()
        # self.arm_vel_queue[:, 0, :] = self.arm_dof_vel.clone()

        # self.delta_noisy_queue[:, 1:] = self.delta_noisy_queue[:, :-1].clone().detach()
        # self.delta_noisy_queue[:, 0, :] = self.noisy_gripper_goal_pos - self.fingertip_centered_pos

        self.arm_joint_queue[:, 1:] = self.arm_joint_queue[:, :-1].clone().detach()
        self.arm_joint_queue[:, 0, :] = self.arm_dof_pos.clone()

        self.eef_queue[:, 1:] = self.eef_queue[:, :-1].clone().detach()
        self.eef_queue[:, 0, :] = torch.cat(self.pose_world_to_robot_base(self.fingertip_centered_pos.clone(),
                                                                          self.fingertip_centered_quat.clone()), dim=-1)

        self.arm_joint_queue_student[:, 1:] = self.arm_joint_queue_student[:, :-1].clone().detach()
        self.arm_joint_queue_student[:, 0, :] = self.arm_dof_pos.clone()

        self.eef_queue_student[:, 1:] = self.eef_queue_student[:, :-1].clone().detach()
        self.eef_queue_student[:, 0, :] = torch.cat(self.pose_world_to_robot_base(self.fingertip_centered_pos.clone(),
                                                                                  self.fingertip_centered_quat.clone()),
                                                    dim=-1)

        # Compute tactile
        self.tactile_queue[:, 1:] = self.tactile_queue[:, :-1].clone().detach()
        self.tactile_queue[:, 0, :] = self.tactile_imgs

        self.ft_queue[:, 1:] = self.ft_queue[:, :-1].clone().detach()
        self.ft_queue[:, 0, :] = 0.1 * self.ft_sensor_tensor.clone()

        # Compute Observation and state at current timestep
        # delta_pos = self.socket_tip - self.fingertip_centered_pos
        noisy_delta_pos = self.noisy_gripper_goal_pos - self.fingertip_centered_pos
        self.delta_pos_noisy_queue[:, 1:] = self.delta_pos_noisy_queue[:, :-1].clone().detach()
        self.delta_pos_noisy_queue[:, 0, :] = noisy_delta_pos

        eef_pos = torch.cat(self.pose_world_to_robot_base(self.fingertip_centered_pos.clone(),
                                                                          self.fingertip_centered_quat.clone()), dim=-1)
        actions = self.actions.clone()
        targets = self.targets.clone()
        
        obs = torch.cat([eef_pos, actions, targets], dim=-1)
        self.obs_queue[:, :-self.num_observations] = self.obs_queue[:, self.num_observations:]
        self.obs_queue[:, -self.num_observations:] = obs

        # Define observations (for actor)
        obs_tensors = [
            # self.arm_joint_queue.reshape(self.num_envs, -1),  # 7 * hist
            self.eef_queue.reshape(self.num_envs, -1),  # (envs, 12 * hist)
            # self.goal_noisy_queue.reshape(self.num_envs, -1),  # (envs, 12 * hist)
            # self.delta_pos_noisy_queue.reshape(self.num_envs, -1),
            self.actions_queue.reshape(self.num_envs, -1),  # (envs, 6 * hist)
            # self.delta_noisy_queue.reshape(self.num_envs, -1),  # (envs, 3 * hist)
            self.targets_queue.reshape(self.num_envs, -1),  # (envs, 6 * hist)
        ]

        obs_tensors_student = [
            # self.arm_joint_queue_student.reshape(self.num_envs, -1),  # 7 * hist
            self.eef_queue_student.reshape(self.num_envs, -1),  # (envs, 12 * hist)
            # self.goal_noisy_queue_student.reshape(self.num_envs, -1),  # (envs, 3 * hist)
            self.actions_queue_student.reshape(self.num_envs, -1),  # (envs, 6 * hist)
            self.targets_queue_student.reshape(self.num_envs, -1),  # (envs, 6 * hist)
        ]

        # Define state (for teacher)
        # eef_pose_wrt_robot = self.pose_world_to_robot_base(self.fingertip_centered_pos.clone(),
        #                                                    self.fingertip_centered_quat.clone(), as_matrix=False)
        # socket_pos_wrt_robot = self.pose_world_to_robot_base(self.socket_pos.clone(), self.socket_quat.clone(),
        #                                                      as_matrix=False)
        # plug_bottom_wrt_robot = self.pose_world_to_robot_base(self.plug_pos.clone(), self.plug_quat.clone(), as_matrix=False)
        # plug_hand_pos, plug_hand_quat = fc.get_pose_error(
        #     fingertip_midpoint_pos=plug_bottom_wrt_robot[0],
        #     fingertip_midpoint_quat=plug_bottom_wrt_robot[1],
        #     ctrl_target_fingertip_midpoint_pos=eef_pose_wrt_robot[0],
        #     ctrl_target_fingertip_midpoint_quat=eef_pose_wrt_robot[1],
        #     jacobian_type=self.cfg_ctrl['jacobian_type'],
        #     rot_error_type='quat')

        # Actually this is the right representation
        plug_hand_pos, plug_hand_quat = self.pose_world_to_hand_base(self.plug_pos, self.plug_quat, as_matrix=False)
        self.plug_hand_pos[...] = plug_hand_pos
        self.plug_hand_quat[...] = plug_hand_quat

        # plug mass
        plug_mass = [self.gym.get_actor_rigid_body_properties(e, p)[0].mass for e, p in
                     zip(self.envs, self.plug_handles)]
        # plug friction
        plug_friction = [self.gym.get_actor_rigid_shape_properties(e, p)[0].friction for e, p in
                         zip(self.envs, self.plug_handles)]
        # socket friction
        socket_friction = [self.gym.get_actor_rigid_shape_properties(e, p)[0].friction for e, p in
                           zip(self.envs, self.socket_handles)]

        # fingertip frictions
        left_finger_friction = []
        right_finger_friction = []
        middle_finger_friction = []
        for e, p in zip(self.envs, self.kuka_handles):
            props = self.gym.get_actor_rigid_shape_properties(e, p)
            left_finger_friction.append(props[self.left_finger_id].friction)
            right_finger_friction.append(props[self.right_finger_id].friction)
            middle_finger_friction.append(props[self.middle_finger_id].friction)

        physics_params = torch.transpose(to_torch([plug_mass,
                                                   plug_friction,
                                                   socket_friction,
                                                   left_finger_friction,
                                                   right_finger_friction,
                                                   middle_finger_friction]), 0, 1)
        self.rigid_physics_params[...] = physics_params

        if self.cfg_task.env.compute_contact_gt:
            display_key = 'triangle' # ['triangle', 'red_round_peg_1_5in', 'yellow_round_peg_2in']
            for k, v in self.subassembly_extrinsic_contact.items():
                self.gt_extrinsic_contact[self.subassembly_to_env_ids[k], ...] = v.get_extrinsic_contact(
                    obj_pos=self.plug_pos[self.subassembly_to_env_ids[k], ...].clone(), 
                    obj_quat=self.plug_quat[self.subassembly_to_env_ids[k], ...].clone(), 
                    socket_pos=self.socket_pos[self.subassembly_to_env_ids[k], ...].clone(),
                    socket_quat=self.socket_quat[self.subassembly_to_env_ids[k], ...].clone(),
                    display = display_key == k
                ).to(self.device)
            
            # self.gt_extrinsic_contact = self.extrinsic_contact_gt.get_extrinsic_contact(
            #     obj_pos=self.plug_pos, obj_quat=self.plug_quat, socket_pos=self.socket_pos,
            #     socket_quat=self.socket_quat
            # )


        # fingertip forces
        # e = 0.9 if self.cfg_task.env.smooth_force else 0
        # normalize_forces = lambda x: (torch.clamp(torch.norm(x, dim=-1), 0, 50) / 50).view(-1)
        # self.finger_normalized_forces[:, 0] = (1 - e) * normalize_forces(
        #     self.left_finger_force.clone()) + e * self.finger_normalized_forces[:, 0]
        # self.finger_normalized_forces[:, 1] = (1 - e) * normalize_forces(
        #     self.right_finger_force.clone()) + e * self.finger_normalized_forces[:, 1]
        # self.finger_normalized_forces[:, 2] = (1 - e) * normalize_forces(
        #     self.middle_finger_force.clone()) + e * self.finger_normalized_forces[:, 2]
        # print(self.plug_pcd.device, plug_hand_pos.device)
        state_tensors = [
            #  add delta error
            # socket_pos_wrt_robot[0],  # 3
            # socket_pos_wrt_robot[1],  # 4
            # plug_bottom_wrt_robot[0],  # 3
            # plug_bottom_wrt_robot[1],  # 4
            # self.socket_pos.clone(), # 3
            # self.socket_quat.clone(), # 4

            plug_hand_pos,   # 3
            plug_hand_quat,  # 4
            physics_params,  # 6
            self.finger_normalized_forces,  # 3
            # self.plug_pcd.view(self.num_envs, -1),  # 3 * num_points =  3 * 10 = 30

            # self.assembly_one_hot, # 4
            # self.socket_contact_force.clone()  # 3
            # TODO: add object shapes -- bring diameter
            # self.plug_heights,  # 1?
            # TODO: add extrinsics contact (point cloud) -> this will encode the shape (check this)
        ]

        self.obs_buf = self.obs_queue.clone() # torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)
        self.obs_student_buf = torch.cat(obs_tensors_student, dim=-1)  # shape = (num_envs, num_observations_student)
        self.states_buf = torch.cat(state_tensors, dim=-1)  # shape = (num_envs, num_states)

        return self.obs_buf  # shape = (num_envs, num_observations)

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        self._update_reset_buf()
        self._update_rew_buf()

    def _update_rew_buf(self):
        """Compute reward at current timestep."""

        action_penalty = torch.norm(self.actions, p=2, dim=-1)
        action_reward = self.cfg_task.rl.action_penalty_scale * action_penalty

        action_delta_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
        action_delta_reward = self.cfg_task.rl.action_delta_scale * action_delta_penalty

        plug_ori_penalty = torch.norm(self.plug_quat - self.identity_quat, p=2, dim=-1)
        ori_reward = plug_ori_penalty * self.cfg_task.rl.ori_reward_scale

        keypoint_dist = self._get_keypoint_dist()
        keypoint_reward = keypoint_dist * self.cfg_task.rl.keypoint_reward_scale

        is_plug_engaged_w_socket = self._check_plug_engaged_w_socket()
        engagement = self._get_engagement_reward_scale(is_plug_engaged_w_socket, self.cfg_task.rl.success_height_thresh)
        engagement_reward = engagement * self.cfg_task.rl.engagement_reward_scale

        # print(keypoint_reward[0], engagement_reward[0], ori_reward[0])


        self.rew_buf[:] = keypoint_reward + engagement_reward + ori_reward # + action_reward + action_delta_reward

        distance_reset_buf = (self.far_from_goal_buf | self.degrasp_buf)
        early_reset_reward = distance_reset_buf * self.cfg_task.rl.early_reset_reward_scale

        # self.rew_buf[:] += (early_reset_reward * self.timeout_reset_buf)
        # self.rew_buf[:] += (self.timeout_reset_buf * self.success_reset_buf) * self.cfg_task.rl.success_bonus
        self.extras['successes'] = ((self.timeout_reset_buf | distance_reset_buf) * self.success_reset_buf) * 1.0
        self.extras['keypoint_reward'] = keypoint_reward
        self.extras['engagement_reward'] = engagement_reward
        self.extras['ori_reward'] = ori_reward

        self.reward_log_buf[:] = self.rew_buf[:]

        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step:
            if not self.cfg_task.data_logger.collect_data:
                success_dones = self.success_reset_buf.nonzero()
                failure_dones = (1.0 - self.success_reset_buf).nonzero()

                print('Success Rate:', torch.mean(self.success_reset_buf * 1.0).item(),
                      'Avg Ep Reward:', torch.mean(self.reward_log_buf).item(),
                      ' Success Reward:', self.rew_buf[success_dones].mean().item(),
                      ' Failure Reward:', self.rew_buf[failure_dones].mean().item())

    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""

        # if successfully inserted to a certain threshold
        self.success_reset_buf[:] = self._check_plug_inserted_in_socket()

        # if we are collecting data, reset at insertion
        if self.cfg_task.data_logger.collect_data or self.cfg_task.data_logger.collect_test_sim:
            self.reset_buf[:] |= self.success_reset_buf[:]

        # If max episode length has been reached
        self.timeout_reset_buf[:] = torch.where(self.progress_buf[:] >= (self.cfg_task.rl.max_episode_length - 1),
                                                torch.ones_like(self.reset_buf),
                                                self.reset_buf)

        self.reset_buf[:] = self.timeout_reset_buf[:]

        # check is object is grasped and reset if not
        roll, pitch, _ = get_euler_xyz(self.plug_quat.clone())
        roll[roll > np.pi] -= 2 * np.pi
        pitch[pitch > np.pi] -= 2 * np.pi
        self.degrasp_buf[:] = (torch.abs(roll) > 0.4) | (torch.abs(pitch) > 0.4)

        # check if object is too far from gripper
        fingertips_plug_dist = (torch.norm(self.left_finger_pos - self.plug_pos, p=2, dim=-1) > 0.12) | (
                torch.norm(self.right_finger_pos - self.plug_pos, p=2, dim=-1) > 0.12) | (
                                       torch.norm(self.middle_finger_pos - self.plug_pos, p=2, dim=-1) > 0.12)
        self.degrasp_buf[:] |= fingertips_plug_dist

        # Reset at grasping fails
        # self.reset_buf[:] |= self.degrasp_buf[:]

        # If plug is too far from socket pos
        self.dist_plug_socket = torch.norm(self.plug_pos - self.socket_pos, p=2, dim=-1)
        self.far_from_goal_buf[:] = self.dist_plug_socket > 0.2  # self.cfg_task.rl.far_error_thresh,
        # self.reset_buf[:] |= self.far_from_goal_buf[:]

    def _reset_environment(self, env_ids):

        random_init_idx = {}
        
        for subassembly in self.cfg_env.env.desired_subassemblies:
            random_init_idx[subassembly] = torch.randint(0, self.total_init_poses[subassembly], size=(len(env_ids),))
        subassemblies = [self.envs_asset[e_id] for e_id in range(self.num_envs)]

        kuka_dof_pos = torch.zeros((len(env_ids), 15))
        socket_pos = torch.zeros((len(env_ids), 3))
        socket_quat = torch.zeros((len(env_ids), 4))
        plug_pos = torch.zeros((len(env_ids), 3))
        plug_quat = torch.zeros((len(env_ids), 4))

        for i, e in enumerate(env_ids):
            subassembly = subassemblies[e]
            kuka_dof_pos[i] = self.init_dof_pos[subassembly][random_init_idx[subassembly][i]]
            socket_pos[i] = self.init_socket_pos[subassembly][random_init_idx[subassembly][i]]
            socket_quat[i] = self.init_socket_quat[subassembly][random_init_idx[subassembly][i]]
            plug_pos[i] = self.init_plug_pos[subassembly][random_init_idx[subassembly][i]]
            plug_quat[i] = self.init_plug_quat[subassembly][random_init_idx[subassembly][i]]

        self._reset_kuka(env_ids, new_pose=kuka_dof_pos)

        for _, v  in self.all_rendering_camera.items():
            self.init_plug_pos_cam[v[0], :] = plug_pos[v[0], :]

        object_pose = {
            'socket_pose': socket_pos,
            'socket_quat': socket_quat,
            'plug_pose': plug_pos,
            'plug_quat': plug_quat
        }

        self._reset_object(env_ids, new_pose=object_pose)
        self._close_gripper(torch.arange(self.num_envs))

        # self._simulate_and_refresh()

    def reset_idx(self, env_ids):
        """Reset specified environments."""
        # self.test_plot = []
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # self.rew_buf[:] = 0
        # self._reset_kuka(env_ids)
        # self._reset_object(env_ids)
        # TODO: change this to reset dof and reset root states
        self._reset_environment(env_ids)
        # self.gym.simulate(self.sim)
        # self.render()
        # self._zero_velocities(env_ids)
        self._zero_velocities(env_ids)
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors(update_tactile=True)

        if self.cfg_task.env.record_video and 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                # print('Saving video')
                self.complete_video_frames = self.video_frames[:]
            self.video_frames = []

        if self.cfg_task.env.record_ft and 0 in env_ids:
            if self.complete_ft_frames is None:
                self.complete_ft_frames = []
            else:
                self.complete_ft_frames = self.ft_frames[:]
            self.ft_frames = []

        self._reset_buffers(env_ids)

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

    def _reset_object(self, env_ids, new_pose=None):
        """Reset root state of plug."""

        # Randomize root state of plug
        # plug_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        # plug_noise_xy = plug_noise_xy @ torch.diag(
        #     torch.tensor(self.cfg_task.randomize.plug_pos_xy_noise, device=self.device))

        # self.root_pos[env_ids, self.plug_actor_id_env, 0] = self.cfg_task.randomize.plug_pos_xy_initial[0] \
        #                                                     + plug_noise_xy[env_ids, 0]
        # self.root_pos[env_ids, self.plug_actor_id_env, 1] = self.cfg_task.randomize.plug_pos_xy_initial[1] \
        #                                                     + plug_noise_xy[env_ids, 1]
        # self.root_pos[env_ids, self.plug_actor_id_env, 2] = self.cfg_base.env.table_height

        # self.root_quat[env_ids, self.plug_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32,
        #                                                                device=self.device).repeat(len(env_ids), 1)

        plug_pose = new_pose['plug_pose']
        plug_quat = new_pose['plug_quat']

        self.root_pos[env_ids, self.plug_actor_id_env, :] = plug_pose.to(device=self.device)
        self.root_quat[env_ids, self.plug_actor_id_env, :] = plug_quat.to(device=self.device)

        # Stabilize plug
        self.root_linvel[env_ids, self.plug_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.plug_actor_id_env] = 0.0

        plug_actor_ids_sim_int32 = self.plug_actor_ids_sim.to(dtype=torch.int32, device=self.device)
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.root_state),
        #                                              gymtorch.unwrap_tensor(plug_actor_ids_sim_int32[env_ids]),
        #                                              len(plug_actor_ids_sim_int32[env_ids]))

        # self._simulate_and_refresh()

        # Randomize root state of socket
        # socket_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        # socket_noise_xy = socket_noise_xy @ torch.diag(
        #     torch.tensor(self.cfg_task.randomize.socket_pos_xy_noise, device=self.device))

        # self.root_pos[env_ids, self.socket_actor_id_env, 0] = self.cfg_task.randomize.socket_pos_xy_initial[0] \
        #                                                     + socket_noise_xy[env_ids, 0]
        # self.root_pos[env_ids, self.socket_actor_id_env, 1] = self.cfg_task.randomize.socket_pos_xy_initial[1] \
        #                                                     + socket_noise_xy[env_ids, 1]
        # self.root_pos[env_ids, self.socket_actor_id_env, 2] = self.cfg_base.env.table_height

        # self.root_quat[env_ids, self.socket_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32,
        #                                                                device=self.device).repeat(len(env_ids), 1)

        socket_pose = new_pose['socket_pose']
        socket_quat = new_pose['socket_quat']

        self.root_pos[env_ids, self.socket_actor_id_env, :] = socket_pose.to(device=self.device)
        self.root_quat[env_ids, self.socket_actor_id_env, :] = socket_quat.to(device=self.device)

        # Stabilize socket
        self.root_linvel[env_ids, self.socket_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.socket_actor_id_env] = 0.0

        socket_actor_ids_sim_int32 = self.socket_actor_ids_sim.to(dtype=torch.int32, device=self.device)

        # print(torch.cat([plug_actor_ids_sim_int32[env_ids], socket_actor_ids_sim_int32[env_ids]]), plug_actor_ids_sim_int32, socket_actor_ids_sim_int32)
        # print(self.root_state[:, plug_actor_ids_sim_int32, :])
        # print(self.root_state[:, socket_actor_ids_sim_int32, :])

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(torch.cat(
                                                         [plug_actor_ids_sim_int32[env_ids],
                                                          socket_actor_ids_sim_int32[env_ids]])),
                                                     len(torch.cat([plug_actor_ids_sim_int32[env_ids],
                                                                    socket_actor_ids_sim_int32[env_ids]])))

        # Simulate one step to apply changes
        self._simulate_and_refresh()

    def _move_arm_to_desired_pose(self, env_ids, desired_pos, desired_rot=None, sim_steps=30):
        """Move gripper to desired pose."""

        # Set target pos above object
        self.ctrl_target_fingertip_centered_pos[env_ids] = desired_pos[env_ids].clone()

        # Set target rot
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
            self.ctrl_target_dof_pos[:, 7:] = self.ctrl_target_gripper_dof_pos
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
        self.render()
        self._refresh_task_tensors()

    def _reset_buffers(self, env_ids):
        """Reset buffers. """

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.time_complete_task = torch.zeros_like(self.progress_buf)
        self.rew_buf[env_ids] = 0
        self.reward_log_buf[env_ids] = 0

        self.degrasp_buf[env_ids] = 0
        self.success_reset_buf[env_ids] = 0
        self.far_from_goal_buf[env_ids] = 0
        self.timeout_reset_buf[env_ids] = 0

        # Reset history
        self.ft_queue[env_ids] = 0
        self.tactile_queue[env_ids] = 0

        self.obs_queue[env_ids, ...] = 0.
        self.actions_queue[env_ids] *= 0
        self.arm_joint_queue[env_ids] *= 0
        self.arm_vel_queue[env_ids] *= 0
        self.eef_queue[env_ids] *= 0
        # self.delta_noisy_queue[env_ids] *= 0
        self.goal_noisy_queue[env_ids] *= 0
        self.targets_queue[env_ids] *= 0
        self.prev_targets[env_ids] *= 0
        self.prev_actions[env_ids] *= 0
        self.delta_pos_noisy_queue[env_ids] *= 0

        self.actions_queue_student[env_ids] *= 0
        self.arm_joint_queue_student[env_ids] *= 0
        self.eef_queue_student[env_ids] *= 0
        self.goal_noisy_queue_student[env_ids] *= 0
        self.targets_queue_student[env_ids] *= 0

        if self.cfg_task.env.compute_contact_gt:
            self.gt_extrinsic_contact[env_ids] *= 0

    def _set_viewer_params(self):
        """Set viewer parameters."""
        bx, by, bz = 0.5, 0.0, 0.05
        cam_pos = gymapi.Vec3(bx - 0.1, by - 0.1, bz + 0.07)
        cam_target = gymapi.Vec3(bx, by, bz)
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

        self.ctrl_target_gripper_dof_pos = gripper_dof_pos

        # self._move_gripper_to_dof_pos(env_ids=env_ids, gripper_dof_pos=gripper_dof_pos, sim_steps=sim_steps)

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
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device)  # - 0.5
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

        # self._simulate_and_refresh()

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
        return torch.norm(self.plug_pos[:, :2] - self.socket_tip[:, :2], p=2,
                          dim=-1) < self.cfg_task.rl.close_error_thresh

    def _check_plug_inserted_in_socket(self):
        """Check if plug is inserted in socket."""

        # Check if plug is within threshold distance of assembled state
        is_plug_below_insertion_height = (
                self.plug_pos[:, 2] <= (self.socket_tip[:, 2] - self.cfg_task.rl.success_height_thresh)
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
                (self.plug_pos[:, 2]) < self.socket_tip[:, 2]
        )

        # Check if plug is close to socket
        # NOTE: This check addresses edge case where base of plug is below top of socket,
        # but plug is outside socket
        is_plug_close_to_socket = self._check_plug_close_to_socket()  # torch.norm(self.plug_pos[:, :2] - self.socket_tip[:, :2], p=2, dim=-1) < 0.005 # self._check_plug_close_to_socket()
        # print(is_plug_below_engagement_height[0], is_plug_close_to_socket[0])

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
        self.obs_dict['tactile_hist'] = self.tactile_queue.clone().to(self.rl_device)
        self.obs_dict['ft_hist'] = self.ft_queue.clone().to(self.rl_device)
        self.obs_dict['priv_info'] = self.obs_dict['states'].to(self.rl_device)
        self.obs_dict['student_obs'] = self.obs_student_buf.to(self.rl_device)
        self.obs_dict['contacts'] = self.gt_extrinsic_contact.to(self.rl_device)
        self.obs_dict['socket_pos'] = self.socket_pos.clone().to(self.rl_device)
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        # self._refresh_task_tensors(update_tactile=True)
        super().reset()
        self.obs_dict['priv_info'] = self.obs_dict['states'].to(self.rl_device)
        self.obs_dict['student_obs'] = self.obs_student_buf.to(self.rl_device)
        self.obs_dict['tactile_hist'] = self.tactile_queue.clone().to(self.rl_device)
        self.obs_dict['ft_hist'] = self.ft_queue.clone().to(self.rl_device)
        self.obs_dict['contacts'] = self.gt_extrinsic_contact.to(self.rl_device)
        self.obs_dict['socket_pos'] = self.socket_pos.clone().to(self.rl_device)

        return self.obs_dict
