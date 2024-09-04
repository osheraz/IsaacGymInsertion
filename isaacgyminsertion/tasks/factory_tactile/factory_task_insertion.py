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
import random

import hydra
import omegaconf
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from isaacgym import gymapi, gymtorch
from isaacgyminsertion.tasks.factory_tactile.factory_env_insertion import FactoryEnvInsertionTactile
from isaacgyminsertion.tasks.factory_tactile.schema.factory_schema_class_task import FactoryABCTask
from isaacgyminsertion.tasks.factory_tactile.schema.factory_schema_config_task import FactorySchemaConfigTask
import isaacgyminsertion.tasks.factory_tactile.factory_control as fc
from isaacgyminsertion.tasks.factory_tactile.factory_utils import *
from isaacgyminsertion.utils import torch_jit_utils
import cv2
from isaacgyminsertion.allsight.tacto_allsight_wrapper.util.util import tensor2im
from torchvision.utils import save_image
from collections import defaultdict

torch.set_printoptions(sci_mode=False)
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi * 0.12, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi * 0.12, y_unit_tensor))

@torch.no_grad()
def filter_pts(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    valid1 = (z >= 0.005) & (z <= 0.5)
    valid2 = (x >= 0.4) & (x <= 0.65)
    valid3 = (y >= -0.2) & (y <= 0.2)

    valid = valid1 & valid3 & valid2
    pts = pts[valid]
    return pts


class FactoryTaskInsertionTactile(FactoryEnvInsertionTactile, FactoryABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize task superclass."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        self.ax = plt.axes(projection='3d')
        self.cfg = cfg
        self._get_task_yaml_params()
        self.display_id = random.randint(0, self.num_envs - 1)

        self._acquire_task_tensors()
        self.parse_controller_spec()

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

        self.tact_hist_len = self.cfg_task.env.tactile_history_len
        self.img_hist_len = self.cfg_task.env.img_history_len

        self.depth_process = DepthImageProcessor(cfg=self.cfg_task,
                                                 dis_noise=self.dis_noise,
                                                 far_clip=self.far_clip,
                                                 near_clip=self.near_clip)

        self.pcl_process = PointCloudAugmentations()

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.act_moving_average_range = self.cfg_task.env.actionsMovingAverage.range
        self.act_moving_average_scheduled_steps = self.cfg_task.env.actionsMovingAverage.schedule_steps
        self.act_moving_average_scheduled_freq = self.cfg_task.env.actionsMovingAverage.schedule_freq

        self.act_moving_average_lower = self.act_moving_average_range[0]
        self.act_moving_average_upper = self.act_moving_average_range[1]

        self.action_delay_prob_max = self.cfg_task.env.actionDelayProbMax
        self.action_delay_prob = self.action_delay_prob_max * torch.rand(self.num_envs,
                                                                         dtype=torch.float, device=self.device)
        self.action_latency_max = self.cfg_task.env.actionLatencyMax
        self.action_latency_scheduled_steps = self.cfg_task.env.actionLatencyScheduledSteps
        # We have action latency MIN and MAX (declared in _read_cfg() function reading from a config file)
        self.action_latency_min = 1
        self.action_latency = torch.randint(0, self.action_latency_min + 1,
                                            size=(self.num_envs,), dtype=torch.long, device=self.device)

        self.plug_obs_delay_prob = self.cfg_task.env.plugObsDelayProb
        self.img_delay_prob = self.cfg_task.env.ImgDelayProb
        self.seg_delay_prob = self.cfg_task.env.SegDelayProb
        self.seg_noise_prob = self.cfg_task.env.SegProbNoise
        self.pcl_noise_prob = self.cfg_task.env.PclProbNoise

        self.tactile_delay_prob = self.cfg_task.env.TactileDelayProb
        self.scale_pos_prob = self.cfg_task.env.scalePosProb
        self.scale_rot_prob = self.cfg_task.env.scaleRotProb

        self.max_skip_obs = self.cfg_task.env.maxObjectSkipObs
        self.max_skip_img = self.cfg_task.env.maxSkipImg
        self.max_skip_tactile = self.cfg_task.env.maxSkipTactile

        self.frame = 0
        self.reset_flag = True

        self.goal_ori = self.identity_quat.clone()

        # inverse refresh rate for each environment
        self.plug_pose_refresh_rates = torch.randint(1, self.max_skip_obs + 1, size=(self.num_envs,),
                                                     device=self.device)
        self.plug_pose_refresh_offset = torch.randint(0, self.max_skip_obs, size=(self.num_envs,), device=self.device)
        self.img_refresh_rates = torch.randint(1, self.max_skip_img + 1, size=(self.num_envs,), device=self.device)
        self.img_refresh_offset = torch.randint(0, self.max_skip_img, size=(self.num_envs,), device=self.device)
        self.tactile_refresh_rates = torch.randint(1, self.max_skip_tactile + 1, size=(self.num_envs,),
                                                   device=self.device)
        self.tactile_refresh_offset = torch.randint(0, self.max_skip_tactile, size=(self.num_envs,), device=self.device)

        # buffer storing object poses which are only refreshed every n steps
        self.obs_plug_pos_freq = self.plug_pos.clone()
        self.obs_plug_quat_freq = self.plug_quat.clone()

        # buffer storing object poses with added delay which are only refreshed every n steps
        self.obs_plug_pos = self.plug_pos.clone()
        self.obs_plug_quat = self.plug_quat.clone()

        self.act_moving_average = self.act_moving_average_upper
        self.force_scale = self.cfg_task.randomize.force_scale
        self.force_prob_range = [0.001, 0.1]
        self.force_decay = 0.99
        self.force_decay_interval = 0.08
        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(
            self.force_prob_range[1]))

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        self.hand_joints = torch.zeros((self.num_envs, 6), device=self.device)

        self.plug_grasp_pos_local = self.plug_heights * 0.9 * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1))
        self.plug_grasp_pos_local[:, 0] = -0.1 * self.plug_widths.squeeze()

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
        self.prev_actions_queue = torch.zeros(self.num_envs,
                                              self.cfg_task.env.action_latency_max + 1,
                                              self.num_actions, dtype=torch.float, device=self.device)

        self.targets = torch.zeros((self.num_envs, self.cfg_task.env.numTargets), device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.cfg_task.env.numTargets), dtype=torch.float,
                                        device=self.device)

        # Keep track of history
        self.obs_queue = torch.zeros((self.num_envs, self.cfg_task.env.numObsHist * self.num_observations),
                                     dtype=torch.float, device=self.device)
        self.obs_stud_queue = torch.zeros((self.num_envs, self.cfg_task.env.numObsStudentHist * self.num_obs_stud),
                                          dtype=torch.float, device=self.device)

        if self.cfg_task.env.compute_contact_gt:
            self.gt_extrinsic_contact = torch.zeros((self.num_envs, self.cfg_task.env.num_points),
                                                    device=self.device, dtype=torch.float)
            self.contact_queue = torch.zeros((self.num_envs, self.cfg_task.env.numObsStudentHist,
                                              self.cfg_task.env.num_points),
                                             dtype=torch.float, device=self.device)

        if self.pcl_cam:

            num_points = self.cfg_task.env.num_points

            if self.cfg_task.env.merge_socket_pcl:
                num_points += self.cfg_task.env.num_points_socket

            if self.cfg_task.env.merge_goal_pcl:
                self.goal_pcl = torch.zeros((self.num_envs, self.cfg_task.env.num_points_goal, 3),
                                            device=self.device, dtype=torch.float)

                num_points += self.cfg_task.env.num_points_goal

            self.pcl_queue = torch.zeros((self.num_envs, self.cfg_task.env.numObsStudentHist, num_points * 3),
                                         dtype=torch.float, device=self.device)

            self.pcl = torch.zeros((self.num_envs, num_points * 3), device=self.device, dtype=torch.float)

            self.rot_pcl_angle = torch.deg2rad(torch.FloatTensor(self.num_envs).uniform_(*(-self.cfg_task.randomize.pcl_rot,
                                                                                           self.cfg_task.randomize.pcl_rot)).to(self.device))
            self.pcl_pos_noise = torch.randn(self.num_envs, 1, 3, device=self.device)
            self.axes = torch.randint(0, 3, (self.num_envs,), device=self.device)

        if self.cfg_task.env.tactile:
            # tactile buffers
            self.num_channels = self.cfg_tactile.encoder.num_channels
            self.width = self.cfg_tactile.encoder.width // 2 if self.cfg_tactile.crop_roi else self.cfg_tactile.encoder.width
            self.height = self.cfg_tactile.encoder.height

            self.tactile_imgs = torch.zeros(
                (self.num_envs, len(self.fingertips), self.num_channels * self.width * self.height),
                device=self.device,
                dtype=torch.float,
            )
            self.tactile_queue = torch.zeros(
                (self.num_envs, self.tact_hist_len, len(self.fingertips), self.num_channels * self.width * self.height),
                device=self.device,
                dtype=torch.float,
            )

        if self.external_cam:

            if self.depth_cam:
                self.img_queue = torch.zeros(
                    (self.num_envs, self.img_hist_len, self.res[1] * self.res[0]),
                    device=self.device,
                    dtype=torch.float,
                )
                self.image_buf = torch.zeros(self.num_envs, self.res[1] * self.res[0]).to(self.device)

            if self.seg_cam:
                self.seg_queue = torch.zeros(
                    (self.num_envs, self.img_hist_len, self.res[1] * self.res[0]), device=self.device,
                    dtype=torch.float, )
                self.seg_buf = torch.zeros(self.num_envs, self.res[1] * self.res[0], dtype=torch.int32).to(self.device)

            self.init_external_cam()

        # reset tensors
        self.timeout_reset_buf = torch.zeros_like(self.reset_buf)
        self.degrasp_buf = torch.zeros_like(self.reset_buf)
        self.far_from_goal_buf = torch.zeros_like(self.reset_buf)
        self.success_reset_buf = torch.zeros_like(self.reset_buf)

        # state tensors
        self.plug_hand_pos, self.plug_hand_quat = torch.zeros((self.num_envs, 3), device=self.device), \
                                                  torch.zeros((self.num_envs, 4), device=self.device)
        self.plug_pos_error, self.plug_quat_error = torch.zeros((self.num_envs, 3), device=self.device), \
                                                    torch.zeros((self.num_envs, 4), device=self.device)
        self.plug_hand_pos_diff, self.plug_hand_quat_diff = torch.zeros((self.num_envs, 3), device=self.device), \
                                                            torch.zeros((self.num_envs, 4), device=self.device)
        self.plug_hand_pos_init, self.plug_hand_quat_init = torch.zeros((self.num_envs, 3), device=self.device), \
                                                            torch.zeros((self.num_envs, 4), device=self.device)

        self.rigid_physics_params = torch.zeros((self.num_envs, 14), device=self.device, dtype=torch.float)
        self.finger_normalized_forces = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        # reward tensor
        self.reward_log_buf = torch.zeros_like(self.rew_buf)
        self.ep_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.ep_success_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.ep_failure_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        self.reset_goal_buf = self.reset_buf.clone()

    def _refresh_task_tensors(self):
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
        self.socket_obs_pos_noise = socket_obs_pos_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.env.socket_pos_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.socket_priv_pos_noise = socket_obs_pos_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.env.socket_priv_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.noisy_socket_pos[:, 0] = self.socket_pos[:, 0] + self.socket_obs_pos_noise[:, 0]
        self.noisy_socket_pos[:, 1] = self.socket_pos[:, 1] + self.socket_obs_pos_noise[:, 1]
        self.noisy_socket_pos[:, 2] = self.socket_pos[:, 2] + self.socket_obs_pos_noise[:, 2]

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
            self.keypoints_socket[:, idx] = torch_jit_utils.tf_combine(self.fingertip_centered_quat,
                                                                       self.fingertip_centered_pos,
                                                                       self.identity_quat,
                                                                       (keypoint_offset * self.socket_heights)
                                                                       + socket_tip_pos_local)[1]

        # hand joints
        joint_names = [
            'finger_1_1_to_finger_1_2',
            'finger_2_1_to_finger_2_2',
            'base_to_finger_3_2',
            'finger_1_2_to_finger_1_3',
            'finger_2_2_to_finger_2_3',
            'finger_3_2_to_finger_3_3'
        ]
        joint_indices = [list(self.dof_dict.values()).index(name) for name in joint_names]
        # Extract values for specific joints using indexing
        self.hand_joints = self.dof_pos[:, joint_indices].clone()

        # fingertip forces
        e = 0.9 if self.cfg_task.env.smooth_force else 0
        normalize_forces = lambda x: torch.norm(x, dim=-1)  # (torch.clamp(torch.norm(x, dim=-1), 0, 10) / 10).view(-1)
        self.finger_normalized_forces[:, 0] = (1 - e) * normalize_forces(
            self.left_finger_force.clone()) + e * self.finger_normalized_forces[:, 0]
        self.finger_normalized_forces[:, 1] = (1 - e) * normalize_forces(
            self.right_finger_force.clone()) + e * self.finger_normalized_forces[:, 1]
        self.finger_normalized_forces[:, 2] = (1 - e) * normalize_forces(
            self.middle_finger_force.clone()) + e * self.finger_normalized_forces[:, 2]

    def update_tactile(self, update_freq, update_delay):

        left_allsight_poses = torch.cat((self.left_finger_pos, self.left_finger_quat), dim=-1)
        right_allsight_poses = torch.cat((self.right_finger_pos, self.right_finger_quat), dim=-1)
        middle_allsight_poses = torch.cat((self.middle_finger_pos, self.middle_finger_quat), dim=-1)
        object_poses = torch.cat((self.plug_pos, self.plug_quat), dim=-1)

        tf = np.eye(4)
        tf[0:3, 0:3] = euler_angles_to_matrix(
            euler_angles=torch.tensor([[0, 0, 0]]), convention="XYZ"
        ).numpy()

        left_finger_poses = xyzquat_to_tf_numpy(left_allsight_poses.cpu().numpy())

        left_finger_poses = left_finger_poses @ tf

        right_finger_poses = xyzquat_to_tf_numpy(right_allsight_poses.cpu().numpy())

        right_finger_poses = right_finger_poses @ tf

        middle_finger_poses = xyzquat_to_tf_numpy(middle_allsight_poses.cpu().numpy())

        middle_finger_poses = middle_finger_poses @ tf

        object_pose = xyzquat_to_tf_numpy(object_poses.cpu().numpy())

        self._render_tactile(left_finger_poses,
                             right_finger_poses,
                             middle_finger_poses,
                             object_pose,
                             update_freq,
                             update_delay)

        self.tactile_queue[:, 1:] = self.tactile_queue[:, :-1].clone().detach()
        self.tactile_queue[:, 0, ...] = self.tactile_imgs

    def _render_tactile(self, left_finger_pose, right_finger_pose, middle_finger_pose, object_pose, update_freq,
                        update_delay):

        tactile_imgs_list, height_maps = [], []  # only for display.
        finger_normalized_forces = self.finger_normalized_forces.clone()

        for e in range(self.num_envs):

            if update_freq[e] and update_delay[e]:

                self.tactile_handles[e][0].update_pose_given_sim_pose(left_finger_pose[e], object_pose[e])
                self.tactile_handles[e][1].update_pose_given_sim_pose(right_finger_pose[e], object_pose[e])
                self.tactile_handles[e][2].update_pose_given_sim_pose(middle_finger_pose[e], object_pose[e])

                tactile_imgs_per_env, height_maps_per_env = [], []
                # TODO: find a parallel solution
                for n in range(3):
                    if self.cfg_task.env.tactile_wrt_force:
                        force = 100 * finger_normalized_forces[e, n].cpu().detach().numpy()
                    else:
                        force = 70

                    tactile_img, height_map = self.tactile_handles[e][n].render(object_pose[e], force)

                    if self.cfg_tactile.sim2real:
                        # Reducing FPS by half
                        color_tensor = self.transform(tactile_img).unsqueeze(0).to(self.device)
                        tactile_img = self.model_G(color_tensor)
                        tactile_img = tensor2im(tactile_img)

                    # Subtract background
                    if self.cfg_tactile.diff:
                        tactile_img = self.tactile_handles[e][n].remove_bg(tactile_img,
                                                                           self.tactile_handles[e][n].bg_img)
                        tactile_img *= self.tactile_handles[e][n].mask

                    # flip to match real camera
                    tactile_img = np.flipud(tactile_img).copy()

                    # crop region of interest
                    if self.cfg_tactile.crop_roi:
                        w = tactile_img.shape[0]
                        tactile_img = tactile_img[:w // 2, :, :]
                        height_map = height_map[:w // 2]

                    # Resizing to encoder size
                    if tactile_img.shape[:2] != (self.width, self.height):
                        resized_img = cv2.resize(tactile_img, (self.height, self.width), interpolation=cv2.INTER_AREA)
                    else:
                        resized_img = tactile_img

                    if self.num_channels == 3:
                        tac_img = to_torch(resized_img).permute(2, 0, 1).to(self.device)
                        self.tactile_imgs[e, n] = tac_img.flatten()
                    else:
                        resized_img = cv2.cvtColor(resized_img.astype('float32'), cv2.COLOR_BGR2GRAY)
                        self.tactile_imgs[e, n] = to_torch(resized_img).to(self.device).flatten()

                    tactile_imgs_per_env.append(tactile_img)
                    height_maps_per_env.append(height_map)

                height_maps.append(height_maps_per_env)
                tactile_imgs_list.append(tactile_imgs_per_env)
            else:
                self.tactile_imgs[e] = self.tactile_queue[e, 0, ...].clone()

        env_to_show = 0
        if self.cfg_task.env.tactile_display_viz and update_freq[env_to_show] and update_delay[env_to_show]:
            self.tactile_handles[env_to_show][0].updateGUI(tactile_imgs_list[env_to_show], height_maps[env_to_show])

    def update_action_moving_average(self):

        # scheduling action moving average

        if self.last_step > 0 and self.last_step % self.act_moving_average_scheduled_freq == 0:
            sched_scaling = 1.0 / self.act_moving_average_scheduled_steps * min(self.last_step,
                                                                                self.act_moving_average_scheduled_steps)
            self.act_moving_average = self.act_moving_average_upper + (
                    self.act_moving_average_lower - self.act_moving_average_upper) * \
                                      sched_scaling

            # print('action moving average: {}'.format(self.act_moving_average))
            # print('last_step: {}'.format(self.last_step),
            #       ' scheduled steps: {}'.format(self.act_moving_average_scheduled_steps))

            self.extras['annealing/action_moving_average_scalar'] = self.act_moving_average

    def apply_action_noise_latency(self):

        # anneal action latency
        if self.randomize:
            self.cur_action_latency = 1.0 / self.action_latency_scheduled_steps \
                                      * min(self.last_step, self.action_latency_scheduled_steps)

            self.cur_action_latency = min(max(int(self.cur_action_latency), self.action_latency_min),
                                          self.action_latency_max)

            self.extras['annealing/cur_action_latency_max'] = self.cur_action_latency

            self.action_latency = torch.randint(0, self.cur_action_latency + 1,
                                                size=(self.num_envs,), dtype=torch.long,
                                                device=self.device)

        # probability of not updating the action this step (on top of the delay)
        action_delay_mask = (torch.rand(self.num_envs, device=self.device) > self.action_delay_prob).view(-1, 1)

        actions_delayed = \
            self.prev_actions_queue[
                torch.arange(self.prev_actions_queue.shape[0]), self.action_latency] * action_delay_mask \
            + self.prev_actions * ~action_delay_mask

        return actions_delayed

    def pre_physics_step(self, actions, with_delay=True):
        """Reset environments. Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # test actions for whenever we want to see some axis motion
        actions[:, :6] = 0.
        # actions[:, 0] = 1.
        # actions[:, 1] = 1.
        # actions[:, 2] = 1.

        self.prev_actions[:] = self.actions.clone()
        self.prev_plug_quat = self.plug_quat.clone()
        self.prev_plug_pos = self.plug_pos.clone()

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

        if with_delay:
            self.update_action_moving_average()
            # update the actions queue
            self.prev_actions_queue[:, 1:] = self.prev_actions_queue[:, :-1].clone().detach()
            self.prev_actions_queue[:, 0, :] = actions
            actions_delayed = self.apply_action_noise_latency()
            actions = actions_delayed

        if self.cfg_task.env.hand_action:
            delta_targets = torch.cat([
                self.actions[:, :3] @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device)),
                # 3
                self.actions[:, 3:6] @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device)),
                # 3
                self.actions[:, 6:9] @ torch.diag(
                    torch.tensor(self.cfg_task.rl.gripper_action_scale, device=self.device))  # 3
            ], dim=-1).clone()
        else:
            delta_targets = torch.cat([
                self.actions[:, :3] @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device)),
                # 3
                self.actions[:, 3:6] @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device)),
                # 3
            ], dim=-1).clone()

        # Update targets
        self.targets = self.prev_targets + delta_targets

        self._apply_actions_as_ctrl_targets(actions=actions,
                                            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
                                            do_scale=True)

        self.prev_targets[:] = self.targets.clone()

        if self.force_scale > 0.0:
            """Applies random forces to the object.
            Forces are applied as in https://arxiv.org/abs/1808.00177
            """
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device) * self.object_rb_masses * self.force_scale

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None,
                                                    gymapi.LOCAL_SPACE)

        if self.cfg_task.collect_rotate:
            self.rotate_plug()

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

        self.progress_buf[:] += 1
        self.randomize_buf[:] += 1

        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

        if self.viewer or True:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            rotate_vec = lambda q, x: quat_apply(q, to_torch(x, device=self.device) * 0.2).cpu().numpy()
            num_envs = 0
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

        self._render_headless()

    def compute_observations(self):
        """Compute observations."""

        # Compute Observation and state at current timestep
        # noisy_delta_pos = self.noisy_gripper_goal_pos - self.fingertip_centered_pos
        #
        # eef_pos = torch.cat(self.pose_world_to_robot_base(self.fingertip_centered_pos.clone(),
        #                                                   self.fingertip_centered_quat.clone(),
        #                                                   to_rep='rot6d'), dim=-1)
        actions = self.actions.clone()

        obs = torch.cat([self.rel_act_angles,  # 3
                         self.goal_ori,        # 4
                         actions[:, -3:],  # 3
                         ], dim=-1)

        self.obs_queue[:, :-self.num_observations] = self.obs_queue[:, self.num_observations:]
        self.obs_queue[:, -self.num_observations:] = obs

        # eef_stud = torch.cat(self.pose_world_to_robot_base(self.fingertip_centered_pos.clone(),
        #                                                    self.fingertip_centered_quat.clone(),
        #                                                    to_rep='matrix'), dim=-1)
        # # fix bug
        # eef_stud = torch.cat((self.fingertip_centered_pos,
        #                       self.stud_tf.forward(eef_stud[:, 3:].reshape(eef_stud.shape[0], 3, 3))), dim=1)

        # test = torch.cat(self.pose_world_to_robot_base(self.plug_pos.clone(),
        #                                                   self.plug_quat.clone(),
        #                                                   to_rep='rot6d'), dim=-1)

        # obs_tensors_student = torch.cat([
        #     eef_stud,
        #     noisy_delta_pos,  # 3
        #     actions,
        # ], dim=-1)

        self.obs_stud_queue[:, :-self.num_obs_stud] = self.obs_stud_queue[:, self.num_obs_stud:]
        self.obs_stud_queue[:, -self.num_obs_stud:] = obs

        # Define state (for teacher)
        if self.randomize:
            # Update the plug observation every update_freq
            obs_update_freq = torch.remainder(self.frame + self.plug_pose_refresh_offset,
                                              self.plug_pose_refresh_rates) == 0
            self.obs_plug_pos_freq[obs_update_freq] = self.plug_pos[obs_update_freq]
            self.obs_plug_quat_freq[obs_update_freq] = self.plug_quat[obs_update_freq]

            # Simulate adding delay on top
            update_delay = torch.randn(self.num_envs, device=self.device) > self.plug_obs_delay_prob
            self.obs_plug_pos[update_delay] = self.obs_plug_pos_freq[update_delay]
            self.obs_plug_quat[update_delay] = self.obs_plug_quat_freq[update_delay]

        plug_hand_pos, plug_hand_quat = self.pose_world_to_hand_base(self.obs_plug_pos, self.obs_plug_quat)

        self.plug_hand_pos[...] = plug_hand_pos
        self.plug_hand_quat[...] = plug_hand_quat

        self.plug_hand_pos_diff[...] = plug_hand_pos - self.plug_hand_pos_init
        self.plug_hand_quat_diff[...] = plug_hand_quat - self.plug_hand_quat_init

        plug_pos_error, plug_quat_error = fc.get_pose_error(
            fingertip_midpoint_pos=self.obs_plug_pos.clone(),
            fingertip_midpoint_quat=self.obs_plug_quat.clone(),
            ctrl_target_fingertip_midpoint_pos=self.socket_pos.clone() + self.socket_priv_pos_noise,
            ctrl_target_fingertip_midpoint_quat=self.identity_quat.clone(),
            jacobian_type='geometric',
            rot_error_type='quat')

        self.plug_pos_error[...] = plug_pos_error
        self.plug_quat_error[...] = plug_quat_error

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
                                                   middle_finger_friction,
                                                   self.plug_heights,  # 1
                                                   self.plug_depths,  # 1
                                                   self.plug_widths,  # 1
                                                   self.socket_heights,  # 1
                                                   self.socket_depths,  # 1
                                                   self.socket_widths,  # 1
                                                   self.plug_scale,  # 1
                                                   self.socket_scale, # 1
                                                   ]), 0, 1).to(self.device)

        self.rigid_physics_params[...] = physics_params

        if self.cfg_task.env.compute_contact_gt:
            for k, v in self.subassembly_extrinsic_contact.items():
                self.gt_extrinsic_contact[self.subassembly_to_env_ids[k], ...] = v.get_extrinsic_contact(
                    obj_pos=self.plug_pos[self.subassembly_to_env_ids[k], ...].clone(),
                    obj_quat=self.plug_quat[self.subassembly_to_env_ids[k], ...].clone(),
                    socket_pos=self.socket_pos[self.subassembly_to_env_ids[k], ...].clone(),
                    socket_quat=self.socket_quat[self.subassembly_to_env_ids[k], ...].clone(),
                    plug_scale=self.plug_scale[self.subassembly_to_env_ids[k]],
                    display='square_peg_hole_32mm_loose' == k and False
                ).to(self.device)

            self.contact_queue[:, 1:] = self.contact_queue[:, :-1].clone().detach()
            self.contact_queue[:, 0, ...] = self.gt_extrinsic_contact

        state_tensors = [
            #  add delta error
            # socket_pos_wrt_robot[0],  # 3
            # socket_pos_wrt_robot[1],  # 4
            # plug_bottom_wrt_robot[0],  # 3
            # plug_bottom_wrt_robot[1],  # 4
            # self.socket_pos.clone(),   # 3
            # self.socket_quat.clone(),  # 4
            self.hand_joints,          # 6
            # self.plug_hand_pos_diff,   # 3
            # self.plug_hand_quat_diff,  # 4
            plug_hand_pos,  # 3
            plug_hand_quat,  # 4
            # plug_pos_error,  # 3
            # plug_quat_error,  # 4
            physics_params,  # 13
            # self.plug_pcd.view(self.num_envs, -1),  # 3 * num_points
        ]

        if self.cfg_task.env.tactile:
            tactile_update_freq = torch.remainder(self.frame + self.tactile_refresh_offset,
                                                  self.tactile_refresh_rates) == 0
            tactile_update_delay = torch.rand(self.num_envs, device=self.device) > self.tactile_delay_prob
            self.update_tactile(tactile_update_freq, tactile_update_delay)

        if self.external_cam:
            img_update_freq = torch.remainder(self.frame + self.img_refresh_offset, self.img_refresh_rates) == 0
            is_initial_update = self.progress_buf < 10

            img_update = torch.rand(self.num_envs, device=self.device) > self.img_delay_prob
            seg_update = torch.rand(self.num_envs, device=self.device) > self.seg_delay_prob

            seg_add_noise = torch.rand(self.num_envs, device=self.device) > 1 - self.seg_noise_prob
            pcl_add_noise = torch.rand(self.num_envs, device=self.device) > 1 - self.pcl_noise_prob

            img_update = is_initial_update | img_update
            seg_update = is_initial_update | seg_update
            seg_add_noise = ~is_initial_update & seg_add_noise
            pcl_add_noise = ~is_initial_update & pcl_add_noise

            self.update_external_cam(img_update_freq,
                                     img_update,
                                     seg_update,
                                     seg_add_noise,
                                     pcl_add_noise)

        self.frame += 1
        self.obs_buf = self.obs_queue.clone()  # torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)
        self.obs_student_buf = self.obs_stud_queue.clone()  # shape = (num_envs, num_observations_student)
        self.states_buf = torch.cat(state_tensors, dim=-1)  # shape = (num_envs, num_states)

        return self.obs_buf  # shape = (num_envs, num_observations)

    def update_external_cam(self, update_freq, update_delay, seg_update_delay, seg_noise, pcl_noise):

        self.gym.step_graphics(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        update = torch.logical_and(update_freq, update_delay)
        update_seg = torch.logical_and(update_freq, seg_update_delay)
        seg_noise = torch.logical_and(seg_noise, update_seg)

        if self.cam_type == "rgb":
            processed_images = torch.stack(self.cam_renders).unsqueeze(1)
            self.image_buf[update] = processed_images[update]

            if self.cfg_task.external_cam.display:
                img = cv2.cvtColor(self.image_buf[0].cpu().numpy(), cv2.COLOR_RGB2BGR)
                cv2.imshow("Follow camera", img)
                cv2.waitKey(1)

        elif self.cam_type == "d":

            depth = torch.stack(self.cam_renders)
            seg = torch.stack(self.seg_renders)
            # self.rot_pcl_angle = torch.deg2rad(torch.FloatTensor(self.num_envs).uniform_(*(-self.cfg_task.randomize.pcl_rot,
            #                                                                                self.cfg_task.randomize.pcl_rot)).to(self.device))
            # self.pcl_pos_noise = torch.randn(self.num_envs, 1, 3, device=self.device)
            # self.axes = torch.randint(0, 3, (self.num_envs,), device=self.device)

            if self.depth_cam:
                if update.any():
                    if update.sum() == 1:
                        depth_to_update = depth[update].unsqueeze(0)
                    else:
                        depth_to_update = depth[update]
                    self.image_buf[update] = self.depth_process.process_depth_image(depth_to_update).flatten(
                        start_dim=1)

            if self.seg_cam:
                if update_seg.any():
                    if update_seg.sum() == 1:
                        seg_to_update = seg[update_seg].unsqueeze(0)
                    else:
                        seg_to_update = seg[update_seg]
                    self.seg_buf[update_seg] = seg_to_update.flatten(start_dim=1)

                if seg_noise.any():
                    self.seg_buf[seg_noise] = self.depth_process.add_seg_noise(self.seg_buf[seg_noise])

            if self.pcl_cam:
                if self.seg_cam:
                    plug_depth = depth.flatten(start_dim=1) * (seg.flatten(start_dim=1) == 2)

                    if self.cfg_task.env.merge_socket_pcl or self.cfg_task.env.relative_pcl:
                        socket_depth = depth.flatten(start_dim=1) * (seg.flatten(start_dim=1) == 3)
                else:
                    plug_depth = depth.flatten(start_dim=1)

                plug_pts = self.pcl_generator.get_point_cloud(
                    depths=plug_depth.reshape(self.num_envs, self.res[1], self.res[0]),
                    filter_func=filter_pts, sample_num=self.cfg_task.env.num_points).to(self.device)

                plug_pts[pcl_noise] = self.pcl_process.augment(plug_pts[pcl_noise],
                                                               self.rot_pcl_angle[pcl_noise],
                                                               self.axes[pcl_noise],
                                                               self.pcl_pos_noise[pcl_noise])

                self.pcl = plug_pts.flatten(start_dim=1)

                if self.cfg_task.env.merge_socket_pcl or self.cfg_task.env.relative_pcl:

                    socket_pts = self.pcl_generator.get_point_cloud(
                        depths=socket_depth.reshape(self.num_envs, self.res[1], self.res[0]),
                        filter_func=filter_pts, sample_num=self.cfg_task.env.num_points_socket).to(self.device)

                    socket_pts[pcl_noise] = self.pcl_process.augment(socket_pts[pcl_noise],
                                                                     self.rot_pcl_angle[pcl_noise],
                                                                     self.axes[pcl_noise],
                                                                     self.pcl_pos_noise[pcl_noise])

                    if self.cfg_task.env.relative_pcl:
                        socket_mean = self.noisy_socket_pos.unsqueeze(1)  # socket_pts.mean(dim=1, keepdim=True)
                        plug_pts -= socket_mean
                        socket_pts -= socket_mean

                    if self.cfg_task.env.merge_socket_pcl:
                        self.pcl = torch.cat([plug_pts, socket_pts], dim=1).flatten(start_dim=1)
                    else:
                        self.pcl = plug_pts.flatten(start_dim=1)

                else:
                    socket_pts = None

                if self.cfg_task.env.merge_goal_pcl:

                    for k, v in self.subassembly_extrinsic_contact.items():
                        self.goal_pcl[self.subassembly_to_env_ids[k], ...] = v.get_goal_pcl(
                            socket_pos=self.socket_pos[self.subassembly_to_env_ids[k], ...].clone(),
                            socket_quat=self.socket_quat[self.subassembly_to_env_ids[k], ...].clone(),
                            plug_scale=self.plug_scale[self.subassembly_to_env_ids[k]],
                            display='square_peg_hole_32mm_loose' == k and self.cfg_task.external_cam.display).to(
                            self.device)

                    self.goal_pcl[pcl_noise] = self.pcl_process.augment(self.goal_pcl[pcl_noise],
                                                                        self.rot_pcl_angle[pcl_noise],
                                                                        self.axes[pcl_noise],
                                                                        self.pcl_pos_noise[pcl_noise])
                    if socket_pts is not None:
                        self.pcl = torch.cat([plug_pts, socket_pts, self.goal_pcl], dim=1).flatten(start_dim=1)
                    else:
                        self.pcl = torch.cat([plug_pts, self.goal_pcl], dim=1).flatten(start_dim=1)

            if self.cfg_task.external_cam.display:
                if self.depth_cam:
                    img = self.image_buf[self.display_id].cpu().clone().reshape(1, self.res[1], self.res[0])
                    cv2.imshow(f"Depth Image: {self.display_id}", img.numpy().transpose(1, 2, 0))

                if self.seg_cam:
                    mask = self.seg_buf[self.display_id].cpu().clone().reshape(self.res[1], self.res[0])
                    forward_mask = ((mask == 2) | (mask == 3)).float()

                if self.seg_cam and self.depth_cam:
                    img = torch.where(forward_mask > 0.5, img, 0)
                    cv2.imshow(f"Mask Image: {self.display_id}", img.numpy().transpose(1, 2, 0))

                cv2.waitKey(1)

        self.gym.end_access_image_tensors(self.sim)

        if self.pcl_cam:
            self.pcl_queue[:, 1:] = self.pcl_queue[:, :-1].clone().detach()
            self.pcl_queue[:, 0, ...] = self.pcl

        if self.depth_cam:
            self.img_queue[:, 1:] = self.img_queue[:, :-1].clone().detach()
            self.img_queue[:, 0, ...] = self.image_buf

        if self.seg_cam:
            self.seg_queue[:, 1:] = self.seg_queue[:, :-1].clone().detach()
            self.seg_queue[:, 0, ...] = self.seg_buf

    def init_external_cam(self, with_seg=True):

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.cam_renders = []
        self.seg_renders = []

        for i in range(self.num_envs):
            if self.cam_type == "rgb":
                im = self.gym.get_camera_image_gpu_tensor(
                    self.sim,
                    self.envs[i],
                    self.camera_handles[i],
                    gymapi.IMAGE_COLOR,
                )
                im = gymtorch.wrap_tensor(im)

            elif self.cam_type == "d":
                im = self.gym.get_camera_image_gpu_tensor(
                    self.sim,
                    self.envs[i],
                    self.camera_handles[i],
                    gymapi.IMAGE_DEPTH,
                )
                im = gymtorch.wrap_tensor(im)

            self.cam_renders.append(im)

            if with_seg:
                im = self.gym.get_camera_image_gpu_tensor(
                    self.sim,
                    self.envs[i],
                    self.camera_handles[i],
                    gymapi.IMAGE_SEGMENTATION,
                )
                im = gymtorch.wrap_tensor(im)
                self.seg_renders.append(im)

        self.gym.end_access_image_tensors(self.sim)

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        self._update_reset_buf()
        self._update_rew_buf()

    def _update_rew_buf(self):
        """Compute reward at current timestep."""

        act = self.actions[:, -3:]
        prv = self.prev_actions[:, -3:]

        action_penalty = torch.norm(act, p=2, dim=-1)
        action_reward = self.cfg_task.rl.action_penalty_scale * action_penalty

        action_delta_penalty = torch.norm(act - prv, p=2, dim=-1)
        action_delta_reward = self.cfg_task.rl.action_delta_scale * action_delta_penalty

        # hand_joints_penalty = torch.norm(self.hand_joints - self.init_hand_joints, p=2, dim=-1)
        # hand_joint_reward = self.cfg_task.rl.tension_penalty * hand_joints_penalty

        # plug_ori_penalty = torch.norm(self.plug_quat - self.identity_quat, p=2, dim=-1)
        # ori_reward = plug_ori_penalty * self.cfg_task.rl.ori_reward_scale

        quat_diff = torch_jit_utils.quat_mul(self.plug_quat, torch_jit_utils.quat_conjugate(self.goal_ori))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
        ori_reward = self.cfg_task.rl.ori_reward_scale / (torch.abs(rot_dist) + self.cfg_task.rl.orientation_threshold)

        # keypoint_dist = self._get_keypoint_dist()
        # keypoint_reward = keypoint_dist * self.cfg_task.rl.keypoint_reward_scale

        # is_plug_engaged_w_socket = self._check_plug_engaged_w_socket()
        # engagement = self._get_engagement_reward_scale(is_plug_engaged_w_socket, self.cfg_task.rl.success_height_thresh)
        # engagement_reward = engagement * self.cfg_task.rl.engagement_reward_scale

        distance_reset_buf = (self.far_from_goal_buf | self.degrasp_buf)
        early_reset_reward = distance_reset_buf * self.cfg_task.rl.early_reset_reward_scale

        fingertip_pos = torch.cat((self.left_finger_pos,
                                   self.right_finger_pos,
                                   self.middle_finger_pos), dim=0)

        ftip_diff = (fingertip_pos.view(self.num_envs, -1, 3) - self.plug_pos[:, None, :])
        ftip_dist = torch.linalg.norm(ftip_diff, dim=-1).view(self.num_envs, -1)
        ftip_dist_mean = ftip_dist.mean(dim=-1)
        ftip_reward = ftip_dist_mean * self.cfg_task.rl.fingertip_reward_scale

        self.rew_buf[:] = ftip_reward + ori_reward + action_reward + action_delta_reward
        self.rew_buf[:] += early_reset_reward

        # self.rew_buf[:] += (early_reset_reward * self.timeout_reset_buf)
        # self.rew_buf[:] += (self.timeout_reset_buf * self.success_reset_buf) * self.cfg_task.rl.success_bonus

        # Find out which envs hit the goal and update successes count
        goal_resets = torch.where(torch.abs(rot_dist) <= self.cfg_task.rl.success_tolerance,
                                  torch.ones_like(self.success_reset_buf),
                                  self.success_reset_buf)

        self.success_reset_buf[:] = goal_resets
        # self.rew_buf[:] = torch.where(goal_resets == 1, self.rew_buf + self.cfg_task.rl.success_bonus, self.rew_buf)

        self.extras['successes'] = ((self.timeout_reset_buf | distance_reset_buf) * self.success_reset_buf) * 1.0
        # self.extras['keypoint_reward'] = keypoint_reward
        # self.extras['engagement_reward'] = engagement_reward
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
        # self.success_reset_buf[:] = self._check_plug_inserted_in_socket()

        # if we are collecting data, reset at insertion
        if (self.cfg_task.data_logger.collect_data or
                self.cfg_task.data_logger.collect_test_sim or
                self.cfg_task.reset_at_success):
            self.reset_buf[:] |= self.success_reset_buf[:]

        # If max episode length has been reached
        self.timeout_reset_buf[:] = torch.where(self.progress_buf[:] >= (self.cfg_task.rl.max_episode_length - 1),
                                                torch.ones_like(self.reset_buf),
                                                self.reset_buf)

        self.reset_buf[:] = self.timeout_reset_buf[:]

        # check is object is grasped and reset if not
        roll, pitch, yaw = get_euler_xyz(self.plug_quat.clone())
        roll[roll > np.pi] -= 2 * np.pi
        pitch[pitch > np.pi] -= 2 * np.pi
        yaw[yaw > np.pi] -= 2 * np.pi

        self.degrasp_buf[:] = (torch.abs(roll) > 0.7) | (torch.abs(pitch) > 0.7) | (torch.abs(yaw) > 0.7)

        # Check if object is too far from gripper
        fingertips_plug_dist = (torch.norm(self.left_finger_pos - self.plug_pos, p=2, dim=-1) > 0.2) | (
                                torch.norm(self.right_finger_pos - self.plug_pos, p=2, dim=-1) > 0.2) | (
                                torch.norm(self.middle_finger_pos - self.plug_pos, p=2, dim=-1) > 0.2)

        self.far_from_goal_buf[:] = fingertips_plug_dist

        if self.cfg_task.reset_at_fails:
            self.reset_buf[:] |= self.degrasp_buf[:]
            self.reset_buf[:] |= self.far_from_goal_buf[:]

        if ((self.cfg_task.data_logger.collect_data or
             self.cfg_task.data_logger.collect_test_sim) and not self.cfg_task.collect_rotate):
            self.reset_buf[:] |= self.degrasp_buf[:]

    def _reset_predefined_environment(self, env_ids):

        random_init_idx = {}

        for subassembly in self.cfg_env.env.desired_subassemblies:
            random_init_idx[subassembly] = torch.randint(0, self.total_init_poses[subassembly], size=(len(env_ids),))
        subassemblies = [self.envs_asset[e_id] for e_id in range(self.num_envs)]

        kuka_dof_pos = torch.zeros((len(env_ids), 13))
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

        object_pose = {
            'socket_pose': socket_pos,
            'socket_quat': socket_quat,
            'plug_pose': plug_pos,
            'plug_quat': plug_quat
        }

        self._reset_object(env_ids, new_pose=object_pose)

        # for k, v in self.subassembly_extrinsic_contact.items():
        #     v.reset_socket_pos(socket_pos=socket_pos.cpu().detach())

    def reset_target_pose(self, env_ids, apply_reset=False, random_rot=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        new_rot = new_rot if random_rot else self.identity_quat[env_ids, :]

        self.goal_ori[env_ids, :] = new_rot

        self.root_quat[env_ids, self.goal_actor_id_env, :] = new_rot.to(device=self.device)
        self.root_linvel[env_ids, self.goal_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.goal_actor_id_env] = 0.0

        if apply_reset:
            goal_actor_ids_sim_int32 = self.goal_actor_ids_sim.clone().to(dtype=torch.int32, device=self.device)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state),
                                                         gymtorch.unwrap_tensor(goal_actor_ids_sim_int32[env_ids]),
                                                         len(goal_actor_ids_sim_int32[env_ids]))

            self._simulate_and_refresh()

        self.reset_goal_buf[env_ids] = 0

    def _reset_environment(self, env_ids):

        random_init_idx = {}

        for subassembly in self.cfg_env.env.desired_subassemblies:
            random_init_idx[subassembly] = torch.randint(0, self.total_init_poses[subassembly], size=(len(env_ids),))

        subassemblies = [self.envs_asset[e_id] for e_id in range(self.num_envs)]

        kuka_dof_pos = torch.zeros((len(env_ids), 13), device=self.device)
        socket_pos = torch.zeros((len(env_ids), 3), device=self.device)
        socket_quat = torch.zeros((len(env_ids), 4), device=self.device)
        plug_pos = torch.zeros((len(env_ids), 3), device=self.device)
        plug_quat = torch.zeros((len(env_ids), 4), device=self.device)

        if self.cfg_task.randomize.same_socket:
            # socket around within x[-1cm, 1cm], y[-1cm, 1cm], z[-2mm, 3mm]
            socket_pos[:, 0] = self.cfg_task.randomize.socket_pos_xy_initial[0]
            socket_pos[:, 1] = self.cfg_task.randomize.socket_pos_xy_initial[1]
            socket_pos[:, 2] = self.cfg_task.randomize.socket_pos_xy_initial[2]
            socket_pos_noise = np.random.uniform(-self.cfg_task.randomize.socket_pos_xy_noise[0],
                                                 self.cfg_task.randomize.socket_pos_xy_noise[1], 3)
            socket_pos_noise[2] = np.random.uniform(self.cfg_task.randomize.socket_pos_z_noise_bounds[0],
                                                    self.cfg_task.randomize.socket_pos_z_noise_bounds[1], 1)
            socket_pos[:, :] += torch.from_numpy(socket_pos_noise)
        else:
            if self.cfg_task.env.compute_contact_gt:
                assert self.cfg_task.randomize.same_socket == self.cfg_task.env.compute_contact_gt

            # Randomize socket pos
            socket_noise_xy = 2 * (
                    torch.rand((len(env_ids), 2), dtype=torch.float32, device=self.device)
                    - 0.5
            )
            socket_noise_xy = socket_noise_xy @ torch.diag(
                torch.tensor(
                    self.cfg_task.randomize.socket_pos_xy_noise,
                    dtype=torch.float32,
                    device=self.device,
                )
            )

            socket_noise_z_mag = (
                    self.cfg_task.randomize.socket_pos_z_noise_bounds[1]
                    - self.cfg_task.randomize.socket_pos_z_noise_bounds[0]
            )
            socket_noise_z = (
                    socket_noise_z_mag
                    * torch.rand((len(env_ids)), dtype=torch.float32, device=self.device)
                    + self.cfg_task.randomize.socket_pos_z_noise_bounds[0]
            )

            socket_pos[:, 0] = (
                    + self.cfg_task.randomize.socket_pos_xy_initial[0]
                    + socket_noise_xy[:, 0]
            )
            socket_pos[:, 1] = (
                    + self.cfg_task.randomize.socket_pos_xy_initial[1]
                    + socket_noise_xy[:, 1]
            )
            socket_pos[:, 2] = self.cfg_base.env.table_height + socket_noise_z

        socket_euler_w_noise = np.array([0, 0, 0])
        # socket_euler_w_noise[2] = np.random.uniform(-self.cfg_task.randomize.socket_rot_euler_noise,
        #                                              self.cfg_task.randomize.socket_rot_euler_noise, 1)  # -2 to 2 deg
        socket_quat[:, :] = torch.from_numpy(R.from_euler('xyz', socket_euler_w_noise).as_quat())

        # above socket with overlap
        # Set plug pos to assembled state, but offset plug Z-coordinate by height of socket,
        plug_pos = socket_pos.clone()
        plug_noise_xy = 2 * (
                torch.rand((len(env_ids), 2), dtype=torch.float32, device=self.device)
                - 0.5
        )
        plug_noise_xy = plug_noise_xy @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.plug_pos_xy_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        plug_noise_z_mag = (
                self.cfg_task.randomize.plug_pos_z_noise_bounds[1]
                - self.cfg_task.randomize.plug_pos_z_noise_bounds[0]
        )
        plug_noise_z = (
                plug_noise_z_mag
                * torch.rand((len(env_ids)), dtype=torch.float32, device=self.device)
                + self.cfg_task.randomize.plug_pos_z_noise_bounds[0]
        )

        plug_pos[:, :2] += plug_noise_xy[:, :2]
        plug_pos[:, 2] = self.cfg_base.env.table_height + self.socket_heights.view(-1)[env_ids] + plug_noise_z

        # plug_pos_noise = torch.rand((len(env_ids), 3)) * self.cfg_task.randomize.plug_pos_xy_noise[0]  # 0 to 0.0254
        # plug_pos_noise[:, 2] = ((torch.rand((len(env_ids),)) * (0.007 - 0.003)) + 0.003) + 0.02  # 0.003 to 0.01
        # plug_pos[:, :] += plug_pos_noise
        plug_rot_noise = 2 * (
                torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
                - 0.5
        )
        plug_rot_noise = plug_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.plug_rot_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )
        # plug_euler_w_noise = np.array([0., 0., 0.])
        # plug_euler_w_noise += np.random.uniform(-0.03, 0.03, plug_euler_w_noise.shape)
        # plug_euler_w_noise[-1] = 0

        plug_quat[:, :] = torch.from_numpy(R.from_euler('xyz', plug_rot_noise.cpu().detach().numpy()).as_quat())

        for i, e in enumerate(env_ids):
            subassembly = subassemblies[e]
            kuka_dof_pos[i] = self.init_dof_pos[subassembly][random_init_idx[subassembly][i]]

        # open gripper
        kuka_dof_pos[:, 7:] = 0.
        self._reset_kuka(env_ids, new_pose=kuka_dof_pos)

        # for _, v in self.all_rendering_camera.items():
        #     self.init_plug_pos_cam[v[0], :] = plug_pos[v[0], :]

        object_pose = {
            'socket_pose': socket_pos,
            'socket_quat': socket_quat,
            'plug_pose': plug_pos,
            'plug_quat': plug_quat
        }

        self._reset_object(env_ids, new_pose=object_pose)

        for k, v in self.subassembly_extrinsic_contact.items():
            v.reset_socket_pos(socket_pos=socket_pos[0].cpu().detach())

        # self._open_gripper(torch.arange(self.num_envs))

        # self._simulate_and_refresh()

    def randomisation_callback(self, param_name, param_val, env_id=None, actor=None):
        if param_name == "scale" and actor == "plug":
            self.plug_scale[env_id] = param_val.mean()
        if param_name == "scale" and actor == "socket":
            self.socket_scale[env_id] = param_val.mean()
        elif param_name == "mass" and actor == "plug":
            self.rigid_physics_params[env_id, 0] = np.mean(param_val)
        elif param_name == "friction" and actor == "plug":
            self.rigid_physics_params[env_id, 1] = np.mean(param_val)
        elif param_name == "friction" and actor == "socket":
            self.rigid_physics_params[env_id, 2] = np.mean(param_val)

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        if self.randomize:
            self.apply_randomizations(dr_params=self.randomization_params,
                                      randomisation_callback=self.randomisation_callback)

        self.reset_target_pose(env_ids, apply_reset=True)

        self.disable_gravity()
        if self.cfg_task.grasp_at_init:
            self._reset_environment(env_ids)
            # Move arm to grasp pose
            plug_pos_noise = (2 * (torch.rand((len(env_ids), 3),
                                              device=self.device) - 0.5)) * self.cfg_task.randomize.grasp_plug_noise
            first_plug_pose = self.plug_grasp_pos.clone()

            first_plug_pose[env_ids, :2] += plug_pos_noise[:, :2] * 0
            self._move_arm_to_desired_pose(env_ids, first_plug_pose,
                                           sim_steps=self.cfg_task.env.num_gripper_move_sim_steps)
            self._refresh_task_tensors()
            self._close_gripper(env_ids)
            self.enable_gravity(-9.81)
        else:
            self._reset_predefined_environment(env_ids)
            self._close_gripper(env_ids)
            self.enable_gravity(-9.81)

        self._zero_velocities(env_ids)
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()

        # Update init grasp pos
        plug_hand_pos, plug_hand_quat = self.pose_world_to_hand_base(self.plug_pos, self.plug_quat)
        self.plug_hand_pos_init[...] = plug_hand_pos
        self.plug_hand_quat_init[...] = plug_hand_quat
        self.init_hand_joints = self.hand_joints.clone()

        if self.cfg_task.env.record_video and 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
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

        self.dof_pos[env_ids, :] = new_pose.to(device=self.device)

        if self.cfg_task.grasp_at_init:
            self.dof_pos[env_ids, :7] = torch.tensor(self.cfg_task.randomize.kuka_arm_initial_dof_pos,
                                                     device=self.device).repeat((len(env_ids), 1))
            # dont play with these joints (no actuation here)#
            self.dof_pos[
                env_ids, list(self.dof_dict.values()).index(
                    'base_to_finger_1_1')] = self.cfg_task.env.openhand.base_angle
            self.dof_pos[
                env_ids, list(self.dof_dict.values()).index(
                    'base_to_finger_2_1')] = -self.cfg_task.env.openhand.base_angle
            # dont play with these joints (no actuation here)#
            self.dof_pos[env_ids, list(self.dof_dict.values()).index(
                'finger_1_1_to_finger_1_2')] = self.cfg_task.env.openhand.proximal_open
            self.dof_pos[env_ids, list(self.dof_dict.values()).index(
                'finger_2_1_to_finger_2_2')] = self.cfg_task.env.openhand.proximal_open
            self.dof_pos[env_ids, list(self.dof_dict.values()).index(
                'base_to_finger_3_2')] = self.cfg_task.env.openhand.proximal_open + 0.4
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

        # Simulate one step to apply changes
        # self._simulate_and_refresh()

    def rotate_plug(self, interval=np.radians(5)):
        if self.cfg_task.data_logger.collect_data:
            from algo.ppo.experience import SimLogger
            self.data_logger = SimLogger(env=self)
            self.data_logger.data_logger = self.data_logger.data_logger_init(None)

        for _ in range(self.cfg_task.data_logger.total_trajectories):
            self.reset_idx(torch.arange(self.num_envs, device=self.device))

            initial_pos = self.plug_pos.clone()
            tip_pos = self.plug_tip.clone()
            initial_quat = self.plug_quat.clone()

            # Randomly select an axis and a direction for each environment
            axes = torch.randint(0, 3, (self.num_envs,), device=self.device)
            directions = torch.randint(0, 2, (self.num_envs,),
                                       device=self.device) * 2 - 1  # -1 for decreasing, 1 for increasing
            num_steps = self.max_episode_length

            for step in range(num_steps):

                cycle_angle = interval * np.sin((2 * np.pi / num_steps) * step)
                cycle_angle = cycle_angle * directions

                cos_half_angle = torch.cos(cycle_angle / 2)
                sin_half_angle = torch.sin(cycle_angle / 2)

                incremental_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
                for i in range(self.num_envs):
                    if axes[i] == 0:  # x-axis
                        incremental_quat[i] = torch.tensor([sin_half_angle[i], 0.0, 0.0, cos_half_angle[i]],
                                                           dtype=torch.float32, device=self.device)
                    elif axes[i] == 1:  # y-axis
                        incremental_quat[i] = torch.tensor([0.0, sin_half_angle[i], 0.0, cos_half_angle[i]],
                                                           dtype=torch.float32, device=self.device)
                    elif axes[i] == 2:  # z-axis
                        incremental_quat[i] = torch.tensor([0.0, 0.0, sin_half_angle[i], cos_half_angle[i]],
                                                           dtype=torch.float32, device=self.device)

                # Calculate the rotation quaternion relative to the initial orientation
                new_quat = quat_mul(initial_quat, incremental_quat)

                # Calculate new base position using the tip as the pivot
                rotated_offset = quat_apply(incremental_quat, initial_pos - tip_pos)
                new_base_pos = tip_pos + rotated_offset

                # Update root quaternion and position for the base
                self.root_quat[:, self.plug_actor_id_env] = new_quat
                self.root_pos[:, self.plug_actor_id_env] = new_base_pos

                self.root_linvel[:, self.plug_actor_id_env] = 0.0
                self.root_angvel[:, self.plug_actor_id_env] = 0.0

                plug_actor_ids_sim_int32 = self.plug_actor_ids_sim.clone().to(dtype=torch.int32, device=self.device)
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self.root_state),
                    gymtorch.unwrap_tensor(plug_actor_ids_sim_int32[:]),
                    len(plug_actor_ids_sim_int32[:])
                )

                self._simulate_and_refresh()
                self.post_physics_step()
                self._update_reset_buf()

                if self.cfg_task.data_logger.collect_data:
                    self.data_logger.log_trajectory_data(self.actions,
                                                         None,
                                                         self.timeout_reset_buf.clone(),
                                                         save_trajectory=True)

    def _reset_object(self, env_ids, new_pose=None):
        """Reset root state of plug."""

        plug_pose = new_pose['plug_pose']
        plug_quat = new_pose['plug_quat']

        self.root_pos[env_ids, self.plug_actor_id_env, :] = plug_pose.to(device=self.device)
        self.root_quat[env_ids, self.plug_actor_id_env, :] = plug_quat.to(device=self.device)

        # Stabilize plug
        self.root_linvel[env_ids, self.plug_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.plug_actor_id_env] = 0.0

        # Set plug root state
        plug_actor_ids_sim_int32 = self.plug_actor_ids_sim.clone().to(dtype=torch.int32, device=self.device)
        if self.cfg_task.grasp_at_init:
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state),
                                                         gymtorch.unwrap_tensor(plug_actor_ids_sim_int32[env_ids]),
                                                         len(plug_actor_ids_sim_int32[env_ids]))
            self._simulate_and_refresh()
        # RESET SOCKET
        socket_pose = new_pose['socket_pose']
        socket_quat = new_pose['socket_quat']

        self.root_pos[env_ids, self.socket_actor_id_env, :] = socket_pose.to(device=self.device)
        self.root_quat[env_ids, self.socket_actor_id_env, :] = socket_quat.to(device=self.device)

        # Stabilize socket
        self.root_linvel[env_ids, self.socket_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.socket_actor_id_env] = 0.0

        socket_actor_ids_sim_int32 = self.socket_actor_ids_sim.clone().to(dtype=torch.int32, device=self.device)

        if self.cfg_task.grasp_at_init:
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state),
                gymtorch.unwrap_tensor(socket_actor_ids_sim_int32),
                len(socket_actor_ids_sim_int32),
            )
            self._simulate_and_refresh()
        else:
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state),
                                                         gymtorch.unwrap_tensor(torch.cat(
                                                             [plug_actor_ids_sim_int32[env_ids],
                                                              socket_actor_ids_sim_int32[env_ids]])),
                                                         len(torch.cat([plug_actor_ids_sim_int32[env_ids],
                                                                        socket_actor_ids_sim_int32[env_ids]])))

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

        if (not self.cfg_task.grasp_at_init and
                self.reset_flag and
                self.cfg_task.reset_at_success):
            print('\n\n\n Rand Inits \n\n\n')
            self.reset_flag = False
            # Cuz it's easier to shuffle here
            self.progress_buf[env_ids] = torch.randint(
                low=0,
                high=self.cfg_task.rl.max_episode_length + 1,
                size=(len(env_ids),),
                device=self.device,
                dtype=torch.long
            )

        self.time_complete_task = torch.zeros_like(self.progress_buf)
        self.rew_buf[env_ids] = 0
        self.reward_log_buf[env_ids] = 0

        self.degrasp_buf[env_ids] = 0
        self.success_reset_buf[env_ids] = 0
        self.far_from_goal_buf[env_ids] = 0
        self.timeout_reset_buf[env_ids] = 0

        # Reset history
        self.obs_queue[env_ids, ...] = 0.
        self.prev_targets[env_ids] *= 0
        self.prev_actions[env_ids] *= 0
        self.prev_actions_queue[env_ids] *= 0
        self.prev_plug_quat[env_ids] *= 0
        self.prev_plug_pos[env_ids] *= 0
        self.rel_act_angles[env_ids] *= 0
        self.rb_forces[env_ids, :, :] = 0.0

        # object pose is represented with respect to the wrist
        self.obs_plug_pos[env_ids] = self.plug_pos[env_ids].clone()
        self.obs_plug_pos_freq[env_ids] = self.plug_pos[env_ids].clone()
        self.obs_plug_quat[env_ids] = self.plug_quat[env_ids].clone()
        self.obs_plug_quat_freq[env_ids] = self.plug_quat[env_ids].clone()

        if self.cfg_task.env.tactile:
            self.tactile_queue[env_ids, ...] = 0
            self.tactile_imgs[env_ids, ...] = 0.
        if self.external_cam:
            if self.depth_cam:
                self.img_queue[env_ids] = 0
                self.image_buf[env_ids] *= 0
            if self.seg_cam:
                self.seg_buf[env_ids] *= 0
                self.seg_queue[env_ids] *= 0
        if self.pcl_cam:
            rand_angles = torch.FloatTensor(self.num_envs).uniform_(*(-self.cfg_task.randomize.pcl_rot,
                                                                      self.cfg_task.randomize.pcl_rot)).to(self.device)
            self.rot_pcl_angle[env_ids] = torch.deg2rad(rand_angles)[env_ids]
            self.pcl_pos_noise[env_ids] = torch.randn(self.num_envs, 1, 3, device=self.device)[env_ids]
            self.axes[env_ids] = torch.randint(0, 3, (self.num_envs,), device=self.device)[env_ids]
            self.pcl_queue[env_ids] *= 0
            self.pcl[env_ids] *= 0
            if self.cfg_task.env.merge_goal_pcl:
                self.goal_pcl[env_ids] *= 0
        if self.cfg_task.env.compute_contact_gt:
            self.gt_extrinsic_contact[env_ids] *= 0
            self.contact_queue[env_ids] *= 0

    def _set_viewer_params(self):
        """Set viewer parameters."""
        bx, by, bz = 0.5, 0.0, 0.05
        cam_pos = gymapi.Vec3(bx - 0.1, by - 0.1, bz + 0.07)
        cam_target = gymapi.Vec3(bx, by, bz)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale, init_grasp=False):
        """Apply actions from policy as position/rotation targets."""

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_action_scale = torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device)

            if self.randomize:
                pos_action_scale = pos_action_scale.unsqueeze(0).repeat(self.num_envs, 1)
                scale_noise_pos = 2 * (
                        torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
                        - 0.5
                )
                scale_noise_pos = scale_noise_pos @ torch.diag(
                    torch.tensor(
                        self.cfg_task.randomize.scale_noise_pos,
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                scale_pos_envs = torch.randn(self.num_envs, device=self.device) > self.scale_pos_prob
                pos_action_scale[scale_pos_envs] += scale_noise_pos[scale_pos_envs]
                pos_action_scale = torch.clamp(pos_action_scale, min=0)
                pos_actions = pos_actions * pos_action_scale
            else:
                pos_actions = pos_actions @ torch.diag(pos_action_scale)

        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:

            rot_action_scale = torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device)
            if self.randomize:
                rot_action_scale = rot_action_scale.unsqueeze(0).repeat(self.num_envs, 1)

                scale_noise_rot = 2 * (
                        torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
                        - 0.5
                )
                scale_noise_rot = scale_noise_rot @ torch.diag(
                    torch.tensor(
                        self.cfg_task.randomize.scale_noise_rot,
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                scale_rot_envs = torch.randn(self.num_envs, device=self.device) > self.scale_rot_prob
                rot_action_scale[scale_rot_envs] += scale_noise_rot[scale_rot_envs]
                rot_action_scale = torch.clamp(rot_action_scale, min=0)

                rot_actions = rot_actions * rot_action_scale
            else:
                rot_actions = rot_actions @ torch.diag(rot_action_scale)

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

        if self.cfg_task.env.hand_action and not init_grasp:

            gripper_actions = actions[:, 6:9]
            # gripper_actions[:, :2] = 1.0

            if do_scale:
                gripper_actions = gripper_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.gripper_action_scale, device=self.device))

            idx = [list(self.dof_dict.values()).index('finger_1_1_to_finger_1_2') - 7,
                   list(self.dof_dict.values()).index('finger_1_2_to_finger_1_3') - 7,
                   list(self.dof_dict.values()).index('finger_2_1_to_finger_2_2') - 7,
                   list(self.dof_dict.values()).index('finger_2_2_to_finger_2_3') - 7,
                   list(self.dof_dict.values()).index('base_to_finger_3_2') - 7,
                   list(self.dof_dict.values()).index('finger_3_2_to_finger_3_3') - 7]

            # (delta) Tendon string constraint r_a * q_a = r_p * q_p + r_d * d_p (constant length)
            # self.act_angles *= 0
            self.act_angles = (
                        self.R.permute(0, 2, 1) @ self.gripper_dof_pos[:, idx].unsqueeze(-1) / self.r_act).squeeze(-1)
            self.act_angles += gripper_actions  # add the action
            self.rel_act_angles += gripper_actions
            # self.act_angles = torch.clamp(self.act_angles, min=0 * self.act_angles)

            tendon_forces = self.Q @ self.act_angles.unsqueeze(-1)  # linear mapping between act_angle to tension
            self.act_torque = self.R @ tendon_forces - self.K @ self.gripper_dof_pos[:, idx].unsqueeze(-1)
            self.act_torque = self.act_torque.squeeze(-1)  # sum torque@each joint

            # to_plot = gripper_actions.clone().cpu().numpy()[0, :]
            # if self.frame == 1:
            #     self.force_hist = gripper_actions.clone().cpu().numpy()[0, :]
            # else:
            #     self.force_hist = np.vstack((self.force_hist, to_plot))
            #     plt.plot(self.force_hist[:, :])
            #     plt.pause(0.0001)

        elif init_grasp or not self.cfg_task.env.hand_action:

            self.act_torque = None
            self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos  # directly putting desired gripper_pos

        self.generate_ctrl_signals()

    def _open_gripper(self, env_ids, sim_steps=20):
        """Fully open gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        gripper_dof_pos = 0 * self.gripper_dof_pos.clone()
        # gripper_dof_pos[env_ids,
        #                 list(self.dof_dict.values()).index(
        #                     'base_to_finger_1_1') - 7] = self.cfg_task.env.openhand.base_angle
        # gripper_dof_pos[env_ids,
        #                 list(self.dof_dict.values()).index(
        #                     'base_to_finger_2_1') - 7] = -self.cfg_task.env.openhand.base_angle

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

    def _close_gripper(self, env_ids, percentage=1.0, sim_steps=20):
        """
        Close gripper to a specified percentage of the maximum closing using controller.
        Args:
            env_ids: IDs of the environments.
            percentage: Float representing the percentage of the max closing (0 to 100).
            sim_steps: Number of simulation steps.
        """
        scale = percentage / 1  # Scale percentage to a value between 0 and 1

        gripper_dof_pos = self.gripper_dof_pos.clone()

        def scaled_position(index, open_value, close_value, noise):
            target_value = open_value + scale * (close_value - open_value) + noise
            current_value = self.gripper_dof_pos[env_ids, index]
            target_tensor = torch.tensor(target_value, dtype=current_value.dtype, device=current_value.device)
            return torch.max(current_value, target_tensor)

        gripper_proximal_close_noise = np.random.uniform(0.0, self.cfg_task.env.openhand.proximal_noise, 3)
        gripper_distal_close_noise = np.random.uniform(0., self.cfg_task.env.openhand.distal_noise, 3)

        positions_indices = [
            # ('base_to_finger_1_1', 0.0,
            #  self.cfg_task.env.openhand.base_angle, 0.0),
            # ('base_to_finger_2_1', 0.0,
            #  -self.cfg_task.env.openhand.base_angle, 0.0),
            ('finger_1_1_to_finger_1_2',
             self.cfg_task.env.openhand.proximal_open,
             self.cfg_task.env.openhand.proximal_close, gripper_proximal_close_noise[0]),
            ('finger_2_1_to_finger_2_2',
             self.cfg_task.env.openhand.proximal_open,
             self.cfg_task.env.openhand.proximal_close, gripper_proximal_close_noise[1]),
            ('base_to_finger_3_2',
             self.cfg_task.env.openhand.proximal_open,
             self.cfg_task.env.openhand.proximal_close,
             gripper_proximal_close_noise[2]),
            (
                'finger_1_2_to_finger_1_3', self.cfg_task.env.openhand.distal_open,
                self.cfg_task.env.openhand.distal_close,
                gripper_distal_close_noise[0]),
            (
                'finger_2_2_to_finger_2_3', self.cfg_task.env.openhand.distal_open,
                self.cfg_task.env.openhand.distal_close,
                gripper_distal_close_noise[1]),
            (
                'finger_3_2_to_finger_3_3', self.cfg_task.env.openhand.distal_open,
                self.cfg_task.env.openhand.distal_close,
                gripper_distal_close_noise[2]),
        ]

        for dof, open_value, close_value, noise in positions_indices:
            index = list(self.dof_dict.values()).index(dof) - 7
            gripper_dof_pos[env_ids, index] = scaled_position(index, open_value, close_value, noise)

        # Slowly grasp the plug
        if self.cfg_task.grasp_at_init:
            for i in range(100):
                diff = gripper_dof_pos[env_ids, :] - self.gripper_dof_pos[env_ids, :]

                self.ctrl_target_gripper_dof_pos[env_ids, :] = self.gripper_dof_pos[env_ids, :] + diff * 0.1
                self._move_gripper_to_dof_pos(env_ids=env_ids, gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
                                              sim_steps=1)
        else:
            # diff = gripper_dof_pos[env_ids, :] - self.gripper_dof_pos[env_ids, :]
            # self.ctrl_target_gripper_dof_pos = gripper_dof_pos.clone()
            # self._move_gripper_to_dof_pos(env_ids=env_ids, gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
            #                               sim_steps=1)
            for i in range(1):
                diff = gripper_dof_pos[env_ids, :] - self.gripper_dof_pos[env_ids, :]

                self.ctrl_target_gripper_dof_pos[env_ids, :] = self.gripper_dof_pos[env_ids, :] + diff * 0.3
                self._move_gripper_to_dof_pos(env_ids=env_ids, gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
                                              sim_steps=1)

    def _move_gripper_to_dof_pos(self, env_ids, gripper_dof_pos, sim_steps=20):
        """Move gripper fingers to specified DOF position using controller."""

        delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
                                      device=self.device)  # no arm motion
        self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False, init_grasp=True)

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

        self.obs_dict['priv_info'] = self.obs_dict['states'].clone().to(self.rl_device)
        self.obs_dict['student_obs'] = self.obs_student_buf.clone().to(self.rl_device)

        if self.cfg_task.env.tactile:
            self.obs_dict['tactile'] = self.tactile_queue.clone().to(self.rl_device)
        if self.pcl_cam:
            self.obs_dict['pcl'] = self.pcl_queue.clone().to(self.rl_device)
        if self.external_cam:
            if self.depth_cam:
                self.obs_dict['img'] = self.img_queue.clone().to(self.rl_device)
            if self.seg_cam:
                self.obs_dict['seg'] = self.seg_queue.clone().to(self.rl_device)
        if self.cfg_task.env.compute_contact_gt:
            self.obs_dict['contacts'] = self.contact_queue.clone().to(self.rl_device)

        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def reset(self, reset_at_success=None, reset_at_fails=None):

        if reset_at_success is not None and reset_at_fails is not None:
            print('reset_at_success:', reset_at_success)
            print('reset_at_fails:', reset_at_fails)
            self.reset_flag = reset_at_success and reset_at_fails
            self.cfg_task.reset_at_success = reset_at_success
            self.cfg_task.reset_at_fails = reset_at_fails

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        super().reset()

        self.obs_dict['priv_info'] = self.obs_dict['states'].clone().to(self.rl_device)
        self.obs_dict['student_obs'] = self.obs_student_buf.clone().to(self.rl_device)

        if self.cfg_task.env.tactile:
            self.obs_dict['tactile'] = self.tactile_queue.clone().to(self.rl_device)
        if self.pcl_cam:
            self.obs_dict['pcl'] = self.pcl_queue.clone().to(self.rl_device)
        if self.external_cam:
            if self.depth_cam:
                self.obs_dict['img'] = self.img_queue.clone().to(self.rl_device)
            if self.seg_cam:
                self.obs_dict['seg'] = self.seg_queue.clone().to(self.rl_device)
        if self.cfg_task.env.compute_contact_gt:
            self.obs_dict['contacts'] = self.contact_queue.clone().to(self.rl_device)

        return self.obs_dict
