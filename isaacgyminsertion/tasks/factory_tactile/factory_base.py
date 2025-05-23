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

"""Factory: base class.

Inherits Gym's VecTask class and abstract base class. Inherited by environment classes. Not directly executed.

Configuration defined in FactoryBaseTactile.yaml. Asset info defined in factory_asset_info_kuka_table.yaml.
"""

import hydra
import math
import numpy as np
import os
import sys
import torch

from gym import logger
from isaacgym import gymapi, gymtorch
from isaacgyminsertion.utils import torch_jit_utils as torch_utils
from isaacgyminsertion.tasks.base.vec_task import VecTask
import isaacgyminsertion.tasks.factory_tactile.factory_control as fc
from isaacgyminsertion.tasks.factory_tactile.schema.factory_schema_class_base import FactoryABCBase
from isaacgyminsertion.tasks.factory_tactile.schema.factory_schema_config_base import FactorySchemaConfigBase
from isaacgyminsertion.tasks.factory_tactile.factory_utils import quat2R, RotationTransformer


class FactoryBaseTactile(VecTask, FactoryABCBase):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize VecTask superclass."""

        self.cfg = cfg
        self.cfg['headless'] = headless

        self._get_base_yaml_params()

        if self.cfg_base.mode.export_scene:
            sim_device = 'cpu'

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture,
                         force_render)  # create_sim() is called here

        # defining video recording params, todo: where do we put this?
        self.record_now = False
        self.complete_video_frames = None
        self.video_frames = []
        self.rot_tf = RotationTransformer()
        self.stud_tf = RotationTransformer(from_rep='matrix', to_rep='rotation_6d')

    def _get_base_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_base', node=FactorySchemaConfigBase)

        config_path = 'task/FactoryBaseTactile.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_base = hydra.compose(config_name=config_path)
        self.cfg_base = self.cfg_base['task']  # strip superfluous nesting

        asset_info_path = '../../assets/factory/yaml/factory_asset_info_kuka_table.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_kuka_table = hydra.compose(config_name=asset_info_path)
        self.asset_info_kuka_table = self.asset_info_kuka_table['']['']['']['']['']['']['assets']['factory'][
            'yaml']  # strip superfluous nesting

    def create_sim(self):
        """Set sim and PhysX params. Create sim object, ground plane, and envs."""

        if self.cfg_base.mode.export_scene:
            self.sim_params.use_gpu_pipeline = False

        self.sim = super().create_sim(compute_device=self.device_id,
                                      graphics_device=self.graphics_device_id,
                                      physics_engine=self.physics_engine,
                                      sim_params=self.sim_params)
        self._create_ground_plane()
        self.create_envs()  # defined in subclass
        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params, randomisation_callback=self.randomisation_callback)

    def _create_ground_plane(self):
        """Set ground plane params. Add plane."""

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.0  # default = 0.0
        plane_params.static_friction = 1.0  # default = 1.0
        plane_params.dynamic_friction = 1.0  # default = 1.0
        plane_params.restitution = 0.0  # default = 0.0

        self.gym.add_ground(self.sim, plane_params)

    def import_kuka_assets(self):
        """Set kuka and table asset options. Import assets."""

        urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'factory', 'urdf')
        kuka_file = 'factory_kuka.urdf'  # 'kuka_kuka_digit.urdf'

        kuka_options = gymapi.AssetOptions()
        kuka_options.flip_visual_attachments = False
        kuka_options.fix_base_link = True
        kuka_options.collapse_fixed_joints = False
        kuka_options.thickness = 0.0  # default = 0.02
        kuka_options.density = 1000.0  # default = 1000.0
        kuka_options.armature = 0.01  # default = 0.0
        kuka_options.use_physx_armature = True
        if self.cfg_base.sim.add_damping:
            kuka_options.linear_damping = 1.0  # default = 0.0; increased to improve stability
            kuka_options.max_linear_velocity = 1.0  # default = 1000.0; reduced to prevent CUDA errors
            kuka_options.angular_damping = 5.0  # default = 0.5; increased to improve stability
            kuka_options.max_angular_velocity = 2 * math.pi  # default = 64.0; reduced to prevent CUDA errors
        else:
            kuka_options.linear_damping = 0.0  # default = 0.0
            kuka_options.max_linear_velocity = 1000.0  # default = 1000.0
            kuka_options.angular_damping = 0.5  # default = 0.5
            kuka_options.max_angular_velocity = 64.0  # default = 64.0
        kuka_options.disable_gravity = True
        kuka_options.enable_gyroscopic_forces = True
        kuka_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE  # DOF_MODE_NONE
        kuka_options.use_mesh_materials = True
        if self.cfg_base.mode.export_scene:
            kuka_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        table_options = gymapi.AssetOptions()
        table_options.flip_visual_attachments = False  # default = False
        table_options.fix_base_link = True
        table_options.thickness = 0.0  # default = 0.02
        table_options.density = 1000.0  # default = 1000.0
        table_options.armature = 0.0  # default = 0.0
        table_options.use_physx_armature = True
        table_options.linear_damping = 0.0  # default = 0.0
        table_options.max_linear_velocity = 1000.0  # default = 1000.0
        table_options.angular_damping = 0.0  # default = 0.5
        table_options.max_angular_velocity = 64.0  # default = 64.0
        table_options.disable_gravity = False
        table_options.enable_gyroscopic_forces = True
        table_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        table_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            table_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        kuka_asset = self.gym.load_asset(self.sim, urdf_root, kuka_file, kuka_options)

        table_asset = self.gym.create_box(self.sim, self.asset_info_kuka_table.table_depth,
                                          self.asset_info_kuka_table.table_width, self.cfg_base.env.table_height,
                                          table_options)

        return kuka_asset, table_asset

    def acquire_base_tensors(self):
        """Acquire and wrap tensors. Create views."""

        '''
        The buffer has shape (num_actors, 13).
        State for each actor root contains:
        position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        '''

        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)  # shape = (num_envs * num_actors, 13)
        _body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)  # shape = (num_envs * num_bodies, 13)
        _dof_state = self.gym.acquire_dof_state_tensor(self.sim)  # shape = (num_envs * num_dofs, 2)
        _dof_force = self.gym.acquire_dof_force_tensor(self.sim)  # shape = (num_envs * num_dofs, 1)
        _contact_force = self.gym.acquire_net_contact_force_tensor(self.sim)  # shape = (num_envs * num_bodies, 3)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, 'kuka')  # shape = (num envs, num_bodies, 6, num_dofs)
        _mass_matrix = self.gym.acquire_mass_matrix_tensor(self.sim, 'kuka')  # shape = (num_envs, num_dofs, num_dofs)
        _ft_sensors = self.gym.acquire_force_sensor_tensor(self.sim)

        self.ft_sensors = gymtorch.wrap_tensor(_ft_sensors)
        self.root_state = gymtorch.wrap_tensor(_root_state)
        self.body_state = gymtorch.wrap_tensor(_body_state)
        self.dof_state = gymtorch.wrap_tensor(_dof_state)
        self.dof_force = gymtorch.wrap_tensor(_dof_force)
        self.contact_force = gymtorch.wrap_tensor(_contact_force)
        self.jacobian = gymtorch.wrap_tensor(_jacobian)
        self.mass_matrix = gymtorch.wrap_tensor(_mass_matrix)

        self.root_pos = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 0:3]
        self.root_quat = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 3:7]
        self.root_linvel = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 7:10]
        self.root_angvel = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 10:13]
        self.body_pos = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self.body_quat = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self.body_linvel = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self.body_angvel = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.dof_force_view = self.dof_force.view(self.num_envs, self.num_dofs, 1)[..., 0]
        self.contact_force = self.contact_force.view(self.num_envs, self.num_bodies, 3)[..., 0:3]
        self.ft_sensor_tensor = self.ft_sensors.view(self.num_envs, (len(self.fingertip_handles)) * 6)
        # self.ft_sensor_tensor = self.ft_sensors.view(self.num_envs, 1 * 6)

        self.arm_dof_pos = self.dof_pos[:, 0:7]
        self.arm_dof_vel = self.dof_vel[:, 0:7]
        self.arm_mass_matrix = self.mass_matrix[:, 0:7, 0:7]  # for kuka arm (not gripper)

        self.robot_base_pos = self.body_pos[:, self.robot_base_body_id_env, 0:3]
        self.robot_base_quat = self.body_quat[:, self.robot_base_body_id_env, 0:4]

        self.hand_pos = self.body_pos[:, self.hand_body_id_env, 0:3]
        self.hand_quat = self.body_quat[:, self.hand_body_id_env, 0:4]
        self.hand_linvel = self.body_linvel[:, self.hand_body_id_env, 0:3]
        self.hand_angvel = self.body_angvel[:, self.hand_body_id_env, 0:3]
        self.hand_jacobian = self.jacobian[:, self.hand_body_id_env - self.robot_base_body_id_env - 1, 0:6,
                             0:7]  # minus 1 because base is fixed

        self.left_finger_pos = self.body_pos[:, self.left_finger_body_id_env, 0:3]
        self.left_finger_quat = self.body_quat[:, self.left_finger_body_id_env, 0:4]
        self.left_finger_linvel = self.body_linvel[:, self.left_finger_body_id_env, 0:3]
        self.left_finger_angvel = self.body_angvel[:, self.left_finger_body_id_env, 0:3]
        self.left_finger_jacobian = self.jacobian[:, self.left_finger_body_id_env - self.robot_base_body_id_env - 1,
                                    0:6,
                                    0:7]  # minus 1 because base is fixed

        self.right_finger_pos = self.body_pos[:, self.right_finger_body_id_env, 0:3]
        self.right_finger_quat = self.body_quat[:, self.right_finger_body_id_env, 0:4]
        self.right_finger_linvel = self.body_linvel[:, self.right_finger_body_id_env, 0:3]
        self.right_finger_angvel = self.body_angvel[:, self.right_finger_body_id_env, 0:3]
        self.right_finger_jacobian = self.jacobian[:, self.right_finger_body_id_env - self.robot_base_body_id_env - 1,
                                     0:6,
                                     0:7]  # minus 1 because base is fixed

        self.middle_finger_pos = self.body_pos[:, self.middle_finger_body_id_env, 0:3]
        self.middle_finger_quat = self.body_quat[:, self.middle_finger_body_id_env, 0:4]
        self.middle_finger_linvel = self.body_linvel[:, self.middle_finger_body_id_env, 0:3]
        self.middle_finger_angvel = self.body_angvel[:, self.middle_finger_body_id_env, 0:3]
        self.middle_finger_jacobian = self.jacobian[:, self.middle_finger_body_id_env - self.robot_base_body_id_env - 1,
                                      0:6,
                                      0:7]  # minus 1 because base is fixed

        self.left_finger_force = self.contact_force[:, self.left_finger_body_id_env, 0:3]
        self.right_finger_force = self.contact_force[:, self.right_finger_body_id_env, 0:3]
        self.middle_finger_force = self.contact_force[:, self.middle_finger_body_id_env, 0:3]

        self.gripper_dof_pos = self.dof_pos[:, 7:]
        self.dof_dict = {index: value for index, value in enumerate(self.kuka_joints_names)}

        self.fingertip_centered_pos = self.body_pos[:, self.fingertip_centered_body_id_env, 0:3]
        self.fingertip_centered_quat = self.body_quat[:, self.fingertip_centered_body_id_env, 0:4]
        self.fingertip_centered_linvel = self.body_linvel[:, self.fingertip_centered_body_id_env, 0:3]
        self.fingertip_centered_angvel = self.body_angvel[:, self.fingertip_centered_body_id_env, 0:3]
        self.fingertip_centered_jacobian = self.jacobian[:,
                                           self.fingertip_centered_body_id_env - self.robot_base_body_id_env - 1, 0:6,
                                           0:7]  # minus 1 because base is fixed

        self.fingertip_midpoint_pos = self.fingertip_centered_pos.detach().clone()  # initial value
        self.fingertip_midpoint_quat = self.fingertip_centered_quat  # always equal
        self.fingertip_midpoint_linvel = self.fingertip_centered_linvel.detach().clone()  # initial value

        self.init_plug_pos_cam = torch.zeros((self.num_envs, 3), device=self.device)
        self.shape_one_hot = torch.zeros((self.num_envs, 4), device=self.device)

        # From sum of angular velocities
        # (https://physics.stackexchange.com/questions/547698/understanding-addition-of-angular-velocity),
        # angular velocity of midpoint w.r.t. world is equal to sum of
        # angular velocity of midpoint w.r.t. hand and angular velocity of hand w.r.t. world.
        # Midpoint is in sliding contact (i.e., linear relative motion) with hand;
        # angular velocity of midpoint w.r.t. hand is zero.
        # Thus, angular velocity of midpoint w.r.t. world is equal to angular velocity of hand w.r.t. world.

        self.fingertip_midpoint_angvel = self.fingertip_centered_angvel  # always equal
        self.fingertip_midpoint_jacobian = (
                                                       self.left_finger_jacobian + self.right_finger_jacobian + self.middle_finger_jacobian) * 1 / 3  # approximation

        self.dof_torque = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.fingertip_contact_wrench = torch.zeros((self.num_envs, 6), device=self.device)

        self.ctrl_target_fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.ctrl_target_dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.ctrl_target_gripper_dof_pos = torch.zeros((self.num_envs, self.gripper_dof_pos.shape[-1]),
                                                       device=self.device)
        self.ctrl_target_fingertip_contact_wrench = torch.zeros((self.num_envs, 6), device=self.device)

        self.ctrl_target_fingertip_centered_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ctrl_target_fingertip_centered_quat = torch.zeros((self.num_envs, 4), device=self.device)

        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        self.gripper_normal_quat = (torch.tensor([-1 / 2 ** 0.5, -1 / 2 ** 0.5, 0.0, 0.0],
                                                 device=self.device).unsqueeze(0).repeat(self.num_envs, 1))

        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs,
                                                                                                        1)
        

        # self.rigid_contacts = self.gym.get_rigid_contacts(self.sim)
        # print(self.rigid_contacts)

        # self.test_plot = []

    def refresh_base_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        if self.record_now:
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

        # Privileged
        self.finger_midpoint_pos = (self.left_finger_pos + self.right_finger_pos + self.middle_finger_pos) * (1 / 3)
        self.fingertip_midpoint_pos = fc.translate_along_local_z(pos=self.finger_midpoint_pos,
                                                                 quat=self.gripper_normal_quat,
                                                                 offset=self.asset_info_kuka_table.openhand_finger_length,
                                                                 device=self.device)

        self.fingertip_midpoint_linvel = self.fingertip_centered_linvel + torch.cross(self.fingertip_centered_angvel,
                                                                                      (self.fingertip_midpoint_pos -
                                                                                       self.fingertip_centered_pos),
                                                                                      dim=1)

        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian
                                            + self.middle_finger_jacobian) * (1 / 3)  # approximation

    def parse_controller_spec(self):
        """Parse controller specification into lower-level controller configuration."""

        cfg_ctrl_keys = {'num_envs',
                         'jacobian_type',
                         'gripper_prop_gains',
                         'gripper_deriv_gains',
                         'motor_ctrl_mode',
                         'gain_space',
                         'ik_method',
                         'joint_prop_gains',
                         'joint_deriv_gains',
                         'do_motion_ctrl',
                         'task_prop_gains',
                         'task_deriv_gains',
                         'do_inertial_comp',
                         'motion_ctrl_axes',
                         'do_force_ctrl',
                         'force_ctrl_method',
                         'wrench_prop_gains',
                         'force_ctrl_axes'}

        self.cfg_ctrl = {cfg_ctrl_key: None for cfg_ctrl_key in cfg_ctrl_keys}

        self.cfg_ctrl['num_envs'] = self.num_envs
        self.cfg_ctrl['jacobian_type'] = self.cfg_task.ctrl.all.jacobian_type
        self.cfg_ctrl['gripper_prop_gains'] = torch.tensor(self.cfg_task.ctrl.all.gripper_prop_gains,
                                                           device=self.device).repeat((self.num_envs, 1))
        self.cfg_ctrl['gripper_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.all.gripper_deriv_gains,
                                                            device=self.device).repeat((self.num_envs, 1))

        ctrl_type = self.cfg_task.ctrl.ctrl_type
        if ctrl_type == 'gym_default':
            self.cfg_ctrl['motor_ctrl_mode'] = 'gym'
            self.cfg_ctrl['gain_space'] = 'joint'
            self.cfg_ctrl['ik_method'] = self.cfg_task.ctrl.gym_default.ik_method
            self.cfg_ctrl['joint_prop_gains'] = torch.tensor(self.cfg_task.ctrl.gym_default.joint_prop_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['joint_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.gym_default.joint_deriv_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['gripper_prop_gains'] = torch.tensor(self.cfg_task.ctrl.gym_default.gripper_prop_gains,
                                                               device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['gripper_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.gym_default.gripper_deriv_gains,
                                                                device=self.device).repeat((self.num_envs, 1))
        elif ctrl_type == 'joint_space_ik':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'joint'
            self.cfg_ctrl['ik_method'] = self.cfg_task.ctrl.joint_space_ik.ik_method
            self.cfg_ctrl['joint_prop_gains'] = torch.tensor(self.cfg_task.ctrl.joint_space_ik.joint_prop_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['joint_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.joint_space_ik.joint_deriv_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = False

        elif ctrl_type == 'joint_space_id':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'joint'
            self.cfg_ctrl['ik_method'] = self.cfg_task.ctrl.joint_space_id.ik_method
            self.cfg_ctrl['joint_prop_gains'] = torch.tensor(self.cfg_task.ctrl.joint_space_id.joint_prop_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['joint_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.joint_space_id.joint_deriv_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = True

        elif ctrl_type == 'task_space_impedance':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = True
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(self.cfg_task.ctrl.task_space_impedance.task_prop_gains,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.task_space_impedance.task_deriv_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = False
            self.cfg_ctrl['motion_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.task_space_impedance.motion_ctrl_axes,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_force_ctrl'] = False

        elif ctrl_type == 'operational_space_motion':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = True
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(self.cfg_task.ctrl.operational_space_motion.task_prop_gains,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(
                self.cfg_task.ctrl.operational_space_motion.task_deriv_gains, device=self.device).repeat(
                (self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = True
            self.cfg_ctrl['motion_ctrl_axes'] = torch.tensor(
                self.cfg_task.ctrl.operational_space_motion.motion_ctrl_axes, device=self.device).repeat(
                (self.num_envs, 1))
            self.cfg_ctrl['do_force_ctrl'] = False

        elif ctrl_type == 'open_loop_force':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = False
            self.cfg_ctrl['do_force_ctrl'] = True
            self.cfg_ctrl['force_ctrl_method'] = 'open'
            self.cfg_ctrl['force_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.open_loop_force.force_ctrl_axes,
                                                            device=self.device).repeat((self.num_envs, 1))
        elif ctrl_type == 'closed_loop_force':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = False
            self.cfg_ctrl['do_force_ctrl'] = True
            self.cfg_ctrl['force_ctrl_method'] = 'closed'
            self.cfg_ctrl['wrench_prop_gains'] = torch.tensor(self.cfg_task.ctrl.closed_loop_force.wrench_prop_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['force_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.closed_loop_force.force_ctrl_axes,
                                                            device=self.device).repeat((self.num_envs, 1))
        elif ctrl_type == 'hybrid_force_motion':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = True
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.task_prop_gains,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.task_deriv_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = True
            self.cfg_ctrl['motion_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.motion_ctrl_axes,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_force_ctrl'] = True
            self.cfg_ctrl['force_ctrl_method'] = 'closed'
            self.cfg_ctrl['wrench_prop_gains'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.wrench_prop_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['force_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.force_ctrl_axes,
                                                            device=self.device).repeat((self.num_envs, 1))

        if self.cfg_ctrl['motor_ctrl_mode'] == 'gym':
            prop_gains = torch.cat((self.cfg_ctrl['joint_prop_gains'],
                                    self.cfg_ctrl['gripper_prop_gains']), dim=-1).to('cpu')
            deriv_gains = torch.cat((self.cfg_ctrl['joint_deriv_gains'],
                                     self.cfg_ctrl['gripper_deriv_gains']), dim=-1).to('cpu')
            # No tensor API for getting/setting actor DOF props; thus, loop required
            for env_ptr, kuka_handle, prop_gain, deriv_gain in zip(self.envs, self.kuka_handles, prop_gains,
                                                                   deriv_gains):
                kuka_dof_props = self.gym.get_actor_dof_properties(env_ptr, kuka_handle)
                kuka_dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
                kuka_dof_props['stiffness'] = prop_gain
                kuka_dof_props['damping'] = deriv_gain
                self.gym.set_actor_dof_properties(env_ptr, kuka_handle, kuka_dof_props)
        elif self.cfg_ctrl['motor_ctrl_mode'] == 'manual':
            # No tensor API for getting/setting actor DOF props; thus, loop required
            for env_ptr, kuka_handle in zip(self.envs, self.kuka_handles):
                kuka_dof_props = self.gym.get_actor_dof_properties(env_ptr, kuka_handle)
                kuka_dof_props['driveMode'][:] = gymapi.DOF_MODE_EFFORT
                kuka_dof_props['stiffness'][:] = 0.0  # zero passive stiffness
                kuka_dof_props['damping'][:] = 0.0  # zero passive damping
                self.gym.set_actor_dof_properties(env_ptr, kuka_handle, kuka_dof_props)

    def generate_ctrl_signals(self):
        """Get Jacobian. Set kuka DOF position targets or DOF torques."""

        # Get desired Jacobian
        if self.cfg_ctrl['jacobian_type'] == 'geometric':
            self.fingertip_centered_jacobian_tf = self.fingertip_centered_jacobian

        elif self.cfg_ctrl['jacobian_type'] == 'analytic':
            self.fingertip_centered_jacobian_tf = fc.get_analytic_jacobian(
                fingertip_quat=self.fingertip_centered_quat,
                fingertip_jacobian=self.fingertip_centered_jacobian,
                num_envs=self.num_envs,
                device=self.device)

        # Set PD joint pos target or joint torque
        if self.cfg_ctrl['motor_ctrl_mode'] == 'gym':
            self._set_dof_pos_target()
        elif self.cfg_ctrl['motor_ctrl_mode'] == 'manual':
            self._set_dof_torque()

    def _set_dof_pos_target(self):
        """Set kuka DOF position target to move fingertips towards target pose."""

        self.ctrl_target_dof_pos = fc.compute_dof_pos_target(
            cfg_ctrl=self.cfg_ctrl,
            arm_dof_pos=self.arm_dof_pos,
            fingertip_midpoint_pos=self.fingertip_centered_pos,
            fingertip_midpoint_quat=self.fingertip_centered_quat,
            jacobian=self.fingertip_centered_jacobian_tf,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_centered_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_centered_quat,
            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
            device=self.device)

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.ctrl_target_dof_pos),
                                                        gymtorch.unwrap_tensor(self.kuka_actor_ids_sim),
                                                        len(self.kuka_actor_ids_sim))

    def _set_dof_torque(self):
        """Set kuka DOF torque to move fingertips towards target pose."""

        self.dof_torque = fc.compute_dof_torque(
            cfg_ctrl=self.cfg_ctrl,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            fingertip_midpoint_pos=self.fingertip_centered_pos,
            fingertip_midpoint_quat=self.fingertip_centered_quat,
            fingertip_midpoint_linvel=self.fingertip_centered_linvel,
            fingertip_midpoint_angvel=self.fingertip_centered_angvel,
            left_finger_force=self.left_finger_force,
            right_finger_force=self.right_finger_force,
            jacobian=self.fingertip_centered_jacobian_tf,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_centered_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_centered_quat,
            ctrl_target_fingertip_contact_wrench=self.ctrl_target_fingertip_contact_wrench,
            device=self.device)

        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.dof_torque),
                                                        gymtorch.unwrap_tensor(self.kuka_actor_ids_sim),
                                                        len(self.kuka_actor_ids_sim))

    def print_sdf_warning(self):
        """Generate SDF warning message."""

        logger.warn(
            'Please be patient: SDFs may be generating, which may take a few minutes. Terminating prematurely may result in a corrupted SDF cache.')

    def print_sdf_finish(self):
        """Generate SDF warning message."""

        logger.warn(
            'Finished generating SDFS.')

    def enable_gravity(self, gravity_mag):
        """Enable gravity."""

        sim_params = self.gym.get_sim_params(self.sim)
        sim_params.gravity.z = -9.81
        sim_params.gravity.x = 0.0
        sim_params.gravity.y = 0.0
        self.gym.set_sim_params(self.sim, sim_params)

    def disable_gravity(self):
        """Disable gravity."""

        sim_params = self.gym.get_sim_params(self.sim)
        sim_params.gravity.x = 0.0
        sim_params.gravity.y = 0.0
        sim_params.gravity.z = 0.0
        self.gym.set_sim_params(self.sim, sim_params)
        # print(sim_params.gravity.x, sim_params.gravity.y, sim_params.gravity.z)

    def export_scene(self, label):
        """Export scene to USD."""

        usd_export_options = gymapi.UsdExportOptions()
        usd_export_options.export_physics = False

        usd_exporter = self.gym.create_usd_exporter(usd_export_options)
        self.gym.export_usd_sim(usd_exporter, self.sim, label)
        sys.exit()

    def extract_poses(self):
        """Extract poses of all bodies."""

        if not hasattr(self, 'export_pos'):
            self.export_pos = []
            self.export_rot = []
            self.frame_count = 0

        pos = self.body_pos
        rot = self.body_quat

        self.export_pos.append(pos.cpu().numpy().copy())
        self.export_rot.append(rot.cpu().numpy().copy())
        self.frame_count += 1

        if len(self.export_pos) == self.max_episode_length:
            output_dir = self.__class__.__name__
            save_dir = os.path.join('usd', output_dir)
            os.makedirs(output_dir, exist_ok=True)

            print(f'Exporting poses to {output_dir}...')
            np.save(os.path.join(save_dir, 'body_position.npy'), np.array(self.export_pos))
            np.save(os.path.join(save_dir, 'body_rotation.npy'), np.array(self.export_rot))
            print('Export completed.')
            sys.exit()

    def pose_world_to_robot_base(self, pos, quat, to_rep='matrix'):
        """Convert pose from world frame to robot base frame."""

        # convert
        robot_base_transform_inv = torch_utils.tf_inverse(
            self.robot_base_quat, self.robot_base_pos
        )
        quat_in_robot_base, pos_in_robot_base = torch_utils.tf_combine(
            robot_base_transform_inv[0], robot_base_transform_inv[1], quat, pos
        )

        if to_rep == 'matrix':
            return pos_in_robot_base, quat2R(quat_in_robot_base).reshape(self.num_envs, -1)
        elif to_rep == 'rot6d':
            return pos_in_robot_base, self.rot_tf.forward(quat_in_robot_base)
        else:
            return pos_in_robot_base, quat_in_robot_base

    def pose_world_to_socket_base(self, pos, quat, to_rep='matrix'):
        """Convert pose from world frame to robot base frame."""

        # convert
        robot_base_transform_inv = torch_utils.tf_inverse(
            self.socket_quat, self.socket_pos
        )
        quat_in_robot_base, pos_in_robot_base = torch_utils.tf_combine(
            robot_base_transform_inv[0], robot_base_transform_inv[1], quat, pos
        )

        if to_rep == 'matrix':
            return pos_in_robot_base, quat2R(quat_in_robot_base).reshape(self.num_envs, -1)
        elif to_rep == 'rot6d':
            return pos_in_robot_base, self.rot_tf.forward(quat_in_robot_base)
        else:
            return pos_in_robot_base, quat_in_robot_base

    def pose_world_to_hand_base(self, pos, quat, to_rep=None):
        """Convert pose from world frame to robot base frame."""

        torch_pi = torch.tensor(
            np.pi,
            dtype=torch.float32,
            device=self.device
        )
        rotation_quat_x = torch_utils.quat_from_angle_axis(torch_pi,
                          torch.tensor([1, 0, 0], dtype=torch.float32, device=self.device)).repeat((self.num_envs, 1))
        rotation_quat_z = torch_utils.quat_from_angle_axis(-torch_pi * 0.5,
                          torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)).repeat((self.num_envs, 1))

        q_rotated = torch_utils.quat_mul(rotation_quat_x, self.fingertip_centered_quat.clone())
        q_rotated = torch_utils.quat_mul(rotation_quat_z, q_rotated)

        robot_base_transform_inv = torch_utils.tf_inverse(
            q_rotated, self.fingertip_centered_pos.clone()
        )
        quat_in_robot_base, pos_in_robot_base = torch_utils.tf_combine(
            robot_base_transform_inv[0], robot_base_transform_inv[1], quat, pos
        )

        if to_rep == 'matrix':
            return pos_in_robot_base, quat2R(quat_in_robot_base).reshape(self.num_envs, -1)
        elif to_rep == 'rot6d':
            return pos_in_robot_base, self.rot_tf.forward(quat_in_robot_base)
        else:
            return pos_in_robot_base, quat_in_robot_base