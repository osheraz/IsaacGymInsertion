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

"""Factory: class for insertion env.

** Modified version of the insertion env, including the kuka and the Openhand **

Inherits base class and abstract environment class. Inherited by insertion task class. Not directly executed.

Configuration defined in FactoryEnvInsertionTactile.yaml. Asset info defined in factory_asset_info_insertion.yaml.
"""

import hydra
import numpy as np
import os
import torch

from isaacgym import gymapi, gymtorch
from isaacgyminsertion.tasks.factory_tactile.factory_base import FactoryBaseTactile
from isaacgyminsertion.tasks.factory_tactile.factory_schema_class_env import FactoryABCEnv
from isaacgyminsertion.tasks.factory_tactile.factory_schema_config_env import FactorySchemaConfigEnv
from isaacgyminsertion.allsight.experiments.allsight_render import allsight_renderer
from isaacgyminsertion.allsight.experiments.digit_render import digit_renderer
import isaacgyminsertion.tasks.factory_tactile.factory_control as fc
import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import omegaconf

class ExtrinsicContact:
    def __init__(
            self,
            mesh_obj,
            mesh_socket,
            obj_scale,
            socket_scale,
            socket_pos,
            num_points = 50
    ) -> None:
        self.object_trimesh = trimesh.load(mesh_obj)
        self.object_trimesh = self.object_trimesh.apply_scale(obj_scale)
        T = np.eye(4)
        T[0:3, 0:3] = R.from_euler("xyz", [0, 0, 90], degrees=True).as_matrix()
        self.object_trimesh = self.object_trimesh.apply_transform(T)

        self.socket_trimesh = trimesh.load(mesh_socket)
        self.socket_trimesh = self.socket_trimesh.apply_scale(socket_scale)
        T = np.eye(4)
        T[0:3, -1] = socket_pos
        self.socket_trimesh.apply_transform(T)

        self.socket = o3d.t.geometry.RaycastingScene()
        self.socket.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(self.socket_trimesh.as_open3d)
        )

        import pyvista as pv
        # Create a PyVista mesh from the trimesh object
        # Randomly sample X points from the mesh's point cloud
        mesh = pv.PolyData(self.object_trimesh.vertices, self.object_trimesh.faces)
        sampled_indices = np.random.choice(mesh.n_points, num_points, replace=False)
        sampled_points = mesh.points[sampled_indices]
        self.pointcloud_obj = sampled_points
        self.n_points = sampled_points. shape[0]

        self.gt_extrinsic_contact = torch.zeros((1, self.n_points))

    def _xyzquat_to_tf_numpy(self, position_quat: np.ndarray) -> np.ndarray:
        """
        convert [x, y, z, qx, qy, qz, qw] to 4 x 4 transformation matrices
        """
        # try:
        position_quat = np.atleast_2d(position_quat)  # (N, 7)
        N = position_quat.shape[0]
        T = np.zeros((N, 4, 4))
        T[:, 0:3, 0:3] = R.from_quat(position_quat[:, 3:]).as_matrix()
        T[:, :3, 3] = position_quat[:, :3]
        T[:, 3, 3] = 1
        # except ValueError:
        #     print("Zero quat error!")
        return T.squeeze()

    def reset_extrinsic_contact(self):
        self.gt_extrinsic_contact *= 0
        self.step = 0

    def get_extrinsic_contact(self, obj_pos, obj_quat, socket_pos):
        object_poses = torch.cat((obj_pos, obj_quat), dim=1)
        object_poses = self._xyzquat_to_tf_numpy(object_poses.cpu().numpy())

        object_pc_i = trimesh.points.PointCloud(self.pointcloud_obj.copy())
        object_pc_i.apply_transform(object_poses)
        coords = np.array(object_pc_i.vertices)

        T = np.eye(4)
        T[0:3, -1] = socket_pos.cpu().numpy()
        self.socket_trimesh.apply_transform(T)

        self.socket = o3d.t.geometry.RaycastingScene()
        self.socket.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(self.socket_trimesh.as_open3d)
        )
        d = self.socket.compute_distance(
            o3d.core.Tensor.from_numpy(coords.astype(np.float32))
        ).numpy()

        c = 0.008
        d = d.flatten()
        idx_2 = np.where(d > c)[0]
        d[idx_2] = c
        d = np.clip(d, 0.0, c)

        d = 1.0 - d / c
        d = np.clip(d, 0.0, 1.0)
        d[d > 0.1] = 1.0
        d = d.reshape((1, self.n_points))

        self.gt_extrinsic_contact = torch.tensor(d, dtype=torch.float32)

        return self.gt_extrinsic_contact


class FactoryEnvInsertionTactile(FactoryBaseTactile, FactoryABCEnv):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass. Acquire tensors."""
        self.cfg = cfg
        self._get_env_yaml_params()

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.acquire_base_tensors()  # defined in superclass
        self._acquire_env_tensors()
        self.refresh_base_tensors()  # defined in superclass
        self.refresh_env_tensors()

        # defining video recording params, todo: where do we put this?
        self.record_now = False
        self.record_now_ft = False
        self.complete_video_frames = None
        self.complete_ft_frames = None

        self.video_frames = []
        self.ft_frames = []

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_env', node=FactorySchemaConfigEnv)

        config_path = 'task/FactoryEnvInsertionTactile.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # strip superfluous nesting

        asset_info_path = '../../assets/factory/yaml/factory_asset_info_insertion.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_insertion = hydra.compose(config_name=asset_info_path)
        self.asset_info_insertion = self.asset_info_insertion['']['']['']['']['']['']['assets']['factory'][
            'yaml']  # strip superfluous nesting

        self.cfg_tactile = omegaconf.OmegaConf.create(self.cfg['tactile'])

        self.randomize = self.cfg_env.randomize.domain_randomize
        self.randomization_params = self.cfg_env.randomize.randomization_params

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing, -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        self.print_sdf_warning()
        kuka_asset, table_asset = self.import_kuka_assets()
        plug_assets, socket_assets = self._import_env_assets()
        self._create_actors(lower, upper, num_per_row, kuka_asset, plug_assets, socket_assets, table_asset)
        self.print_sdf_finish()

    def _import_env_assets(self):
        """Set plug and socket asset options. Import assets."""
        self.plug_files, self.socket_files = [], []
        urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'factory', 'urdf')

        plug_options = gymapi.AssetOptions()
        plug_options.flip_visual_attachments = False
        plug_options.fix_base_link = False
        plug_options.thickness = 0.0  # default = 0.02
        plug_options.armature = 0.0  # default = 0.0
        plug_options.use_physx_armature = True
        plug_options.linear_damping = 0.5  # default = 0.0  TODO
        plug_options.max_linear_velocity = 1000.0  # default = 1000.0
        plug_options.angular_damping = 0.5  # default = 0.5
        plug_options.max_angular_velocity = 64.0  # default = 64.0
        plug_options.disable_gravity = False
        plug_options.enable_gyroscopic_forces = True
        plug_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # plug_options.vhacd_enabled = True  # convex decomposition
        plug_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            plug_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        socket_options = gymapi.AssetOptions()
        socket_options.flip_visual_attachments = False
        socket_options.fix_base_link = True
        socket_options.thickness = 0.0  # default = 0.02
        socket_options.armature = 0.0  # default = 0.0
        socket_options.use_physx_armature = True
        socket_options.linear_damping = 0.0  # default = 0.0
        socket_options.max_linear_velocity = 1.0  # default = 1000.0
        socket_options.angular_damping = 0.0  # default = 0.5
        socket_options.max_angular_velocity = 64.0  # default = 64.0
        socket_options.disable_gravity = False
        socket_options.enable_gyroscopic_forces = True
        socket_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        socket_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            socket_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        plug_assets = []
        socket_assets = []
        for subassembly in self.cfg_env.env.desired_subassemblies:
            components = list(self.asset_info_insertion[subassembly])
            plug_file = self.asset_info_insertion[subassembly][components[0]]['urdf_path'] + '.urdf'
            socket_file = self.asset_info_insertion[subassembly][components[1]]['urdf_path'] + '.urdf'
            plug_options.density = self.asset_info_insertion[subassembly][components[0]]['density']
            socket_options.density = self.asset_info_insertion[subassembly][components[1]]['density']
            plug_asset = self.gym.load_asset(self.sim, urdf_root, plug_file, plug_options)
            socket_asset = self.gym.load_asset(self.sim, urdf_root, socket_file, socket_options)
            plug_assets.append(plug_asset)
            socket_assets.append(socket_asset)

            # Save URDF file paths (for loading appropriate meshes during SAPU and SDF-Based Reward calculations)
            self.plug_files.append(os.path.join(urdf_root, plug_file))
            self.socket_files.append(os.path.join(urdf_root, socket_file))

        return plug_assets, socket_assets

    def _create_actors(self, lower, upper, num_per_row, kuka_asset, plug_assets, socket_assets, table_asset):
        """Set initial actor poses. Create actors. Set shape and DOF properties."""

        kuka_pose = gymapi.Transform()
        # kuka_pose.p.x = self.cfg_base.env.kuka_depth
        # kuka_pose.p.y = 0.0
        # kuka_pose.p.z = 0.0
        # kuka_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
        kuka_pose.p.x = 0
        kuka_pose.p.y = 0.0
        kuka_pose.p.z = 0.0
        kuka_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_pose = gymapi.Transform()
        # table_pose.p.x = 0.0
        # table_pose.p.y = 0.0
        # table_pose.p.z = self.cfg_base.env.table_height * 0.5
        # table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        table_pose.p.x = self.cfg_base.env.kuka_depth
        table_pose.p.y = 0.0
        table_pose.p.z = self.cfg_base.env.table_height * 0.5
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.envs_asset = {}
        self.envs = []
        self.kuka_handles = []
        self.plug_handles = []
        self.socket_handles = []
        self.table_handles = []
        self.shape_ids = []

        self.kuka_actor_ids_sim = []  # within-sim indices
        self.plug_actor_ids_sim = []  # within-sim indices
        self.socket_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices

        self.fingertips = ['finger_1_3', 'finger_2_3', 'finger_3_3']  # left, right, bottom. same for all envs
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(kuka_asset, name) for name in self.fingertips]
        self.left_fingertip_handle = []
        self.right_fingertip_handle = []
        self.middle_fingertip_handle = []
        self.tactile_handles = []  # [num_envs , 3]

        actor_count = 0

        self.plug_heights = []
        self.plug_widths = []
        self.plug_depths = []

        self.socket_heights = []
        self.socket_widths = []
        self.socket_depths = []

        self.asset_indices = []
        self.extrinsic_contact_gt = []
        # Create wrist and fingertip force sensors
        sensor_pose = gymapi.Transform()
        # for ft_handle in self.fingertip_handles:
        #     self.gym.create_asset_force_sensor(kuka_asset, ft_handle, sensor_pose)
        wrist_ft_handle = self.gym.find_asset_rigid_body_index(kuka_asset, 'iiwa7_link_7')
        self.gym.create_asset_force_sensor(kuka_asset, wrist_ft_handle, sensor_pose)
        from tqdm import tqdm

        for i in tqdm(range(self.num_envs)):

            # sample random subassemblies
            j = np.random.randint(0, len(self.cfg_env.env.desired_subassemblies))

            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # compute aggregate size
            num_kuka_bodies = self.gym.get_asset_rigid_body_count(kuka_asset)
            num_kuka_shapes = self.gym.get_asset_rigid_shape_count(kuka_asset)
            num_plug_bodies = self.gym.get_asset_rigid_body_count(plug_assets[j])
            num_plug_shapes = self.gym.get_asset_rigid_shape_count(plug_assets[j])
            num_socket_bodies = self.gym.get_asset_rigid_body_count(socket_assets[j])
            num_socket_shapes = self.gym.get_asset_rigid_shape_count(socket_assets[j])
            num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
            num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)

            max_agg_bodies = num_kuka_bodies + num_plug_bodies + num_socket_bodies + num_table_bodies
            max_agg_shapes = num_kuka_shapes + num_plug_shapes + num_socket_shapes + num_table_shapes

            # begin aggregation mode if enabled - this can improve simulation performance
            if self.cfg_env.env.aggregate_mode:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.cfg_env.sim.disable_kuka_collisions:
                kuka_handle = self.gym.create_actor(env_ptr, kuka_asset, kuka_pose, 'kuka', i + self.num_envs, 0, 0)
            else:
                kuka_handle = self.gym.create_actor(env_ptr, kuka_asset, kuka_pose, 'kuka', i, 0, 0)
            self.kuka_actor_ids_sim.append(actor_count)
            actor_count += 1

            subassembly = self.cfg_env.env.desired_subassemblies[j]
            components = list(self.asset_info_insertion[subassembly])

            plug_pose = gymapi.Transform()
            # plug_pose.p.x = 0.0
            # plug_pose.p.y = self.cfg_env.env.plug_lateral_offset
            # plug_pose.p.z = self.cfg_base.env.table_height
            # plug_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            plug_pose.p.x = self.cfg_base.env.kuka_depth
            plug_pose.p.y = self.cfg_env.env.plug_lateral_offset
            plug_pose.p.z = self.cfg_base.env.table_height
            plug_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            plug_handle = self.gym.create_actor(env_ptr, plug_assets[j], plug_pose, 'plug', i, 0, 0)
            self.plug_actor_ids_sim.append(actor_count)
            actor_count += 1

            socket_pose = gymapi.Transform()
            # socket_pose.p.x = 0.0
            # socket_pose.p.y = 0.0
            # socket_pose.p.z = self.cfg_base.env.table_height
            # socket_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            socket_pose.p.x = self.cfg_base.env.kuka_depth
            socket_pose.p.y = 0.0
            socket_pose.p.z = self.cfg_base.env.table_height
            socket_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            socket_handle = self.gym.create_actor(env_ptr, socket_assets[j], socket_pose, 'socket', i, 0, 0)
            self.socket_actor_ids_sim.append(actor_count)
            actor_count += 1

            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, 'table', i, 0, 0)
            self.table_actor_ids_sim.append(actor_count)
            actor_count += 1

            link7_id = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'iiwa7_link_7', gymapi.DOMAIN_ACTOR)
            hand_id = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'gripper_base_link',
                                                           gymapi.DOMAIN_ACTOR)

            left_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'finger_1_3',
                                                                  gymapi.DOMAIN_ACTOR)
            right_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'finger_2_3',
                                                                   gymapi.DOMAIN_ACTOR)
            middle_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'finger_3_3',
                                                                    gymapi.DOMAIN_ACTOR)

            # useful for measuring the friction parameters (privileged information)
            self.left_finger_id = left_finger_id
            self.right_finger_id = right_finger_id
            self.middle_finger_id = middle_finger_id

            self.shape_ids = [link7_id, hand_id, left_finger_id - 1, right_finger_id - 1, middle_finger_id - 1]

            kuka_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, kuka_handle)


            for shape_id in self.shape_ids:
                kuka_shape_props[shape_id].friction = self.cfg_base.env.kuka_friction
                kuka_shape_props[shape_id].rolling_friction = 0.0  # default = 0.0
                kuka_shape_props[shape_id].torsion_friction = 0.0  # default = 0.0
                kuka_shape_props[shape_id].restitution = 0.0  # default = 0.0
                kuka_shape_props[shape_id].compliance = 0.0  # default = 0.0
                kuka_shape_props[shape_id].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, kuka_handle, kuka_shape_props)

            plug_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, plug_handle)
            plug_shape_props[0].friction = self.cfg_env.env.plug_friction # todo osher changed, WAS 1
            plug_shape_props[0].rolling_friction = 0.0  # default = 0.0
            plug_shape_props[0].torsion_friction = 0.0  # default = 0.0
            plug_shape_props[0].restitution = 0.0  # default = 0.0
            plug_shape_props[0].compliance = 0.0  # default = 0.0
            plug_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, plug_handle, plug_shape_props)

            socket_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, socket_handle)
            socket_shape_props[0].friction = self.asset_info_insertion[subassembly][components[1]]['friction']
            socket_shape_props[0].rolling_friction = 0.0  # default = 0.0
            socket_shape_props[0].torsion_friction = 0.0  # default = 0.0
            socket_shape_props[0].restitution = 0.0  # default = 0.0
            socket_shape_props[0].compliance = 0.0  # default = 0.0
            socket_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, socket_handle, socket_shape_props)
            # self.gym.set_actor_scale(env_ptr, socket_handle, 5)


            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            table_shape_props[0].friction = self.cfg_base.env.table_friction
            table_shape_props[0].rolling_friction = 0.0  # default = 0.0
            table_shape_props[0].torsion_friction = 0.0  # default = 0.0
            table_shape_props[0].restitution = 0.0  # default = 0.0
            table_shape_props[0].compliance = 0.0  # default = 0.0
            table_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            self.kuka_num_dofs = self.gym.get_actor_dof_count(env_ptr, kuka_handle)

            self.gym.enable_actor_dof_force_sensors(env_ptr, kuka_handle)

            self.plug_heights.append(self.asset_info_insertion[subassembly][components[0]]['length'])
            self.socket_heights.append(self.asset_info_insertion[subassembly][components[1]]['height'])
            if any('rectangular' in sub for sub in components):
                self.plug_depths.append(self.asset_info_insertion[subassembly][components[0]]['width'])
                self.plug_widths.append(self.asset_info_insertion[subassembly][components[0]]['depth'])
                self.socket_widths.append(self.asset_info_insertion[subassembly][components[1]]['width'])
                self.socket_depths.append(self.asset_info_insertion[subassembly][components[1]]['depth'])
            else:
                self.plug_widths.append(self.asset_info_insertion[subassembly][components[0]]['diameter'])
                self.socket_widths.append(self.asset_info_insertion[subassembly][components[1]]['diameter'])

            self.asset_indices.append(j)
            self.envs.append(env_ptr)
            self.kuka_handles.append(kuka_handle)
            self.plug_handles.append(plug_handle)
            self.socket_handles.append(socket_handle)
            self.table_handles.append(table_handle)

            # creating a camera on environment 0
            if (i == 0) and self.cfg['env']['record_video']:
                self.camera_props = gymapi.CameraProperties()
                self.camera_props.width = 1280
                self.camera_props.height = 720
                self.rendering_camera1 = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                print('CAMERA:', self.rendering_camera1)
                self.gym.set_camera_location(self.rendering_camera1, env_ptr, gymapi.Vec3(1.5, 1, 3.0),
                                             gymapi.Vec3(0, 0, 0))
                
                self.rendering_camera2 = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                print('CAMERA:', self.rendering_camera2)
                self.gym.set_camera_location(self.rendering_camera2, env_ptr, gymapi.Vec3(1.5, 1, 3.0),
                                             gymapi.Vec3(0, 0, 0))

            # add Tactile modules for the tips
            self.envs_asset[i] = {'subassembly': subassembly, 'components': components}
            plug_file = self.asset_info_insertion[subassembly][components[0]]['urdf_path']
            plug_file += '_subdiv_3x.obj' if 'rectangular' in plug_file else '.obj'
            socket_file = self.asset_info_insertion[subassembly][components[0]]['urdf_path']
            socket_file += '_subdiv_3x.obj' if 'rectangular' in plug_file else '.obj'

            mesh_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'factory', 'mesh',
                                     'factory_insertion')

            if self.cfg['env']['tactile']:
                self.tactile_handles.append([allsight_renderer(self.cfg_tactile,
                                                               os.path.join(mesh_root, plug_file), randomize=True,
                                                               finger_idx=i) for i in range(len(self.fingertips))])
            if self.cfg['env']['compute_contact_gt']:
                socket_pos = [0, 0, self.cfg_base.env.table_height]
                self.extrinsic_contact_gt.append(ExtrinsicContact(mesh_obj=os.path.join(mesh_root, plug_file),
                                                                  mesh_socket=os.path.join(mesh_root, socket_file),
                                                                  obj_scale=1.0,
                                                                  socket_scale=1.0,
                                                                  socket_pos=socket_pos
                                                                  )
                                                 )

            if self.cfg_env.env.aggregate_mode:
                self.gym.end_aggregate(env_ptr)

        # Get indices
        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.kuka_actor_ids_sim = torch.tensor(self.kuka_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.plug_actor_ids_sim = torch.tensor(self.plug_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.socket_actor_ids_sim = torch.tensor(self.socket_actor_ids_sim, dtype=torch.int32, device=self.device)

        # For extracting root pos/quat
        self.plug_actor_id_env = self.gym.find_actor_index(env_ptr, 'plug', gymapi.DOMAIN_ENV)
        self.socket_actor_id_env = self.gym.find_actor_index(env_ptr, 'socket', gymapi.DOMAIN_ENV)

        # For extracting body pos/quat, force, and Jacobian
        self.robot_base_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, "iiwa7_link_0",
                                                                           gymapi.DOMAIN_ENV)
        self.plug_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, plug_handle, 'plug', gymapi.DOMAIN_ENV)
        self.socket_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, socket_handle, 'socket',
                                                                       gymapi.DOMAIN_ENV)
        self.hand_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'gripper_base_link',
                                                                     gymapi.DOMAIN_ENV)
        self.wrist_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'dummy_ft_link',
                                                                      gymapi.DOMAIN_ENV)

        self.left_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'finger_1_3',
                                                                            gymapi.DOMAIN_ENV)
        self.right_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'finger_2_3',
                                                                             gymapi.DOMAIN_ENV)
        self.middle_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'finger_3_3',
                                                                              gymapi.DOMAIN_ENV)
        # Robot motion will be w.r.t this tf.
        self.fingertip_centered_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle,
                                                                                   'kuka_fingertip_centered',
                                                                                   gymapi.DOMAIN_ENV)

        self.kuka_joints_names = self.gym.get_asset_dof_names(kuka_asset)

        # For computing body COM pos
        self.plug_heights = torch.tensor(self.plug_heights, device=self.device).unsqueeze(-1)
        self.socket_heights = torch.tensor(self.socket_heights, device=self.device).unsqueeze(-1)

        # For setting initial state

        # For defining success or failure
        self.plug_widths = torch.tensor(self.plug_widths, device=self.device).unsqueeze(-1)

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""

        self.plug_pos = self.root_pos[:, self.plug_actor_id_env, 0:3]
        self.plug_quat = self.root_quat[:, self.plug_actor_id_env, 0:4]
        self.plug_linvel = self.root_linvel[:, self.plug_actor_id_env, 0:3]
        self.plug_angvel = self.root_angvel[:, self.plug_actor_id_env, 0:3]

        self.socket_pos = self.root_pos[:, self.socket_actor_id_env, 0:3]
        self.socket_quat = self.root_quat[:, self.socket_actor_id_env, 0:4]

        # TODO: Define socket height and plug height params in asset info YAML.
        self.plug_com_pos = fc.translate_along_local_z(pos=self.plug_pos,
                                                       quat=self.plug_quat,
                                                       offset=self.socket_heights + self.plug_heights * 1.0,
                                                       device=self.device)

        self.above_socket_pos = fc.translate_along_local_z(pos=self.socket_pos,
                                                           quat=self.socket_quat,
                                                           offset=self.socket_heights + self.plug_heights,
                                                           device=self.device)

        self.plug_com_quat = self.plug_quat  # always equal
        self.plug_com_linvel = self.plug_linvel + torch.cross(self.plug_angvel,
                                                              (self.plug_com_pos - self.plug_pos),
                                                              dim=1)
        self.plug_com_angvel = self.plug_angvel  # always equal

        self.socket_contact_force = self.contact_force[:, self.socket_actor_id_env, :3]

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.

        self.plug_com_pos = fc.translate_along_local_z(pos=self.plug_pos,
                                                       quat=self.plug_quat,
                                                       offset=self.plug_heights * 0.5,
                                                       device=self.device)

        self.plug_com_linvel = self.plug_linvel + torch.cross(self.plug_angvel,
                                                              (self.plug_com_pos - self.plug_pos),
                                                              dim=1)

        self.above_socket_pos = fc.translate_along_local_z(pos=self.socket_pos,
                                                           quat=self.socket_quat,
                                                           offset=self.socket_heights + self.plug_heights * 1.0,
                                                           device=self.device)

        self.socket_tip = fc.translate_along_local_z(pos=self.socket_pos,
                                                     quat=self.socket_quat,
                                                     offset=self.socket_heights,
                                                     device=self.device)

    ### start code for logging videos while training ###
    # record camera (does not matter if headless)
    def _render_headless(self):
        # if self.record_now:
        #     # print('should record soon..')
        #     if self.complete_video_frames is not None:
        #         print(len(self.complete_video_frames))
        if self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0:
            # bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
            # bx, by, bz = -0.0012, -0.0093, 0.4335  # self.root_pos[0, self.plug_actor_id_env, 0], self.root_pos[0, self.plug_actor_id_env, 1], self.root_pos[0, self.plug_actor_id_env, 2]
            # bx, by, bz = 0, 1, 0
            bx, by, bz = self.init_plug_pos_cam[0, 0], self.init_plug_pos_cam[0, 1], self.init_plug_pos_cam[0, 2]

            self.gym.set_camera_location(self.rendering_camera1, self.envs[0], gymapi.Vec3(bx - 0.1, by - 0.1, bz+0.1),
                                         gymapi.Vec3(bx, by, bz))
            self.video_frame1 = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera1,
                                                         gymapi.IMAGE_COLOR)
            self.video_frame1 = self.video_frame1.reshape((self.camera_props.height, self.camera_props.width, 4))

            self.gym.set_camera_location(self.rendering_camera2, self.envs[0], gymapi.Vec3(bx - 0.1, by + 0.1, bz+0.1),
                                         gymapi.Vec3(bx, by, bz))
            self.video_frame2 = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera2,
                                                         gymapi.IMAGE_COLOR)
            self.video_frame2 = self.video_frame2.reshape((self.camera_props.height, self.camera_props.width, 4))

            # print('video frame shape', self.video_frame.shape)
            self.video_frames.append(np.concatenate((self.video_frame1, self.video_frame2), axis=1))

        if self.record_now_ft and self.complete_ft_frames is not None and len(self.complete_ft_frames) == 0:
            self.ft_frames.append(0.1 * self.ft_sensor_tensor[:1].cpu().numpy().squeeze())

    def start_recording(self):
        self.complete_video_frames = None
        self.record_now = True

    def start_recording_ft(self):
        self.complete_ft_frames = None
        self.record_now_ft = True

    def pause_recording(self):
        self.complete_video_frames = []
        self.video_frames = []
        self.record_now = False

    def pause_recording_ft(self):
        self.complete_ft_frames = []
        self.ft_frames = []
        self.record_now_ft = False

    def get_complete_frames(self):
        if self.complete_video_frames is None:
            return []
        return self.complete_video_frames

    def get_ft_frames(self):
        if self.complete_ft_frames is None:
            return []
        return self.complete_ft_frames

    # def start_recording_eval(self):
    #     self.complete_video_frames_eval = None
    #     self.record_eval_now = True

    # def pause_recording_eval(self):
    #     self.complete_video_frames_eval = []
    #     self.video_frames_eval = []
    #     self.record_eval_now = False

    # def get_complete_frames_eval(self):
    #     if self.complete_video_frames_eval is None:
    #         return []
    #     return self.complete_video_frames_eval
    ### end code for logging videos while training ###
