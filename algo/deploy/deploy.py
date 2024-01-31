##### !/usr/bin/env /home/osher/Desktop/isaacgym/venv/bin/python3
# --------------------------------------------------------
# now its our turn.
# https://arxiv.org/abs/todo
# Copyright (c) 2023 Osher & Co.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import rospy
# from algo.models.models import ActorCritic
from algo.models.models_split import ActorCriticSplit as ActorCritic
from algo.models.running_mean_std import RunningMeanStd
import torch
import os
import hydra
import cv2
from isaacgyminsertion.utils import torch_jit_utils
import isaacgyminsertion.tasks.factory_tactile.factory_control as fc
from isaacgyminsertion.tasks.factory_tactile.factory_utils import quat2R
import time
import numpy as np
import omegaconf
import zmq
import json
import trimesh
import open3d as o3d
from matplotlib import pyplot as plt
from tf.transformations import quaternion_matrix, identity_matrix, quaternion_from_matrix
from tqdm import tqdm
from iiwa_msgs.srv import SetPathParameters
from hand_control.srv import TargetAngles

from scipy.spatial.transform import Rotation as R
# import extrinsic contact here, this way we can do privileged information inference in real world.

np.random.seed(3)

class ExtrinsicContact:
    def __init__(
            self,
            mesh_plug,
            mesh_socket,
            socket_pos,
            num_envs=1,
            num_points=50,
            device='cuda:0'
    ) -> None:

        self.object_trimesh = trimesh.load(mesh_plug)
        self.reset_object_trimesh = self.object_trimesh.copy()
        self.socket_trimesh = trimesh.load(mesh_socket)
        
        # T = np.eye(4)
        # T[0:3, -1] = socket_pos
        self.socket_trimesh.apply_transform(socket_pos)
       
        self.socket_pos = socket_pos
        self.socket = o3d.t.geometry.RaycastingScene()
        self.socket.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(self.socket_trimesh.as_open3d)
        )
        self.socket_pcl = trimesh.sample.sample_surface_even(self.socket_trimesh, num_points, seed=42)[0]

        self.pointcloud_obj = trimesh.sample.sample_surface(self.object_trimesh, num_points, seed=42)[0]
        self.object_pc = trimesh.points.PointCloud(self.pointcloud_obj.copy())

        self.n_points = num_points
        self.constant_socket = False
        self.fig = plt.figure(figsize=(10, 10))
        self.ax1 = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.ax2 = self.fig.add_subplot(1, 2, 2, projection='3d')
        self.num_envs = num_envs
        self.device = device
        self.dec = torch.zeros((num_envs, self.n_points)).to(device)

    def _xyzquat_to_tf_numpy(self, position_quat: np.ndarray) -> np.ndarray:
        """
        convert [x, y, z, qx, qy, qz, qw] to 4 x 4 transformation matrices
        """
        position_quat = np.atleast_2d(position_quat)  # (N, 7)
        N = position_quat.shape[0]
        T = np.zeros((N, 4, 4))
        T[:, 0:3, 0:3] = R.from_quat(position_quat[:, 3:]).as_matrix()
        T[:, :3, 3] = position_quat[:, :3]
        T[:, 3, 3] = 1
        return T.squeeze()

    def reset_extrinsic_contact(self):
        self.gt_extrinsic_contact *= 0
        self.step = 0

    def apply_transform(self, poses, pc_vertices):
        count, dim = pc_vertices.shape
        pc_vertices_wtrans = np.column_stack((pc_vertices, np.ones(count)))
        stack = np.repeat(pc_vertices_wtrans[np.newaxis, ...], poses.shape[0], axis=0)
        transformed = np.matmul(poses, np.transpose(stack, (0, 2, 1)))
        transformed = np.transpose(transformed, (0, 2, 1))[..., :3]
        return transformed

    def reset_socket_pos(self, socket_pos):
        self.socket_pos = socket_pos
        self.socket = o3d.t.geometry.RaycastingScene()
        self.socket.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(self.socket_trimesh.as_open3d)
        )
        self.socket_pcl = trimesh.sample.sample_surface_even(self.socket_trimesh, self.n_points, seed=42)[0]

    def get_extrinsic_contact(self, object_pos, threshold=0.005, display=False, dec=None):
        
        self.object_trimesh = self.reset_object_trimesh.copy()
        self.object_trimesh.apply_transform(object_pos)
        query_points = trimesh.points.PointCloud(trimesh.sample.sample_surface(self.object_trimesh, self.n_points, seed=42)[0]).vertices
        d = self.socket.compute_distance(o3d.core.Tensor.from_numpy(query_points.astype(np.float32))).numpy()
        intersecting_indices = d < threshold
        contacts = query_points[intersecting_indices]

        if display:
            self.ax1.clear()
            self.ax1.plot(self.socket_pcl[:, 0], self.socket_pcl[:, 1], self.socket_pcl[:, 2], 'yo')
            self.ax1.plot(query_points[:, 0], query_points[:, 1], query_points[:, 2], 'ko')
            self.ax1.set_xlabel('X')
            self.ax1.set_ylabel('Y')
            intersecting_indices = d < threshold
            contacts = query_points[intersecting_indices]
            self.ax1.plot(contacts[:, 0], contacts[:, 1], contacts[:, 2], 'ro')
            # for c in contacts: 
            #     if np.linalg.norm(c, axis=0):
            #         self.ax.plot(c[0], c[1], c[2], 'ro')

            if dec is not None:
                self.ax2.clear()
                self.ax2.plot(self.socket_pcl[:, 0], self.socket_pcl[:, 1], self.socket_pcl[:, 2], 'yo')
                self.ax2.plot(query_points[:, 0], query_points[:, 1], query_points[:, 2], 'ko')
                intersecting_indices = (torch.sigmoid(dec) > 0.5).reshape(-1).numpy()
                contacts = query_points[intersecting_indices]
                for c in contacts:
                    if np.linalg.norm(c, axis=0):
                        self.ax2.plot(c[0], c[1], c[2], 'ro')
                # self.ax2.plot(contacts[:, 0], contacts[:, 1], contacts[:, 2], 'ro')
                self.ax2.set_xlabel('X')
                self.ax2.set_ylabel('Y')
            plt.pause(0.00001)
            self.fig.savefig('/home/robotics/dhruv/object_tracking/contact.png')
        else:
            plt.close(self.fig)

        d = d.flatten()
        idx_2 = np.where(d > threshold)[0]
        d[idx_2] = threshold
        d = np.clip(d, 0.0, threshold)

        d = 1.0 - d / threshold
        d = np.clip(d, 0.0, 1.0)
        d[d > 0.1] = 1.0

        return np.array(d).copy(), query_points, contacts, self.socket_pcl



class HardwarePlayer(object):
    def __init__(self, output_dir, full_config):

        self.deploy_config = full_config.deploy
        self.full_config = full_config
        self.pos_scale_deploy = self.deploy_config.rl.pos_action_scale
        self.rot_scale_deploy = self.deploy_config.rl.rot_action_scale

        self.pos_scale = full_config.task.rl.pos_action_scale
        self.rot_scale = full_config.task.rl.rot_action_scale

        self.device = 'cuda:0' # full_config["rl_device"]

        # ---- build environment ----
        self.obs_shape = (self.deploy_config.env.numObservations,) # 86
        self.obs_stud_shape = (self.deploy_config.env.numObsStudent,)

        self.num_actions = self.deploy_config.env.numActions
        self.num_targets = self.deploy_config.env.numTargets
        self.num_contact_points = self.deploy_config.env.num_points

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
            'extrin_adapt': self.extrin_adapt,
            'priv_info_dim': self.priv_info_dim,
            'priv_info': self.priv_info,
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
            "merge_units": self.deploy_config.network.merge_mlp.units,
            "obs_units": self.deploy_config.network.obs_mlp.units,
            "num_contact_points": self.num_contact_points,
            "gt_contacts_info": False,
            "contacts_mlp_units": 0,
            'shared_parameters': False
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

        # tactile_info_path = '../allsight/experiments/conf/test.yaml'  # relative to Gym's Hydra search path (cfg dir)
        # self.cfg_tactile = hydra.compose(config_name=tactile_info_path)['']['']['']['allsight']['experiments']['conf']
        self.cfg_tactile = full_config.task.tactile

        asset_info_path = '../../assets/factory/yaml/factory_asset_info_insertion.yaml'
        self.asset_info_insertion = hydra.compose(config_name=asset_info_path)
        self.asset_info_insertion = self.asset_info_insertion['']['']['']['']['']['']['assets']['factory']['yaml']

        self.force_torque = []

        self.plug_pose = np.eye(4)
        self.plug_pose_world = np.eye(4)
        self.socket_pose_world = np.eye(4)

        self.prev_palm_plug_pose_cam = None
        self.palm_plug_pose_cam = np.eye(4)

        self.plug_pose_diff = np.eye(4)

        

        # load initial socket pose
        with open(f"/home/robotics/dhruv/object_tracking/test_data/poses/socket.txt", "r") as f:
            self.socket_pose = np.array(json.load(f)).reshape(4, 4)

        # load initial socket pose
        with open(f"/home/robotics/dhruv/object_tracking/test_data/poses/robot_base.txt", "r") as f:
            self.robot_base_pose = np.array(json.load(f)).reshape(4, 4)
        self.R_camera_world = np.linalg.inv(self.robot_base_pose.copy())
        self.R_world_camera = self.robot_base_pose.copy()
        # setup extrinsic contact
        
        
        self.extrinsic_contact = None
        self.decoded_contact = None
        # check if force torque sensor is calibrated
        self.ft_calibrated = False

        rospy.wait_for_service('/iiwa/configuration/pathParameters')
        self.path_parameters_srv = rospy.ServiceProxy('/iiwa/configuration/pathParameters', SetPathParameters)
        self.move_gripper_srv = rospy.ServiceProxy('/MoveGripper', TargetAngles)
        self.squeeze_in_srv = rospy.ServiceProxy('/SqueezeIn', TargetAngles)
        

    def restore(self, fn):
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        # self.running_mean_std_stud.load_state_dict(checkpoint['running_mean_std_stud'])
        self.model.load_state_dict(checkpoint['model'])
        self.set_eval()

    def set_eval(self):
        self.model.eval()
        self.running_mean_std.eval()
        # self.running_mean_std_stud.eval()

    def _initialize_grasp_poses(self, gp='yellow_round_peg_2in'):

        self.initial_grasp_poses = np.load(f'initial_grasp_data/{gp}.npz')

        self.total_init_poses = self.initial_grasp_poses['socket_pos'].shape[0]
        self.init_dof_pos = torch.zeros((self.total_init_poses, 15))
        self.init_dof_pos = self.init_dof_pos[:, :7]
        dof_pos = self.initial_grasp_poses['dof_pos'][:, :7]

        from tqdm import tqdm
        print("Loading init grasping poses for:", gp)
        for i in tqdm(range(self.total_init_poses)):
            self.init_dof_pos[i] = torch.from_numpy(dof_pos[i])

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

        # Gripper pointing down w.r.t the world frame
        gripper_goal_euler = torch.tensor(self.deploy_config.env.fingertip_midpoint_rot_initial,
                                          device=self.device).unsqueeze(0)

        self.gripper_goal_quat = torch_jit_utils.quat_from_euler_xyz(gripper_goal_euler[:, 0],
                                                                     gripper_goal_euler[:, 1],
                                                                     gripper_goal_euler[:, 2])

        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0)

        self.plug_grasp_pos_local = self.plug_height * 0.5 * torch.tensor([0.0, 0.0, 1.0],
                                                                          device=self.device).unsqueeze(0)
        self.plug_grasp_quat_local = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0)

        self.plug_tip_pos_local = self.plug_height * torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0)
        self.socket_tip_pos_local = self.socket_height * torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0)

        self.actions = torch.zeros((1, self.num_actions), device=self.device)
        self.targets = torch.zeros((1, self.deploy_config.env.numTargets), device=self.device)
        self.contacts = torch.zeros((1, self.num_contact_points), device=self.device)
        self.prev_targets = torch.zeros((1, self.deploy_config.env.numTargets), dtype=torch.float, device=self.device)
        # self.tactile_imgs = torch.zeros((1, self.deploy_config.env.obs_seq_length, 1, 64, 64), dtype=torch.float,
                                        # device=self.device)

        # Keep track of history
        self.arm_joint_queue = torch.zeros((1, self.deploy_config.env.obs_seq_length, 7), dtype=torch.float,
                                           device=self.device)
        self.arm_vel_queue = torch.zeros((1, self.deploy_config.env.obs_seq_length, 7), dtype=torch.float,
                                         device=self.device)
        self.actions_queue = torch.zeros((1, self.deploy_config.env.obs_seq_length, self.num_actions),
                                         dtype=torch.float, device=self.device)
        self.targets_queue = torch.zeros((1, self.deploy_config.env.obs_seq_length, self.num_targets),
                                         dtype=torch.float, device=self.device)
        self.eef_queue = torch.zeros((1, self.deploy_config.env.obs_seq_length, 12),
                                     dtype=torch.float, device=self.device)
        self.goal_noisy_queue = torch.zeros((1, self.deploy_config.env.obs_seq_length, 12),
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
        self.eef_queue_student = torch.zeros((1, self.deploy_config.env.stud_obs_seq_length, 12),
                                             dtype=torch.float, device=self.device)
        self.goal_noisy_queue_student = torch.zeros((1, self.deploy_config.env.stud_obs_seq_length, 12),
                                                    dtype=torch.float, device=self.device)

        self.num_channels = 1 # self.cfg_tactile.decoder.num_channels
        self.width = self.cfg_tactile.decoder.width // 2 if self.cfg_tactile.half_image else self.cfg_tactile.decoder.width
        self.height = self.cfg_tactile.decoder.height

        self.d = []
        self.qp = []
        self.socket_pcl = None

        # tactile buffers
        self.tactile_imgs = torch.zeros(
            (1, 3,  # left, right, bottom
             self.width, self.height, self.num_channels),
            device=self.device,
            dtype=torch.float,
        )
        # Way too big tensor.
        self.tactile_queue = torch.zeros(
            (1, self.tactile_seq_length, 3,  # left, right, bottom
             self.width, self.height, self.num_channels),
            device=self.device,
            dtype=torch.float,
        )

        self.ft_queue = torch.zeros((1, self.ft_seq_length, 6), device=self.device, dtype=torch.float)

        self.ee_pos = torch.zeros((1, 7), device=self.device, dtype=torch.float)

        self.obs_buf = torch.zeros((1, self.obs_shape[0]), device=self.device, dtype=torch.float)
        self.obs_student_buf = torch.zeros((1, self.obs_stud_shape[0]), device=self.device, dtype=torch.float)
        self.socket_top_pos_world_list = []
        self.socket_top_pos_world = torch.zeros((1, 3), device=self.device, dtype=torch.float)
        self.episode_length = torch.zeros((1, 1), device=self.device, dtype=torch.float)
        self.max_episode_length = 1999

        self.done = torch.zeros((1, 1), device=self.device, dtype=torch.bool)

        self.ft_data = torch.zeros((1, 6), device=self.device, dtype=torch.float)

    def _set_socket_pose(self, pos):

        self.socket_pos = torch.tensor(pos, device=self.device).unsqueeze(0)

    def _set_plug_pose(self, pos):
        self.plug_pos = torch.tensor(pos, device=self.device).unsqueeze(0)

    def _update_socket_pose(self):
        """ Update the noisy estimation of the socket pos"""

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

        # Add observation noise to socket rot
        socket_rot_euler = torch.zeros(
            (1, 3), dtype=torch.float32, device=self.device
        )
        socket_obs_rot_noise = 2 * (
                torch.rand((1, 3), dtype=torch.float32, device=self.device)
                - 0.5
        )
        socket_obs_rot_noise = socket_obs_rot_noise @ torch.diag(
            torch.tensor(
                self.deploy_config.env.socket_rot_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        socket_obs_rot_euler = socket_rot_euler + socket_obs_rot_noise
        self.noisy_socket_quat = torch_jit_utils.quat_from_euler_xyz_deploy(
            socket_obs_rot_euler[:, 0],
            socket_obs_rot_euler[:, 1],
            socket_obs_rot_euler[:, 2],
        )

        # Compute observation noise on socket
        (
            self.noisy_gripper_goal_quat,
            self.noisy_gripper_goal_pos,
        ) = torch_jit_utils.tf_combine_deploy(
            self.noisy_socket_quat,
            self.noisy_socket_pos,
            self.gripper_goal_quat,
            self.socket_tip_pos_local,
        )

    def compute_observations(self, dec=None, display_image=False):
        
        try:
            # st = time()
            obses = self.env.get_obs()
            # arm_joints = obses['joints']
            self.ee_pos[:, :]  = torch.tensor(obses['ee_pose'], device=self.device, dtype=torch.float)

            ee_pose = self.pose_world_to_robot_base(self.ee_pos.clone())
           
            d = obses['extrinsic_contact']
            self.contacts[0, :] = torch.tensor(d).to(self.device)

            self.plug_pose_world = obses['plug_pose_world']
            self.socket_pose_world = obses['socket_pose_world']

            # try:
            self.ft_data[0, :] = torch.tensor(obses['ft']).to(torch.float).to(self.device)
                # print(self.ft_data.shape)
            # [:]
            # except Exception as e:
                # print(e)
                # print(obses['ft'])

            # self.palm_plug_pose_cam = self.env.get_palm_plug_pose_camera()
            # if self.prev_palm_plug_pose_cam is None:
            #     self.prev_palm_plug_pose_cam = self.palm_plug_pose_cam.copy()
            
            # self.plug_pose_diff[...] = self.palm_plug_pose_cam @ np.linalg.inv(self.prev_palm_plug_pose_cam)
            # self.prev_palm_plug_pose_cam = self.palm_plug_pose_cam.copy()

            # print(R.from_matrix(self.plug_pose_diff[:3, :3]).as_euler('xyz', degrees=True))
            # ee_pose = torch.tensor(ee_pose, device=self.device, dtype=torch.float)
           
            obs = torch.cat([ee_pose, self.actions], dim=-1)
            self.obs_buf[...] = obs

            left, right, bottom = obses['frames']
            # Cutting by half
            if self.cfg_tactile.half_image:
                w = left.shape[0]
                left = left[:w // 2, :, :]
                right = right[:w // 2, :, :]
                bottom = bottom[:w // 2, :, :]


            # Resizing to decoder size
            left = cv2.resize(left, (self.height, self.width), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (self.height, self.width), interpolation=cv2.INTER_AREA)
            bottom = cv2.resize(bottom, (self.height, self.width), interpolation=cv2.INTER_AREA)

            self.tactile_imgs[0, 0] = torch_jit_utils.gray_transform(
                cv2.cvtColor(left.astype('float32'), cv2.COLOR_BGR2GRAY)).to(self.device).permute(1, 2, 0)
            self.tactile_imgs[0, 1] = torch_jit_utils.gray_transform(
                cv2.cvtColor(right.astype('float32'), cv2.COLOR_BGR2GRAY)).to(self.device).permute(1, 2, 0)
            self.tactile_imgs[0, 2] = torch_jit_utils.gray_transform(
                cv2.cvtColor(bottom.astype('float32'), cv2.COLOR_BGR2GRAY)).to(self.device).permute(1, 2, 0)

            # if display_image:
            #     cv2.imshow("Hand View\tLeft\tRight\tMiddle", np.concatenate((left, right, bottom), axis=1))
            #     cv2.waitKey(1)

            # self.tactile_imgs[0, 0, ...] = torch.tensor(left, device=self.device, dtype=torch.float)
            # self.tactile_imgs[0, 1, ...] = torch.tensor(right, device=self.device, dtype=torch.float)
            # self.tactile_imgs[0, 2, ...] = torch.tensor(bottom, device=self.device, dtype=torch.float)

            
            # some-like taking a new socket pose measurement
            # self._update_socket_pose()

            # st = time()
            # euler_angs = R.from_matrix(plug_pose_world[:3, :3]).as_euler('xyz', degrees=True)
            # euler_angs[2] = 0.
            # # print(euler_angs)
            # plug_pose_world_no_rot = plug_pose_world.copy()
            # plug_pose_world_no_rot[:3, :3] = R.from_euler('xyz', euler_angs, degrees=True).as_matrix()
            # # print(plug_pose_world_no_rot, plug_pose_world)
            # plug_pose_camera_no_rot = np.matmul(self.R_world_camera, plug_pose_world)

            # d, qp, contacts, socket = self.extrinsic_contact.get_extrinsic_contact(plug_pose_camera_no_rot, dec=dec)
            # self.contacts[...] = torch.from_numpy(d).reshape(-1, self.num_contact_points)
            # print("Extrinsic contact time: ", (time() - st))
            # # self.qp.append(qp)
            # # self.d.append(d)
            # # if self.socket_pcl is None:
            # #     self.socket_pcl = socket.copy()

        except KeyboardInterrupt:
            raise KeyboardInterrupt("")
        except Exception as e:
            print(e)
        
        return self.obs_buf, self.contacts, self.obs_student_buf, self.tactile_queue
    
    def _update_reset_buf(self):
        # socket_top_pose_world = self.socket_pose_world.copy()
        # socket_top_pose_world[:3, 3] += np.array([0., 0., 0.02])
        # plug_socket_distance = np.linalg.norm(self.ee_pos[:, :3] - socket_top_pose_world[:3, 3])
        plug_socket_xy_distance = torch.norm(self.ee_pos[:, :2] - self.socket_top_pos_world[:, :2])

        is_very_close = plug_socket_xy_distance < 0.03
        below_socket_top = (self.ee_pos[:, 2] < 0.085)

        really_below_socket = (self.ee_pos[:, 2] < 0.075)

        inserted = is_very_close & below_socket_top

        is_too_far =  (plug_socket_xy_distance > 0.08) | (self.ee_pos[:, 2] > 0.125)

        timeout = (self.episode_length >= self.max_episode_length)

        # if inserted:
        #     print('YAY!')

        self.done = is_too_far | timeout | really_below_socket | inserted
        if self.done[0, 0].item():
            print('reset because ', "far away" if is_too_far[0].item() else "", "timeoout" if timeout.item() else "", "inserted" if inserted.item() else "", "really_below_socket" if really_below_socket[0].item() else "")
            self.env.reset_pose()


    def _move_arm_to_desired_pose(self, desired_pos=None, desired_rot=None, regularize_force=False):
        """Move gripper to desired pose."""

        info = self.env.get_info_for_control()
        ee_pose = info['ee_pose']

        if desired_pos is None:
            self.ctrl_target_fingertip_centered_pos = torch.tensor(ee_pose[:3], device=self.device).unsqueeze(0)
        else:
            self.ctrl_target_fingertip_centered_pos = torch.tensor(desired_pos, device=self.device).unsqueeze(0)
    
        if desired_rot is None:
            self.ctrl_target_fingertip_centered_quat = torch.tensor(ee_pose[3:], device=self.device).unsqueeze(0)
            # ctrl_target_fingertip_centered_euler = torch.tensor(self.deploy_config.env.fingertip_midpoint_rot_initial,
            #                                                     device=self.device).unsqueeze(0)

            # self.ctrl_target_fingertip_centered_quat = torch_jit_utils.quat_from_euler_xyz_deploy(
            #     ctrl_target_fingertip_centered_euler[:, 0],
            #     ctrl_target_fingertip_centered_euler[:, 1],
            #     ctrl_target_fingertip_centered_euler[:, 2])
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
            # print(delta_hand_pose)
            actions = torch.zeros((1, self.num_actions), device=self.device)
            actions[:, :6] = delta_hand_pose
            # Apply the action, keep fingers in the same status
            self.apply_action(actions=actions, do_scale=False, do_clamp=False, wait=True, regularize_force=regularize_force)

    def update_and_apply_action(self, actions, wait=True, regularize_force=True):

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

        self.apply_action(self.actions.clone(), wait=wait, regularize_force=regularize_force)

    def apply_action(self, actions, do_scale=True, do_clamp=False, regularize_force=True, wait=True):

        # Apply the action
        if regularize_force:
            ft = torch.tensor(self.env.get_ft(), device=self.device, dtype=torch.float).unsqueeze(0)
            # actions = torch.where(torch.abs(ft) > 2.0, actions * 0, actions)
            check_ft = (torch.abs(ft) > 0.5)
            # print(ft)
            actions[:, 2] = (((check_ft[:, 2]) & (actions[:, 2] > 0.0)) | ~check_ft[:, 2]) * 1.0 * actions[:, 2]
            # actions[:, :2] = (torch.abs(ft[:, :2]) > 1.0) * 1.0 * actions[:, :2]
            
            # print(before[:, 2], torch.abs(ft)[:, 2], actions[:, 2], ~check_ft[:, 2])

        if do_clamp:
            actions = torch.clamp(actions, -1.0, 1.0)
        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.pos_scale_deploy, device=self.device))
        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.rot_scale_deploy, device=self.device))

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
        
        # print(self.fingertip_centered_pos, self.ctrl_target_fingertip_centered_pos)
        # s = time()
        self.generate_ctrl_signals(wait=wait)
        # print('generate_ctrl_signals', time() - s)
    
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
            device=self.device) # inverse kinematics

        self.ctrl_target_dof_pos = self.ctrl_target_dof_pos[:, :7]
        target_joints = self.ctrl_target_dof_pos.cpu().detach().numpy().squeeze().tolist()

        try:
            self.env.move_to_joint_values(target_joints, wait=wait)
        except:
            print(f'failed to reach {target_joints}')
    
    def convert_msg_to_matrix(self, msg):
        mat = quaternion_matrix([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        mat[:3, 3] = np.array([msg.position.x, msg.position.y, msg.position.z])
        return mat

    def reset_from_socket(self):
        self._move_arm_to_desired_pose([0.5, 0.0, 0.15])
        self._move_arm_to_desired_pose([0.5, 0.0, 0.075])
        self.grasp()
        self.reset()

    def grasp(self, x=-0.015, y=0., z_scale=0.98):

        resp1 = self.path_parameters_srv(joint_relative_velocity=0.04, joint_relative_acceleration=0.04, override_joint_acceleration=1.)
        if resp1.success:
            print('path_parameters_srv success')
        rospy.sleep(0.5)

        self.plug_pose_world = self.env.get_plug_pose_world()
        pregrasp = self.plug_pose_world[:2, 3]  
        pregrasp[0] += x
        pregrasp[1] += y
        plug_height = 0.0762
        plug_grasp_height = plug_height*z_scale
        plug_grasp_height = max(0.05, plug_grasp_height)
        plug_grasp_height = min(plug_grasp_height, 0.08)
        self._move_arm_to_desired_pose([*pregrasp, plug_grasp_height])
        self.env.grasp()

    def reset(self):
        resp1 = self.path_parameters_srv(joint_relative_velocity=0.04, joint_relative_acceleration=0.04, override_joint_acceleration=1.)
        if resp1.success:
            print('path_parameters_srv success')
        rospy.sleep(0.5)

        self._move_arm_to_desired_pose([0.5, 0.0, 0.13], regularize_force=False)
        self._move_arm_to_desired_pose(desired_rot=[0.7071068, -0.7071068, 0, 0], regularize_force=False)
        self._move_arm_to_desired_pose([0.5, -0.11, 0.13], regularize_force=False)
        self._move_arm_to_desired_pose([0.5, -0.11, 0.075], regularize_force=False)
        self.env.release()

        self.done[...] = False
        self.episode_length[...] = 0.
    
    def squeeze_in(self, angles):
        # suc = self.squeeze_in_srv(angles=angles.tolist())
        return

    def init_above_socket(self):
        resp1 = self.path_parameters_srv(joint_relative_velocity=0.04, joint_relative_acceleration=0.04, override_joint_acceleration=1.)
        if resp1.success:
            print('path_parameters_srv success')
        rospy.sleep(0.5)

        self._move_arm_to_desired_pose([0.5, -0.11, 0.13]) 
        self._move_arm_to_desired_pose(desired_rot=[0.7071068, -0.7071068, 0, 0])
        self.grasp()

        gripper_pos = self.env.get_gripper_pos()
        gripper_pos[1:] += 0.03
        self.squeeze_in(gripper_pos)

        self._move_arm_to_desired_pose([0.5, -0.11, 0.13], regularize_force=False) # post grasp
        if not self.ft_calibrated:
            self.env.arm.calib_robotiq()
            rospy.sleep(0.5)
            self.ft_calibrated = self.env.arm.calib_robotiq()
            # self.ft_calibrated = True
            print('F/T Calibrated', self.ft_calibrated)
        
        
        self._move_arm_to_desired_pose([0.5, -0.00, 0.13], regularize_force=False)
        rand_x = np.random.uniform(-0.0254, 0.0254)
        rand_y = np.random.uniform(-0.0254, 0.0254)
        rand_z = np.random.uniform(0.10, 0.10)
        socket_top = [self.socket_pose_world[0, 3] + rand_x, self.socket_pose_world[1, 3] + rand_y, 0.095]
        print('socket pose', socket_top)
        self._move_arm_to_desired_pose(socket_top, regularize_force=False)

        return gripper_pos

    # def move_around_socket(self):
    #     self._move_arm_to_desired_pose([0.5, -0.00, 0.098])
    #     self._move_arm_to_desired_pose([0.53, -0.00, 0.098])
        
    #     self._move_arm_to_desired_pose([0.53, -0.03, 0.098])
    #     self._move_arm_to_desired_pose([0.47, -0.03, 0.098])

    #     self._move_arm_to_desired_pose([0.47, 0.03, 0.098])
    #     self._move_arm_to_desired_pose([0.53, 0.03, 0.098])

    #     self._move_arm_to_desired_pose([0.50, 0.03, 0.098])
    #     self._move_arm_to_desired_pose([0.50, -0.03, 0.098])
        

    # def up_down(self):
    #     self._move_arm_to_desired_pose(desired_rot=[0.7071068, -0.7071068, 0, 0])
    #     self._move_arm_to_desired_pose([0.5, -0.15, 0.15]) 
    #     self._move_arm_to_desired_pose([0.5, -0.15, 0.075]) # pre grasp
    #     self.env.grasp()
    #     self._move_arm_to_desired_pose([0.5, -0.15, 0.15]) # post grasp

    # def up_down_contact(self):
    #     self._move_arm_to_desired_pose([0.5, -0.00, 0.098])
    #     self._move_arm_to_desired_pose([0.5, -0.00, 0.085])
    #     self._move_arm_to_desired_pose([0.5, -0.00, 0.098])

    #     self._move_arm_to_desired_pose([0.53, -0.00, 0.098])
    #     self._move_arm_to_desired_pose([0.53, -0.00, 0.085])
    #     self._move_arm_to_desired_pose([0.53, -0.00, 0.098])

    #     self._move_arm_to_desired_pose([0.53, -0.03, 0.098])
    #     self._move_arm_to_desired_pose([0.53, -0.03, 0.085])
    #     self._move_arm_to_desired_pose([0.53, -0.03, 0.098])

    #     self._move_arm_to_desired_pose([0.47, -0.03, 0.098])
    #     self._move_arm_to_desired_pose([0.47, -0.03, 0.085])
    #     self._move_arm_to_desired_pose([0.47, -0.03, 0.098])

    #     self._move_arm_to_desired_pose([0.47, 0.03, 0.098])
    #     self._move_arm_to_desired_pose([0.47, 0.03, 0.085])
    #     self._move_arm_to_desired_pose([0.47, 0.03, 0.098]) 
    
    def deploy(self):

        # self._initialize_grasp_poses()
        from algo.deploy.env.env import ExperimentEnv
        rospy.init_node('DeployEnv', disable_signals=True)
        self.env = ExperimentEnv()
        rospy.sleep(2)
        # socket_pose_camera = self.env.get_socket_pose_camera()
        self.socket_pose_world =  self.env.get_socket_pose_world()
        # self.socket_pose_world = torch.from_numpy(np.matmul(np.linalg.inv(robot_base_pose_camera), socket_pose_camera)).float().to(self.device)

        mesh_plug = "/home/robotics/osher3_workspace/src/isaacgym/python/IsaacGymInsertion/assets/factory/mesh/factory_insertion/yellow_round_peg_2in.obj"
        mesh_socket = "/home/robotics/dhruv/object_tracking/test_data/model/socket/socket.obj"
        # self.extrinsic_contact = ExtrinsicContact(mesh_plug, mesh_socket, socket_pos=socket_pose_camera, num_points=self.num_contact_points, device=self.device)
        # Wait for connections.
        rospy.sleep(0.5)
        rospy.logwarn('Finished setting the env, lets play.')

        hz = 10
        ros_rate = rospy.Rate(hz)

        self._create_asset_info()
        self._acquire_task_tensors()

        socket_top_pose_world = self.socket_pose_world.copy()
        socket_top_pose_world[3, 3] += 0.02
        self.socket_top_pos_world[...] = torch.from_numpy(socket_top_pose_world).float().to(self.device)[:3, 3]


        true_socket_pose = [self.deploy_config.env.kuka_depth, 0.0, self.deploy_config.env.table_height]
        self._set_socket_pose(pos=true_socket_pose)
        # above_socket_pose = [x + y for x, y in zip(true_socket_pose, [0, 0, 0.1])]

        true_plug_pose = [self.deploy_config.env.kuka_depth, 0.0, self.deploy_config.env.table_height]
        self._set_plug_pose(pos=true_plug_pose)
        # above_plug_pose = [x + y for x, y in zip(true_plug_pose, [0, 0, 0.2])]

        # ---- Data Logger ----
        if self.deploy_config.task.collect_data:
            from algo.ppo.experience import RealLogger
            data_logger = RealLogger(env=self)
            data_logger.data_logger = data_logger.data_logger_init(None)

        # self.reset_from_socket()  
        # exit()
        # self.init_above_socket()
        # self.move_around_socket()
        # self.reset()
        # exit()

        # check grasp here
        # while True:
        #     _ = input('grasp')
        #     try:
        #         x = float(input('x'))
        #     except:
        #         x = -0.015
        #     try:
        #         y = float(input('y'))
        #     except:
        #         y = 0.0
        #     try:
        #         z = float(input('z'))
        #     except:
        #         z = 0.98
        #     print(x, y, z)
        #     self.grasp(x=x, y=y, z_scale=z)
        #     # self.grasp(z_scale=1.0)
        #     rospy.sleep(2)
        #     _ = input('release')
        #     self.env.release()
        # exit()

        for run_idx in range(20):
            try:
                if run_idx % 5 == 0:
                    self.ft_calibrated = False
                gripper_pos = self.init_above_socket()
                self.env.reset_pose()
                resp1 = self.path_parameters_srv(joint_relative_velocity=0.01, joint_relative_acceleration=0.01, override_joint_acceleration=1)
                if resp1.success:
                    print('path_parameters_srv success')
                else:
                    rospy.logerr('path_parameters_srv failed')
                    exit()
                
                # REGULARIZE FORCES
                # self.env.regularize_force(True)
                # print(self.socket_pose_world)

                obs, contacts, obs_stud, tactile = self.compute_observations()
                self._update_reset_buf()

                action, latent = None, None

                start_time = time.time()
                try:
                    while True:
                        if self.done[0, 0]:
                            if self.deploy_config.task.collect_data:
                                data_logger.log_trajectory_data(action=None, latent=None, done=self.done.clone())
                            break

                        obs_mean = self.running_mean_std(obs.clone())

                        input_dict = {
                            'obs': obs_mean,
                            'contacts': contacts.clone(),
                        }

                        action, latent, dec = self.model.act_inference(input_dict)
                        action = torch.clamp(action, -1.0, 1.0)
                        
                        self.update_and_apply_action(action, wait=False, regularize_force=True)

                        if self.deploy_config.task.collect_data:
                            data_logger.log_trajectory_data(action, latent, self.done.clone())

                        obs, contacts, obs_stud, tactile = self.compute_observations() 
                        self.episode_length += 1
                        self._update_reset_buf()

                        sleep_time = max(0, (1.0/10.0) - (time.time() - start_time))
                        time.sleep(sleep_time)
                        # print('HZ:', 1/(time.time() - start_time))
                        start_time = time.time()
                       
                except KeyboardInterrupt:
                    raise KeyboardInterrupt("")

                self.reset()
                
            except KeyboardInterrupt:
                print("Program was interrupted by the user.")
                self.reset()
                break


    def pose_world_to_robot_base(self, pose, as_matrix=True):
        """Convert pose from world frame to robot base frame."""

        if as_matrix:
            return torch.cat([pose[:, :3], quat2R(pose[:, 3:]).reshape(1, -1)], dim=1)
        else:
            return pose