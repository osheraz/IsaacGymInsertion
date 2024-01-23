import rospy
from std_msgs.msg import Bool, Float64MultiArray
from geometry_msgs.msg import PoseStamped
import trimesh
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
from tf.transformations import quaternion_matrix
from scipy.spatial.transform import Rotation as R
import time
from std_msgs.msg import String, Float32MultiArray, Bool, Int16


class ExtrinsicContact:
    def __init__(
            self,
            mesh_plug,
            mesh_socket,
            socket_pos,
            num_points=50,
            device='cpu'
    ) -> None:

        self.obj_pos, self.obj_rpy = [], []
        self.object_trimesh = trimesh.load(mesh_plug)
        self.reset_object_trimesh = self.object_trimesh.copy()
        self.socket_trimesh = trimesh.load(mesh_socket)

        self.socket_pose_world = np.eye(4)
        self.socket_pose_world[:3, 3] = socket_pos

        self.n_points = num_points
        self.device = device
        self.plug_pose_world_no_rot = np.eye(4)

        rospy.Subscriber('/external_tracker/plug_pose_world', PoseStamped, self.plug_pose_world_callback)
        rospy.Subscriber('/external_tracker/socket_pose_world', PoseStamped, self.socket_pose_world_callback)
        rospy.Subscriber('/hand_control/obj_pos', Float32MultiArray, self._object_pose_callback)
        rospy.Subscriber('/hand_control/obj_rpy', Float32MultiArray, self._object_rpy_callback)
        self.extrinsic_contact_pub = rospy.Publisher('/extrinsic_contact', Float64MultiArray, queue_size=1)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def initialize(self):

        self.socket_trimesh.apply_transform(self.socket_pose_world)

        self.socket_pos = self.socket_pose_world
        self.socket = o3d.t.geometry.RaycastingScene()
        self.socket.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(self.socket_trimesh.as_open3d)
        )
        self.socket_pcl = trimesh.sample.sample_surface_even(self.socket_trimesh, self.n_points, seed=42)[0]

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

    def estimate_pose(self, curr_pose, prev_pos=None):
        '''
        this function removes any noise in the yaw estimation from the tracker by using the previous pose.
        Source: https://github.com/shiyoung77/tensegrity_perception/blob/main/tracking.py#L570
        Credits: Shiyang Liu
        '''

        curr_pos = curr_pose[:3, 3]
        curr_rot = curr_pose[:3, :3]

        curr_z_dir = curr_rot[:, 2]
        if prev_pos is None:
            prev_pos = np.eye(4)
        prev_rot = prev_pos[:3, :3]
        prev_z_dir = prev_rot[:, 2]
        delta_rot = np.eye(3)
        cos_dist = prev_z_dir @ curr_z_dir
        if not np.allclose(cos_dist, 1):
            axis = np.cross(prev_z_dir, curr_z_dir)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(cos_dist)
            delta_rot = R.from_rotvec(angle * axis).as_matrix()

        curr_rod_pose = np.eye(4)
        curr_rod_pose[:3, :3] = delta_rot @ prev_rot
        curr_rod_pose[:3, 3] = curr_pos
        return curr_rod_pose

    def get_extrinsic_contact(self, threshold=0.002, display=False, dec=None):

        self.object_trimesh = self.reset_object_trimesh.copy()

        # self.plug_pose_world_no_rot = self.estimate_pose(self.plug_pose_world.copy(),
        #                                                  self.plug_pose_world_no_rot)  # removes yaw noise

        self.plug_pose_world_no_rot = np.eye(4)
        self.plug_pose_world_no_rot[:3, :3] = R.from_euler('xyz', self.obj_rpy).as_matrix()
        self.plug_pose_world_no_rot[:3, 3] = self.obj_pos

        print(R.from_matrix(self.plug_pose_world_no_rot[:3, :3]).as_euler('xyz', degrees=True))

        self.object_trimesh.apply_transform(self.plug_pose_world_no_rot)
        query_points = trimesh.points.PointCloud(
            trimesh.sample.sample_surface(self.object_trimesh, self.n_points, seed=42)[0]).vertices
        d = self.socket.compute_distance(o3d.core.Tensor.from_numpy(query_points.astype(np.float32))).numpy()

        self.socket_pcl = trimesh.sample.sample_surface_even(self.socket_trimesh, self.n_points, seed=42)[0]

        intersecting_indices = d < threshold
        contacts = query_points[intersecting_indices]

        if display:
            self.ax.scatter(query_points[:, 0], query_points[:, 1], query_points[:, 2], c='yellow')
            self.ax.scatter(self.socket_pcl[:, 0], self.socket_pcl[:, 1], self.socket_pcl[:, 2], c='green')
            self.ax.scatter(contacts[:, 0], contacts[:, 1], contacts[:, 2], c='r')

            plt.pause(0.01)
            plt.cla()

        d = d.flatten()
        idx_2 = np.where(d > threshold)[0]
        d[idx_2] = threshold
        d = np.clip(d, 0.0, threshold)

        d = 1.0 - d / threshold
        d = np.clip(d, 0.0, 1.0)
        d[d > 0.1] = 1.0
        return np.array(d)

    def run(self):
        d = self.get_extrinsic_contact()
        self.extrinsic_contact_pub.publish(Float64MultiArray(data=d))

    def convert_msg_to_matrix(self, msg):
        mat = quaternion_matrix([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        mat[:3, 3] = np.array([msg.position.x, msg.position.y, msg.position.z])
        return mat

    def plug_pose_world_callback(self, msg):
        self.plug_pose_world = self.convert_msg_to_matrix(msg.pose)

    def socket_pose_world_callback(self, msg):
        self.socket_pose_world = self.convert_msg_to_matrix(msg.pose)

    def robot_base_pose_camera_callback(self, msg):
        self.robot_base_pose_camera = self.convert_msg_to_matrix(msg.pose)

    def _object_pose_callback(self, msg):
        self.obj_pos = np.array(msg.data) if not np.isnan(np.sum(np.array(msg.data))) else self.obj_pos

    def _object_rpy_callback(self, msg):
        self.obj_rpy = np.array(msg.data) if not np.isnan(np.sum(np.array(msg.data))) else self.obj_rpy


if __name__ == '__main__':

    mesh_plug = "/home/robotics/osher3_workspace/src/isaacgym/python/IsaacGymInsertion/assets/factory/mesh/factory_insertion/yellow_round_peg_2in.obj"
    mesh_socket = "/home/robotics/dhruv/object_tracking/test_data/model/socket/socket.obj"
    num_points = 500

    rospy.init_node('extrinsic_contact')
    rate = rospy.Rate(30)

    tracker = ExtrinsicContact(mesh_plug=mesh_plug, mesh_socket=mesh_socket, num_points=num_points)
    # print('Waiting for socket pose camera')
    # while True:
    #     if tracker.robot_base_pose_camera is None or tracker.socket_pose_world is None:
    #         continue
    #     else:
    #         tracker.initialize()
    #         break
    tracker.initialize()
    print('Extrinsic contact tracker initialized')

    while not rospy.is_shutdown():
        st = time.time()
        tracker.run()
        rate.sleep()
        print('Time taken: ', 1 / (time.time() - st))
