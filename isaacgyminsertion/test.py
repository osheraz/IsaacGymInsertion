from std_msgs.msg import Float64MultiArray, String
import rospy
import open3d as o3d
import numpy as np
import math

_EPS = np.finfo(float).eps * 4.0

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

class o3dNode():
    def __init__(self):
        
        self.plugMat = np.zeros((4, 4))
        self.socketMat = np.zeros((4, 4))

    def callback_socket(self, msg):
        data = np.array(msg.data).reshape(4, 4)
        self.socketMat[:, :] = data

    def callback_plug(self, msg):
        data = np.array(msg.data).reshape(4, 4)
        self.plugMat[:, :] = data

        self.origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # self.plug_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.plug_mesh = o3d.io.read_triangle_mesh("/home/robotics/osher3_workspace/src/isaacgym/python/IsaacGymInsertion/assets/factory/mesh/factory_insertion/yellow_round_peg_2in.obj")
        self.socket_mesh = o3d.io.read_triangle_mesh("/home/robotics/osher3_workspace/src/isaacgym/python/IsaacGymInsertion/assets/factory/mesh/factory_insertion/green_round_hole_2in.obj")
        self.socket_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.plug_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

        socketMat = quaternion_matrix(np.array([0, -0.7071068, 0, 0.7071068]))
        socketMat[:3, 3] = np.array([-0.04, 0.0, -0.01])

        plugMat = quaternion_matrix(np.array([0, -0.7071068, 0, 0.7071068]))
        plugMat[:3, 3] = np.array([-0.025, 0.0, -0.038])
        # socketMat = quaternion_matrix(np.array([0, 0., 0., 1.]))

        originMat = quaternion_matrix(np.array([0., 0., 0., 1.]))
        originMat[:3, 3] = np.array([0.0, 0.0, 0.0])  
        self.origin = self.origin.transform(originMat).scale(0.02, center=(0, 0, 0))

        worldSocketMat = quaternion_matrix(np.array([ 0, 0.7071068, 0, 0.7071068]))
        worldSocketMat[:3, 3] = np.array([0.04, 0.0, 0.01])
        cam2world_matrix = np.matmul(worldSocketMat, np.linalg.inv(self.socketMat))

        # print(np.matmul(plugMat, np.matmul(cam2world_matrix, self.plugMat)))

        self.socket_mesh = self.socket_mesh.transform(self.socketMat).transform(cam2world_matrix).transform(socketMat)
        self.socket_mesh_frame = self.socket_mesh_frame.transform(self.socketMat).transform(cam2world_matrix).transform(socketMat).scale(0.05, center=self.socket_mesh.get_center())
        self.plug_mesh = self.plug_mesh.transform(self.plugMat).transform(cam2world_matrix).transform(plugMat)
        self.plug_mesh_frame = self.plug_mesh_frame.transform(self.plugMat).transform(cam2world_matrix).transform(plugMat).scale(0.05, center=self.plug_mesh.get_center())


        o3d.visualization.draw_geometries([self.socket_mesh, self.plug_mesh, self.socket_mesh_frame, self.plug_mesh_frame])

    def main(self):
        rospy.init_node('o3d_listener', anonymous=True)
        rospy.Subscriber('/hand_control/socket_camera_frame', Float64MultiArray, self.callback_socket)
        rospy.Subscriber('/hand_control/plug_camera_frame', Float64MultiArray, self.callback_plug)
        # spin() simply keeps python from exiting until this node is stopped
        # self.rate = rospy.Rate(100)
        # while not not rospy.is_shutdown():
        #     self.rate.sleep()
        rospy.spin()

if __name__ == "__main__":
    node = o3dNode()
    try:
        node.main()
    except rospy.ROSInterruptException:
        pass