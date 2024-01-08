from typing import Any
import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
import zmq
import numpy as np
import json
from tf.transformations import quaternion_matrix, quaternion_from_matrix, euler_from_quaternion, quaternion_from_euler

class ExternalTracker:
    def __init__(self):
        
        rospy.init_node('external_tracker', anonymous=True)

        context = zmq.Context()
        self.subscriber = context.socket(zmq.SUB)
        self.subscriber.connect("tcp://localhost:5555")
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, '')
        self.subscriber.setsockopt(zmq.CONFLATE, 1)

        self.plug_pose = np.eye(4)

        # load initial socket pose
        with open(f"/home/robotics/dhruv/object_tracking/test_data/poses/socket.txt", "r") as f:
            self.socket_pose = np.array(json.load(f)).reshape(4, 4)

        # load initial socket pose
        with open(f"/home/robotics/dhruv/object_tracking/test_data/poses/robot_base.txt", "r") as f:
            self.robot_base_pose = np.array(json.load(f)).reshape(4, 4)
        
        self.R_camera_world = np.linalg.inv(self.robot_base_pose.copy())

        self.plug_pose_pub = rospy.Publisher('/external_tracker/pose', PoseStamped, queue_size=1)
        self.socket_pose_pub = rospy.Publisher('/external_tracker/pose', PoseStamped, queue_size=1)
        self.robot_base_pose_pub = rospy.Publisher('/external_tracker/pose', PoseStamped, queue_size=1)

    def get_pose_msg(self, pose):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'world'
        msg.pose.position.x = pose[0, 3]
        msg.pose.position.y = pose[1, 3]
        msg.pose.position.z = pose[2, 3]
        quat = quaternion_from_matrix(pose)
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        return msg

    def run(self):
        rospy.Rate(200)
        skip = 0
        last_message = None
        
        self.socket_pose_pub.publish(self.get_pose_msg(self.socket_pose))
        self.robot_base_pose_pub.publish(self.get_pose_msg(self.robot_base_pose))

        while True:
            try:
                message = self.subscriber.recv(zmq.NOBLOCK)
                last_message = message
                skip += 1
            except zmq.Again:
                skip = 0
                break
        
        if last_message is not None:
            pose_matrices = np.frombuffer(last_message, dtype=np.float32).reshape(-1, 16)
            plug_pose = pose_matrices[0].reshape(4, 4).T
            print('plug_pose', self.plug_pose)
            # self.plug_pose = plug_pose.copy()
            self.plug_pose = np.matmul(self.R_camera_world, plug_pose.copy())
            self.plug_pose_pub.publish(self.get_pose_msg(self.plug_pose))

        
        