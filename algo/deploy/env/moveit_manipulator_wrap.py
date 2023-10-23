import rospy
from tactile_insertion.srv import MoveitJacobian, MoveitMoveJointPosition, MoveitPose, VelAndAcc, MoveitJoints
from tactile_insertion.srv import MoveitJacobianResponse, MoveitMoveJointPositionResponse, MoveitPoseResponse, \
    VelAndAccResponse, MoveitJointsResponse, VelAndAccRequest, MoveitMoveEefPose, MoveitMoveEefPoseRequest, MoveitMoveJointPositionRequest
import copy
import sys
import numpy as np
from std_srvs.srv import Empty, EmptyResponse
from iiwa_msgs.msg import JointQuantity, JointPosition
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

class MoveManipulatorServiceWrap():
    """
    Python 2.7 - 3 issues with melodic.
    This class uses all the manipulator modules by ROS services
    """
    # TODO add response to all of the moving requests.
    def __init__(self):

        rospy.logdebug("===== In MoveManipulatorServiceWrap")

        self.jacobian_srv = rospy.ServiceProxy('/MoveItJacobian', MoveitJacobian)
        self.scale_vel_acc_srv = rospy.ServiceProxy('/MoveItScaleVelAndAcc', VelAndAcc)
        self.moveit_move_joints_srv = rospy.ServiceProxy("/MoveItMoveJointPosition", MoveitMoveJointPosition)
        self.moveit_move_eef_pose_srv = rospy.ServiceProxy("/MoveItMoveEefPose", MoveitMoveEefPose)
        self.moveit_get_pose_srv = rospy.ServiceProxy("/MoveItPose", MoveitPose)
        self.moveit_get_joints_srv = rospy.ServiceProxy("/MoveItJoints", MoveitJoints)
        self.moveit_stop_motion_srv = rospy.ServiceProxy('/Stop', Empty)

        self.pose = None
        self.joints = None
        self.jacob = None

        rospy.Subscriber('/iiwa/Jacobian', Float32MultiArray, self.callback_jacob)
        rospy.Subscriber('/iiwa/Joints', JointPosition, self.callback_joints)
        rospy.Subscriber('/iiwa/Pose', PoseStamped, self.callback_pose)

        rospy.wait_for_message('/iiwa/Joints', JointPosition)
        rospy.wait_for_message('/iiwa/Pose', PoseStamped)

        rospy.logdebug("===== Out MoveManipulatorServiceWrap")

    def callback_joints(self, msg):

        joints = [msg.position.a1,
                  msg.position.a2,
                  msg.position.a3,
                  msg.position.a4,
                  msg.position.a5,
                  msg.position.a6,
                  msg.position.a7]

        self.joints = joints

    def joint_values(self):
        # rospy.wait_for_message('/iiwa/Joints', JointPosition)

        return self.joints

    def joint_values_moveit(self):

        res = self.moveit_get_joints_srv()

        joints = [res.pos.a1,
                  res.pos.a2,
                  res.pos.a3,
                  res.pos.a4,
                  res.pos.a5,
                  res.pos.a6,
                  res.pos.a7]

        return joints

    def callback_pose(self, msg):
        self.pose = msg.pose

    def get_cartesian_pose(self):
        # rospy.wait_for_message('/iiwa/Pose', PoseStamped)
        return self.pose

    def get_cartesian_pose_moveit(self,):

        res = self.moveit_get_pose_srv()

        return res.pose

    def callback_jacob(self, msg):

        self.jacob = np.array(msg.data).reshape(6, 7)

    def get_jacobian_matrix(self):
        # rospy.wait_for_message('/iiwa/Jacobian', JointPosition)

        return self.jacob

    def get_jacobian_matrix_moveit(self):

        jacob = np.array(self.jacobian_srv().data).reshape(6, 7)

        return jacob


    def scale_vel(self, scale_vel, scale_acc):
        req = VelAndAccRequest()
        req.vel = scale_vel
        req.acc = scale_acc
        self.scale_vel_acc_srv(req)

    def ee_traj_by_pose_target(self, pose, wait=True):
        req = MoveitMoveEefPoseRequest()
        req.pose = pose
        req.wait = wait
        self.moveit_move_eef_pose_srv(req)

    def joint_traj(self, positions_array, wait=True):
        js = JointQuantity()

        js.a1 = positions_array[0]
        js.a2 = positions_array[1]
        js.a3 = positions_array[2]
        js.a4 = positions_array[3]
        js.a5 = positions_array[4]
        js.a6 = positions_array[5]
        js.a7 = positions_array[6]

        req = MoveitMoveJointPositionRequest()
        req.pos = js
        req.wait = wait

        self.moveit_move_joints_srv(req)

        return True

    def ee_pose(self):
        gripper_pose = self.get_cartesian_pose()
        return gripper_pose

    def ee_rpy(self):
        gripper_pose = self.get_cartesian_pose()
        return gripper_pose.orientation


    def stop_motion(self):

        self.moveit_stop_motion_srv()



if __name__ == '__main__':

    rospy.init_node('moveit_test')

    rate = rospy.Rate(100)
    moveit_test = MoveManipulatorServiceWrap()

    # Init tests

    print(moveit_test.get_jacobian_matrix())
    print(moveit_test.get_cartesian_pose())
    print(moveit_test.joint_values())

    from time import time
    np.set_printoptions(5)
    last = 0
    while True:
        start_time = time()

        a = moveit_test.get_cartesian_pose()
        b = moveit_test.get_jacobian_matrix()
        c = moveit_test.joint_values()
        rate.sleep()
        print("FPS: ", 1.0 / (time() - start_time))  # FPS = 1 / time to process loop

    #
    # # Move stuff
    # init_pose = [0.0064, 0.2375,  -0.0075,  -1.2022, 0.0015,  1.6900,  -1.5699]
    # moveit_test.joint_traj(init_pose, wait=True)

