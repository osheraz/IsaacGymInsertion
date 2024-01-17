import rospy
from tactile_insertion.srv import MoveitJacobian, MoveitMoveJointPosition, MoveitPose, VelAndAcc, MoveitJoints
from tactile_insertion.srv import MoveitJacobianResponse, MoveitMoveJointPositionResponse, MoveitPoseResponse, \
    VelAndAccResponse, MoveitJointsResponse, VelAndAccRequest, MoveitMoveEefPose, MoveitMoveEefPoseRequest, \
    MoveitMoveJointPositionRequest
import copy
import sys
import numpy as np
from std_srvs.srv import Empty, EmptyResponse
from iiwa_msgs.msg import JointQuantity, JointPosition
from kortex_driver.msg import Base_JointSpeeds, JointSpeed
from geometry_msgs.msg import PoseStamped

from kortex_driver.srv import *
from kortex_driver.msg import Empty as Empty_K
from kortex_driver.msg import ActionNotification, CartesianReferenceFrame, CartesianSpeed, ActionEvent, JointAngle, BaseCyclic_Feedback, WaypointList
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from std_msgs.msg import String, Float32MultiArray, Bool
import tf

class MoveKinovaServiceWrap():
    """
    Python 2.7 - 3 issues with melodic.
    This class uses all the manipulator modules by ROS services
    """

    # TODO add response to all of the moving requests.
    def __init__(self):

        rospy.logdebug("===== In MoveKinovaServiceWrap")

        self.robot_name = ''

        # MoveIt related
        self.jacobian_srv = rospy.ServiceProxy('/MoveItJacobian', MoveitJacobian)
        self.scale_vel_acc_srv = rospy.ServiceProxy('/MoveItScaleVelAndAcc', VelAndAcc)
        self.moveit_move_joints_srv = rospy.ServiceProxy("/MoveItMoveJointPosition", MoveitMoveJointPosition)
        self.moveit_move_eef_pose_srv = rospy.ServiceProxy("/MoveItMoveEefPose", MoveitMoveEefPose)
        self.moveit_get_pose_srv = rospy.ServiceProxy("/MoveItPose", MoveitPose)
        self.moveit_get_joints_srv = rospy.ServiceProxy("/MoveItJoints", MoveitJoints)
        self.moveit_stop_motion_srv = rospy.ServiceProxy('/Stop', Empty)

        self.pose = None  # We update by tf
        self.joints = None  # We update by kuka api
        self.jacob = None  # We update by moveit callback

        # Published by moveit_manipulator in tactile_insertion
        rospy.Subscriber('/kinova/Jacobian', Float32MultiArray, self.callback_jacob)
        rospy.Subscriber('/kinova/Joints', JointPosition, self.callback_joints)
        rospy.Subscriber('/kinova/Pose', PoseStamped, self.callback_pose)

        # Kinova-API related
        rospy.Subscriber('/ft_stop', Bool, self._stop_callback)
        self.stop_pub = rospy.Publisher('/in/stop', Empty_K, queue_size=10)
        self.pub_joints_vel_api = rospy.Publisher('/in/joint_velocity', Base_JointSpeeds, queue_size=10)

        self.pub_joint_request = rospy.Publisher('/kinova/desired_joints', Float32MultiArray, queue_size=10)

        # Init the action topic subscriber
        self.action_topic_sub = rospy.Subscriber("/" + self.robot_name + "/action_topic", ActionNotification,
                                                 self.cb_action_topic)
        self.last_action_notif_type = None

        clear_faults_full_name = '/' + self.robot_name + '/base/clear_faults'
        rospy.wait_for_service(clear_faults_full_name)
        self.clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)

        # TODO: Kinova old API
        play_cartesian_trajectory_full_name = '/' + self.robot_name + '/base/play_cartesian_trajectory'
        rospy.wait_for_service(play_cartesian_trajectory_full_name)
        self.play_cartesian_trajectory = rospy.ServiceProxy(play_cartesian_trajectory_full_name,
                                                            PlayCartesianTrajectory)

        play_joint_trajectory_full_name = '/' + self.robot_name + '/base/play_joint_trajectory'
        rospy.wait_for_service(play_joint_trajectory_full_name)
        self.play_joint_trajectory = rospy.ServiceProxy(play_joint_trajectory_full_name, PlayJointTrajectory)

        activate_publishing_of_action_notification_full_name = '/' + self.robot_name + '/base/activate_publishing_of_action_topic'
        rospy.wait_for_service(activate_publishing_of_action_notification_full_name)
        self.activate_publishing_of_action_notification = rospy.ServiceProxy(
            activate_publishing_of_action_notification_full_name, OnNotificationActionTopic)

        get_product_configuration_full_name = '/' + self.robot_name + '/base/get_product_configuration'
        rospy.wait_for_service(get_product_configuration_full_name)
        self.get_product_configuration = rospy.ServiceProxy(get_product_configuration_full_name,
                                                            GetProductConfiguration)

        self.clear_faults()
        self.subscribe_to_a_robot_notification()
        self.stop_force_request = False
        self.vx = 0.05
        self.vw = 1

        # rospy.wait_for_message('/kinova/Joints', JointPosition)
        # rospy.wait_for_message('/kinova/Pose', PoseStamped)
        # rospy.wait_for_message('/kinova/Jacobian', Float32MultiArray)

        rospy.logdebug("===== Out MoveKinovaServiceWrap")

    def clear_faults(self):
        try:
            self.clear_faults()
        except rospy.ServiceException:
            rospy.logerr("Failed to call ClearFaults")
            return False
        else:
            rospy.loginfo("Cleared the faults successfully")
            return True

    def subscribe_to_a_robot_notification(self):
        # Activate the publishing of the ActionNotification
        req = OnNotificationActionTopicRequest()
        rospy.loginfo("Activating the action notifications...")
        try:
            self.activate_publishing_of_action_notification(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call OnNotificationActionTopic")
            return False
        else:
            rospy.loginfo("Successfully activated the Action Notifications!")

        rospy.sleep(1.0)
        return True

    def _stop_callback(self, msg):
        if msg.data:
            self.stop_force_request = True
        else:
            self.stop_force_request = False

    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event

    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if self.stop_force_request:
                self.stop_motion()
                return True
            if (self.last_action_notif_type == ActionEvent.ACTION_END):
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif (self.last_action_notif_type == ActionEvent.ACTION_ABORT):
                rospy.loginfo("Received ACTION_ABORT notification")
                return False
            else:
                rospy.sleep(0.01)

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

    def get_cartesian_pose_moveit(self, ):

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

        self.vx = scale_vel
        self.vw = scale_acc

    def ee_traj_by_pose_target(self, pose, wait=True, by_moveit=True):

        if by_moveit:
            req = MoveitMoveEefPoseRequest()
            req.pose = pose
            req.wait = wait
            self.moveit_move_eef_pose_srv(req)
        else:
            self.last_action_notif_type = None

            req = PlayCartesianTrajectoryRequest()

            if isinstance(pose, PoseStamped):
                pose = pose.pose

            req.input.target_pose.x = pose.position.x
            req.input.target_pose.y = pose.position.y
            req.input.target_pose.z = pose.position.z

            roll, pitch, yaw = tf.transformations.euler_from_quaternion((pose.orientation.x,
                                                                         pose.orientation.y,
                                                                         pose.orientation.z,
                                                                         pose.orientation.w))
            req.input.target_pose.theta_x = np.rad2deg(roll)
            req.input.target_pose.theta_y = np.rad2deg(pitch)
            req.input.target_pose.theta_z = np.rad2deg(yaw)

            pose_speed = CartesianSpeed()
            pose_speed.translation = self.vx
            pose_speed.orientation = self.vw

            # The constraint is a one_of in Protobuf. The one_of concept does not exist in ROS
            # To specify a one_of, create it and put it in the appropriate list of the oneof_type member of the ROS object :
            req.input.constraint.oneof_type.speed.append(pose_speed)

            # Call the service
            try:
                self.play_cartesian_trajectory(req)
            except rospy.ServiceException:
                rospy.logerr("Failed to call PlayCartesianTrajectory")
                return False
            else:
                return self.wait_for_action_end_or_abort() if wait else True

    def joint_traj(self, positions_array, wait=True, by_moveit=True, by_vel=False):

        if by_moveit:
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
            if wait:
                msg = rospy.wait_for_message('/kinova/Joints', JointPosition)
                joints = [msg.position.a1,
                          msg.position.a2,
                          msg.position.a3,
                          msg.position.a4,
                          msg.position.a5,
                          msg.position.a6,
                          msg.position.a7]

                self.joints = joints
                rospy.logwarn('Reached {}'.format([['%.3f' % n for n in joints]]))

        else:
            if by_vel:
                msg = Float32MultiArray()
                msg.data = positions_array
                self.pub_joint_request.publish(msg)

            else:
                self.last_action_notif_type = None
                # Create the list of angles
                req = PlayJointTrajectoryRequest()
                # Here the arm is vertical (all zeros)
                for i in range(7):
                    temp_angle = JointAngle()
                    temp_angle.joint_identifier = i
                    temp_angle.value = np.rad2deg(positions_array[i])
                    req.input.joint_angles.joint_angles.append(temp_angle)

                # Send the angles
                try:
                    self.play_joint_trajectory(req)
                except rospy.ServiceException:
                    rospy.logerr("Failed to call PlayJointTrajectory")
                    return False
                else:
                    return self.wait_for_action_end_or_abort()

        return True

    def joint_vel(self, vel_array):

        req = Base_JointSpeeds()

        for i in range(7):
            temp_speed = JointSpeed()
            temp_speed.joint_identifier = i
            temp_speed.value = vel_array[i]
            req.joint_speeds.append(temp_speed)

        self.pub_joints_vel_api.publish(req)

    def ee_pose(self):
        gripper_pose = self.get_cartesian_pose()
        return gripper_pose

    def ee_rpy(self):
        gripper_pose = self.get_cartesian_pose()
        return gripper_pose.orientation

    def stop_motion(self):

        # self.moveit_stop_motion_srv()
        self.stop_pub.publish()
        self.clear_faults()

if __name__ == '__main__':

    rospy.init_node('moveit_test')

    rate = rospy.Rate(40)
    moveit_test = MoveKinovaServiceWrap()
    # Init tests

    print(moveit_test.get_jacobian_matrix())
    print(moveit_test.get_cartesian_pose())
    print(moveit_test.joint_values())

    from time import time
    np.set_printoptions(5)
    last = 0
    # while True:
    #     start_time = time()
    #
    #     # a = moveit_test.get_cartesian_pose()
    #     b = moveit_test.get_jacobian_matrix()
    #     # c = moveit_test.get_jacobian_matrix_moveit()
    #     # print(moveit_test.get_cartesian_pose())
    #
    #     # c = moveit_test.joint_values()
    #     rate.sleep()
    #     print("FPS: ", 1.0 / (time() - start_time))  # FPS = 1 / time to process loop

    ############################################################
    # # Move stuff
    pos1 = [0.28, -0.18855233780982772, -3.120710008534155, -2.5599484178204546, 0.0026864879365296754, 0.9624248060626042, 1.5682030736834318]
    pos2 = np.array(pos1) + np.random.uniform(-0.05, 0.05, size=(7,))
    pos3 = np.array(pos1) + np.random.uniform(-0.05, 0.05, size=(7,))

    import random

    moveit_test.joint_traj(pos1, by_moveit=False, wait=True)
    #
    all_lists = [pos1, pos2, pos3]
    # # # # Randomly select one list
    while True:
        start_time = time()
        random_list = random.choice(all_lists)
        moveit_test.joint_traj(random_list, by_moveit=False, wait=True)
        rate.sleep()
        print("FPS: ", 1.0 / (time() - start_time))  # FPS = 1 / time to process loop

# rosservice call /iiwa/configuration/pathParameters "{joint_relative_velocity: 0.05, joint_relative_acceleration: 0.05, override_joint_acceleration: 1}"
