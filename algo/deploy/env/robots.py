import numpy
import rospy
import tf

from control_msgs.msg import JointTrajectoryControllerState
import geometry_msgs.msg
from geometry_msgs.msg import WrenchStamped
import copy
from robotiq_force_torque_sensor.srv import sensor_accessor
import robotiq_force_torque_sensor.srv as ft_srv
from robotiq_force_torque_sensor.msg import *
from algo.deploy.env.moveit_manipulator import MoveManipulator


class RobotWithFtEnv():
    """ Robot with Ft sensor
    """

    def __init__(self):

        self.wait_env_ready()
        self.robotiq_wrench_filtered_state = numpy.array([0, 0, 0, 0, 0, 0])
        self.init_success = self._check_all_systems_ready()

        # Define feedback callbacks
        self.ft_zero = rospy.ServiceProxy('/robotiq_force_torque_sensor_acc', sensor_accessor)
        rospy.Subscriber("/arm_controller/state", JointTrajectoryControllerState, self._joint_state_callback)
        rospy.Subscriber('/robotiq_force_torque_wrench_filtered_exp', WrenchStamped,
                         self._robotiq_wrench_states_callback)

        self.move_manipulator = MoveManipulator()

    def calib_robotiq(self):

        rospy.sleep(0.5)
        msg = ft_srv.sensor_accessorRequest()
        msg.command = "SET ZRO"
        suc_zero = True
        self.robotiq_wrench_filtered_state *= 0
        for _ in range(5):
            result = self.ft_zero(msg)
            rospy.sleep(0.5)
            if 'Done' not in str(result):
                suc_zero &= False
                rospy.logerr('Failed calibrating the F\T')
            else:
                suc_zero &= True
        return suc_zero

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other robot systems are
        operational.
        """
        rospy.logdebug("ManipulatorOpenhandRealEnv check_all_systems_ready...")
        self._check_arm_joint_state_ready()
        self._check_robotiq_connection()
        rospy.logdebug("END ManipulatorOpenhandRealEnv _check_all_systems_ready...")
        return True

    def _check_arm_joint_state_ready(self):

        self.arm_joint_state = None
        rospy.logdebug(
            "Waiting for arm_controller/state to be READY...")
        while self.arm_joint_state is None and not rospy.is_shutdown():
            try:
                self.arm_joint_state = rospy.wait_for_message(
                    "arm_controller/state", JointTrajectoryControllerState, timeout=5.0)
                rospy.logdebug(
                    "Current arm_controller/state READY=>")

            except:
                rospy.logerr(
                    "Current arm_controller/state not ready yet, retrying for getting laser_scan")
        return self.arm_joint_state

    def _check_robotiq_connection(self):

        self.robotiq_wrench_filtered_state = numpy.array([0, 0, 0, 0, 0, 0])
        rospy.logdebug(
            "Waiting for robotiq_force_torque_wrench_filtered to be READY...")
        while not numpy.sum(self.robotiq_wrench_filtered_state) and not rospy.is_shutdown():
            try:
                self.robotiq_wrench_filtered_state = rospy.wait_for_message(
                    "robotiq_force_torque_wrench_filtered", WrenchStamped, timeout=5.0)
                self.robotiq_wrench_filtered_state = numpy.array([self.robotiq_wrench_filtered_state.wrench.force.x,
                                                                  self.robotiq_wrench_filtered_state.wrench.force.y,
                                                                  self.robotiq_wrench_filtered_state.wrench.force.z,
                                                                  self.robotiq_wrench_filtered_state.wrench.torque.x,
                                                                  self.robotiq_wrench_filtered_state.wrench.torque.y,
                                                                  self.robotiq_wrench_filtered_state.wrench.torque.z])
                rospy.logdebug(
                    "Current robotiq_force_torque_wrench_filtered READY=>")
            except:
                rospy.logerr(
                    "Current robotiq_force_torque_wrench_filtered not ready yet, retrying")

        return self.robotiq_wrench_filtered_state

    def wait_env_ready(self):

        import time

        for i in range(1):
            print("WAITING..." + str(i))
            sys.stdout.flush()
            time.sleep(1.0)

        print("WAITING...DONE")

    def _joint_state_callback(self, data):
        self.arm_joint_state = data

    def _robotiq_wrench_states_callback(self, data):

        self.robotiq_wrench_filtered_state = numpy.array([data.wrench.force.x,
                                                          data.wrench.force.y,
                                                          data.wrench.force.z,
                                                          data.wrench.torque.x,
                                                          data.wrench.torque.y,
                                                          data.wrench.torque.z])

    def rotate_pose_by_rpy(self, in_pose, roll, pitch, yaw, wait=True):
        """
        Apply an RPY rotation to a pose in its parent coordinate system.
        """
        try:
            if in_pose.header:  # = in_pose is a PoseStamped instead of a Pose.
                in_pose.pose = self.rotate_pose_by_rpy(in_pose.pose, roll, pitch, yaw, wait)
                return in_pose
        except:
            pass
        q_in = [in_pose.orientation.x, in_pose.orientation.y, in_pose.orientation.z, in_pose.orientation.w]
        q_rot = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        q_rotated = tf.transformations.quaternion_multiply(q_in, q_rot)

        rotated_pose = copy.deepcopy(in_pose)
        rotated_pose.orientation = geometry_msgs.msg.Quaternion(*q_rotated)
        result = self.move_manipulator.ee_traj_by_pose_target(rotated_pose, wait)
        return rotated_pose

    def apply_delta_action(self, action_xyz=None, action_theta=None, wait=True):
        """
        Sets the Pose of the EndEffector based on the action variable.
        The action variable contains the position and orientation of the EndEffector.
        See create_action
        """
        # Set up a trajectory message to publish.
        if action_theta is None:
            action_theta = [0.0, 0.0, 0.0]
        if action_xyz is None:
            action_theta = [0.0, 0.0, 0.0]

        action_xyz = action_xyz if isinstance(action_xyz, list) else action_xyz.tolist()
        if len(action_xyz) < 3: action_xyz.append(0.0)  # xy -> xyz
        ee_target = self.get_ee_pose(as_message=True).pose
        cur_ee_pose = self.get_ee_pose(as_message=True)

        if numpy.any(action_theta):
            roll, pitch, yaw = tf.transformations.euler_from_quaternion((cur_ee_pose.pose.orientation.x,
                                                                         cur_ee_pose.pose.orientation.y,
                                                                         cur_ee_pose.pose.orientation.z,
                                                                         cur_ee_pose.pose.orientation.w))
            roll = roll + action_theta[0] if action_theta[0] else roll
            pitch = pitch + action_theta[1] if action_theta[1] else pitch
            yaw = yaw + action_theta[2] if action_theta[2] else yaw
            quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
            ee_target.orientation.x, ee_target.orientation.y, ee_target.orientation.z, ee_target.orientation.w = quaternion

        if numpy.any(action_xyz):
            ee_target.position.x = cur_ee_pose.pose.position.x + action_xyz[0] if action_xyz[
                0] else cur_ee_pose.pose.position.x
            ee_target.position.y = cur_ee_pose.pose.position.y + action_xyz[1] if action_xyz[
                1] else cur_ee_pose.pose.position.y
            ee_target.position.z = cur_ee_pose.pose.position.z + action_xyz[2] if action_xyz[
                2] else cur_ee_pose.pose.position.z

        result = self.move_manipulator.ee_traj_by_pose_target(ee_target, wait)
        return result

    def set_ee_pose(self, pose, wait=True):

        # Set up a trajectory message to publish.
        ee_target = geometry_msgs.msg.Pose()
        if isinstance(pose, dict):
            ee_target.orientation.x = pose["qx"]
            ee_target.orientation.y = pose["qy"]
            ee_target.orientation.z = pose["qz"]
            ee_target.orientation.w = pose["qw"]

            ee_target.position.x = pose["x"]
            ee_target.position.y = pose["y"]
            ee_target.position.z = pose["z"]
        else:
            ee_target = pose

        result = self.move_manipulator.ee_traj_by_pose_target(ee_target, wait)
        return result

    def set_ee_pose_from_trans_rot(self, trans, rot, wait=True):
        # Set up a trajectory message to publish.

        req_pose = self.get_ee_pose(as_message=True)

        req_pose.pose.position.x = trans[0]
        req_pose.pose.position.y = trans[1]
        req_pose.pose.position.z = trans[2]
        req_pose.pose.orientation.x = rot[0]
        req_pose.pose.orientation.y = rot[1]
        req_pose.pose.orientation.z = rot[2]
        req_pose.pose.orientation.w = rot[3]

        result = self.set_ee_pose(req_pose, wait=wait)
        return result

    def set_trajectory_joints(self, positions_array):

        result = self.move_manipulator.joint_traj(positions_array)

        return result

    def get_ee_pose(self, rot_as_euler=False, as_message=False):
        """
        """
        if not as_message:
            gripper_pose = self.move_manipulator.ee_pose().pose
            x, y, z = gripper_pose.position.x, gripper_pose.position.y, gripper_pose.position.z

            if rot_as_euler:
                roll, pitch, yaw = tf.transformations.euler_from_quaternion((gripper_pose.orientation.x,
                                                                             gripper_pose.orientation.y,
                                                                             gripper_pose.orientation.z,
                                                                             gripper_pose.orientation.w))

                rot = [roll, pitch, yaw]
            else:
                rot = [gripper_pose.orientation.x,
                       gripper_pose.orientation.y,
                       gripper_pose.orientation.z,
                       gripper_pose.orientation.w]

            return [x, y, z], rot
        else:
            self.move_manipulator.ee_pose()

    def get_ee_rpy(self):
        gripper_rpy = self.move_manipulator.ee_rpy()
        return gripper_rpy
