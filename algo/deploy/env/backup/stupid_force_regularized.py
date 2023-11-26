#!/usr/bin/env python

import rospy
from robotiq_force_torque_sensor.srv import sensor_accessor
import robotiq_force_torque_sensor.srv as ft_srv
from robotiq_force_torque_sensor.msg import *
import numpy as np
from geometry_msgs.msg import WrenchStamped
from iiwa_msgs.msg import JointQuantity, JointPosition
from std_msgs.msg import String, Float32MultiArray, Bool, Float32
from std_msgs.msg import Float32MultiArray, Time

class RobotWithFt():
    """ Robot with Ft sensor
    """

    def __init__(self):

        self.wait_env_ready()
        self.robotiq_wrench_filtered_state = np.array([0, 0, 0, 0, 0, 0])
        self.init_success = self._check_all_systems_ready()

        # Define feedback callbacks
        self.ft_zero = rospy.ServiceProxy('/robotiq_force_torque_sensor_acc', sensor_accessor)
        rospy.Subscriber('/iiwa/state/JointPosition', JointPosition, self._joint_state_callback)
        rospy.Subscriber('/robotiq_force_torque_wrench_filtered_exp', WrenchStamped,
                         self._robotiq_wrench_states_callback)
        rospy.Subscriber('/iiwa/regularize', Bool, self.callback_regularize)
        self.pub_joints_api = rospy.Publisher('/iiwa/command/JointPosition', JointPosition, queue_size=10)

        self.regularize = False

    def callback_regularize(self, msg):

        self.regularize = msg.data

    def wait_env_ready(self):

        import time

        for i in range(1):
            print("WAITING..." + str(i))
            sys.stdout.flush()
            time.sleep(1.0)

        print("WAITING...DONE")

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other robot systems are
        operational.
        """
        rospy.logdebug("Manipulator check_all_systems_ready...")
        self._check_arm_joint_state_ready()
        self._check_robotiq_connection()
        rospy.logdebug("END Manipulator _check_all_systems_ready...")
        return True

    def _check_arm_joint_state_ready(self):

        self.arm_joint_state = None
        rospy.logdebug(
            "Waiting for arm_controller/state to be READY...")
        while self.arm_joint_state is None and not rospy.is_shutdown():
            try:
                self.arm_joint_state = rospy.wait_for_message(
                    "/iiwa/state/JointPosition", JointPosition, timeout=5.0)
                rospy.logdebug(
                    "Current arm_controller/state READY=>")

            except:
                rospy.logerr(
                    "Current /iiwa/state/JointPosition not ready yet, retrying for getting State")
        return self.arm_joint_state

    def _joint_state_callback(self, msg):

        joints = [msg.position.a1,
                  msg.position.a2,
                  msg.position.a3,
                  msg.position.a4,
                  msg.position.a5,
                  msg.position.a6,
                  msg.position.a7]

        self.arm_joint_state = joints

    def stop_motion(self, wait=False):

        msg = rospy.wait_for_message('/iiwa/state/JointPosition', JointPosition)

        js = JointPosition()
        js.header.seq = 0
        js.header.stamp = rospy.Time(0)
        js.header.frame_id = "world"

        positions_array = self.arm_joint_state
        js.position.a1 = positions_array[0]
        js.position.a2 = positions_array[1]
        js.position.a3 = positions_array[2]
        js.position.a4 = positions_array[3]
        js.position.a5 = positions_array[4]
        js.position.a6 = positions_array[5]
        js.position.a7 = positions_array[6]

        self.pub_joints_api.publish(js)

        if wait:
            rospy.wait_for_message('/iiwa/state/DestinationReached', Time)


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

    def _check_robotiq_connection(self):

        self.robotiq_wrench_filtered_state = np.array([0, 0, 0, 0, 0, 0])
        rospy.logdebug(
            "Waiting for robotiq_force_torque_wrench_filtered to be READY...")
        while not np.sum(self.robotiq_wrench_filtered_state) and not rospy.is_shutdown():
            try:
                self.robotiq_wrench_filtered_state = rospy.wait_for_message(
                    "robotiq_force_torque_wrench_filtered", WrenchStamped, timeout=5.0)
                self.robotiq_wrench_filtered_state = np.array([self.robotiq_wrench_filtered_state.wrench.force.x,
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

    def _robotiq_wrench_states_callback(self, data):

        self.robotiq_wrench_filtered_state = np.array([data.wrench.force.x,
                                                       data.wrench.force.y,
                                                       data.wrench.force.z,
                                                       data.wrench.torque.x,
                                                       data.wrench.torque.y,
                                                       data.wrench.torque.z])


if __name__ == "__main__":

    rospy.init_node('stupid_force_regularized')

    rate = rospy.Rate(100)
    max_pressure = 3.0
    robot = RobotWithFt()

    while not rospy.is_shutdown():

        if robot.regularize:

            rospy.loginfo('Calibrating FT')
            robot.calib_robotiq()
            rospy.sleep(0.5)
            robot.calib_robotiq()

            ft_init = robot.robotiq_wrench_filtered_state

            while not rospy.is_shutdown() and robot.regularize:

                ft = robot.robotiq_wrench_filtered_state - ft_init

                if abs(ft[2]) > max_pressure:
                    robot.stop_motion()
                    rospy.loginfo('Stopping motion')

                rate.sleep()

        rate.sleep()
