import rospy
from algo.deploy.env.hand_ros import HandROSSubscriberFinger
from algo.deploy.env.openhand_env import OpenhandEnv
from algo.deploy.env.robots import RobotWithFtEnv
from algo.deploy.env.apriltag_tracker import Tracker
from std_msgs.msg import Bool
import numpy as np

class ExperimentEnv:
    """ Superclass for all Robots environments.
    """

    def __init__(self, with_arm=True, with_hand=True, with_tactile=True, with_ext_cam=True):
        rospy.logwarn('Setting up the environment')

        if with_hand:
            self.hand = OpenhandEnv()
            self.hand.set_gripper_joints_to_init()
        if with_tactile:
            self.tactile = HandROSSubscriberFinger()
        if with_arm:
            self.arm = RobotWithFtEnv()
        if with_ext_cam:
            self.tracker = Tracker()
            self.tracker.set_object_id(6)
        rospy.sleep(2)
        if with_arm and with_tactile:
            self.ready = self.arm.init_success and self.tactile.init_success

        self.pub_regularize = rospy.Publisher('/manipulator/regularize', Bool, queue_size=10)

    def regularize_force(self, status):
        self.pub_regularize.publish(status)

    def get_obs(self):
        ft = self.arm.robotiq_wrench_filtered_state.tolist()
        left, right, bottom = self.tactile.get_frames()
        pos, quat = self.arm.get_ee_pose()
        joints = self.arm.get_joint_values()

        return {'joints': joints,
                'ee_pose': pos + quat,
                'ft': ft,
                'frames': (left, right, bottom),
                }

    def get_info_for_control(self):
        pos, quat = self.arm.get_ee_pose()
        joints = self.arm.get_joint_values()
        jacob = self.arm.get_jacobian_matrix().tolist()

        return {'joints': joints,
                'ee_pose': pos + quat,
                'jacob': jacob,
                }

    def get_frames(self):
        left, right, bottom = self.tactile.get_frames()

        return left, right, bottom

    def get_ft(self):
        ft = self.arm.robotiq_wrench_filtered_state.tolist()

        return ft

    def move_to_init_state(self):
        self.arm.move_to_init()
        self.hand.set_gripper_joints_to_init()

    def grasp(self, ):

        self.hand.grasp()

    def adapt_and_grasp(self,):

        for i in range(5):

            ee_pos, ee_quat = self.arm.get_ee_pose()
            obj_pos = self.tracker.get_obj_relative_pos()

            if not np.isnan(np.sum(obj_pos)):
                # added delta_x/delta_y to approximately center the object
                ee_pos[0] -= obj_pos[0] + 0.025
                ee_pos[1] -= obj_pos[1]
                ee_pos[2] -= obj_pos[2] - 0.07

                self.arm_movement_result = self.set_ee_pose(tp)

                self.grasp()

                tp.pose.position.z += 0.03
                self.arm_movement_result = self.set_ee_pose(tp)

                self.init_load = self.get_gripper_load_state()
                self.start_obj_rpy = self.get_obj_rpy()
                return True
            else:
                rospy.logerr('Object is undetectable, attempt: ' + str(i))

        return False
    def release(self):

        self.hand.set_gripper_joints_to_init()

    def move_to_joint_values(self, values, wait=False):

        result = self.arm.set_trajectory_joints(values, wait=wait)

