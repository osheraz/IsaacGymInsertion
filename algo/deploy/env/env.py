import numpy
import rospy
import tf
from hand import Hand
from openhand_env import OpenhandEnv
from robots import RobotWithFtEnv

class ExperimentEnv:
    """ Superclass for all Robots environments.
    """

    def __init__(self, ):
        rospy.logwarn('Setting up the environment')
        self.hand = OpenhandEnv()
        self.tactile = Hand()
        self.arm = RobotWithFtEnv()
        self.listener = tf.TransformListener()
        # (trans_ee, rot_ee) = self.listener.lookupTransform('/base_link', 'end_effector_link', rospy.Time(0))
        rospy.sleep(2)
        self.ready = self.arm.init_success and self.tactile.init_success

    def get_obs(self):
        ft = self.arm.robotiq_wrench_filtered_state
        left, right, bottom = self.tactile.get_frames()
        pos, quat = self.arm.get_ee_pose()
        joints = self.arm.move_manipulator.joint_values()

        return {'joints': joints,
                'ee_pose': pos + quat,
                'ft': ft,
                'frames': (left, right, bottom),
                }

    def get_info_for_control(self):
        pos, quat = self.arm.get_ee_pose()
        joints = self.arm.move_manipulator.joint_values()
        jacob = self.arm.move_manipulator.get_jacobian_matrix()

        return {'joints': joints,
                'ee_pose': pos + quat,
                'jacob': jacob,
                }

    def get_frames(self):

        left, right, bottom = self.tactile.get_frames()

        return left, right, bottom

    def get_ft(self):

        ft = self.arm.robotiq_wrench_filtered_state

        return ft

    def move_to_init_state(self):

        pass