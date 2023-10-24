import rospy
from algo.deploy.env.hand_ros import HandROSSubscriberFinger
from algo.deploy.env.openhand_env import OpenhandEnv
from algo.deploy.env.robots import RobotWithFtEnv


class ExperimentEnv:
    """ Superclass for all Robots environments.
    """

    def __init__(self, ):
        rospy.logwarn('Setting up the environment')
        self.hand = OpenhandEnv()
        self.tactile = HandROSSubscriberFinger()
        self.arm = RobotWithFtEnv()
        rospy.sleep(2)
        self.ready = self.arm.init_success and self.tactile.init_success

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

    def move_to_joint_values(self, values, wait=False):

        result = self.arm.set_trajectory_joints(values, wait=wait)

