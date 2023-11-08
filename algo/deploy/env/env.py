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
        self.hand.set_gripper_joints_to_init()
        self.joint_start_insert = [0.00462375348434 ,  0.413038998842, -0.00556655274704, -1.79681813717, 0.00278532551602,  0.931868672371,  -1.57314860821]
        self.joints_socket_pos = [-0.0564706847072, 0.476379305124, 0.0717663317919,  -1.82776355743, -0.0441524721682, 0.838986992836, -1.53421509266]
        self.joints_grasp_pos = [0.214260026813, 0.469324231148, 0.174929410219, -1.6954228878, -0.0941470190883, 0.984972000122, -1.14854979515]
        self.joints_above_plug = [0.20592649281,0.389553636312, 0.184911131859,  -1.61935830116,  -0.0766384452581,1.13970589638, -1.1620862484]
        self.joints_above_socket = [0.00512505369261, 0.306541800499, -0.00611614901572, -1.69656717777, 0.00215246272273, 1.13829433918, -1.57246220112]

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

