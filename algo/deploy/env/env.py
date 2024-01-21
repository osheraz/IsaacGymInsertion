import rospy
from algo.deploy.env.hand_ros import HandROSSubscriberFinger
from algo.deploy.env.openhand_env import OpenhandEnv
from algo.deploy.env.robots import RobotWithFtEnv
from std_msgs.msg import Bool, Float64MultiArray, Float32MultiArray
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_matrix, quaternion_from_matrix, euler_from_quaternion, quaternion_from_euler
from isaacgyminsertion.tasks.factory_tactile.factory_utils import quat2R
import numpy as np

class ExperimentEnv:
    """ Superclass for all Robots environments.
    """

    def __init__(self, with_arm=True, with_hand=True, with_tactile=True):
        rospy.logwarn('Setting up the environment')


        if with_hand:
            self.hand = OpenhandEnv()
            self.hand.set_gripper_joints_to_init()
        if with_tactile:
            self.tactile = HandROSSubscriberFinger()
        if with_arm:
            self.arm = RobotWithFtEnv()

        rospy.sleep(2)
        if with_arm and with_tactile:
            self.ready = self.arm.init_success and self.tactile.init_success

        # TODO: move to a yaml or something..
        self.joint_start_insert = [0.00462375348434,  0.413038998842, -0.00556655274704, -1.79681813717, 0.00278532551602,  0.931868672371,  -1.57314860821]
        self.joints_socket_pos = [-0.0564706847072, 0.476379305124, 0.0717663317919,  -1.82776355743, -0.0441524721682, 0.838986992836, -1.53421509266]
        self.joints_grasp_pos = [0.214260026813, 0.469324231148, 0.174929410219, -1.6954228878, -0.0941470190883, 0.984972000122, -1.14854979515]
        self.joints_above_plug = [0.20592649281,0.389553636312, 0.184911131859,  -1.61935830116,  -0.0766384452581,1.13970589638, -1.1620862484]
        self.joints_above_socket = [0.00512505369261, 0.306541800499, -0.00611614901572, -1.69656717777, 0.00215246272273, 1.13829433918, -1.57246220112]

        self.pub_regularize = rospy.Publisher('/iiwa/regularize', Bool, queue_size=10)
        # rospy.Subscriber('/hand_control/plug_camera_pose', PoseStamped, self.plug_camera_pose_callback)
        rospy.Subscriber('/external_tracker/plug_pose_world', PoseStamped, self.external_plug_pose_world_callback)
        rospy.Subscriber('/external_tracker/plug_pose_camera', PoseStamped, self.external_plug_pose_camera_callback)
        rospy.Subscriber('/external_tracker/socket_pose_camera', PoseStamped, self.external_socket_pose_camera_callback)
        rospy.Subscriber('/external_tracker/robot_base_pose_camera', PoseStamped, self.external_robot_base_pose_camera_callback)
        rospy.Subscriber('/extrinsic_contact', Float64MultiArray, self.extrinsic_contact_callback)
        rospy.Subscriber('/gripper/load', Float32MultiArray, self.gripper_load_callback)
        rospy.Subscriber('/gripper/pos', Float32MultiArray, self.gripper_pos_callback)

        self.plug_pose_camera = None
        self.plug_pose_world = None
        self.socket_pose_camera = None
        self.robot_base_pose_camera = None
        self.extrinsic_contact = None
        self.gripper_load = None
        self.gripper_pos = None

    def regularize_force(self, status):
        self.pub_regularize.publish(status)

    def convert_msg_to_matrix(self, msg):
        mat = quaternion_matrix([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        mat[:3, 3] = np.array([msg.position.x, msg.position.y, msg.position.z])
        return mat

    def get_obs(self):
        ft = self.arm.robotiq_wrench_filtered_state.tolist()
        left, right, bottom = self.tactile.get_frames()
        pos, quat = self.arm.get_ee_pose()
        joints = self.arm.get_joint_values()
        plug_pose_world = self.get_plug_pose_world()
        plug_pose_camera = self.get_plug_pose_camera()
        extrinsic_contact = self.get_extrinsic_contact()

        return {'joints': joints,
                'ee_pose': pos + quat,
                'ft': ft,
                'frames': (left, right, bottom),
                'plug_pose_world': plug_pose_world,
                'plug_pose_camera': plug_pose_camera,
                'extrinsic_contact': extrinsic_contact,
                }

    def get_info_for_control(self):
        pos, quat = self.arm.get_ee_pose()
        joints = self.arm.get_joint_values()
        jacob = self.arm.get_jacobian_matrix().tolist()
        camera_pose = self.arm.get_camera_pose()
        plug_pose_world = self.get_plug_pose_world()
        plug_pose_camera = self.get_plug_pose_camera()
        extrinsic_contact = self.get_extrinsic_contact()

        return {'joints': joints,
                'ee_pose': pos + quat,
                'jacob': jacob,
                'camera_pose': camera_pose,
                'plug_pose_world': plug_pose_world,
                'plug_pose_camera': plug_pose_camera,
                'extrinsic_contact': extrinsic_contact,
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

    def release(self):

        self.hand.set_gripper_joints_to_init()

    def move_to_joint_values(self, values, wait=False):

        result = self.arm.set_trajectory_joints(values, wait=wait)

    def external_plug_pose_camera_callback(self, msg):
        self.plug_pose_camera = self.convert_msg_to_matrix(msg.pose)

    def get_plug_pose_camera(self):
        return self.plug_pose_camera
    
    def external_plug_pose_world_callback(self, msg):
        self.plug_pose_world = self.convert_msg_to_matrix(msg.pose)
    
    def get_plug_pose_world(self):
        return self.plug_pose_world
    
    def external_socket_pose_camera_callback(self, msg):
        self.socket_pose_camera = self.convert_msg_to_matrix(msg.pose)
    
    def get_socket_pose_camera(self):
        return self.socket_pose_camera
    
    def external_robot_base_pose_camera_callback(self, msg):
        self.robot_base_pose_camera = self.convert_msg_to_matrix(msg.pose)
    
    def get_robot_base_pose_camera(self):
        return self.robot_base_pose_camera
    
    def extrinsic_contact_callback(self, msg):
        self.extrinsic_contact = np.array(msg.data)
    
    def get_extrinsic_contact(self):
        return self.extrinsic_contact
    
    def gripper_load_callback(self, msg):
        self.gripper_load = np.array(msg.data)
    
    def get_gripper_load(self):
        return self.gripper_load
    
    def gripper_pos_callback(self, msg):
        self.gripper_pos = np.array(msg.data)
    
    def get_gripper_pos(self):
        return self.gripper_pos

