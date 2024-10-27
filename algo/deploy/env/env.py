import rospy
import torch

from algo.deploy.env.hand_ros import HandROSSubscriberFinger
from algo.deploy.env.openhand_env import OpenhandEnv
from algo.deploy.env.robots import RobotWithFtEnv
from algo.deploy.env.apriltag_tracker import Tracker
from algo.deploy.env.zed_camera import ZedCameraSubscriber
from algo.deploy.env.pcl_camera import ZedPointCloudSubscriber

from std_msgs.msg import Bool
import geometry_msgs.msg

import numpy as np


class ExperimentEnv:
    """ Superclass for all Robots environments.
    """

    def __init__(self, with_arm=True, with_hand=True, with_tactile=True,
                 with_ext_cam=False, with_depth=False, with_pcl=False):

        rospy.logwarn('Setting up the environment')

        self.with_depth = with_depth
        self.with_hand = with_hand
        self.with_arm = with_arm
        self.with_tactile = with_tactile
        self.with_ext_cam = with_ext_cam
        self.with_pcl = with_pcl

        self.ready = True  # Start with the assumption that everything will be initialized correctly

        if with_hand:
            self.hand = OpenhandEnv()
            self.hand.set_gripper_joints_to_init()
            self.ready = self.ready and self.hand.init_success

        if with_tactile:
            self.tactile = HandROSSubscriberFinger()
            self.ready = self.ready and self.tactile.init_success

        if with_arm:
            self.arm = RobotWithFtEnv()
            self.ready = self.ready and self.arm.init_success

        if with_ext_cam:
            self.tracker = Tracker()
            self.tracker.set_object_id(6)
            self.ready = self.ready and self.tracker.init_success

        if with_depth:
            self.zed = ZedCameraSubscriber()
            self.ready = self.ready and self.zed.init_success

        if with_pcl:
            self.pcl_gen = ZedPointCloudSubscriber()
            self.ready = self.ready and self.pcl_gen.zed_init

        rospy.sleep(2)

        self.pub_regularize = rospy.Publisher('/manipulator/regularize', Bool, queue_size=10)
        rospy.logwarn('Env is ready')

    def regularize_force(self, status):
        self.pub_regularize.publish(status)

    def get_obs(self):

        obs = {}
        if self.with_arm:
            ft = self.arm.robotiq_wrench_filtered_state.tolist()
            pos, quat = self.arm.get_ee_pose()
            joints = self.arm.get_joint_values()

            obs.update({
                'joints': joints,
                'ee_pose': pos + quat,
                'ft': ft
            })

        if self.with_tactile:
            left, right, bottom = self.tactile.get_frames()
            obs['frames'] = (left, right, bottom)

        if self.with_depth:
            img, seg = self.zed.get_frame()
            obs['img'] = img
            obs['seg'] = seg

        if self.with_pcl:
            pcl, sync_rgb, sync_depth, sync_seg, sync_rec_rgb = self.pcl_gen.get_pcl()
            obs['pcl'] = pcl
            obs['rgb'] = sync_rec_rgb
            obs['img'] = sync_depth
            obs['seg'] = sync_seg

        return obs

    def get_extrinsic(self):

        return self.tracker.extrinsic_contact

    def get_info_for_control(self):
        pos, quat = self.arm.get_ee_pose()

        joints = self.arm.get_joint_values()
        jacob = self.arm.get_jacobian_matrix().tolist()

        return {'joints': joints,
                'ee_pose': pos + quat,
                'jacob': jacob,
                }

    def get_img(self):

        img, seg = self.zed.get_frame()
        return img, seg

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

    def get_hand_motor_state(self, normalized=True):

        motors = self.hand.get_gripper_motor_state()

        if not normalized:
            motors = self.hand.to_radians(motors)

        return motors

    def align_and_grasp(self, ):

        # TODO change align and grasp to dof_relative funcs without moveit

        for i in range(5):

            # ee_pos, ee_quat = self.arm.get_ee_pose()
            ee_pose = self.arm.move_manipulator.get_cartesian_pose_moveit()
            ee_pos = [ee_pose.position.x, ee_pose.position.y, ee_pose.position.z]
            ee_quat = [ee_pose.orientation.x, ee_pose.orientation.y, ee_pose.orientation.z, ee_pose.orientation.w]
            obj_pos = self.tracker.get_obj_pos()
            obj_pos[-1] += 0.075

            if not np.isnan(np.sum(obj_pos)):

                # added delta_x/delta_y to approximately center the object
                ee_pos[0] = obj_pos[0] - 0.02
                ee_pos[1] = obj_pos[1] - 0.01
                ee_pos[2] = obj_pos[2] - 0.01

                # Orientation is different due to moveit orientation, kinova/orientation ( -0.707,0.707,0,0 ~ 0.707,-0.707,0,0)
                ee_target = geometry_msgs.msg.Pose()
                ee_target.orientation.x = ee_quat[0]
                ee_target.orientation.y = ee_quat[1]
                ee_target.orientation.z = ee_quat[2]
                ee_target.orientation.w = ee_quat[3]

                ee_target.position.x = ee_pos[0]
                ee_target.position.y = ee_pos[1]
                ee_target.position.z = ee_pos[2]
                self.arm_movement_result = self.arm.set_ee_pose(ee_target)

                self.grasp()

                return True
            else:
                rospy.logerr('Object is undetectable, attempt: ' + str(i))

        return False

    def align_and_release(self, init_plug_pose):

        # TODO change align and grasp to dof_relative funcs without moveit

        for i in range(5):

            ee_pose = self.arm.move_manipulator.get_cartesian_pose_moveit()
            ee_pos = [ee_pose.position.x, ee_pose.position.y, ee_pose.position.z]
            ee_quat = [ee_pose.orientation.x, ee_pose.orientation.y, ee_pose.orientation.z, ee_pose.orientation.w]
            obj_pos = self.tracker.get_obj_pos()
            obj_height = 0
            init_delta_height = 0.03

            if not np.isnan(np.sum(obj_pos)):

                # added delta_x/delta_y to approximately center the object
                ee_pos[0] = init_plug_pose[0] + (ee_pos[0] - obj_pos[0])
                ee_pos[1] = init_plug_pose[1] + (ee_pos[1] - obj_pos[1])
                ee_pos[2] = init_plug_pose[2] + obj_pos[2] - obj_height + init_delta_height

                # Orientation is different due to moveit orientation,
                # kinova/orientation ( -0.707,0.707,0,0 ~ 0.707,-0.707,0,0)

                ee_target = geometry_msgs.msg.Pose()
                ee_target.orientation.x = ee_quat[0]
                ee_target.orientation.y = ee_quat[1]
                ee_target.orientation.z = ee_quat[2]
                ee_target.orientation.w = ee_quat[3]

                ee_target.position.x = ee_pos[0]
                ee_target.position.y = ee_pos[1]
                ee_target.position.z = ee_pos[2]
                self.arm_movement_result = self.arm.set_ee_pose(ee_target)

                self.release()

                return True
            else:
                rospy.logerr('Object is undetectable, attempt: ' + str(i))

        return False

    def randomize_grasp(self,):

        # TODO change align and grasp to dof_relative funcs without moveit
        try:
            self.hand.jiggle_jiggle()

            return True

        except Exception as e:
            print(e)
            return False

    def set_random_init_error(self, true_socket_pose, with_tracker=True):

        # TODO change motion to be without moveit
        if torch.is_tensor(true_socket_pose):
            true_socket_pose = true_socket_pose[0].cpu().detach().numpy()

        for i in range(5):

            ee_pose = self.arm.move_manipulator.get_cartesian_pose_moveit()
            ee_pos = [ee_pose.position.x, ee_pose.position.y, ee_pose.position.z]
            ee_quat = [ee_pose.orientation.x, ee_pose.orientation.y, ee_pose.orientation.z, ee_pose.orientation.w]

            if with_tracker:
                obj_pos = self.tracker.get_obj_pos()  # tracker already gives the bottom of the object
                obj_height = 0  # 0.07
                init_delta_height = 0.08
            else:
                obj_pos = true_socket_pose
                obj_pos[-1] += 0.015
                obj_height = 0  # 0.07
                init_delta_height = 0.1

            if not np.isnan(np.sum(obj_pos)):

                rand_add = np.random.uniform(-0.01, 0.01, 2)
                # added delta_x/delta_y to approximately center the object
                ee_pos[0] = true_socket_pose[0] + (ee_pos[0] - obj_pos[0]) + rand_add[0]
                ee_pos[1] = true_socket_pose[1] + (ee_pos[1] - obj_pos[1]) + rand_add[1]
                ee_pos[2] = true_socket_pose[2] + obj_pos[2] - obj_height + init_delta_height

                # Orientation is different due to moveit orientation, kinova/orientation ( -0.707,0.707,0,0 ~ 0.707,-0.707,0,0)
                ee_target = geometry_msgs.msg.Pose()
                ee_target.orientation.x = ee_quat[0]
                ee_target.orientation.y = ee_quat[1]
                ee_target.orientation.z = ee_quat[2]
                ee_target.orientation.w = ee_quat[3]

                ee_target.position.x = ee_pos[0]
                ee_target.position.y = ee_pos[1]
                ee_target.position.z = ee_pos[2]
                self.arm_movement_result = self.arm.set_ee_pose(ee_target)

                return True
            else:
                rospy.logerr('Object is undetectable, attempt: ' + str(i))

        return False

    def release(self):

        self.hand.set_gripper_joints_to_init()

    def move_to_joint_values(self, values, wait=True, by_moveit=True, by_vel=False):

        result = self.arm.set_trajectory_joints(values, wait=wait, by_moveit=by_moveit, by_vel=by_vel)

    def move_to_pose(self, values, wait=True):

        result = self.arm.set_ee_pose(values, wait=wait)