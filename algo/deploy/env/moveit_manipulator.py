#!/usr/bin/python

import rospy
from moveit_msgs.msg import JointLimits
import moveit_msgs.msg
import moveit_commander
import geometry_msgs.msg
from tactile_insertion.srv import MoveitJacobian, MoveitMoveJointPosition, MoveitPose, VelAndAcc, MoveitJoints, \
    MoveitMoveEefPose
from tactile_insertion.srv import MoveitJacobianResponse, MoveitMoveJointPositionResponse, MoveitPoseResponse, \
    VelAndAccResponse, MoveitJointsResponse, MoveitMoveEefPoseResponse
from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyResponse

from iiwa_msgs.msg import JointQuantity, JointPosition
from std_msgs.msg import Header, Float32MultiArray
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
import copy
import sys
import numpy as np
import tf


def avoid_jumps(q_Current, q_Prev):
    q_Current = np.array(q_Current)
    q_Prev = np.array(q_Prev)

    # norm_diff = np.linalg.norm(q_Prev - q_Current) ** 2
    # norm_sum = np.linalg.norm(q_Prev + q_Current) ** 2

    # if norm_diff < norm_sum:

    if np.sign(q_Current[0]) != np.sign(q_Prev[0]):
        return -q_Current
    else:
        return q_Current

def VecToso3(omg):
    """Converts a 3-vector to an so(3) representation

    :param omg: A 3-vector
    :return: The skew symmetric representation of omg

    Example Input:
        omg = np.array([1, 2, 3])
    Output:
        np.array([[ 0, -3,  2],
                  [ 3,  0, -1],
                  [-2,  1,  0]])
    """
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])
def TransToRp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector

    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        (np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]]),
         np.array([0, 0, 3]))
    """
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]

def Adjoint(T):
    """Computes the adjoint representation of a homogeneous transformation
    matrix

    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1, 0,  0, 0, 0,  0],
                  [0, 0, -1, 0, 0,  0],
                  [0, 1,  0, 0, 0,  0],
                  [0, 0,  3, 1, 0,  0],
                  [3, 0,  0, 0, 0, -1],
                  [0, 0,  0, 0, 1,  0]])
    """
    R, p = TransToRp(T)
    return np.r_[np.c_[R, np.zeros((3, 3))],
                 np.c_[np.dot(VecToso3(p), R), R]]

class MoveManipulator():

    def __init__(self):
        rospy.logdebug("===== In MoveManipulator")
        moveit_commander.roscpp_initialize(sys.argv)
        arm_group_name = "manipulator"
        self.tl = tf.TransformListener()

        self.robot = moveit_commander.RobotCommander("/iiwa/robot_description")
        self.scene = moveit_commander.PlanningSceneInterface(ns='/iiwa')
        self.group = moveit_commander.MoveGroupCommander(arm_group_name, ns='/iiwa')
        self.planning_frame = self.group.get_planning_frame()
        self.eef_link = self.group.get_end_effector_link()
        self.group_names = self.robot.get_group_names()
        rospy.logdebug("===== Out MoveManipulator")

    def tf_trans(self, target_frame, source_frame):
        try:
            # listen to transform, from source to target. if source is 0 and target is 1 than A_0^1
            (trans, rot) = self.tl.lookupTransform(target_frame, source_frame, rospy.Time(0))
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 'couldnt find mat', None, None

    def get_jacobian_matrix(self, joints=None):

        if joints is None:
            joints = self.group.get_current_joint_values()
            print(joints)

        jacob = self.group.get_jacobian_matrix(joints)
        # trans, rot = self.tf_trans('world', 'iiwa_link_ee')
        # T = self.tl.fromTranslationRotation(trans, rot)
        # Adj = Adjoint(T)
        # ret = np.dot(Adj, jacob)
        # print((ret - jacob).max())

        return jacob

    def scale_vel(self, scale_vel, scale_acc):
        self.group.set_max_velocity_scaling_factor(scale_vel)
        self.group.set_max_acceleration_scaling_factor(scale_acc)

    def set_constraints(self):
        joint_constraint_list = []
        above = below = 0.007
        # joint_constraint = moveit_msgs.msg.JointConstraint()
        # joint_constraint.joint_name = self.group.get_joints()[1]
        # joint_constraint.position = 0.0
        # joint_constraint.tolerance_above = above
        # joint_constraint.tolerance_below = below
        # joint_constraint.weight = 1.0
        # joint_constraint_list.append(joint_constraint)

        joint_constraint = moveit_msgs.msg.JointConstraint()
        joint_constraint.joint_name = self.group.get_joints()[4]
        joint_constraint.position = 0.0
        joint_constraint.tolerance_above = above
        joint_constraint.tolerance_below = below
        joint_constraint.weight = 1.0
        joint_constraint_list.append(joint_constraint)

        # joint_constraint = moveit_msgs.msg.JointConstraint()
        # joint_constraint.joint_name = self.group.get_joints()[6]
        # joint_constraint.position = 0.0
        # joint_constraint.tolerance_above = above
        # joint_constraint.tolerance_below = below
        # joint_constraint.weight = 1.0
        # joint_constraint_list.append(joint_constraint)

        constraint_list = moveit_msgs.msg.Constraints()
        constraint_list.name = 'todo'
        constraint_list.joint_constraints = joint_constraint_list

        self.group.set_path_constraints(constraint_list)

    def clear_all_constraints(self):

        self.group.clear_path_constraints()

    def get_planning_feedback(self):
        planning_frame = self.group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # print the name of the end-effector link for this group:
        eef_link = self.group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # get a list of all the groups in the robot:
        group_names = self.robot.get_group_names()
        print("============ Available Planning Groups:", group_names)

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(self.robot.get_current_state())

    def define_workspace_at_init(self):
        # Walls are defined with respect to the coordinate frame of the robot base, with directions
        # corresponding to standing behind the robot and facing into the table.
        rospy.sleep(0.6)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world'

        # self.robot.get_planning_frame()
        table_pose = PoseStamped()
        table_pose.header = header
        table_pose.pose.position.x = 0
        table_pose.pose.position.y = 0
        table_pose.pose.position.z = -0.0001
        self.scene.remove_world_object('bottom')
        self.scene.add_plane(name='bottom', pose=table_pose, normal=(0, 0, 1))

        upper_pose = PoseStamped()
        upper_pose.header = header
        upper_pose.pose.position.x = 0
        upper_pose.pose.position.y = 0
        upper_pose.pose.position.z = 0.6
        self.scene.remove_world_object('upper')
        self.scene.add_plane(name='upper', pose=upper_pose, normal=(0, 0, 1))

        back_pose = PoseStamped()
        back_pose.header = header
        back_pose.pose.position.x = 0
        back_pose.pose.position.y = -0.4  # -0.25
        back_pose.pose.position.z = 0
        self.scene.remove_world_object('rightWall')
        self.scene.add_plane(name='rightWall', pose=back_pose, normal=(0, 1, 0))

        front_pose = PoseStamped()
        front_pose.header = header
        front_pose.pose.position.x = -0.25
        front_pose.pose.position.y = 0.0  # 0.52 # Optimized (0.55 NG)
        front_pose.pose.position.z = 0
        self.scene.remove_world_object('backWall')
        # self.scene.add_plane(name='backWall', pose=front_pose, normal=(1, 0, 0))

        right_pose = PoseStamped()
        right_pose.header = header
        right_pose.pose.position.x = 0.45  # 0.2
        right_pose.pose.position.y = 0
        right_pose.pose.position.z = 0
        self.scene.remove_world_object('frontWall')
        self.scene.add_plane(name='frontWall', pose=right_pose, normal=(1, 0, 0))

        left_pose = PoseStamped()
        left_pose.header = header
        left_pose.pose.position.x = 0.0  # -0.54
        left_pose.pose.position.y = 0.4
        left_pose.pose.position.z = 0
        self.scene.remove_world_object('leftWall')
        self.scene.add_plane(name='leftWall', pose=left_pose, normal=(0, 1, 0))
        rospy.sleep(0.6)

    def all_close(self, goal, actual, tolerance):
        """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
        all_equal = True
        if type(goal) is list:
            for index in range(len(goal)):
                if abs(actual[index] - goal[index]) > tolerance:
                    return False

        elif type(goal) is geometry_msgs.msg.PoseStamped:
            return self.all_close(goal.pose, actual.pose, tolerance)

        # elif type(goal) is geometry_msgs.msg.Pose:
        #     return self.all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

        return True

    def reach_named_position(self, target):
        arm_group = self.group

        # Going to one of those targets
        rospy.loginfo("Going to named target " + target)
        # Set the target
        arm_group.set_named_target(target)
        return arm_group.go(wait=True)

    def ee_traj_by_cartesian_path(self, pose, wait=True):
        # self.group.set_pose_target(pose)
        # result = self.execute_trajectory(wait)
        cartesian_plan, fraction = self.plan_cartesian_path(pose)
        result = self.execute_plan(cartesian_plan, wait)
        return result

    def ee_traj_by_pose_target(self, pose, wait=True, tolerance=0.0001):  # 0.0001

        self.group.set_goal_position_tolerance(tolerance)
        self.group.set_pose_target(pose)
        result = self.execute_trajectory(wait)
        return result

    def get_cartesian_pose(self, verbose=False):
        arm_group = self.group

        # Get the current pose and display it
        pose = arm_group.get_current_pose()
        if verbose:
            rospy.loginfo("Actual cartesian pose is : ")
            rospy.loginfo(pose.pose)

        return pose.pose

    def joint_traj(self, positions_array, wait=True):

        self.group_variable_values = self.group.get_current_joint_values()
        self.group_variable_values[0] = positions_array[0]
        self.group_variable_values[1] = positions_array[1]
        self.group_variable_values[2] = positions_array[2]
        self.group_variable_values[3] = positions_array[3]
        self.group_variable_values[4] = positions_array[4]
        self.group_variable_values[5] = positions_array[5]
        self.group_variable_values[6] = positions_array[6]

        self.group.set_joint_value_target(self.group_variable_values)
        result = self.execute_trajectory(wait)

        return result

    def execute_trajectory(self, wait=True):
        """
        Assuming that the trajecties has been set to the self objects appropriately
        Make a plan to the destination in Homogeneous Space(x,y,z,yaw,pitch,roll)
        and returns the result of execution
        """
        self.plan = self.group.plan()
        result = self.group.go(wait=wait)
        self.group.clear_pose_targets()

        return result

    def ee_pose(self):
        gripper_pose = self.group.get_current_pose()
        return gripper_pose

    def ee_rpy(self):
        gripper_rpy = self.group.get_current_rpy()
        return gripper_rpy

    def joint_values(self):

        joints = self.group.get_current_joint_values()

        return joints

    def plan_cartesian_path(self, pose, eef_step=0.001):

        waypoints = []
        # start with the current pose
        # waypoints.append(self.arm_group.get_current_pose().pose)

        wpose = self.group.get_current_pose().pose  # geometry_msgs.msg.Pose()
        wpose.position.x = pose.position.x
        wpose.position.y = pose.position.y
        wpose.position.z = pose.position.z
        # wpose.orientation.x = pose.orientation.x
        # wpose.orientation.y = pose.orientation.y
        # wpose.orientation.z = pose.orientation.z
        # wpose.orientation.w = pose.orientation.w

        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = self.group.compute_cartesian_path(
            waypoints,  # waypoints to follow
            eef_step,  # eef_step
            2.0)  # jump_threshold

        # Note: We are just planning, not asking move_group to actually move the robot yet
        return plan, fraction

    def execute_plan(self, plan, wait=True):
        ## Executing a Plan
        ## ^^^^^^^^^^^^^^^^
        ## Use execute if you would like the robot to follow
        ## the plan that has already been computed:
        result = self.group.execute(plan, wait=wait)
        self.group.clear_pose_targets()
        return result

    def stop_motion(self):

        self.group.stop()
        self.group.clear_pose_targets()

class ServiceWrap:
    def __init__(self):
        """
        Just wraping everything cuz melodic don't like python3+
        """
        self.joints = None
        self.pose = None
        self.rot = None
        self.tl = tf.TransformListener()

        self.moveit = MoveManipulator()

    def tf_trans(self, target_frame, source_frame):
        try:
            # listen to transform, from source to target. if source is 0 and target is 1 than A_0^1
            (trans, rot) = self.tl.lookupTransform(target_frame, source_frame, rospy.Time(0))
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 'couldnt find mat', None, None

    def callback_iiwa_pose(self, msg):

        """
        Overwrite but keep syncing the update
        iiwa api publisher is w.r.t link7
        our publisher is w.r.t to eef.
        """

        trans, rot = self.tf_trans('world', 'iiwa_link_ee')

        # T = self.tl.fromTranslationRotation(trans, rot)
        # rot = tf.transformations.quaternion_from_matrix(T)
        # print(np.array(np.round(T[:3,:3],2)))

        # roll, pitch, yaw = tf.transformations.euler_from_quaternion((rot[0],
        #                                                              rot[1],
        #                                                              rot[2],
        #                                                             rot[3]))
        #
        # if roll < 0:
        #     roll = 2 * np.pi + roll
        # if pitch < 0:
        #     pitch = 2 * np.pi + pitch
        # if yaw < 0:
        #     yaw = 2 * np.pi + yaw
        #
        # rot = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

        # print(np.array(np.round([roll, pitch, yaw],2)))
        # if self.rot is None:
        #     self.prev_rot = rot
        #     self.rot = rot
        # else:
        #     rot = avoid_jumps(rot, self.prev_rot)
        #     self.rot = rot
        #     self.prev_rot = rot

        msg.pose.position.x = trans[0]
        msg.pose.position.y = trans[1]
        msg.pose.position.z = trans[2]

        msg.pose.orientation.x = rot[0]
        msg.pose.orientation.y = rot[1]
        msg.pose.orientation.z = rot[2]
        msg.pose.orientation.w = rot[3]

        self.pose = msg


    def callback_iiwa_joints(self, msg):
        self.joints = msg

    def callback_set_pose(self, req):
        res = MoveitMoveEefPoseResponse()
        self.moveit.ee_traj_by_pose_target(req.pose, wait=req.wait)
        return res

    def callback_set_joints(self, req):

        res = MoveitMoveJointPositionResponse()
        joints = [req.pos.a1,
                  req.pos.a2,
                  req.pos.a3,
                  req.pos.a4,
                  req.pos.a5,
                  req.pos.a6,
                  req.pos.a7]

        current_js = [self.joints.position.a1,
                  self.joints.position.a2,
                  self.joints.position.a3,
                  self.joints.position.a4,
                  self.joints.position.a5,
                  self.joints.position.a6,
                  self.joints.position.a7]

        rospy.logwarn('-------New Request -------')
        rospy.logwarn('Current {}'.format([['%.3f' % n for n in current_js]]))
        rospy.logwarn('Desired {}'.format([['%.3f' % n for n in joints]]))

        self.moveit.joint_traj(joints, wait=req.wait)
        return res

    def callback_joints(self, req):
        res = MoveitJointsResponse()
        values = self.moveit.joint_values()

        res.pos.a1 = values[0]
        res.pos.a2 = values[1]
        res.pos.a3 = values[2]
        res.pos.a4 = values[3]
        res.pos.a5 = values[4]
        res.pos.a6 = values[5]
        res.pos.a7 = values[6]

        return res

    def callback_jacobian(self, req):
        res = MoveitJacobianResponse()
        j = np.array(self.moveit.get_jacobian_matrix()).flatten()

        for i in range(len(j)):
            res.data[i] = j[i]
        return res

    def callback_pose(self, req):
        res = MoveitPoseResponse()
        p = self.moveit.ee_pose()
        res.pose = p.pose
        return res

    def callback_set_vel_acc(self, req):
        res = VelAndAccResponse()
        self.moveit.scale_vel(req.vel, req.acc)
        return res

    def callback_stop_motion(self, req):

        self.moveit.stop_motion()

        return EmptyResponse()

if __name__ == "__main__":

    '''
    ROS_NAMESPACE=iiwa rosrun tactile_insertion moveit_manipulator.py 
    '''
    rospy.init_node('arm_control', anonymous=True)

    rate = rospy.Rate(200)

    wrap = ServiceWrap()

    rospy.Service("/MoveItMoveJointPosition", MoveitMoveJointPosition, wrap.callback_set_joints)
    rospy.Service("/MoveItMoveEefPose", MoveitMoveEefPose, wrap.callback_set_pose)
    rospy.Service("/MoveItJacobian", MoveitJacobian, wrap.callback_jacobian)
    rospy.Service("/MoveItPose", MoveitPose, wrap.callback_pose)
    rospy.Service("/MoveItJoints", MoveitJoints, wrap.callback_joints)
    rospy.Service("/MoveItScaleVelAndAcc", VelAndAcc, wrap.callback_set_vel_acc)
    rospy.Service("/Stop", Empty, wrap.callback_stop_motion)

    pub_jacob = rospy.Publisher('/iiwa/Jacobian', Float32MultiArray, queue_size=10)
    pub_joints = rospy.Publisher('/iiwa/Joints', JointPosition, queue_size=10)
    pub_pose = rospy.Publisher('/iiwa/Pose', PoseStamped, queue_size=10)

    rospy.Subscriber('/iiwa/state/JointPosition', JointPosition, wrap.callback_iiwa_joints)
    rospy.Subscriber('/iiwa/state/CartesianPose', PoseStamped, wrap.callback_iiwa_pose)

    rospy.wait_for_message('/iiwa/state/JointPosition', JointPosition)
    rospy.wait_for_message('/iiwa/state/CartesianPose', PoseStamped)

    rospy.logwarn('[arm_control] node is ready')

    while not rospy.is_shutdown():

        joints = wrap.joints
        pub_joints.publish(joints)

        joints = [joints.position.a1,
                  joints.position.a2,
                  joints.position.a3,
                  joints.position.a4,
                  joints.position.a5,
                  joints.position.a6,
                  joints.position.a7]

        jacob = np.array(wrap.moveit.get_jacobian_matrix(joints)).flatten().tolist()

        msg = Float32MultiArray()
        msg.data = jacob
        pub_jacob.publish(msg)

        pub_pose.publish(wrap.pose)

        rate.sleep()
