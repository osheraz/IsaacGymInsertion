import rospy
from moveit_msgs.msg import JointLimits
import moveit_msgs.msg
import moveit_commander
import geometry_msgs.msg
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
import copy
import sys


class MoveManipulator():

    def __init__(self):
        rospy.logdebug("===== In MoveManipulator")
        moveit_commander.roscpp_initialize(sys.argv)
        arm_group_name = "manipulator"

        self.robot = moveit_commander.RobotCommander("robot_description")
        self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
        self.group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
        self.planning_frame = self.group.get_planning_frame()
        self.eef_link = self.group.get_end_effector_link()
        self.group_names = self.robot.get_group_names()
        rospy.logdebug("===== Out MoveManipulator")

    def get_jacobian_matrix(self):

        jacob = self.group.get_jacobian_matrix(self.group.get_current_joint_values())

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

    def joint_traj(self, positions_array):

        self.group_variable_values = self.group.get_current_joint_values()
        self.group_variable_values[0] = positions_array[0]
        self.group_variable_values[1] = positions_array[1]
        self.group_variable_values[2] = positions_array[2]
        self.group_variable_values[3] = positions_array[3]
        self.group_variable_values[4] = positions_array[4]
        self.group_variable_values[5] = positions_array[5]
        self.group_variable_values[6] = positions_array[6]

        self.group.set_joint_value_target(self.group_variable_values)
        result = self.execute_trajectory()

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
