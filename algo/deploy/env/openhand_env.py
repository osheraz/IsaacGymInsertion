import numpy
import rospy
# import tf
from hand_control.srv import close, TargetPos, TargetAngles
from std_msgs.msg import Float32MultiArray, Bool
from std_srvs.srv import Empty
import numpy as np


class OpenhandEnv():
    """Superclass for all Openhand environments.
    """

    def __init__(self):
        """
        Do some magic

        Args:
        """
        rospy.logdebug("Start OpenhandEnv ...")

        self.wait_env_ready()
        # Internal Vars
        self.gripper_joint_names = ["base_to_finger_1_1",
                                    "finger_1_1_to_finger_1_2",
                                    "finger_1_2_to_finger_1_3",
                                    "base_to_finger_2_1",
                                    "finger_2_1_to_finger_2_2",
                                    "finger_2_2_to_finger_2_3",
                                    "base_to_finger_3_2",
                                    "finger_3_2_to_finger_3_3"]

        # We launch the init function of the Parent Class robot_real_env.RobotRealEnv

        self._check_all_systems_ready()

        rospy.Subscriber('/gripper/pos', Float32MultiArray, self._gripper_motor_states_callback)
        rospy.Subscriber('/gripper/load', Float32MultiArray, self._gripper_load_states_callback)
        self.init_success = False

        self.start_time = rospy.Time.now()
        # Start Services
        self._setup_tf_listener()
        self._setup_movement_system()
        # self.grasped_object = ObjectTracker()

        rospy.logdebug("Finished OpenhandEnv INIT...")

    # Methods needed by the RobotRealEnv
    # ----------------------------

    def wait_env_ready(self):

        import time
        import sys
        for i in range(3):
            print("OpenhandEnv: WAITING..." + str(i))
            sys.stdout.flush()
            time.sleep(1.0)

        print("OpenhandEnv: WAITING...DONE")

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other robot systems are
        operational.
        """
        rospy.logdebug("OpenhandEnv check_all_systems_ready...")
        self._check_all_sensors_ready()

        rospy.logdebug("END OpenhandEnv _check_all_systems_ready...")
        return True

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        self._check_hand_connection()
        rospy.logdebug("ALL SENSORS READY")

    def _check_hand_connection(self):
        self.gripper_motor_state_check = None
        rospy.logdebug(
            "Waiting for gripper/pos to be READY...")
        while self.gripper_motor_state_check is None and not rospy.is_shutdown():
            try:
                self.gripper_motor_state_check = rospy.wait_for_message(
                    "gripper/pos", Float32MultiArray, timeout=5.0)
                rospy.logdebug(
                    "Current gripper/pos READY=>")

            except:
                rospy.logerr(
                    "Current gripper/pos not ready yet, retrying")
        return self.gripper_motor_state_check

    def _gripper_motor_states_callback(self, msg):
        self.gripper_motor_state = numpy.array(msg.data)[1:]  # without abduction

    def _gripper_load_states_callback(self, msg):
        self.gripper_load_state = numpy.array(msg.data)[1:]  # without abduction

    def _setup_tf_listener(self):
        """
        Set ups the TF listener for getting the transforms you ask for.
        """
        pass
        # self.listener = tf.TransformListener()

    def _setup_movement_system(self):
        """
        Setup of the movement system.
        :return:
        """
        self.gripper_control = GripperTendonController()
        self.init_success = True

    ########################################
    ### Gripper ############################
    ########################################

    def get_gripper_motor_state(self):
        return self.gripper_motor_state

    def get_gripper_load_state(self):
        return self.gripper_load_state

    def to_radians(self, qn):

        # motors returns a normalized angle
        # convert theta from 0 - 1 to 0 - 2* pi, with interpolating from min/max motor setting
        # q_to_angle = np.interp(qn, [0, 1.0], [0.002, 0.99]) - dont think this is necceray
        q_to_angle = np.interp(qn, [0, 1], [0, 2 * np.pi])  # theta_transformed

        return q_to_angle

    def get_act_joint_limits(self):

        up_limits_array = [1.0, 1.0, 1.0]

        down_limits_array = [0.0, 0.0, 0.0]

        return up_limits_array, down_limits_array

    def set_gripper_motors(self, motors_positions_array):
        """
        Moves all the motors to the given position
        :param: joints_positions_array: array that ahas the desired joint positions in radians.
        """
        to_move = motors_positions_array.tolist() if not isinstance(motors_positions_array,
                                                                    list) else motors_positions_array
        suc = self.gripper_control.move_gripper(to_move)

        return suc

    def set_gripper_joints_to_init(self):

        self.gripper_control.init_position()
        return True

    def grasp(self):

        self.gripper_control.grasp()

    def jiggle_jiggle(self):
        """
        shaky shaky
        """
        import random
        def generate_explore_actions(num_actions=3, sequence_length=6):
            explore_action = []
            possible_values = [-1, 1]

            for i in range(sequence_length):
                if i == 0:
                    # Start with a random action ensuring at least one 1
                    action = [random.choice(possible_values) for _ in range(num_actions)]
                    while 1 not in action:
                        action = [random.choice(possible_values) for _ in range(num_actions)]
                else:
                    action = []
                    for j in range(num_actions):
                        if explore_action[-1][j] == -1:
                            # If the previous action had -1, set +1
                            action.append(1)
                        else:
                            # Otherwise, randomly choose between -1 and 1
                            action.append(random.choice(possible_values))

                    # Ensure at least one 1 in the current action
                    if 1 not in action:
                        action[random.randint(0, num_actions - 1)] = 1

                explore_action.append(action)

            return explore_action

        n = 4
        amount = 0.05

        # Generate the exploration action sequence
        explore_action = generate_explore_actions(num_actions=3, sequence_length=6)

        for act in explore_action:
            for _ in range(n):
                if not self.set_gripper_motors(amount * np.array(act)):
                    assert False, "Initial error initialisation is failed...." + str(act)

        self.gripper_control.grasp()


class GripperTendonController(object):

    def __init__(self):

        # create the connection to the action server
        self.num_fingers = 3
        self.act_angles = numpy.zeros(self.num_fingers)  # Normalized actuators angles [0,1]

        self.move_srv = rospy.ServiceProxy('/MoveGripper', TargetAngles)
        self.open_srv = rospy.ServiceProxy('/OpenGripper', Empty)
        self.close_srv = rospy.ServiceProxy('/CloseGripper', close)

    def move_gripper(self, act_pos_array, time_out=3.0, error=0.02):

        # act_pos_array = max(min(act_pos_array, 0.04), 0)
        #                                [ 1, 1 ,1 ]
        act_pos_array = numpy.hstack((0, act_pos_array))  # add abduction
        suc = self.move_srv(act_pos_array).success
        # rospy.logerr(suc)
        # self.wait_for_joints_to_get_there(act_pos_array, error=error, timeout=time_out)
        return suc

    def init_position(self):
        # We wait what it takes to reset pose
        self.open_srv()

    def grasp(self):
        self.close_srv()

    def wait_for_joints_to_get_there(self, desired_pos_array, error=0.2, timeout=3.0):

        time_waiting = 0.0
        frequency = 10.0
        are_equal = False
        is_timeout = False
        rospy.logwarn("Waiting for joint to get to the position")
        while not are_equal and not is_timeout and not rospy.is_shutdown():

            current_pos = [self.act_angles]

            are_equal = numpy.allclose(a=current_pos,
                                       b=desired_pos_array,
                                       atol=error)

            rospy.logdebug("are_equal=" + str(are_equal))
            rospy.logdebug(str(desired_pos_array))
            rospy.logdebug(str(current_pos))
            rate = rospy.Rate(10)

            rate.sleep()
            if timeout == 0.0:
                # We wait what it takes
                time_waiting += 0.0
            else:
                time_waiting += 1.0 / frequency
            is_timeout = time_waiting > timeout

        rospy.logwarn(
            "Actuaturs are in the desired position with an erro of " + str(error))

    def get_act_angels(self):
        msg = rospy.wait_for_message('/gripper/pos', Float32MultiArray)
        return numpy.array(msg.data)

    def delta_joints(self, delta_array):
        """
        :return:
        """
        new_pos_array = len(delta_array) * [0.0]
        i = 0
        act_angles = self.get_act_angels()
        for delta in delta_array:
            new_pos_array[i] = act_angles + delta
            i += 1
        # enforce limit
        new_pos_array = max(min(new_pos_array, 0.04), 0)

        self.move_gripper(new_pos_array)


if __name__ == "__main__":

    rospy.init_node('example')

    hand = OpenhandEnv()

    for i in range(10):
        hand.jiggle_jiggle()