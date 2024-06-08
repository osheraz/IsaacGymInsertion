from std_msgs.msg import String, Float32MultiArray, Bool, Int16, Float64MultiArray
import numpy
import rospy


class Tracker():
    """ simple apriltag tracker
    """

    def __init__(self):

        self.obj_relative_pos, self.obj_relative_rpy, self.obj_pos, self.obj_rpy, self.extrinsic_contact = [], [], [], [], []

        self.drop = False
        self.drop_counter = 0

        rospy.Subscriber('/hand_control/obj_relative_pos', Float32MultiArray, self._object_relative_pose_callback)
        rospy.Subscriber('/hand_control/obj_relative_rpy', Float32MultiArray, self._object_relative_rpy_callback)
        rospy.Subscriber('/hand_control/obj_pos', Float32MultiArray, self._object_pose_callback)
        rospy.Subscriber('/hand_control/obj_rpy', Float32MultiArray, self._object_rpy_callback)
        rospy.Subscriber('/hand_control/drop', Bool, self._object_drop_callback)
        rospy.Subscriber('/extrinsic_contact', Float64MultiArray, self.extrinsic_contact_callback)

        self.pub_obj_id = rospy.Publisher('/object_id', Int16, queue_size=10)
        self.init_success = True

    def extrinsic_contact_callback(self, msg):
        self.extrinsic_contact = numpy.array(msg.data)

    def _object_pose_callback(self, msg):
        self.obj_pos = numpy.array(msg.data, dtype=numpy.float) if not numpy.isnan(numpy.sum(numpy.array(msg.data))) else self.obj_pos

    def _object_rpy_callback(self, msg):
        self.obj_rpy = numpy.array(msg.data, dtype=numpy.float) if not numpy.isnan(numpy.sum(numpy.array(msg.data))) else self.obj_rpy

    def _object_relative_pose_callback(self, msg):
        self.obj_relative_pos = numpy.array(msg.data, dtype=numpy.float) if not numpy.isnan(
            numpy.sum(numpy.array(msg.data))) else self.obj_relative_pos

    def _object_relative_rpy_callback(self, msg):
        self.obj_relative_rpy = numpy.array(msg.data, dtype=numpy.float) if not numpy.isnan(
            numpy.sum(numpy.array(msg.data))) else self.obj_relative_rpy

    def _object_drop_callback(self, msg):
        if msg.data:
            self.drop_counter += 1
            self.drop = self.drop_counter >= 10
        else:
            self.drop_counter = 0
            self.drop = False

    def get_obj_relative_pos(self):
        return self.obj_relative_pos

    def get_obj_relative_rpy(self):
        return self.obj_relative_rpy

    def get_obj_pos(self):
        # msg = rospy.wait_for_message('/hand_control/obj_pos', Float32MultiArray)
        return self.obj_pos

    def get_obj_rpy(self):
        return self.obj_rpy

    def set_object_id(self, obj_id, pub=True):
        #         self.object_map = {'circle': 0,
        #                            'hexagon': 3,
        #                            'ellipse': 4,
        #                            'rectangle': 5,
        #                            'square': 6,
        #                            'box': 9,
        #                            'wire': 11,
        #                            'gum': 10,
        #                            'star': 7}
        self.object_id = obj_id
        # TODO: move to yaml
        # if self.object_id == 0:
        #     self.grasp_area = self.circle_grasp_area
        #     self.object_start = self.circle_start
        #     self.object_start_top = self.circle_start_top
        # elif self.object_id == 6:  # 6
        #     self.grasp_area = self.hexagon_grasp_area
        #     self.object_start = self.hexagon_start
        #     self.object_start_top = self.hexagon_start_top
        # elif self.object_id == 4:
        #     self.grasp_area = self.ellipse_grasp_area
        #     self.object_start = self.ellipse_start
        #     self.object_start_top = self.ellipse_start_top
        # elif self.object_id == 5:
        #     self.grasp_area = self.rectangle_grasp_area
        #     self.object_start = self.rectangle_start
        #     self.object_start_top = self.rectangle_start_top
        # elif self.object_id == 3:
        #     self.grasp_area = self.square_grasp_area
        #     self.object_start = self.square_start
        #     self.object_start_top = self.square_start_top
        # elif self.object_id == 9 or self.object_id == 10 or self.object_id == 11 or self.object_id == 7:
        #     self.grasp_area = self.rectangle_grasp_area
        #     self.object_start = self.rectangle_start
        #     self.object_start_top = self.rectangle_start_top
        # else:
        #     assert False, "Please set up object id"

        if pub:
            obj_msg_id = Int16()
            obj_msg_id.data = obj_id
            for _ in range(10):  # for hand_control TODO: change..
                self.pub_obj_id.publish(obj_msg_id)

        # self.move_manipulator.scene.remove_world_object()
        # rospy.logerr('Changed grasped object to: ' + find_key(self.object_map, self.object_id))

        def find_key(input_dict, value):
            for key, val in input_dict.items():
                if val == value: return key
            return "None"
