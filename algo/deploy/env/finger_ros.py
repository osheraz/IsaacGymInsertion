import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from algo.deploy.env.finger import Finger
from algo.deploy.env.env_utils.img_utils import ContactArea, circle_mask, align_center


class TactileSubscriberFinger:

    def __init__(self, dev_name, fix):
        """
        Finger Device class for a single Finger
        :param serial: Finger device serial
        :param name: Human friendly identifier name for the device
        """

        self._cv_bridge = CvBridge()
        self.dev_name = dev_name
        self.fix = fix
        self._topic_name = rospy.get_param('~topic_name', 'allsight{}/usb_cam/image_raw'.format(dev_name))
        rospy.loginfo("(topic_name) Subscribing to Images to topic  %s", self._topic_name)
        self.init_success = False
        self._check_finger_ready()

        self._image_subscriber = rospy.Subscriber(self._topic_name, Image, self.image_callback, queue_size=2)

    def _check_finger_ready(self):

        self.last_frame = None
        rospy.logdebug(
            "Waiting for 'allsight{}/usb_cam/image_raw' to be READY...".format(self.dev_name))
        while self.last_frame is None and not rospy.is_shutdown():
            try:
                self.last_frame = rospy.wait_for_message(
                    '/allsight{}/usb_cam/image_raw'.format(self.dev_name), Image, timeout=5.0)
                rospy.logdebug(
                    "Current '/allsight{}/usb_cam/image_raw' READY=>".format(self.dev_name))
                self.init_success = True
                self.start_time = rospy.get_time()
            except:
                rospy.logerr(
                    "Current '/allsight{}/usb_cam/image_raw' not ready yet, retrying for getting image".format(
                        self.dev_name))
        return self.last_frame

    def image_callback(self, msg):
        try:
            cv2_img = self._cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        else:
            self.last_frame = cv2_img

    def get_frame(self):
        if type(self.last_frame) == Image:
            self.last_frame = self._cv_bridge.imgmsg_to_cv2(self.last_frame, "bgr8")
        return self.last_frame


class TactileFingerROSPublisher(Finger):

    def __init__(self, serial=None, dev_name=None, fix=(0, 0)):
        """
        Finger Device class for a single Finger
        :param serial: Finger device serial
        :param name: Human friendly identifier name for the device
        """

        Finger.__init__(self, serial=serial, dev_name=dev_name, fix=fix)

        self._cv_bridge = CvBridge()

        self._topic_name = rospy.get_param('~topic_name', 'allsight{}/usb_cam/image_raw'.format(dev_name))
        rospy.loginfo("[%s] (topic_name) Publishing Images to topic  %s", self.name, self._topic_name)

        self._image_publisher = rospy.Publisher(self._topic_name, Image, queue_size=1)

        self._rate = rospy.get_param('~publish_rate', self.fps)
        rospy.loginfo("[%s] (publish_rate) Publish rate set to %s hz", self.name, self._rate)

        self._frame_id = rospy.get_param('~frame_id', 'finger{}'.format(dev_name))
        rospy.loginfo("[%s] (frame_id) Frame ID set to  %s", self.name, self._frame_id)

    def run(self):

        ros_rate = rospy.Rate(self._rate)
        while not rospy.is_shutdown():
            try:
                cv_image = self.get_frame()
                if cv_image is not None:
                    ros_msg = self._cv_bridge.cv2_to_imgmsg(cv_image, "bgr8")
                    ros_msg.header.frame_id = self._frame_id
                    ros_msg.header.stamp = rospy.Time.now()
                    self._image_publisher.publish(ros_msg)
                else:
                    rospy.loginfo("[%s] Invalid image file", self.name)
                ros_rate.sleep()

            except CvBridgeError as e:
                rospy.logerr(e)


if __name__ == "__main__":

    rospy.init_node('Finger')
    tactile = TactileSubscriberFinger(dev_name=0, fix=(0, 0))
    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        rate.sleep()
