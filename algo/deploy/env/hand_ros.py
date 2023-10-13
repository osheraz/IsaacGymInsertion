import numpy as np
import cv2
from finger_ros import TactileSubscriberFinger
from hand import Hand
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy


class HandROSSubscriberFinger():

    def __init__(self, dev_names=None):
        """
        Finger Device class for a single Finger
        :param serial: Finger device serial
        :param name: Human friendly identifier name for the device
        """

        if dev_names is None:
            dev_names = [2, 0, 4]

        self.finger_left = TactileSubscriberFinger(dev_name=dev_names[0])
        self.finger_right = TactileSubscriberFinger(dev_name=dev_names[1])
        self.finger_bottom = TactileSubscriberFinger(dev_name=dev_names[2])

    def get_frames(self):
        """
        Returns a single image frame for the device
        :param transpose: Show direct output from the image sensor, WxH instead of HxW
        :return: Image frame array
        """
        frame_left = self.finger_left.get_frame()
        frame_right = self.finger_right.get_frame()
        frame_bottom = self.finger_bottom.get_frame()

        return frame_left, frame_right, frame_bottom

    def show_fingers_view(self):
        """
        Creates OpenCV named window with live view of Finger device, ESC to close window
        :param ref_frame: Specify reference frame to show image difference
        :return: None
        """

        while True:

            left, right, bottom = self.get_frames()

            cv2.imshow("Hand View", np.concatenate((left, right, bottom), axis=1))

            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()


class HandROSPublisher(Hand):

    def __init__(self, dev_names=None, fix=((0, 0), (0, 0), (0, 0))):
        """
        Finger Device class for a single Finger
        :param serial: Finger device serial
        :param name: Human friendly identifier name for the device
        """

        if dev_names is None:
            dev_names = [2, 0, 4]

        Hand.__init__(self, dev_names=dev_names, fix=fix)
        self.init_hand()

        self._cv_bridge = CvBridge()

        self._topic_names, self._image_publishers, self._frame_ids = [], [], []
        self._rate = rospy.get_param('~publish_rate', self.finger_left.fps)

        for i in dev_names:
            self._topic_names.append('allsight{}/usb_cam/image_raw'.format(i))
            rospy.loginfo("(topic_name) Publishing Images to topic {}".format(self._topic_names[-1]))

            self._image_publishers.append(rospy.Publisher(self._topic_names[-1], Image, queue_size=1))

            rospy.loginfo("(publish_rate) Publish rate set to %s hz", self._rate)

            self._frame_ids.append('finger{}'.format(dev_names))
            rospy.loginfo("(frame_id) Frame ID set to  %s", self._frame_ids[-1])

    def run(self):
        # One thread that publish all
        ros_rate = rospy.Rate(self._rate)
        while not rospy.is_shutdown():
            for i, cv_image in enumerate(self.get_frames()):
                try:
                    if cv_image is not None:
                        ros_msg = self._cv_bridge.cv2_to_imgmsg(cv_image, "bgr8")
                        ros_msg.header.frame_id = self._frame_ids[i]
                        ros_msg.header.stamp = rospy.Time.now()
                        self._image_publishers[i].publish(ros_msg)
                    else:
                        rospy.loginfo("[%s] Invalid image file", self._topic_names[i])
                    ros_rate.sleep()

                except CvBridgeError as e:
                    rospy.logerr(e)


if __name__ == "__main__":
    import os

    pc_name = os.getlogin()

    tactile = HandROSSubscriberFinger()

    tactile.show_fingers_view()
