import numpy as np
import cv2
from algo.deploy.env.finger_ros import TactileSubscriberFinger
from algo.deploy.env.hand import Hand
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy
from algo.deploy.env.env_utils.img_utils import ContactArea, circle_mask, align_center


class HandROSSubscriberFinger():

    def __init__(self, dev_names=None, fix=None):
        """
        Finger Device class for a single Finger
        :param serial: Finger device serial
        :param name: Human friendly identifier name for the device
        """

        # if dev_name == 4:  # left
        #     fix = (2, 12)
        # elif dev_name == 0:  # bottom
        #     fix = (8, 10)
        # elif dev_name == 2:  # right
        #     fix = (15, 15)

        if dev_names is None:
            dev_names = [4, 2, 0]  # left, right, bottom
        if fix is None:
            fix = [(), (), ()]
            fix[0] = (2, 12)  # 5 5
            fix[1] = (15, 15)
            fix[2] = (8, 10)

        self.finger_left = TactileSubscriberFinger(dev_name=dev_names[0], fix=fix[0])
        self.finger_right = TactileSubscriberFinger(dev_name=dev_names[1], fix=fix[1])
        self.finger_bottom = TactileSubscriberFinger(dev_name=dev_names[2], fix=fix[2])
        self.init_success = True
        self.mask_resized = None
        self.left_bg, self.right_bg, self.bottom_bg = self.get_frames(diff=False)

    def get_frames(self, diff=True):
        """
        Returns a single image frame for the device
        :param transpose: Show direct output from the image sensor, WxH instead of HxW
        :return: Image frame array
        """

        left = self.finger_left.get_frame()
        right = self.finger_right.get_frame()
        bottom = self.finger_bottom.get_frame()

        min_width = min(left.shape[1], right.shape[1], bottom.shape[1])
        min_height = min(left.shape[0], right.shape[0], bottom.shape[0])

        if self.mask_resized is None:
            self.mask_resized = circle_mask((min_width, min_height))

        left = cv2.resize(left, (min_width, min_height))
        right = cv2.resize(right, (min_width, min_height))
        bottom = cv2.resize(bottom, (min_width, min_height))

        if diff:
            left = self._subtract_bg(left, self.left_bg) * self.mask_resized
            right = self._subtract_bg(right, self.right_bg) * self.mask_resized
            bottom = self._subtract_bg(bottom, self.bottom_bg) * self.mask_resized

        return left, right, bottom

    def _subtract_bg(self, img1, img2, offset=0.5):

        img1 = np.int32(img1)
        img2 = np.int32(img2)
        diff = img1 - img2
        diff = diff / 255.0 + offset
        return diff

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
    rospy.init_node('TACTILE')

    pc_name = os.getlogin()

    tactile = HandROSSubscriberFinger()
    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        tactile.show_fingers_view()

