import rospy

from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener
# from algo.deploy.env.env_utils.transforms import transform_images
from algo.deploy.env.env_utils.deploy_utils import msg_to_pil, image_msg_to_numpy
import numpy
import cv2



class ZedCameraSubscriber:

    def __init__(self, topic='/zedm/zed_node/depth/depth_registered', display=False):
        """
        Finger Device class for a single Finger
        :param serial: Finger device serial
        :param name: Human friendly identifier name for the device
        """
        self.w = 320
        self.h = 180
        self.cam_type = 'd'
        self.far_clip = 1.0
        self.near_clip = 0.0
        self.dis_noise = 0.001
        self.display = display
        self.init_success = False

        self._topic_name = rospy.get_param('~topic_name', '{}'.format(topic))
        rospy.loginfo("(topic_name) Subscribing to Images to topic  %s", self._topic_name)
        self._image_subscriber = rospy.Subscriber(self._topic_name, Image, self.image_callback, queue_size=2)
        self._check_camera_ready()

    def _check_camera_ready(self):

        self.last_frame = None
        rospy.logdebug(
            "Waiting for '{}' to be READY...".format(self._topic_name))
        while self.last_frame is None and not rospy.is_shutdown():
            try:
                self.last_frame = rospy.wait_for_message(
                    '{}'.format(self._topic_name), Image, timeout=5.0)
                rospy.logdebug(
                    "Current '{}' READY=>".format(self._topic_name))
                self.init_success = True
                self.start_time = rospy.get_time()
            except:
                rospy.logerr(
                    "Current '{}' not ready yet, retrying for getting image".format(self._topic_name))
        return self.last_frame

    def image_callback(self, msg):
        try:
            frame = image_msg_to_numpy(msg)
        except Exception as e:
            print(e)
        else:
            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
            frame = numpy.expand_dims(frame, axis=0)
            frame = self.process_depth_image(frame)
            self.last_frame = frame

            if self.display:
                cv2.imshow("Depth Image", frame.transpose(1, 2, 0) + 0.5)
                key = cv2.waitKey(1)

    def get_frame(self):

        return self.last_frame

    def process_depth_image(self, depth_image):
        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        depth_image += self.dis_noise * 2 * (numpy.random.random(1) - 0.5)[0]
        depth_image = numpy.clip(depth_image, -self.far_clip, -self.near_clip)
        # depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.near_clip) / (self.far_clip - self.near_clip) - 0.5
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image

if __name__ == "__main__":

    rospy.init_node('Finger')
    tactile = ZedCameraSubscriber()
    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        rate.sleep()