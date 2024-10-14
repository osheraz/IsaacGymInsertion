import rospy

from sensor_msgs.msg import Image
from algo.deploy.env.env_utils.deploy_utils import image_msg_to_numpy
import numpy
import cv2


class ZedCameraSubscriber:

    def __init__(self, topic='/zedm/zed_node/depth/depth_registered', with_seg=False, display=False):
        """
        Finger Device class for a single Finger
        :param serial: Finger device serial
        :param name: Human friendly identifier name for the device
        """
        self.w = 320 # 320
        self.h = 180 # 180
        self.cam_type = 'd'
        self.far_clip = 0.5
        self.near_clip = 0.1
        self.dis_noise = 0.00
        self.display = display
        self.zed_init = False
        self.init_success = False
        self.with_seg = with_seg
        self.with_socket = False

        if with_seg:
            from algo.deploy.env.seg_camera import SegCameraSubscriber
            self.seg = SegCameraSubscriber(with_socket=self.with_socket)
            self.socket_id = self.seg.socket_id
            self.plug_id = self.seg.plug_id
            self.distinct = True

        self._topic_name = rospy.get_param('~topic_name', '{}'.format(topic))
        rospy.loginfo("(topic_name) Subscribing to Images to topic  %s", self._topic_name)
        self._check_camera_ready()
        if with_seg:
            self._check_seg_ready()

        self._image_subscriber = rospy.Subscriber(self._topic_name, Image, self.image_callback, queue_size=2)

    def _check_seg_ready(self):
        print('Waiting for SAM to init')
        while not self.seg.init_success and not rospy.is_shutdown():
            self.init_success &= self.seg.init_success
        print('SAM is ready')

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
                self.zed_init = True
                self.last_frame = image_msg_to_numpy(self.last_frame)
                self.last_frame = numpy.expand_dims(self.last_frame, axis=0)
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
            proc_frame = self.process_depth_image(frame)

            try:
                if self.with_seg:

                    seg = self.seg.get_frame()
                    if seg is not None:

                        plug_mask = (seg == self.plug_id).astype(float)
                        socket_mask = (seg == self.socket_id).astype(float)

                        plug_mask = self.seg.shrink_mask(plug_mask)
                        socket_mask = self.seg.shrink_mask(socket_mask)

                        if self.with_socket:
                            mask = plug_mask | socket_mask
                        else:
                            mask = plug_mask

                        frame = proc_frame * numpy.expand_dims(mask, axis=0)
                        self.seg_frame = seg if self.distinct else mask

                    self.last_frame = frame

            except Exception as e:
                print(e)

            if self.display:
                cv2.imshow("Depth Image", proc_frame.transpose(1, 2, 0))

                cv2.imshow("Test Depth Image", self.last_frame.transpose(1, 2, 0))
                key = cv2.waitKey(1)

    def get_frame(self):

        seg_frame = cv2.resize(self.seg_frame.astype(float), (320, 180), interpolation=cv2.INTER_NEAREST)
        last_frame = numpy.expand_dims(cv2.resize(self.last_frame[0], (320, 180), interpolation=cv2.INTER_AREA), axis=0)

        return last_frame, seg_frame

    def process_depth_image(self, depth_image):
        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        # depth_image += self.dis_noise * 2 * (numpy.random.random(1) - 0.5)[0]
        depth_image = numpy.clip(depth_image, -self.far_clip, -self.near_clip)
        # depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.near_clip) / (self.far_clip - self.near_clip)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image


if __name__ == "__main__":

    rospy.init_node('Zed')
    tactile = ZedCameraSubscriber(display=True)
    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        rate.sleep()
