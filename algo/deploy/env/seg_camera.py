import rospy

from sensor_msgs.msg import Image
from algo.deploy.env.env_utils.deploy_utils import msg_to_pil, image_msg_to_numpy
import numpy
import cv2
from algo.models.FastSAM.fastsam import FastSAM, FastSAMPrompt


class SegCameraSubscriber:

    def __init__(self, topic='/zedm/zed_node/depth/depth_registered', display=True, device='cuda:0'):
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
        self.dis_noise = 0.00
        self.display = display
        self.init_success = False
        self.query = [89, 69]
        self.device = device
        self._topic_name = rospy.get_param('~topic_name', '{}'.format(topic))
        rospy.loginfo("(topic_name) Subscribing to Images to topic  %s", self._topic_name)
        self._image_subscriber = rospy.Subscriber(self._topic_name, Image, self.image_callback, queue_size=2)
        self._check_camera_ready()
        self.model = FastSAM('/home/roblab20/osher3_workspace/src/isaacgym/python/'
                             'IsaacGymInsertion/isaacgyminsertion/outputs/weights/FastSAM-x.pt')

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
                # init model overrides
                _ = self.model.predict(source=self.last_frame,
                                       device=self.device,
                                       retina_masks=True,
                                       imgsz=[480, 640],
                                       conf=0.4,
                                       iou=0.9,
                                       first=True)
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
            results = self.model.predict(source=frame, device=self.device)
            prompt_process = FastSAMPrompt(frame, results, device=self.device)
            ann = prompt_process.point_prompt(points=[self.query], pointlabel=[1])[0]
            mask_3d = numpy.repeat(ann[:, :, numpy.newaxis], 3, axis=2)

            self.last_frame = frame * mask_3d

            if self.display:
                cv2.imshow("Depth Image", frame.transpose(1, 2, 0) + 0.5)
                key = cv2.waitKey(1)

    def get_frame(self):

        return self.last_frame


if __name__ == "__main__":

    rospy.init_node('Seg')
    tactile = SegCameraSubscriber()
    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        rate.sleep()
